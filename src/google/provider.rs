//! The Google provider: Claude Code's Messages requests answered by Gemini models through
//! Google's Cloud Code Assist backend.
//!
//! One task per `/v1/messages` request parses the request, opens the upstream call (retrying
//! what is worth retrying) and hands the HTTP head to the waiting handler through a oneshot.
//! A streamed answer then flows through an `mpsc` channel that is the response body; a
//! non-streamed one is collected and sent as one JSON.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::header::{
    HeaderValue, AUTHORIZATION, CACHE_CONTROL, CONTENT_TYPE, RETRY_AFTER, USER_AGENT,
};
use hyper::http::uri::{Authority, Scheme};
use hyper::{Method, Request, Response, StatusCode, Uri};
use serde::Deserialize;
use tokio::sync::{mpsc, oneshot};

use super::auth::{CredentialSource, GoogleAuth, GoogleEndpoints};
use super::client::GoogleClient;
use super::request::{build_request, thinking_level_for, RequestOptions, SchemaMode};
use super::response::{GeminiTranslator, GeminiUsage, Outcome, SseSplitter};
use super::signatures::SignatureStore;
use crate::anthropic::stream::{MessageWriter, StopReason};
use crate::anthropic::types::MessagesRequest;
use crate::provider::{
    json_error, json_response, Provider, ProxyRequest, RespBody, UpstreamClient,
};
use crate::usage::{UsageLog, UsageRecord};

/// Signature given to thinking blocks that Gemini did not sign.
const PLACEHOLDER_SIGNATURE: &str = "claude-mr-unsigned";
/// How many times an empty or malformed answer is requested again.
const EMPTY_RETRIES: u32 = 2;
/// The pause before the single retry of a 500 or 503.
const SERVER_ERROR_PAUSE: Duration = Duration::from_secs(1);
/// Capacity of the channel that carries a streamed answer to the client.
const CHANNEL_CAPACITY: usize = 16;
/// Words in a 400 body that point at the tool schema.
const SCHEMA_MARKERS: [&str; 4] = [
    "parameters_json_schema",
    "parametersJsonSchema",
    "function_declarations",
    "Invalid JSON payload",
];

/// Settings of the Google provider.
pub struct GoogleConfig {
    /// Where the login of the Antigravity CLI is read from.
    pub credentials: CredentialSource,
    /// The OAuth client and `User-Agent` (see [`super::client::resolve`]), or the
    /// explanation why they are missing, which every request then answers with.
    pub client: Result<GoogleClient, String>,
    /// The Google endpoints.
    pub endpoints: GoogleEndpoints,
    /// Upper bound for the answer's tokens.
    pub max_output_tokens: u32,
    /// How many thought signatures are kept.
    pub signature_capacity: usize,
}

impl GoogleConfig {
    /// Defaults around `client`: the Antigravity keyring entry, Google's real endpoints,
    /// 65536 output tokens, 4096 signatures.
    pub fn new(client: Result<GoogleClient, String>) -> Self {
        GoogleConfig {
            credentials: CredentialSource::antigravity(),
            client,
            endpoints: GoogleEndpoints::default(),
            max_output_tokens: 65536,
            signature_capacity: 4096,
        }
    }
}

/// Serves Claude Code's requests with Gemini models.
pub struct GoogleProvider {
    auth: Result<GoogleAuth, String>,
    store: SignatureStore,
    client: UpstreamClient,
    generate_url: Option<Uri>,
    max_output_tokens: u32,
    usage: Arc<UsageLog>,
}

/// An upstream or local failure in Anthropic's terms.
struct Failure {
    status: StatusCode,
    kind: &'static str,
    message: String,
    retry_after: Option<HeaderValue>,
}

impl Failure {
    fn new(status: StatusCode, kind: &'static str, message: impl Into<String>) -> Self {
        Failure {
            status,
            kind,
            message: message.into(),
            retry_after: None,
        }
    }

    fn upstream(message: impl std::fmt::Display) -> Self {
        Failure::new(
            StatusCode::BAD_GATEWAY,
            "api_error",
            format!("router upstream failed: {message}"),
        )
    }

    fn response(&self) -> Response<RespBody> {
        let mut resp = json_error(self.status, self.kind, &self.message);
        if let Some(v) = &self.retry_after {
            resp.headers_mut().insert(RETRY_AFTER, v.clone());
        }
        resp
    }
}

/// What the handler waiting on a request receives.
enum Reply {
    Stream(mpsc::Receiver<Bytes>),
    Json(Bytes),
    Fail(Failure),
}

/// Corrections learned from Google's 400 answers; they persist across the retries of a request.
#[derive(Clone, Copy, Default)]
struct Flags {
    skip_signatures: bool,
    cleaned_schema: bool,
}

/// The figures of one request, written to the usage log and to stderr.
struct Rec {
    status: u16,
    usage: GeminiUsage,
    stop: Option<&'static str>,
    error: Option<String>,
    ttft: Option<Duration>,
}

impl Rec {
    fn new() -> Self {
        Rec {
            status: 200,
            usage: GeminiUsage::default(),
            stop: None,
            error: None,
            ttft: None,
        }
    }

    fn failed(f: &Failure) -> Self {
        Rec {
            status: f.status.as_u16(),
            error: Some(f.message.clone()),
            ..Rec::new()
        }
    }
}

#[derive(Deserialize)]
struct GoogleError {
    error: Option<GoogleErrorBody>,
}

#[derive(Deserialize)]
struct GoogleErrorBody {
    message: Option<String>,
}

/// Google's own message of an error body, or the body's start when it is not Google's shape.
fn google_message(body: &[u8]) -> String {
    if let Ok(GoogleError {
        error: Some(GoogleErrorBody {
            message: Some(m), ..
        }),
    }) = serde_json::from_slice(body)
    {
        return m;
    }
    let text = String::from_utf8_lossy(body);
    let text = text.trim();
    if text.is_empty() {
        "no message".to_owned()
    } else {
        text.chars().take(300).collect()
    }
}

enum Sink {
    Sse(mpsc::Sender<Bytes>),
    Collect,
}

impl GoogleProvider {
    /// A provider that reads the user's login on the first request.
    pub fn new(cfg: GoogleConfig, client: UpstreamClient, usage: Arc<UsageLog>) -> Self {
        let generate_url = generate_url(&cfg.endpoints.code_assist);
        let auth = cfg
            .client
            .map(|g| GoogleAuth::new(cfg.credentials, cfg.endpoints, client.clone(), g));
        GoogleProvider {
            auth,
            store: SignatureStore::new(cfg.signature_capacity),
            client,
            generate_url,
            max_output_tokens: cfg.max_output_tokens,
            usage,
        }
    }

    /// Opens the upstream call for `req`. Retries a 401 once with a new token, a 400 about
    /// signatures or the tool schema once each with the matching correction, and a 500 or 503
    /// once after a pause. Returns the response body and the renamed tools of the translation.
    async fn connect(
        &self,
        req: &MessagesRequest<'_>,
        flags: &mut Flags,
    ) -> Result<(Incoming, Vec<(String, String)>), Failure> {
        let url = self.generate_url.clone().ok_or_else(|| {
            Failure::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
                "router: bad Google endpoint",
            )
        })?;
        let auth = self.auth.as_ref().map_err(|why| {
            Failure::new(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                why.clone(),
            )
        })?;
        let mut unauthorized = false;
        let mut server_error = false;
        loop {
            let project = auth.project().await.map_err(Failure::upstream)?;
            let token = auth.access_token().await.map_err(Failure::upstream)?;
            let mut bearer = HeaderValue::from_str(&format!("Bearer {token}"))
                .map_err(|_| Failure::upstream("bad access token"))?;
            bearer.set_sensitive(true);
            let opts = RequestOptions {
                project: &project,
                model: &req.model,
                thinking_level: thinking_level_for(&req.model),
                schema_mode: if flags.cleaned_schema {
                    SchemaMode::Cleaned
                } else {
                    SchemaMode::JsonSchema
                },
                skip_signatures: flags.skip_signatures,
                max_output_tokens: self.max_output_tokens,
            };
            let translated = build_request(req, &opts, |id| self.store.get(id))
                .map_err(|e| Failure::new(StatusCode::BAD_REQUEST, "invalid_request_error", e.0))?;
            let http = Request::builder()
                .method(Method::POST)
                .uri(url.clone())
                .header(AUTHORIZATION, bearer)
                .header(CONTENT_TYPE, "application/json")
                .header(USER_AGENT, auth.user_agent().clone())
                .body(Full::new(Bytes::from(translated.body)))
                .map_err(Failure::upstream)?;
            let resp = self.client.request(http).await.map_err(Failure::upstream)?;
            let status = resp.status();
            if status.is_success() {
                return Ok((resp.into_body(), translated.renamed_tools));
            }
            let retry_after = resp.headers().get(RETRY_AFTER).cloned();
            let raw = resp
                .into_body()
                .collect()
                .await
                .map(|c| c.to_bytes())
                .unwrap_or_default();
            let text = String::from_utf8_lossy(&raw);
            let message = google_message(&raw);
            match status.as_u16() {
                401 if !unauthorized => {
                    unauthorized = true;
                    auth.invalidate().await;
                }
                401 => {
                    return Err(Failure::new(status, "authentication_error", message));
                }
                400 if !flags.skip_signatures && text.contains("thought_signature") => {
                    flags.skip_signatures = true;
                }
                400 if !flags.cleaned_schema
                    && !req.tools.is_empty()
                    && SCHEMA_MARKERS.iter().any(|m| text.contains(m)) =>
                {
                    flags.cleaned_schema = true;
                }
                400 => {
                    return Err(Failure::new(status, "invalid_request_error", message));
                }
                403 => return Err(Failure::new(status, "permission_error", message)),
                404 => return Err(Failure::new(status, "not_found_error", message)),
                429 => {
                    let mut f = Failure::new(status, "rate_limit_error", message);
                    f.retry_after = retry_after;
                    return Err(f);
                }
                s if s >= 500 && !server_error => {
                    server_error = true;
                    tokio::time::sleep(SERVER_ERROR_PAUSE).await;
                }
                503 => {
                    let overloaded = StatusCode::from_u16(529).unwrap_or(status);
                    return Err(Failure::new(overloaded, "overloaded_error", message));
                }
                s if s >= 500 => {
                    return Err(Failure::new(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "api_error",
                        message,
                    ));
                }
                _ => return Err(Failure::new(status, "api_error", message)),
            }
        }
    }

    /// Writes the bytes the writer holds to the sink; false when the client has gone.
    async fn flush(
        sink: &Sink,
        writer: &mut MessageWriter,
        received: Instant,
        ttft: &mut Option<Duration>,
    ) -> bool {
        let Sink::Sse(tx) = sink else { return true };
        let Some(bytes) = writer.take() else {
            return true;
        };
        if tx.send(bytes).await.is_err() {
            return false;
        }
        ttft.get_or_insert_with(|| received.elapsed());
        true
    }

    /// Serves one `/v1/messages` request to its end and returns the figures for the log.
    async fn run(&self, body: Bytes, received: Instant, reply: oneshot::Sender<Reply>) -> Rec {
        let req: MessagesRequest = match serde_json::from_slice(&body) {
            Ok(r) => r,
            Err(e) => {
                let f = Failure::new(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    format!("invalid request body: {e}"),
                );
                let rec = Rec::failed(&f);
                let _ = reply.send(Reply::Fail(f));
                return rec;
            }
        };
        let model = req.model.as_ref();
        let mut flags = Flags::default();
        let (mut upstream, mut renamed) = match self.connect(&req, &mut flags).await {
            Ok(c) => c,
            Err(f) => {
                let rec = Rec::failed(&f);
                let _ = reply.send(Reply::Fail(f));
                return rec;
            }
        };
        let mut rec = Rec::new();
        let mut reply = Some(reply);
        let sink = if req.stream {
            let (tx, rx) = mpsc::channel(CHANNEL_CAPACITY);
            let delivered = reply.take().map(|r| r.send(Reply::Stream(rx)));
            if !matches!(delivered, Some(Ok(()))) {
                rec.status = 499;
                return rec;
            }
            Sink::Sse(tx)
        } else {
            Sink::Collect
        };

        let mut attempt = 0;
        loop {
            let mut writer = match sink {
                Sink::Sse(_) => MessageWriter::sse(model),
                Sink::Collect => MessageWriter::collect(model),
            }
            .with_placeholder_signature(PLACEHOLDER_SIGNATURE);
            let mut translator = GeminiTranslator::new(&renamed, &self.store);
            let mut splitter = SseSplitter::new();
            let mut closed = false;
            let mut broken: Option<String> = None;
            loop {
                let next = match &sink {
                    Sink::Sse(tx) => tokio::select! {
                        f = upstream.frame() => Some(f),
                        () = tx.closed() => None,
                    },
                    Sink::Collect => Some(upstream.frame().await),
                };
                match next {
                    None => {
                        closed = true;
                        break;
                    }
                    Some(None) => break,
                    Some(Some(Err(e))) => {
                        broken = Some(format!("upstream stream failed: {e}"));
                        break;
                    }
                    Some(Some(Ok(frame))) => {
                        let Some(data) = frame.data_ref() else {
                            continue;
                        };
                        splitter.feed(data, |ev| {
                            // A chunk that does not parse is skipped; the stream's end decides.
                            let _ = translator.on_event(ev, &mut writer);
                        });
                        if !Self::flush(&sink, &mut writer, received, &mut rec.ttft).await {
                            closed = true;
                            break;
                        }
                    }
                }
            }
            if closed {
                rec.status = 499;
                rec.usage = translator.usage();
                return rec;
            }
            if let Some(msg) = broken {
                rec.usage = translator.usage();
                return Self::finish_error(
                    &sink,
                    reply,
                    writer,
                    rec,
                    received,
                    Failure::new(StatusCode::BAD_GATEWAY, "api_error", msg),
                )
                .await;
            }
            splitter.finish(|ev| {
                let _ = translator.on_event(ev, &mut writer);
            });
            let outcome = translator.finish(&mut writer);
            rec.usage = translator.usage();
            let stop = translator.stop_reason().map(StopReason::as_str);
            let failure = match outcome {
                Outcome::Done => {
                    if !Self::flush(&sink, &mut writer, received, &mut rec.ttft).await {
                        rec.status = 499;
                        return rec;
                    }
                    rec.stop = stop;
                    if let Some(reply) = reply {
                        let _ = reply.send(match writer.into_json() {
                            Ok(json) => Reply::Json(json),
                            Err((kind, msg)) => Reply::Fail(Failure::new(
                                StatusCode::BAD_GATEWAY,
                                "api_error",
                                format!("{kind}: {msg}"),
                            )),
                        });
                    }
                    return rec;
                }
                Outcome::Empty | Outcome::Malformed if attempt < EMPTY_RETRIES => {
                    attempt += 1;
                    drop(translator);
                    match self.connect(&req, &mut flags).await {
                        Ok((u, r)) => {
                            upstream = u;
                            renamed = r;
                            continue;
                        }
                        Err(f) => f,
                    }
                }
                Outcome::Empty => Failure::new(
                    StatusCode::BAD_GATEWAY,
                    "api_error",
                    "empty response from Gemini",
                ),
                Outcome::Malformed => Failure::new(
                    StatusCode::BAD_GATEWAY,
                    "api_error",
                    "malformed function call",
                ),
            };
            return Self::finish_error(&sink, reply, writer, rec, received, failure).await;
        }
    }

    /// Ends a request with an error: an SSE `error` event for a stream that is already open, a
    /// JSON error for a collected answer.
    async fn finish_error(
        sink: &Sink,
        reply: Option<oneshot::Sender<Reply>>,
        mut writer: MessageWriter,
        mut rec: Rec,
        received: Instant,
        failure: Failure,
    ) -> Rec {
        rec.error = Some(failure.message.clone());
        match reply {
            Some(reply) => {
                rec.status = failure.status.as_u16();
                let _ = reply.send(Reply::Fail(failure));
            }
            None => {
                writer.error(failure.kind, &failure.message);
                if !Self::flush(sink, &mut writer, received, &mut rec.ttft).await {
                    rec.status = 499;
                }
            }
        }
        rec
    }

    fn log(&self, model: &str, received: Instant, rec: &Rec) {
        let latency = received.elapsed();
        let g = rec.usage;
        self.usage.record(&UsageRecord {
            provider: "google",
            model,
            status: rec.status,
            input_tokens: g.prompt.saturating_sub(g.cached),
            cache_read_tokens: g.cached,
            output_tokens: g.candidates,
            thinking_tokens: g.thoughts,
            latency_ms: millis(latency),
            ttft_ms: rec.ttft.map(millis),
            stop_reason: rec.stop,
            error: rec.error.as_deref(),
        });
        eprintln!(
            "router: POST /v1/messages model={model} side=google status={} in={} cache={} out={} think={} ms={}",
            rec.status,
            g.prompt.saturating_sub(g.cached),
            g.cached,
            g.candidates,
            g.thoughts,
            millis(latency)
        );
    }

    async fn messages(self: Arc<Self>, req: ProxyRequest) -> Response<RespBody> {
        let ProxyRequest {
            body,
            received,
            model,
            ..
        } = req;
        let model = model.unwrap_or_else(|| "None".to_owned());
        let (tx, rx) = oneshot::channel();
        let me = Arc::clone(&self);
        tokio::spawn(async move {
            let rec = me.run(body, received, tx).await;
            me.log(&model, received, &rec);
        });
        match rx.await {
            Ok(Reply::Stream(chan)) => {
                let mut resp = Response::new(RespBody::channel(chan));
                let h = resp.headers_mut();
                h.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
                h.insert(CACHE_CONTROL, HeaderValue::from_static("no-cache"));
                resp
            }
            Ok(Reply::Json(json)) => json_response(StatusCode::OK, json),
            Ok(Reply::Fail(f)) => f.response(),
            Err(_) => json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
                "router: the request task ended without an answer",
            ),
        }
    }
}

fn millis(d: Duration) -> u64 {
    u64::try_from(d.as_millis()).unwrap_or(u64::MAX)
}

fn generate_url(base: &Uri) -> Option<Uri> {
    let scheme: Scheme = base.scheme()?.clone();
    let authority: Authority = base.authority()?.clone();
    let prefix = base.path().trim_end_matches('/');
    Uri::builder()
        .scheme(scheme)
        .authority(authority)
        .path_and_query(format!("{prefix}/v1internal:streamGenerateContent?alt=sse"))
        .build()
        .ok()
}

impl Provider for GoogleProvider {
    fn name(&self) -> &'static str {
        "google"
    }

    fn handle(
        self: Arc<Self>,
        req: ProxyRequest,
    ) -> Pin<Box<dyn Future<Output = Response<RespBody>> + Send>> {
        Box::pin(async move {
            let path = req.parts.uri.path();
            if req.parts.method != Method::POST {
                return json_error(StatusCode::NOT_FOUND, "not_found_error", "unknown route");
            }
            match path {
                "/v1/messages" => self.messages(req).await,
                "/v1/messages/count_tokens" => {
                    // An estimate of four bytes per token over the request body; Claude Code
                    // only uses it to size its context.
                    let tokens = (req.body.len() / 4).max(1);
                    json_response(
                        StatusCode::OK,
                        Bytes::from(format!(r#"{{"input_tokens":{tokens}}}"#)),
                    )
                }
                _ => json_error(StatusCode::NOT_FOUND, "not_found_error", "unknown route"),
            }
        })
    }
}
