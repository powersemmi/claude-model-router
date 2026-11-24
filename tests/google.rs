//! The Google provider against a fake Google: token endpoint, project lookup and
//! `streamGenerateContent`, scripted per test.

use std::collections::VecDeque;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytes::Bytes;
use claude_model_router::google::auth::{CredentialSource, GoogleEndpoints};
use claude_model_router::google::client::{ClientCandidates, GoogleClient, OAuthClient};
use claude_model_router::google::provider::{GoogleConfig, GoogleProvider};
use claude_model_router::passthrough::{PassAuth, Passthrough};
use claude_model_router::provider::upstream_client;
use claude_model_router::routes::Routes;
use claude_model_router::serve;
use claude_model_router::usage::UsageLog;
use http_body_util::{BodyExt, Full};
use hyper::{Request, Response, StatusCode, Uri};
use tokio::io::AsyncWriteExt;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, Notify};
use tokio::time::timeout;

use super::{read_until, recording, spawn_fake, ChanBody, Handler, Log, LIMIT, TB};

const MODEL: &str = "gemini-4-neon-medium";

const THOUGHT: &str = r#"data: {"response": {"candidates": [{"content": {"role": "model","parts": [{"thought": true,"text": "Reading the file.\n"}]}}],"usageMetadata": {"promptTokenCount": 10408,"totalTokenCount": 10535,"thoughtsTokenCount": 127},"modelVersion": "gemini-4-neon","responseId": "sI8F"},"traceId": "eff9","metadata": {}}"#;
const CALL: &str = r#"data: {"response": {"candidates": [{"content": {"role": "model","parts": [{"thoughtSignature": "ErsFCrgFAWkUfRN4","functionCall": {"name": "Read","args": {"file_path": "/x/note.txt"},"id": "call_1343"}}]}}],"usageMetadata": {"promptTokenCount": 10408,"candidatesTokenCount": 78,"totalTokenCount": 10655,"thoughtsTokenCount": 169},"modelVersion": "gemini-4-neon","responseId": "sI8F"},"traceId": "eff9","metadata": {}}"#;
const STOP: &str = r#"data: {"response": {"candidates": [{"content": {"role": "model","parts": [{"text": ""}]},"finishReason": "STOP"}],"usageMetadata": {"promptTokenCount": 10408,"candidatesTokenCount": 81,"totalTokenCount": 10658,"thoughtsTokenCount": 169,"cachedContentTokenCount": 8351},"modelVersion": "gemini-4-neon","responseId": "sI8F"},"traceId": "eff9","metadata": {}}"#;

fn full_stream() -> String {
    format!("{THOUGHT}\n\n{CALL}\n\n{STOP}\n\n")
}

/// What the fake answers to the next `streamGenerateContent`.
enum Step {
    Body(String),
    Status(u16, String, Option<&'static str>),
    /// Sends the first chunk, then holds the stream open and notes when it is dropped.
    Hang(Arc<Notify>),
}

#[derive(Default)]
struct FakeGoogle {
    script: Mutex<VecDeque<Step>>,
    bodies: Mutex<Vec<String>>,
    auth: Mutex<Vec<String>>,
    tokens: AtomicUsize,
}

fn sse_response(body: TB) -> Response<TB> {
    Response::builder()
        .header("content-type", "text/event-stream")
        .body(body)
        .unwrap()
}

fn fake_handler(g: Arc<FakeGoogle>) -> Handler {
    Arc::new(move |req| {
        let g = Arc::clone(&g);
        Box::pin(async move {
            let (parts, body) = req.into_parts();
            let body = body.collect().await.unwrap().to_bytes();
            let json = |s: String| {
                Response::builder()
                    .header("content-type", "application/json")
                    .body(Full::new(Bytes::from(s)).boxed())
                    .unwrap()
            };
            match parts.uri.path() {
                "/token" => {
                    let n = g.tokens.fetch_add(1, Ordering::SeqCst) + 1;
                    json(format!(r#"{{"access_token":"at-{n}","expires_in":3599}}"#))
                }
                "/v1internal:loadCodeAssist" => {
                    json(r#"{"cloudaicompanionProject":"proj-test"}"#.to_owned())
                }
                "/v1internal:streamGenerateContent" => {
                    g.bodies
                        .lock()
                        .unwrap()
                        .push(String::from_utf8(body.to_vec()).unwrap());
                    g.auth.lock().unwrap().push(
                        parts
                            .headers
                            .get("authorization")
                            .and_then(|v| v.to_str().ok())
                            .unwrap_or("")
                            .to_owned(),
                    );
                    let step = g.script.lock().unwrap().pop_front();
                    match step {
                        Some(Step::Body(b)) => sse_response(Full::new(Bytes::from(b)).boxed()),
                        Some(Step::Status(code, b, retry)) => {
                            let mut r = Response::builder()
                                .status(code)
                                .header("content-type", "application/json");
                            if let Some(v) = retry {
                                r = r.header("retry-after", v);
                            }
                            r.body(Full::new(Bytes::from(b)).boxed()).unwrap()
                        }
                        Some(Step::Hang(dropped)) => {
                            let (tx, rx) = mpsc::channel(4);
                            tokio::spawn(async move {
                                let first = format!("{THOUGHT}\n\n");
                                let _ = tx.send(Bytes::from(first)).await;
                                tx.closed().await;
                                dropped.notify_one();
                            });
                            sse_response(ChanBody(rx).boxed())
                        }
                        None => Response::builder()
                            .status(500)
                            .body(Full::new(Bytes::from_static(b"{}")).boxed())
                            .unwrap(),
                    }
                }
                _ => Response::builder()
                    .status(404)
                    .body(Full::new(Bytes::new()).boxed())
                    .unwrap(),
            }
        })
    })
}

struct Rig {
    router: SocketAddr,
    google: Arc<FakeGoogle>,
    usage_path: PathBuf,
    /// Holds the login and usage files; removed when the rig drops.
    _dir: tempfile::TempDir,
}

fn fake_client() -> Result<GoogleClient, String> {
    Ok(GoogleClient {
        candidates: ClientCandidates::single(
            OAuthClient {
                client_id: "1-fakefakefakefakefakefakefakefake.apps.googleusercontent.com"
                    .to_owned(),
                client_secret: "GOCSPX-fakefakefakefakefakefakefake".to_owned(),
            },
            "test",
        ),
        user_agent: "test-agent/1".to_owned(),
    })
}

async fn rig(tag: &str, script: Vec<Step>) -> Rig {
    rig_with(tag, script, fake_client()).await
}

async fn rig_with(
    tag: &str,
    script: Vec<Step>,
    client_values: Result<GoogleClient, String>,
) -> Rig {
    let google = Arc::new(FakeGoogle::default());
    google.script.lock().unwrap().extend(script);
    let g_addr = spawn_fake(fake_handler(Arc::clone(&google))).await;
    let anthropic = spawn_fake(recording("anthropic", Log::default())).await;

    let dir = tempfile::Builder::new()
        .prefix(&format!("cmr-google-{tag}-"))
        .tempdir()
        .unwrap();
    let creds = dir.path().join("creds.json");
    std::fs::write(
        &creds,
        r#"{"token":{"access_token":"x","refresh_token":"rt-test"}}"#,
    )
    .unwrap();
    let usage_path = dir.path().join("usage.jsonl");
    let usage = Arc::new(UsageLog::open(&usage_path).unwrap());

    let client = upstream_client();
    let base = format!("http://{g_addr}");
    let provider = GoogleProvider::new(
        GoogleConfig {
            credentials: CredentialSource::File(creds),
            endpoints: GoogleEndpoints {
                code_assist: base.parse().unwrap(),
                token: format!("{base}/token").parse().unwrap(),
            },
            ..GoogleConfig::new(client_values)
        },
        client.clone(),
        usage,
    );
    let anthropic = Passthrough::new(
        "anthropic",
        format!("http://{anthropic}").parse::<Uri>().unwrap(),
        PassAuth::Forward,
        client,
    )
    .unwrap();
    let mut routes = Routes::new(Arc::new(anthropic));
    routes.add(MODEL, Arc::new(provider));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let router = listener.local_addr().unwrap();
    tokio::spawn(serve(listener, routes));
    Rig {
        router,
        google,
        usage_path,
        _dir: dir,
    }
}

fn body_json(stream: bool, extra_messages: &str) -> String {
    format!(
        r#"{{"model":"{MODEL}","max_tokens":1024,"stream":{stream},"messages":[{{"role":"user","content":"read note.txt"}}{extra_messages}],"tools":[{{"name":"Read","description":"Reads a file","input_schema":{{"type":"object","properties":{{"file_path":{{"type":"string"}}}},"required":["file_path"]}}}}]}}"#
    )
}

const TOOL_TURN: &str = r#",{"role":"assistant","content":[{"type":"tool_use","id":"call_1343","name":"Read","input":{"file_path":"/x/note.txt"}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1343","content":"hello"}]}"#;

/// POSTs `body` to the router; returns status, headers and the whole body text.
async fn post(
    router: SocketAddr,
    path: &str,
    body: String,
) -> (StatusCode, hyper::HeaderMap, String) {
    let req = Request::post(format!("http://{router}{path}"))
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(body)))
        .unwrap();
    timeout(LIMIT, async {
        let resp = upstream_client().request(req).await.unwrap();
        let (parts, body) = resp.into_parts();
        let text = body.collect().await.unwrap().to_bytes();
        (
            parts.status,
            parts.headers,
            String::from_utf8(text.to_vec()).unwrap(),
        )
    })
    .await
    .unwrap()
}

fn done_script() -> Vec<Step> {
    vec![Step::Body(full_stream())]
}

#[tokio::test]
async fn missing_client_values_answer_with_the_instructions() {
    let why = "Google models are unavailable: create /x/google.json".to_owned();
    let rig = rig_with("noclient", done_script(), Err(why)).await;
    let (status, _, text) = post(rig.router, "/v1/messages", body_json(true, "")).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(text.contains("invalid_request_error"), "{text}");
    assert!(text.contains("create /x/google.json"), "{text}");
    assert_eq!(rig.google.tokens.load(Ordering::SeqCst), 0);
    assert!(rig.google.bodies.lock().unwrap().is_empty());
}

#[tokio::test]
async fn streamed_tool_call() {
    let rig = rig("stream", done_script()).await;
    let (status, headers, text) = post(rig.router, "/v1/messages", body_json(true, "")).await;
    assert_eq!(status, 200);
    assert_eq!(headers["content-type"], "text/event-stream");
    for needle in [
        "event: message_start",
        "thinking_delta",
        "Reading the file.",
        "signature_delta",
        r#""id":"call_1343""#,
        r#""name":"Read""#,
        "input_json_delta",
        "/x/note.txt",
        r#""stop_reason":"tool_use""#,
        r#""input_tokens":2057"#,
        r#""cache_read_input_tokens":8351"#,
        "event: message_stop",
    ] {
        assert!(text.contains(needle), "missing {needle:?} in {text}");
    }
    let upstream = rig.google.bodies.lock().unwrap()[0].clone();
    assert!(upstream.contains("proj-test"), "{upstream}");
    assert_eq!(rig.google.auth.lock().unwrap()[0], "Bearer at-1");
}

#[tokio::test]
async fn signature_is_put_back_on_the_next_request() {
    let rig = rig(
        "sig",
        vec![Step::Body(full_stream()), Step::Body(full_stream())],
    )
    .await;
    post(rig.router, "/v1/messages", body_json(true, "")).await;
    let (status, _, _) = post(rig.router, "/v1/messages", body_json(true, TOOL_TURN)).await;
    assert_eq!(status, 200);
    let second = rig.google.bodies.lock().unwrap()[1].clone();
    let v: serde_json::Value = serde_json::from_str(&second).unwrap();
    let parts: Vec<&serde_json::Value> = v["request"]["contents"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|c| c["parts"].as_array().unwrap())
        .collect();
    let call = parts
        .iter()
        .find(|p| p.get("functionCall").is_some())
        .unwrap();
    assert_eq!(call["thoughtSignature"], "ErsFCrgFAWkUfRN4");
    assert_eq!(call["functionCall"]["name"], "Read");
    let resp = parts
        .iter()
        .find(|p| p.get("functionResponse").is_some())
        .unwrap();
    assert_eq!(resp["functionResponse"]["name"], "Read");
}

#[tokio::test]
async fn missing_signature_is_retried_with_the_sentinel() {
    let rig = rig(
        "sentinel",
        vec![
            Step::Status(
                400,
                r#"{"error":{"code":400,"message":"Function call is missing a thought_signature in functionCall parts.","status":"INVALID_ARGUMENT"}}"#.to_owned(),
                None,
            ),
            Step::Body(full_stream()),
        ],
    )
    .await;
    let (status, _, text) = post(rig.router, "/v1/messages", body_json(true, TOOL_TURN)).await;
    assert_eq!(status, 200, "{text}");
    assert!(text.contains("event: message_stop"));
    let bodies = rig.google.bodies.lock().unwrap().clone();
    assert_eq!(bodies.len(), 2);
    assert!(
        bodies[1].contains("skip_thought_signature_validator"),
        "{}",
        bodies[1]
    );
}

#[tokio::test]
async fn empty_stream_is_requested_again() {
    let rig = rig(
        "empty",
        vec![Step::Body(String::new()), Step::Body(full_stream())],
    )
    .await;
    let (status, _, text) = post(rig.router, "/v1/messages", body_json(true, "")).await;
    assert_eq!(status, 200);
    assert!(text.contains(r#""stop_reason":"tool_use""#), "{text}");
    assert_eq!(rig.google.bodies.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn empty_stream_three_times_ends_in_an_error_event() {
    let rig = rig(
        "empty3",
        vec![
            Step::Body(String::new()),
            Step::Body(String::new()),
            Step::Body(String::new()),
        ],
    )
    .await;
    let (status, _, text) = post(rig.router, "/v1/messages", body_json(true, "")).await;
    assert_eq!(status, 200);
    assert!(text.contains("event: error"), "{text}");
    assert!(text.contains("empty response from Gemini"), "{text}");
    assert_eq!(rig.google.bodies.lock().unwrap().len(), 3);
}

#[tokio::test]
async fn unauthorized_refreshes_the_token_once() {
    let rig = rig(
        "401",
        vec![
            Step::Status(401, r#"{"error":{"message":"expired"}}"#.to_owned(), None),
            Step::Body(full_stream()),
        ],
    )
    .await;
    let (status, _, _) = post(rig.router, "/v1/messages", body_json(true, "")).await;
    assert_eq!(status, 200);
    assert_eq!(rig.google.tokens.load(Ordering::SeqCst), 2);
    let auth = rig.google.auth.lock().unwrap().clone();
    assert_eq!(auth, ["Bearer at-1", "Bearer at-2"]);
}

#[tokio::test]
async fn rate_limit_reaches_claude_code() {
    let rig = rig(
        "429",
        vec![Step::Status(
            429,
            r#"{"error":{"code":429,"message":"Quota exhausted","status":"RESOURCE_EXHAUSTED"}}"#
                .to_owned(),
            Some("7"),
        )],
    )
    .await;
    let (status, headers, text) = post(rig.router, "/v1/messages", body_json(true, "")).await;
    assert_eq!(status, 429);
    assert_eq!(headers["retry-after"], "7");
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["type"], "error");
    assert_eq!(v["error"]["type"], "rate_limit_error");
    assert_eq!(v["error"]["message"], "Quota exhausted");
}

#[tokio::test]
async fn non_stream_returns_a_message() {
    let rig = rig("json", done_script()).await;
    let (status, headers, text) = post(rig.router, "/v1/messages", body_json(false, "")).await;
    assert_eq!(status, 200, "{text}");
    assert_eq!(headers["content-type"], "application/json");
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["type"], "message");
    assert_eq!(v["role"], "assistant");
    assert_eq!(v["stop_reason"], "tool_use");
    assert_eq!(v["content"][0]["type"], "thinking");
    assert_eq!(v["content"][1]["type"], "tool_use");
    assert_eq!(v["content"][1]["id"], "call_1343");
    assert_eq!(v["content"][1]["input"]["file_path"], "/x/note.txt");
    assert_eq!(v["usage"]["input_tokens"], 2057);
}

#[tokio::test]
async fn count_tokens_is_answered_locally() {
    let rig = rig("count", vec![]).await;
    let (status, _, text) = post(
        rig.router,
        "/v1/messages/count_tokens",
        body_json(false, ""),
    )
    .await;
    assert_eq!(status, 200);
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert!(v["input_tokens"].as_u64().unwrap() > 0);
    assert!(rig.google.bodies.lock().unwrap().is_empty());
    let (status, _, _) = post(rig.router, "/v1/other", body_json(false, "")).await;
    assert_eq!(status, 404);
}

#[tokio::test]
async fn usage_line_carries_the_figures() {
    let rig = rig("usage", done_script()).await;
    post(rig.router, "/v1/messages", body_json(true, "")).await;
    let text = timeout(LIMIT, async {
        loop {
            let t = std::fs::read_to_string(&rig.usage_path).unwrap_or_default();
            if t.ends_with('\n') {
                return t;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 1);
    let v: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
    assert_eq!(v["provider"], "google");
    assert_eq!(v["model"], MODEL);
    assert_eq!(v["status"], 200);
    assert_eq!(v["input_tokens"], 2057);
    assert_eq!(v["cache_read_tokens"], 8351);
    assert_eq!(v["output_tokens"], 81);
    assert_eq!(v["thinking_tokens"], 169);
    assert_eq!(v["stop_reason"], "tool_use");
    assert!(v["ttft_ms"].is_u64());
}

#[tokio::test]
async fn client_drop_stops_the_upstream_read() {
    let dropped = Arc::new(Notify::new());
    let rig = rig("drop", vec![Step::Hang(Arc::clone(&dropped))]).await;
    timeout(LIMIT, async {
        let mut s = TcpStream::connect(rig.router).await.unwrap();
        let body = body_json(true, "");
        let req = format!(
            "POST /v1/messages HTTP/1.1\r\nhost: x\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
            body.len()
        );
        s.write_all(req.as_bytes()).await.unwrap();
        let mut buf = Vec::new();
        read_until(&mut s, &mut buf, "Reading the file").await;
        drop(s);
        dropped.notified().await;
    })
    .await
    .unwrap();
}
