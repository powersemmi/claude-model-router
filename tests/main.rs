//! Integration tests: the router runs in-process against fake upstreams.
mod cli;
mod google;

use std::convert::Infallible;
use std::future::Future;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;

use bytes::Bytes;
use claude_model_router::passthrough::{PassAuth, Passthrough};
use claude_model_router::provider::{json_error, upstream_client, RespBody};
use claude_model_router::routes::Routes;
use claude_model_router::serve;
use claude_model_router::usage::{UsageLog, UsageRecord};
use http_body_util::combinators::BoxBody;
use http_body_util::{BodyExt, Full};
use hyper::body::{Body, Frame, Incoming};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{HeaderMap, Request, Response, Uri};
use hyper_util::rt::TokioIo;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, Notify};
use tokio::time::timeout;

const LIMIT: Duration = Duration::from_secs(8);
const LOCAL_MODEL: &str = "local-test-model";

type TestBody = BoxBody<Bytes, Infallible>;
type Handler = Arc<
    dyn Fn(Request<Incoming>) -> Pin<Box<dyn Future<Output = Response<TestBody>> + Send>>
        + Send
        + Sync,
>;

struct Seen {
    method: String,
    uri: String,
    headers: HeaderMap,
    body: Bytes,
}

type Log = Arc<Mutex<Vec<Seen>>>;

fn full(s: &'static str) -> TestBody {
    Full::new(Bytes::from_static(s.as_bytes())).boxed()
}

/// Answers every request with `name` as the body and records what it received.
fn recording(name: &'static str, log: Log) -> Handler {
    Arc::new(move |req| {
        let log = Arc::clone(&log);
        Box::pin(async move {
            let (parts, body) = req.into_parts();
            let body = body.collect().await.unwrap().to_bytes();
            log.lock().unwrap().push(Seen {
                method: parts.method.to_string(),
                uri: parts.uri.to_string(),
                headers: parts.headers,
                body,
            });
            Response::builder()
                .header("x-fake", name)
                .body(full(name))
                .unwrap()
        })
    })
}

async fn spawn_fake(handler: Handler) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                return;
            };
            let handler = Arc::clone(&handler);
            tokio::spawn(async move {
                let svc = service_fn(move |req| {
                    let handler = Arc::clone(&handler);
                    async move { Ok::<_, Infallible>(handler(req).await) }
                });
                let _ = http1::Builder::new()
                    .serve_connection(TokioIo::new(stream), svc)
                    .await;
            });
        }
    });
    addr
}

async fn start_router(local: SocketAddr, anthropic: SocketAddr) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let client = upstream_client();
    let uri = |a: SocketAddr| format!("http://{a}").parse::<Uri>().unwrap();
    let anthropic = Passthrough::new(
        "anthropic",
        uri(anthropic),
        PassAuth::Forward,
        client.clone(),
    )
    .unwrap();
    let local = Passthrough::new("local", uri(local), PassAuth::LocalKey, client).unwrap();
    let mut routes = Routes::new(Arc::new(anthropic));
    routes.add(LOCAL_MODEL, Arc::new(local));
    tokio::spawn(serve(listener, routes));
    addr
}

struct Pair {
    router: SocketAddr,
    local_log: Log,
    anthropic_log: Log,
}

async fn pair() -> Pair {
    let local_log = Log::default();
    let anthropic_log = Log::default();
    let local = spawn_fake(recording("local", Arc::clone(&local_log))).await;
    let anthropic = spawn_fake(recording("anthropic", Arc::clone(&anthropic_log))).await;
    Pair {
        router: start_router(local, anthropic).await,
        local_log,
        anthropic_log,
    }
}

/// Sends one request with `Connection: close` and returns (head, body) of the raw reply.
async fn raw(addr: SocketAddr, head: &str, body: &str) -> (String, String) {
    let mut s = TcpStream::connect(addr).await.unwrap();
    let req = format!(
        "{head}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    );
    s.write_all(req.as_bytes()).await.unwrap();
    let mut buf = Vec::new();
    timeout(LIMIT, s.read_to_end(&mut buf))
        .await
        .unwrap()
        .unwrap();
    let text = String::from_utf8_lossy(&buf).into_owned();
    let (h, b) = text.split_once("\r\n\r\n").unwrap();
    (h.to_owned(), b.to_owned())
}

#[tokio::test]
async fn routes_by_model() {
    let p = pair().await;
    let post = "POST /v1/messages HTTP/1.1\r\nhost: x";
    let (h, b) = raw(
        p.router,
        post,
        &format!(r#"{{"model":"{LOCAL_MODEL}","stream":true}}"#),
    )
    .await;
    assert!(h.starts_with("HTTP/1.1 200"), "{h}");
    assert_eq!(b, "local");
    let (_, b) = raw(p.router, post, r#"{"model":"claude-opus","x":[1,2]}"#).await;
    assert_eq!(b, "anthropic");
    let (_, b) = raw(p.router, "GET /api/hello HTTP/1.1\r\nhost: x", "").await;
    assert_eq!(b, "anthropic");
    let (_, b) = raw(p.router, post, "not json").await;
    assert_eq!(b, "anthropic");
    assert_eq!(p.local_log.lock().unwrap().len(), 1);
    assert_eq!(p.anthropic_log.lock().unwrap().len(), 3);
}

#[tokio::test]
async fn headers_path_and_body_forwarded() {
    let p = pair().await;
    let head = "POST /v1/messages?beta=true HTTP/1.1\r\nhost: client.example\r\n\
                authorization: Bearer secret\r\nx-custom: yes";
    let local_body = format!(r#"{{"model":"{LOCAL_MODEL}"}}"#);
    raw(p.router, head, &local_body).await;
    raw(p.router, head, r#"{"model":"claude-opus"}"#).await;

    let local = p.local_log.lock().unwrap();
    let l = &local[0];
    assert_eq!(l.method, "POST");
    assert!(l.uri.ends_with("/v1/messages?beta=true"), "{}", l.uri);
    assert!(l.headers.get("authorization").is_none());
    assert_eq!(l.headers["x-api-key"], "local");
    assert_eq!(l.headers["x-custom"], "yes");
    assert_eq!(l.body, local_body.as_bytes());

    let anthropic = p.anthropic_log.lock().unwrap();
    let a = &anthropic[0];
    assert!(a.uri.ends_with("/v1/messages?beta=true"), "{}", a.uri);
    assert_eq!(a.headers["authorization"], "Bearer secret");
    assert_eq!(a.headers["x-custom"], "yes");
    assert!(a.headers.get("x-api-key").is_none());
    assert_eq!(a.body, br#"{"model":"claude-opus"}"#.as_slice());
}

struct ChanBody(mpsc::Receiver<Bytes>);

impl Body for ChanBody {
    type Data = Bytes;
    type Error = Infallible;
    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, Infallible>>> {
        self.0.poll_recv(cx).map(|o| o.map(|b| Ok(Frame::data(b))))
    }
}

async fn read_until(s: &mut TcpStream, buf: &mut Vec<u8>, needle: &str) {
    let mut chunk = [0u8; 4096];
    while !String::from_utf8_lossy(buf).contains(needle) {
        let n = s.read(&mut chunk).await.unwrap();
        assert!(n > 0, "stream ended before {needle:?}");
        buf.extend_from_slice(&chunk[..n]);
    }
}

fn model_request() -> String {
    let body = format!(r#"{{"model":"{LOCAL_MODEL}"}}"#);
    format!(
        "POST /v1/messages HTTP/1.1\r\nhost: x\r\ncontent-length: {}\r\n\r\n{body}",
        body.len()
    )
}

#[tokio::test]
async fn sse_chunks_arrive_as_they_are_sent() {
    let release = Arc::new(Notify::new());
    let rel = Arc::clone(&release);
    let local: Handler = Arc::new(move |_req| {
        let rel = Arc::clone(&rel);
        Box::pin(async move {
            let (tx, rx) = mpsc::channel(4);
            tokio::spawn(async move {
                let _ = tx.send(Bytes::from_static(b"data: one\n\n")).await;
                rel.notified().await;
                let _ = tx.send(Bytes::from_static(b"data: two\n\n")).await;
            });
            Response::builder()
                .header("content-type", "text/event-stream")
                .body(ChanBody(rx).boxed())
                .unwrap()
        })
    });
    let local = spawn_fake(local).await;
    let anthropic = spawn_fake(recording("anthropic", Log::default())).await;
    let router = start_router(local, anthropic).await;

    timeout(LIMIT, async {
        let mut s = TcpStream::connect(router).await.unwrap();
        s.write_all(model_request().as_bytes()).await.unwrap();
        let mut buf = Vec::new();
        read_until(&mut s, &mut buf, "data: one").await;
        assert!(!String::from_utf8_lossy(&buf).contains("data: two"));
        release.notify_one();
        read_until(&mut s, &mut buf, "data: two").await;
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn head_returns_status_and_headers_without_body() {
    let p = pair().await;
    let (h, b) = raw(p.router, "HEAD /api/hello HTTP/1.1\r\nhost: x", "").await;
    assert!(h.starts_with("HTTP/1.1 200"), "{h}");
    assert!(h.to_lowercase().contains("x-fake: anthropic"), "{h}");
    assert_eq!(b, "");
    assert_eq!(p.anthropic_log.lock().unwrap()[0].method, "HEAD");
}

#[tokio::test]
async fn client_disconnect_cancels_upstream() {
    let closed = Arc::new(Notify::new());
    let sig = Arc::clone(&closed);
    let local: Handler = Arc::new(move |_req| {
        let sig = Arc::clone(&sig);
        Box::pin(async move {
            let (tx, rx) = mpsc::channel(4);
            tokio::spawn(async move {
                if tx
                    .send(Bytes::from_static(b"data: first\n\n"))
                    .await
                    .is_err()
                {
                    return;
                }
                // The receiver is dropped when the upstream connection goes away.
                tx.closed().await;
                sig.notify_one();
            });
            Response::builder().body(ChanBody(rx).boxed()).unwrap()
        })
    });
    let local = spawn_fake(local).await;
    let anthropic = spawn_fake(recording("anthropic", Log::default())).await;
    let router = start_router(local, anthropic).await;

    timeout(LIMIT, async {
        let mut s = TcpStream::connect(router).await.unwrap();
        s.write_all(model_request().as_bytes()).await.unwrap();
        let mut buf = Vec::new();
        read_until(&mut s, &mut buf, "data: first").await;
        drop(s);
        closed.notified().await;
    })
    .await
    .expect("upstream was not cancelled after the client left");
}

#[tokio::test]
async fn dead_upstream_gives_502_json() {
    let dead = {
        let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
        l.local_addr().unwrap()
    };
    let anthropic = spawn_fake(recording("anthropic", Log::default())).await;
    let router = start_router(dead, anthropic).await;
    let body = format!(r#"{{"model":"{LOCAL_MODEL}"}}"#);
    let (h, b) = raw(router, "POST /v1/messages HTTP/1.1\r\nhost: x", &body).await;
    assert!(h.starts_with("HTTP/1.1 502"), "{h}");
    assert!(
        h.to_lowercase().contains("content-type: application/json"),
        "{h}"
    );
    let v: serde_json::Value = serde_json::from_str(&b).unwrap();
    assert_eq!(v["type"], "error");
    assert_eq!(v["error"]["type"], "api_error");
    assert!(v["error"]["message"]
        .as_str()
        .unwrap()
        .starts_with("router upstream failed: "));
}

#[tokio::test]
async fn resp_body_channel_streams_and_ends() {
    let (tx, rx) = mpsc::channel(4);
    let mut body = Box::pin(RespBody::channel(rx));
    tx.send(Bytes::from_static(b"one")).await.unwrap();
    let f = timeout(LIMIT, body.frame())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(f.into_data().unwrap(), "one");
    tx.send(Bytes::from_static(b"two")).await.unwrap();
    let f = timeout(LIMIT, body.frame())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(f.into_data().unwrap(), "two");
    drop(tx);
    assert!(timeout(LIMIT, body.frame()).await.unwrap().is_none());
}

#[tokio::test]
async fn resp_body_channel_drop_fails_sender() {
    let (tx, rx) = mpsc::channel(4);
    drop(RespBody::channel(rx));
    assert!(tx.send(Bytes::from_static(b"x")).await.is_err());
}

#[tokio::test]
async fn json_error_shape() {
    let resp = json_error(
        hyper::StatusCode::BAD_REQUEST,
        "invalid_request_error",
        "bad",
    );
    assert_eq!(resp.status(), 400);
    assert_eq!(resp.headers()["content-type"], "application/json");
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        v,
        serde_json::json!({"type":"error","error":{"type":"invalid_request_error","message":"bad"}})
    );
}

#[test]
fn usage_log_writes_one_json_line_per_record() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("nested/usage.jsonl");
    let log = UsageLog::open(&path).unwrap();
    for status in [200u16, 502] {
        log.record(&UsageRecord {
            provider: "google",
            model: "m",
            status,
            input_tokens: 1,
            cache_read_tokens: 2,
            output_tokens: 3,
            thinking_tokens: 4,
            latency_ms: 5,
            ttft_ms: Some(6),
            stop_reason: Some("end_turn"),
            error: None,
        });
    }
    UsageLog::disabled().record(&UsageRecord {
        provider: "x",
        model: "x",
        status: 0,
        input_tokens: 0,
        cache_read_tokens: 0,
        output_tokens: 0,
        thinking_tokens: 0,
        latency_ms: 0,
        ttft_ms: None,
        stop_reason: None,
        error: None,
    });
    let text = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 2);
    let v: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
    assert_eq!(v["provider"], "google");
    assert_eq!(v["status"], 502);
    assert_eq!(v["input_tokens"], 1);
    assert_eq!(v["cache_read_tokens"], 2);
    assert_eq!(v["output_tokens"], 3);
    assert_eq!(v["thinking_tokens"], 4);
    assert_eq!(v["latency_ms"], 5);
    assert_eq!(v["ttft_ms"], 6);
    assert_eq!(v["stop_reason"], "end_turn");
    assert!(v["error"].is_null());
    let ts = v["ts"].as_str().unwrap().as_bytes();
    assert_eq!(ts.len(), 24);
    for (i, c) in ts.iter().enumerate() {
        match i {
            4 | 7 => assert_eq!(*c, b'-'),
            10 => assert_eq!(*c, b'T'),
            13 | 16 => assert_eq!(*c, b':'),
            19 => assert_eq!(*c, b'.'),
            23 => assert_eq!(*c, b'Z'),
            _ => assert!(c.is_ascii_digit(), "ts {i}"),
        }
    }
}
