//! The provider contract: what a backend receives, what it returns, and the shared pieces
//! (HTTP client, response body, JSON replies).

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use bytes::Bytes;
use http_body_util::Full;
use hyper::body::{Body, Frame, Incoming, SizeHint};
use hyper::header::{HeaderValue, CONTENT_TYPE};
use hyper::{Response, StatusCode};
use hyper_rustls::{HttpsConnector, HttpsConnectorBuilder};
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use pin_project_lite::pin_project;
use tokio::sync::mpsc;

/// Boxed error used by response bodies.
pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// The shared HTTPS/HTTP client type.
pub type UpstreamClient = Client<HttpsConnector<HttpConnector>, Full<Bytes>>;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Builds the shared HTTPS/HTTP client: webpki roots, HTTP/1.1 and HTTP/2, `TCP_NODELAY`,
/// a 10 second connect timeout.
pub fn upstream_client() -> UpstreamClient {
    let mut http = HttpConnector::new();
    http.enforce_http(false);
    http.set_connect_timeout(Some(CONNECT_TIMEOUT));
    http.set_nodelay(true);
    let connector = HttpsConnectorBuilder::new()
        .with_webpki_roots()
        .https_or_http()
        .enable_http1()
        .enable_http2()
        .wrap_connector(http);
    Client::builder(TokioExecutor::new()).build(connector)
}

/// One incoming request with its body already collected.
pub struct ProxyRequest {
    /// Method, URI and headers as received.
    pub parts: hyper::http::request::Parts,
    /// The whole request body.
    pub body: Bytes,
    /// The body's `model` field, when the body parsed.
    pub model: Option<String>,
    /// When the router received the request.
    pub received: Instant,
}

/// A backend that answers requests routed to it.
pub trait Provider: Send + Sync + 'static {
    /// Short label for logs ("anthropic", "local", "google").
    fn name(&self) -> &'static str;
    /// Answers one request. One boxed future per request is the only dynamic dispatch on the
    /// request path.
    fn handle(
        self: Arc<Self>,
        req: ProxyRequest,
    ) -> Pin<Box<dyn Future<Output = Response<RespBody>> + Send>>;
}

pin_project! {
    /// Response body for every provider. Dispatch per frame is static: a `match` over the
    /// variants, no boxing per chunk.
    #[project = RespProj]
    #[allow(missing_docs)]
    pub enum RespBody {
        /// An upstream body passed through untouched.
        Pass {
            #[pin]
            inner: PassBody,
        },
        /// A complete body.
        Fixed {
            #[pin]
            inner: Full<Bytes>,
        },
        /// A body fed by a task through a channel.
        Chan {
            rx: mpsc::Receiver<Bytes>,
        },
    }
}

impl RespBody {
    /// A complete in-memory body.
    pub fn fixed(bytes: Bytes) -> Self {
        RespBody::Fixed {
            inner: Full::new(bytes),
        }
    }

    /// A body fed by a task: each received `Bytes` is one data frame and the body ends when the
    /// sender is dropped. Dropping the body drops the receiver, so the task's next send fails;
    /// that is how a translating provider sees a client disconnect.
    pub fn channel(rx: mpsc::Receiver<Bytes>) -> Self {
        RespBody::Chan { rx }
    }

    /// An upstream body passed through untouched. Dropping it cancels the upstream request; if
    /// that happens before the end of the stream, one stderr line says the client closed early.
    pub fn passthrough(inner: Incoming, model: String, side: &'static str) -> Self {
        RespBody::Pass {
            inner: PassBody {
                inner,
                ended: false,
                model,
                side,
            },
        }
    }
}

impl Body for RespBody {
    type Data = Bytes;
    type Error = BoxError;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, BoxError>>> {
        match self.project() {
            RespProj::Pass { inner } => inner.poll_frame(cx),
            RespProj::Fixed { inner } => inner
                .poll_frame(cx)
                .map(|o| o.map(|r| r.map_err(|e| match e {}))),
            RespProj::Chan { rx } => rx.poll_recv(cx).map(|o| o.map(|b| Ok(Frame::data(b)))),
        }
    }

    fn is_end_stream(&self) -> bool {
        match self {
            RespBody::Pass { inner } => inner.inner.is_end_stream(),
            RespBody::Fixed { inner } => inner.is_end_stream(),
            RespBody::Chan { .. } => false,
        }
    }

    fn size_hint(&self) -> SizeHint {
        match self {
            RespBody::Pass { inner } => inner.inner.size_hint(),
            RespBody::Fixed { inner } => inner.size_hint(),
            RespBody::Chan { .. } => SizeHint::default(),
        }
    }
}

pin_project! {
    /// The passthrough variant's state.
    #[project(!Unpin)]
    pub struct PassBody {
        #[pin]
        inner: Incoming,
        ended: bool,
        model: String,
        side: &'static str,
    }

    impl PinnedDrop for PassBody {
        fn drop(this: Pin<&mut Self>) {
            let this = this.project();
            if crate::verbose() && !*this.ended && !this.inner.as_ref().is_end_stream() {
                eprintln!(
                    "router: client closed the stream (model {}, {})",
                    this.model, this.side
                );
            }
        }
    }
}

impl PassBody {
    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, BoxError>>> {
        let this = self.project();
        match this.inner.poll_frame(cx) {
            Poll::Ready(Some(Ok(f))) => Poll::Ready(Some(Ok(f))),
            Poll::Ready(Some(Err(e))) => {
                *this.ended = true;
                Poll::Ready(Some(Err(Box::new(e))))
            }
            Poll::Ready(None) => {
                *this.ended = true;
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

/// A plain JSON response with the given status.
pub fn json_response(status: StatusCode, body: Bytes) -> Response<RespBody> {
    let mut resp = Response::new(RespBody::fixed(body));
    *resp.status_mut() = status;
    resp.headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    resp
}

/// An Anthropic-shaped JSON error: `{"type":"error","error":{"type":kind,"message":message}}`.
pub fn json_error(status: StatusCode, kind: &str, message: &str) -> Response<RespBody> {
    let payload = serde_json::json!({
        "type": "error",
        "error": { "type": kind, "message": message },
    });
    json_response(status, Bytes::from(payload.to_string()))
}
