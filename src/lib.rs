//! A local reverse proxy that routes Claude Code's API calls by the `model` field of the JSON
//! body. Each backend is a [`provider::Provider`]; the [`routes::Routes`] table maps a model
//! name to one, and a default provider takes every other model.
#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod passthrough;
pub mod provider;
pub mod routes;
pub mod usage;

pub mod anthropic;
pub mod google;

use std::borrow::Cow;
use std::convert::Infallible;
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use bytes::Bytes;
use http_body_util::BodyExt;
use hyper::body::Incoming;
use hyper::service::service_fn;
use hyper::{Request, Response};
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto;
use serde::Deserialize;
use tokio::net::TcpListener;

use crate::provider::{ProxyRequest, RespBody};
use crate::routes::Routes;

#[derive(Deserialize)]
struct ModelOnly<'a> {
    #[serde(borrow)]
    model: Option<Cow<'a, str>>,
}

static VERBOSE: AtomicBool = AtomicBool::new(false);

/// Turns the diagnostic lines on or off (the `--verbose` flag).
pub fn set_verbose(on: bool) {
    VERBOSE.store(on, Ordering::Relaxed);
}

/// Whether diagnostic lines are printed.
pub fn verbose() -> bool {
    VERBOSE.load(Ordering::Relaxed)
}

/// Serves connections accepted on `listener` until accepting fails.
pub async fn serve(listener: TcpListener, routes: Routes) -> io::Result<()> {
    let routes = Arc::new(routes);
    loop {
        let (stream, _) = listener.accept().await?;
        let _ = stream.set_nodelay(true);
        let routes = Arc::clone(&routes);
        tokio::spawn(async move {
            let service = service_fn(move |req| handle(Arc::clone(&routes), req));
            let _ = auto::Builder::new(TokioExecutor::new())
                .serve_connection(TokioIo::new(stream), service)
                .await;
        });
    }
}

async fn handle(
    routes: Arc<Routes>,
    req: Request<Incoming>,
) -> Result<Response<RespBody>, Infallible> {
    let received = Instant::now();
    let (parts, body) = req.into_parts();
    let body = match body.collect().await {
        Ok(c) => c.to_bytes(),
        Err(_) => Bytes::new(),
    };
    let model: Option<String> = serde_json::from_slice::<ModelOnly>(&body)
        .ok()
        .and_then(|m| m.model)
        .map(Cow::into_owned);
    let provider = Arc::clone(routes.resolve(model.as_deref()));
    Ok(provider
        .handle(ProxyRequest {
            parts,
            body,
            model,
            received,
        })
        .await)
}
