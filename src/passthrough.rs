//! The provider that forwards a request unchanged to an Anthropic-format server.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use http_body_util::Full;
use hyper::header::HeaderValue;
use hyper::http::uri::{Authority, Scheme};
use hyper::{Request, Response, Uri, Version};

use crate::provider::{json_error, Provider, ProxyRequest, RespBody, UpstreamClient};
use hyper::StatusCode;

/// Headers that belong to one connection and are never forwarded.
const HOP_BY_HOP: [&str; 8] = [
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
];

/// What the passthrough does with the client's credentials.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PassAuth {
    /// Forward the headers as received.
    Forward,
    /// Remove `authorization` and set `x-api-key: local`, so the subscription token never
    /// reaches a local server.
    LocalKey,
}

/// Forwards each request to `base` with its path, query, headers and body kept.
pub struct Passthrough {
    name: &'static str,
    scheme: Scheme,
    authority: Authority,
    auth: PassAuth,
    client: UpstreamClient,
}

impl Passthrough {
    /// Builds the provider. `base` needs a scheme and an authority.
    pub fn new(
        name: &'static str,
        base: Uri,
        auth: PassAuth,
        client: UpstreamClient,
    ) -> Result<Self, String> {
        match (base.scheme().cloned(), base.authority().cloned()) {
            (Some(scheme), Some(authority)) => Ok(Self {
                name,
                scheme,
                authority,
                auth,
                client,
            }),
            _ => Err(format!("the {name} URL needs a scheme and a host: {base}")),
        }
    }

    async fn forward(&self, req: ProxyRequest) -> Response<RespBody> {
        let ProxyRequest {
            mut parts,
            body,
            model,
            ..
        } = req;
        let side = self.name;
        let path_and_query = parts
            .uri
            .path_and_query()
            .map_or("/", |p| p.as_str())
            .to_owned();
        let mut headers = std::mem::take(&mut parts.headers);
        for name in HOP_BY_HOP {
            headers.remove(name);
        }
        headers.remove("host");
        headers.remove("content-length");
        if self.auth == PassAuth::LocalKey {
            headers.remove("authorization");
            headers.insert("x-api-key", HeaderValue::from_static("local"));
        }
        let uri = Uri::builder()
            .scheme(self.scheme.clone())
            .authority(self.authority.clone())
            .path_and_query(path_and_query.as_str())
            .build();
        let method = parts.method.clone();
        let model_label = model.as_deref().unwrap_or("None");

        let upstream = match uri {
            Ok(uri) => {
                let mut up = Request::new(Full::new(body));
                *up.method_mut() = parts.method;
                *up.uri_mut() = uri;
                *up.headers_mut() = headers;
                self.client.request(up).await.map_err(|e| e.to_string())
            }
            Err(e) => Err(e.to_string()),
        };

        match upstream {
            Ok(resp) => {
                let (mut rparts, rbody) = resp.into_parts();
                for name in HOP_BY_HOP {
                    rparts.headers.remove(name);
                }
                rparts.version = Version::HTTP_11;
                eprintln!(
                    "router: {method} {path_and_query} model={model_label} side={side} status={}",
                    rparts.status.as_u16()
                );
                let body =
                    RespBody::passthrough(rbody, model.unwrap_or_else(|| "None".to_owned()), side);
                Response::from_parts(rparts, body)
            }
            Err(msg) => {
                eprintln!(
                    "router: {method} {path_and_query} model={model_label} side={side} status=502"
                );
                json_error(
                    StatusCode::BAD_GATEWAY,
                    "api_error",
                    &format!("router upstream failed: {msg}"),
                )
            }
        }
    }
}

impl Provider for Passthrough {
    fn name(&self) -> &'static str {
        self.name
    }

    fn handle(
        self: Arc<Self>,
        req: ProxyRequest,
    ) -> Pin<Box<dyn Future<Output = Response<RespBody>> + Send>> {
        Box::pin(async move { self.forward(req).await })
    }
}
