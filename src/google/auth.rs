//! The user's Antigravity CLI login, the token refresh with the CLI's OAuth client, and the
//! Cloud Code project lookup.

use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::header::{HeaderValue, AUTHORIZATION, CONTENT_TYPE, USER_AGENT as UA};
use hyper::{Method, Request, StatusCode, Uri};
use serde::Deserialize;
use tokio::sync::{Mutex, OnceCell};

use super::client::{ClientCandidates, GoogleClient, OAuthClient};
use crate::provider::UpstreamClient;

/// Tokens are refreshed this long before they expire.
const REFRESH_MARGIN: Duration = Duration::from_secs(300);

/// Where the login of the Antigravity CLI is read from. Both hold the same JSON.
#[derive(Clone, Debug)]
pub enum CredentialSource {
    /// A Secret Service keyring entry, read with `secret-tool`.
    Keyring {
        /// The entry's `service` attribute.
        service: String,
        /// The entry's `username` attribute.
        user: String,
    },
    /// A JSON file.
    File(PathBuf),
}

impl CredentialSource {
    /// The keyring entry the Antigravity CLI writes (`service gemini`, `username antigravity`).
    pub fn antigravity() -> Self {
        CredentialSource::Keyring {
            service: "gemini".to_owned(),
            user: "antigravity".to_owned(),
        }
    }

    fn describe(&self) -> String {
        match self {
            CredentialSource::Keyring { service, user } => {
                format!("keyring entry {service}/{user}")
            }
            CredentialSource::File(p) => format!("file {}", p.display()),
        }
    }
}

/// The two Google endpoints the provider talks to.
#[derive(Clone, Debug)]
pub struct GoogleEndpoints {
    /// Base of the Cloud Code Assist API.
    pub code_assist: Uri,
    /// The OAuth token endpoint.
    pub token: Uri,
}

impl Default for GoogleEndpoints {
    fn default() -> Self {
        GoogleEndpoints {
            code_assist: Uri::from_static("https://daily-cloudcode-pa.googleapis.com"),
            token: Uri::from_static("https://oauth2.googleapis.com/token"),
        }
    }
}

/// Why a credential step failed. The text never holds a token.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AuthError {
    /// The credential source could not be read or has no refresh token.
    Credentials(String),
    /// The token endpoint refused or could not be reached.
    Refresh(String),
    /// The project lookup failed.
    Project(String),
}

impl fmt::Display for AuthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AuthError::Credentials(m) => write!(f, "google credentials: {m}"),
            AuthError::Refresh(m) => write!(f, "google token refresh: {m}"),
            AuthError::Project(m) => write!(f, "google project lookup: {m}"),
        }
    }
}

impl std::error::Error for AuthError {}

#[derive(Deserialize)]
struct Stored {
    token: StoredToken,
}

#[derive(Deserialize)]
struct StoredToken {
    refresh_token: String,
}

#[derive(Deserialize)]
struct RefreshReply {
    access_token: String,
    #[serde(default)]
    expires_in: Option<u64>,
    #[serde(default)]
    refresh_token: Option<String>,
}

#[derive(Deserialize)]
struct TokenError {
    #[serde(default)]
    error: String,
}

#[derive(Deserialize)]
struct ProjectReply {
    #[serde(default, rename = "cloudaicompanionProject")]
    project: Option<ProjectField>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ProjectField {
    Plain(String),
    Object { id: String },
}

#[derive(Default)]
struct State {
    refresh_token: Option<String>,
    access: Option<(Arc<str>, Instant)>,
    /// The `(id, secret)` candidate pair Google accepted.
    chosen: Option<(usize, usize)>,
}

/// How one refresh attempt ended.
enum Attempt {
    Token(RefreshReply),
    /// Google rejected the pair: `invalid_client` (wrong secret for the id, or an unknown
    /// id) or `unauthorized_client` (a real client that did not issue this refresh token).
    WrongClient {
        other_client: bool,
    },
    /// Google authenticated the client and refused for another reason (such as a revoked
    /// refresh token).
    Refused(AuthError),
    /// The attempt did not get an answer about the client (network, unexpected reply).
    Failed(AuthError),
}

/// Google access tokens for the Cloud Code Assist API, refreshed on demand.
pub struct GoogleAuth {
    source: CredentialSource,
    endpoints: GoogleEndpoints,
    client: UpstreamClient,
    candidates: ClientCandidates,
    user_agent: HeaderValue,
    state: Mutex<State>,
    project: OnceCell<Arc<str>>,
}

impl GoogleAuth {
    /// Credentials read from `source`, refreshed with the OAuth client of `google`, talking
    /// to `endpoints` through `client`. A `User-Agent` that is no valid header value is sent
    /// empty; [`super::client::resolve`] rejects such values.
    pub fn new(
        source: CredentialSource,
        endpoints: GoogleEndpoints,
        client: UpstreamClient,
        google: GoogleClient,
    ) -> Self {
        GoogleAuth {
            source,
            endpoints,
            client,
            candidates: google.candidates,
            user_agent: HeaderValue::from_str(&google.user_agent)
                .unwrap_or_else(|_| HeaderValue::from_static("")),
            state: Mutex::new(State::default()),
            project: OnceCell::new(),
        }
    }

    /// The `User-Agent` sent to Google.
    pub fn user_agent(&self) -> &HeaderValue {
        &self.user_agent
    }

    async fn read_source(&self) -> Result<String, AuthError> {
        let what = self.source.describe();
        let bytes = match &self.source {
            CredentialSource::Keyring { service, user } => {
                let out = tokio::process::Command::new("secret-tool")
                    .args(["lookup", "service", service, "username", user])
                    .stdin(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .output()
                    .await
                    .map_err(|_| {
                        AuthError::Credentials(format!("cannot run secret-tool for {what}"))
                    })?;
                if !out.status.success() {
                    return Err(AuthError::Credentials(format!("{what} not found")));
                }
                out.stdout
            }
            CredentialSource::File(p) => std::fs::read(p)
                .map_err(|_| AuthError::Credentials(format!("cannot read {what}")))?,
        };
        serde_json::from_slice::<Stored>(&bytes)
            .map(|s| s.token.refresh_token)
            .ok()
            .filter(|t| !t.is_empty())
            .ok_or_else(|| AuthError::Credentials(format!("{what} holds no refresh token")))
    }

    /// Refreshes with the pair Google accepted before. The first refresh tries every client
    /// id with every secret in order; a pair Google rejects as the wrong client moves on to
    /// the next, an id Google knows but that did not issue the token skips its other secrets.
    /// A pair that gets past client authentication is kept for the life of the process.
    async fn refresh(
        &self,
        chosen: &mut Option<(usize, usize)>,
        refresh_token: &str,
    ) -> Result<RefreshReply, AuthError> {
        if let Some((i, j)) = *chosen {
            return match self
                .attempt(&self.candidates.pair(i, j), refresh_token)
                .await
            {
                Attempt::Token(r) => Ok(r),
                Attempt::Refused(e) | Attempt::Failed(e) => Err(e),
                Attempt::WrongClient { .. } => Err(AuthError::Refresh(format!(
                    "Google rejected the OAuth client from {}",
                    self.candidates.origin()
                ))),
            };
        }
        let (ids, secrets) = self.candidates.counts();
        for i in 0..ids {
            for j in 0..secrets {
                match self
                    .attempt(&self.candidates.pair(i, j), refresh_token)
                    .await
                {
                    Attempt::Token(r) => {
                        *chosen = Some((i, j));
                        return Ok(r);
                    }
                    Attempt::Refused(e) => {
                        *chosen = Some((i, j));
                        return Err(e);
                    }
                    Attempt::Failed(e) => return Err(e),
                    Attempt::WrongClient { other_client: true } => break,
                    Attempt::WrongClient {
                        other_client: false,
                    } => {}
                }
            }
        }
        Err(AuthError::Refresh(format!(
            "Google accepted none of the {ids} OAuth client ids with the {secrets} secrets \
             from {}",
            self.candidates.origin()
        )))
    }

    async fn attempt(&self, oauth: &OAuthClient, refresh_token: &str) -> Attempt {
        let form = form_urlencoded::Serializer::new(String::with_capacity(256))
            .append_pair("client_id", &oauth.client_id)
            .append_pair("client_secret", &oauth.client_secret)
            .append_pair("refresh_token", refresh_token)
            .append_pair("grant_type", "refresh_token")
            .finish();
        let req = match Request::builder()
            .method(Method::POST)
            .uri(self.endpoints.token.clone())
            .header(CONTENT_TYPE, "application/x-www-form-urlencoded")
            .header(UA, self.user_agent.clone())
            .body(Full::new(Bytes::from(form)))
        {
            Ok(r) => r,
            Err(_) => return Attempt::Failed(AuthError::Refresh("bad token request".to_owned())),
        };
        let (status, body) = match self.send(req).await {
            Ok(x) => x,
            Err(e) => return Attempt::Failed(AuthError::Refresh(e)),
        };
        if !status.is_success() {
            let code = serde_json::from_slice::<TokenError>(&body)
                .map(|e| e.error)
                .unwrap_or_default();
            return match code.as_str() {
                "invalid_client" => Attempt::WrongClient {
                    other_client: false,
                },
                "unauthorized_client" => Attempt::WrongClient { other_client: true },
                "" => Attempt::Failed(AuthError::Refresh(format!(
                    "token endpoint answered {status}"
                ))),
                // Google's error codes are a fixed vocabulary; the text carries no token.
                c => Attempt::Refused(AuthError::Refresh(format!(
                    "token endpoint answered {status} ({})",
                    c.chars().take(64).collect::<String>()
                ))),
            };
        }
        match serde_json::from_slice(&body) {
            Ok(r) => Attempt::Token(r),
            Err(_) => Attempt::Failed(AuthError::Refresh("unreadable token reply".to_owned())),
        }
    }

    async fn send(&self, req: Request<Full<Bytes>>) -> Result<(StatusCode, Bytes), String> {
        let resp = self
            .client
            .request(req)
            .await
            .map_err(|e| format!("request failed: {e}"))?;
        let status = resp.status();
        let body = resp
            .into_body()
            .collect()
            .await
            .map_err(|_| "reply body failed".to_owned())?
            .to_bytes();
        Ok((status, body))
    }

    /// A valid access token. The first call reads the refresh token from the source. The
    /// token is cached until five minutes before it expires; concurrent callers share one
    /// refresh. A failed refresh reads the source once more (the user may have logged in
    /// again) before giving up.
    pub async fn access_token(&self) -> Result<Arc<str>, AuthError> {
        let mut st = self.state.lock().await;
        if let Some((tok, expiry)) = &st.access {
            if Instant::now() + REFRESH_MARGIN < *expiry {
                return Ok(Arc::clone(tok));
            }
        }
        if st.refresh_token.is_none() {
            st.refresh_token = Some(self.read_source().await?);
        }
        let first = st.refresh_token.clone().unwrap_or_default();
        let reply = match self.refresh(&mut st.chosen, &first).await {
            Ok(r) => r,
            Err(err) => {
                let fresh = match self.read_source().await {
                    Ok(f) if f != first => f,
                    _ => return Err(err),
                };
                st.refresh_token = Some(fresh.clone());
                self.refresh(&mut st.chosen, &fresh).await?
            }
        };
        if let Some(rt) = reply.refresh_token.filter(|t| !t.is_empty()) {
            st.refresh_token = Some(rt);
        }
        let tok: Arc<str> = Arc::from(reply.access_token);
        let life = Duration::from_secs(reply.expires_in.unwrap_or(3600));
        st.access = Some((Arc::clone(&tok), Instant::now() + life));
        Ok(tok)
    }

    /// Forgets the cached access token (after a 401); the next call refreshes.
    pub async fn invalidate(&self) {
        self.state.lock().await.access = None;
    }

    /// The Cloud Code project of the account, looked up once per process.
    pub async fn project(&self) -> Result<Arc<str>, AuthError> {
        self.project
            .get_or_try_init(|| async {
                let base = self.endpoints.code_assist.to_string();
                let uri: Uri = format!("{}/v1internal:loadCodeAssist", base.trim_end_matches('/'))
                    .parse()
                    .map_err(|_| AuthError::Project("bad endpoint".to_owned()))?;
                let mut retried = false;
                loop {
                    let token = self.access_token().await?;
                    let bearer = HeaderValue::from_str(&format!("Bearer {token}"))
                        .map_err(|_| AuthError::Project("bad token".to_owned()))?;
                    let req = Request::builder()
                        .method(Method::POST)
                        .uri(uri.clone())
                        .header(AUTHORIZATION, bearer)
                        .header(CONTENT_TYPE, "application/json")
                        .header(UA, self.user_agent.clone())
                        .body(Full::new(Bytes::from_static(
                            br#"{"metadata":{"ideType":"ANTIGRAVITY"}}"#,
                        )))
                        .map_err(|_| AuthError::Project("bad request".to_owned()))?;
                    let (status, body) = self.send(req).await.map_err(AuthError::Project)?;
                    if status == StatusCode::UNAUTHORIZED && !retried {
                        retried = true;
                        self.invalidate().await;
                        continue;
                    }
                    if !status.is_success() {
                        return Err(AuthError::Project(format!("answered {status}")));
                    }
                    let reply: ProjectReply = serde_json::from_slice(&body)
                        .map_err(|_| AuthError::Project("unreadable reply".to_owned()))?;
                    return match reply.project {
                        Some(ProjectField::Plain(s) | ProjectField::Object { id: s })
                            if !s.is_empty() =>
                        {
                            Ok(Arc::from(s))
                        }
                        _ => Err(AuthError::Project("reply names no project".to_owned())),
                    };
                }
            })
            .await
            .cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::upstream_client;
    use hyper::body::Incoming;
    use hyper::service::service_fn;
    use hyper::Response;
    use hyper_util::rt::TokioIo;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::net::TcpListener;

    const ID_GOOD: &str = "111-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.apps.googleusercontent.com";
    const SEC_GOOD: &str = "GOCSPX-goodgoodgoodgoodgoodgoodgood";
    const ID_OTHER: &str = "222-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb.apps.googleusercontent.com";
    const SEC_OTHER: &str = "GOCSPX-otherotherotherotherotherot";
    const SEC_THIRD: &str = "GOCSPX-thirdthirdthirdthirdthirdth";

    /// The fake token endpoint's answer for a form: the good pair gets a token (or
    /// `invalid_grant` for the refresh token `REVOKED`), the other real pair
    /// `unauthorized_client`, anything else `invalid_client`.
    fn token_answer(form: &str, n: usize) -> (StatusCode, String) {
        let f: std::collections::HashMap<String, String> = form_urlencoded::parse(form.as_bytes())
            .into_owned()
            .collect();
        let pair = (f["client_id"].as_str(), f["client_secret"].as_str());
        let err = |s: StatusCode, e: &str| (s, format!(r#"{{"error":"{e}"}}"#));
        match pair {
            (ID_GOOD, SEC_GOOD) if f["refresh_token"] == "REVOKED" => {
                err(StatusCode::BAD_REQUEST, "invalid_grant")
            }
            (ID_GOOD, SEC_GOOD) => (
                StatusCode::OK,
                format!(r#"{{"access_token":"at-{n}","expires_in":3599}}"#),
            ),
            (ID_OTHER, SEC_OTHER) => err(StatusCode::UNAUTHORIZED, "unauthorized_client"),
            _ => err(StatusCode::UNAUTHORIZED, "invalid_client"),
        }
    }

    fn good_client() -> GoogleClient {
        GoogleClient {
            candidates: ClientCandidates::single(
                OAuthClient {
                    client_id: ID_GOOD.to_owned(),
                    client_secret: SEC_GOOD.to_owned(),
                },
                "test",
            ),
            user_agent: "test-agent/1".to_owned(),
        }
    }

    #[derive(Default)]
    struct Log {
        token_bodies: std::sync::Mutex<Vec<String>>,
        project_calls: AtomicUsize,
        project_json: std::sync::Mutex<String>,
        project_auth: std::sync::Mutex<Vec<String>>,
    }

    async fn fake(log: Arc<Log>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let (s, _) = listener.accept().await.unwrap();
                let log = Arc::clone(&log);
                tokio::spawn(async move {
                    let svc = service_fn(move |req: Request<Incoming>| {
                        let log = Arc::clone(&log);
                        async move {
                            let path = req.uri().path().to_owned();
                            let auth = req
                                .headers()
                                .get(AUTHORIZATION)
                                .and_then(|v| v.to_str().ok())
                                .map(str::to_owned);
                            let body = req.into_body().collect().await.unwrap().to_bytes();
                            let (status, out) = if path == "/token" {
                                let form = String::from_utf8(body.to_vec()).unwrap();
                                let n = {
                                    let mut b = log.token_bodies.lock().unwrap();
                                    b.push(form.clone());
                                    b.iter()
                                        .filter(|f| token_answer(f, 0).0.is_success())
                                        .count()
                                };
                                tokio::time::sleep(Duration::from_millis(50)).await;
                                token_answer(&form, n)
                            } else {
                                log.project_calls.fetch_add(1, Ordering::SeqCst);
                                log.project_auth.lock().unwrap().extend(auth);
                                (StatusCode::OK, log.project_json.lock().unwrap().clone())
                            };
                            let mut resp = Response::new(Full::new(Bytes::from(out)));
                            *resp.status_mut() = status;
                            Ok::<_, std::convert::Infallible>(resp)
                        }
                    });
                    let _ = hyper::server::conn::http1::Builder::new()
                        .serve_connection(TokioIo::new(s), svc)
                        .await;
                });
            }
        });
        format!("http://{addr}")
    }

    /// A login file holding `rt`; the file goes when the returned guard drops.
    fn creds(rt: &str) -> (tempfile::NamedTempFile, CredentialSource) {
        let f = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            f.path(),
            format!(r#"{{"token":{{"access_token":"x","refresh_token":"{rt}"}}}}"#),
        )
        .unwrap();
        let src = CredentialSource::File(f.path().to_path_buf());
        (f, src)
    }

    fn auth(base: &str, src: CredentialSource) -> GoogleAuth {
        auth_with(base, src, good_client())
    }

    fn auth_with(base: &str, src: CredentialSource, google: GoogleClient) -> GoogleAuth {
        GoogleAuth::new(
            src,
            GoogleEndpoints {
                code_assist: base.parse().unwrap(),
                token: format!("{base}/token").parse().unwrap(),
            },
            upstream_client(),
            google,
        )
    }

    fn several(ids: &[&str], secrets: &[&str]) -> GoogleClient {
        GoogleClient {
            candidates: ClientCandidates::new(
                ids.iter().map(|s| s.to_string()).collect(),
                secrets.iter().map(|s| s.to_string()).collect(),
                "fake agy",
            )
            .unwrap(),
            user_agent: "test-agent/1".to_owned(),
        }
    }

    fn sent_pairs(log: &Log) -> Vec<(String, String)> {
        log.token_bodies
            .lock()
            .unwrap()
            .iter()
            .map(|b| {
                let f: std::collections::HashMap<String, String> =
                    form_urlencoded::parse(b.as_bytes()).into_owned().collect();
                (f["client_id"].clone(), f["client_secret"].clone())
            })
            .collect()
    }

    #[tokio::test]
    async fn first_refresh_finds_the_accepted_pair_and_keeps_it() {
        let log = Arc::new(Log::default());
        let base = fake(Arc::clone(&log)).await;
        let google = several(&[ID_OTHER, ID_GOOD], &[SEC_OTHER, SEC_THIRD, SEC_GOOD]);
        let (_login, src) = creds("r");
        let a = auth_with(&base, src, google);
        assert_eq!(&*a.access_token().await.unwrap(), "at-1");
        let s = |a: &str, b: &str| (a.to_owned(), b.to_owned());
        // ID_OTHER is a real client that did not issue the token: its other secrets are
        // skipped. ID_GOOD tries the secrets in order until Google accepts one.
        assert_eq!(
            sent_pairs(&log),
            vec![
                s(ID_OTHER, SEC_OTHER),
                s(ID_GOOD, SEC_OTHER),
                s(ID_GOOD, SEC_THIRD),
                s(ID_GOOD, SEC_GOOD),
            ]
        );
        a.invalidate().await;
        assert_eq!(&*a.access_token().await.unwrap(), "at-2");
        assert_eq!(sent_pairs(&log).len(), 5);
        assert_eq!(sent_pairs(&log)[4], s(ID_GOOD, SEC_GOOD));
    }

    #[tokio::test]
    async fn no_accepted_pair_is_an_error_without_secrets() {
        let log = Arc::new(Log::default());
        let base = fake(Arc::clone(&log)).await;
        let google = several(&[ID_OTHER, ID_GOOD], &[SEC_THIRD]);
        let (_login, src) = creds("SECRET-REFRESH");
        let a = auth_with(&base, src, google);
        let e = a.access_token().await.unwrap_err().to_string();
        assert!(e.contains("accepted none of the 2 OAuth client ids"), "{e}");
        assert!(
            !e.contains("SECRET-REFRESH") && !e.contains("GOCSPX"),
            "{e}"
        );
    }

    #[tokio::test]
    async fn revoked_token_stops_at_the_authenticated_pair() {
        let log = Arc::new(Log::default());
        let base = fake(Arc::clone(&log)).await;
        let google = several(&[ID_GOOD, ID_OTHER], &[SEC_GOOD, SEC_OTHER]);
        let (_login, src) = creds("REVOKED");
        let a = auth_with(&base, src, google);
        let e = a.access_token().await.unwrap_err().to_string();
        assert!(e.contains("invalid_grant"), "{e}");
        // One attempt, then one more after re-reading the (unchanged) login: none.
        assert_eq!(sent_pairs(&log).len(), 1);
    }

    #[tokio::test]
    async fn concurrent_callers_share_one_refresh() {
        let log = Arc::new(Log::default());
        let base = fake(Arc::clone(&log)).await;
        let (_login, src) = creds("1//ref/tok+x");
        let a = Arc::new(auth(&base, src));
        let hs: Vec<_> = (0..8)
            .map(|_| {
                let a = Arc::clone(&a);
                tokio::spawn(async move { a.access_token().await.unwrap() })
            })
            .collect();
        for h in hs {
            assert_eq!(&*h.await.unwrap(), "at-1");
        }
        assert_eq!(&*a.access_token().await.unwrap(), "at-1");
        let bodies = log.token_bodies.lock().unwrap().clone();
        assert_eq!(bodies.len(), 1);
        let b = &bodies[0];
        assert!(b.contains("grant_type=refresh_token"));
        assert!(b.contains("refresh_token=1%2F%2Fref%2Ftok%2Bx"));
        assert_eq!(
            sent_pairs(&log),
            vec![(ID_GOOD.to_owned(), SEC_GOOD.to_owned())]
        );
    }

    #[tokio::test]
    async fn invalidate_forces_refresh() {
        let log = Arc::new(Log::default());
        let base = fake(Arc::clone(&log)).await;
        let (_login, src) = creds("r");
        let a = auth(&base, src);
        assert_eq!(&*a.access_token().await.unwrap(), "at-1");
        a.invalidate().await;
        assert_eq!(&*a.access_token().await.unwrap(), "at-2");
    }

    #[tokio::test]
    async fn project_both_shapes_and_cached() {
        for json in [
            r#"{"cloudaicompanionProject":"proj-a"}"#,
            r#"{"cloudaicompanionProject":{"id":"proj-a"}}"#,
        ] {
            let log = Arc::new(Log::default());
            *log.project_json.lock().unwrap() = json.to_owned();
            let base = fake(Arc::clone(&log)).await;
            let (_login, src) = creds("r");
            let a = auth(&base, src);
            assert_eq!(&*a.project().await.unwrap(), "proj-a");
            assert_eq!(&*a.project().await.unwrap(), "proj-a");
            assert_eq!(log.project_calls.load(Ordering::SeqCst), 1);
            assert_eq!(log.project_auth.lock().unwrap()[0], "Bearer at-1");
        }
    }

    #[tokio::test]
    async fn errors_carry_no_secrets() {
        let src = CredentialSource::File(PathBuf::from("/nonexistent/cmr-creds.json"));
        let a = auth("http://127.0.0.1:1", src);
        let e = a.access_token().await.unwrap_err();
        assert!(matches!(e, AuthError::Credentials(_)));
        let (_login, src) = creds("SECRET-REFRESH");
        let a = auth("http://127.0.0.1:1", src);
        let e = a.access_token().await.unwrap_err().to_string();
        assert!(
            !e.contains("SECRET-REFRESH") && !e.contains("GOCSPX") && !e.contains(SEC_GOOD),
            "{e}"
        );
    }
}
