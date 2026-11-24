//! The OAuth client (id and secret) and the `User-Agent` the Google provider talks to Google
//! with.
//!
//! Each value comes from the JSON config file when the file sets it, otherwise from the
//! installed Antigravity CLI binary (`agy`):
//!
//! - client ids and `GOCSPX-` secrets: every one the binary holds. The binary carries several
//!   OAuth clients, and the first token refresh tries the pairs until Google accepts one (see
//!   [`crate::google::auth`]).
//! - `User-Agent`: built the way the CLI builds it, from `agy --version`, the build changelist
//!   in the binary's source paths, and the operating system and architecture of this machine.
//!
//! Resolution runs once at startup. It reads the binary once in 1 MiB pieces with a SIMD
//! substring search (about a quarter of a second for a 260 MB binary from the page cache) and
//! runs `agy --version` once (about 0.1 s).

use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use memchr::memmem;
use serde::Deserialize;

/// The suffix every Google OAuth client id ends with.
const ID_SUFFIX: &[u8] = b".apps.googleusercontent.com";
/// The prefix of a Google installed-app client secret.
const SECRET_PREFIX: &[u8] = b"GOCSPX-";
/// Characters after [`SECRET_PREFIX`].
const SECRET_BODY: usize = 28;
/// Characters between the `-` and [`ID_SUFFIX`] of a client id.
const ID_HASH: usize = 32;
/// Most digits of a client id's project number, and of a changelist, that the scan accepts.
const MAX_DIGITS: usize = 20;
/// The source path of the build, followed by the changelist number and [`CL_SUFFIX`].
const CL_PREFIX: &[u8] = b"google_src/files/";
const CL_SUFFIX: &[u8] = b"/depot";
/// Bytes carried from one read into the next, so a string across a read boundary is seen
/// whole. Covers the widest look-around: a client id suffix cut at the boundary
/// (`ID_SUFFIX.len()`) plus the `MAX_DIGITS + 1 + ID_HASH` bytes before it.
const OVERLAP: usize = 128;
/// Size of one read of the binary.
const CHUNK: usize = 1 << 20;

/// One OAuth client. `Debug` hides the secret.
#[derive(Clone, PartialEq, Eq)]
pub struct OAuthClient {
    /// The client id (`<digits>-<hash>.apps.googleusercontent.com`).
    pub client_id: String,
    /// The client secret.
    pub client_secret: String,
}

impl fmt::Debug for OAuthClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OAuthClient")
            .field("client_id", &self.client_id)
            .field("client_secret", &"<hidden>")
            .finish()
    }
}

/// The client ids and secrets the token refresh may use, and where they came from. Every
/// id is tried with every secret, in the order found. `Debug` hides the secrets.
#[derive(Clone, PartialEq, Eq)]
pub struct ClientCandidates {
    ids: Vec<String>,
    secrets: Vec<String>,
    origin: String,
}

impl fmt::Debug for ClientCandidates {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ClientCandidates")
            .field("ids", &self.ids)
            .field("secrets", &self.secrets.len())
            .field("origin", &self.origin)
            .finish()
    }
}

impl ClientCandidates {
    /// Exactly one known pair.
    pub fn single(client: OAuthClient, origin: impl Into<String>) -> Self {
        ClientCandidates {
            ids: vec![client.client_id],
            secrets: vec![client.client_secret],
            origin: origin.into(),
        }
    }

    /// Every id with every secret; `None` when either list is empty.
    pub fn new(ids: Vec<String>, secrets: Vec<String>, origin: impl Into<String>) -> Option<Self> {
        (!ids.is_empty() && !secrets.is_empty()).then(|| ClientCandidates {
            ids,
            secrets,
            origin: origin.into(),
        })
    }

    /// Where the candidates were read from, for messages.
    pub fn origin(&self) -> &str {
        &self.origin
    }

    /// How many client ids and secrets there are.
    pub fn counts(&self) -> (usize, usize) {
        (self.ids.len(), self.secrets.len())
    }

    /// The pair `(id index, secret index)`.
    pub fn pair(&self, id: usize, secret: usize) -> OAuthClient {
        OAuthClient {
            client_id: self.ids[id].clone(),
            client_secret: self.secrets[secret].clone(),
        }
    }
}

/// Everything the Google provider needs besides the user's login.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GoogleClient {
    /// The OAuth client candidates.
    pub candidates: ClientCandidates,
    /// The `User-Agent` sent to Google.
    pub user_agent: String,
}

/// The config file: `{"client_id": "...", "client_secret": "...", "user_agent": "..."}`.
/// Each field is optional; a field it sets wins over the Antigravity CLI binary.
#[derive(Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClientConfig {
    /// The OAuth client id.
    #[serde(default)]
    pub client_id: Option<String>,
    /// The OAuth client secret.
    #[serde(default)]
    pub client_secret: Option<String>,
    /// The `User-Agent` sent to Google.
    #[serde(default)]
    pub user_agent: Option<String>,
}

impl ClientConfig {
    fn complete(&self) -> bool {
        self.client_id.is_some() && self.client_secret.is_some() && self.user_agent.is_some()
    }
}

/// `$XDG_CONFIG_HOME/claude-mr/google.json`.
pub fn default_config_path() -> Option<PathBuf> {
    dirs::config_dir().map(|d| d.join("claude-mr").join("google.json"))
}

/// Reads the config file; empty strings count as unset.
pub fn read_config(path: &Path) -> Result<ClientConfig, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let mut c: ClientConfig = serde_json::from_slice(&bytes).map_err(|_| {
        format!(
            "{} is not a JSON object with the string fields client_id, client_secret, \
             user_agent",
            path.display()
        )
    })?;
    for f in [&mut c.client_id, &mut c.client_secret, &mut c.user_agent] {
        if f.as_deref().is_some_and(|s| s.trim().is_empty()) {
            *f = None;
        }
    }
    Ok(c)
}

/// Where `agy` is looked for: each directory of `PATH`, then `~/.local/bin`.
pub fn agy_search_paths() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::env::var_os("PATH")
        .map(|p| std::env::split_paths(&p).map(|d| d.join("agy")).collect())
        .unwrap_or_default();
    if let Some(home) = dirs::home_dir() {
        let local = home.join(".local").join("bin").join("agy");
        if !out.contains(&local) {
            out.push(local);
        }
    }
    out
}

/// What [`scan_binary`] finds in the Antigravity CLI binary.
#[derive(Debug, Default)]
pub struct BinaryScan {
    /// Client ids, each once, in the order found.
    pub ids: Vec<String>,
    /// Client secrets, each once, in the order found.
    pub secrets: Vec<String>,
    /// The changelist that occurs most often in the build's source paths.
    pub changelist: Option<String>,
}

/// Scans a binary for client ids, client secrets and the build changelist. Reads the file
/// once in 1 MiB pieces with a SIMD substring search; memory stays at one piece.
pub fn scan_binary(path: &Path) -> io::Result<BinaryScan> {
    let mut file = File::open(path)?;
    let id_finder = memmem::Finder::new(ID_SUFFIX);
    let secret_finder = memmem::Finder::new(SECRET_PREFIX);
    let cl_finder = memmem::Finder::new(CL_PREFIX);
    let mut out = BinaryScan::default();
    let mut cls: HashMap<String, usize> = HashMap::new();
    // Absolute offset of the last changelist counted, so the overlap does not count it twice.
    let mut last_cl: Option<u64> = None;
    let mut buf = vec![0u8; OVERLAP + CHUNK];
    let mut carried = 0;
    let mut base: u64 = 0;
    loop {
        let n = file.read(&mut buf[carried..])?;
        if n == 0 {
            break;
        }
        let len = carried + n;
        let hay = &buf[..len];
        for end in id_finder.find_iter(hay) {
            if let Some(id) = id_before(hay, end) {
                push_new(&mut out.ids, id);
            }
        }
        for start in secret_finder.find_iter(hay) {
            if let Some(s) = secret_at(hay, start) {
                push_new(&mut out.secrets, s);
            }
        }
        for start in cl_finder.find_iter(hay) {
            let at = base + start as u64;
            if last_cl.is_some_and(|l| at <= l) {
                continue;
            }
            if let Some(cl) = changelist_at(hay, start) {
                last_cl = Some(at);
                *cls.entry(cl).or_default() += 1;
            }
        }
        carried = OVERLAP.min(len);
        buf.copy_within(len - carried..len, 0);
        base += (len - carried) as u64;
    }
    out.changelist = cls
        .into_iter()
        .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(&a.0)))
        .map(|(cl, _)| cl);
    Ok(out)
}

fn push_new(v: &mut Vec<String>, s: String) {
    if !v.contains(&s) {
        v.push(s);
    }
}

/// The client id whose [`ID_SUFFIX`] starts at `suffix`: one to [`MAX_DIGITS`] digits, `-`,
/// [`ID_HASH`] characters of `[a-z0-9]`.
fn id_before(hay: &[u8], suffix: usize) -> Option<String> {
    let hash_start = suffix.checked_sub(ID_HASH)?;
    let hash = &hay[hash_start..suffix];
    if !hash
        .iter()
        .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
    {
        return None;
    }
    let dash = hash_start.checked_sub(1)?;
    if hay[dash] != b'-' {
        return None;
    }
    let digits = hay[..dash]
        .iter()
        .rev()
        .take(MAX_DIGITS + 1)
        .take_while(|b| b.is_ascii_digit())
        .count();
    if digits == 0 || digits > MAX_DIGITS {
        return None;
    }
    let end = suffix + ID_SUFFIX.len();
    std::str::from_utf8(&hay[dash - digits..end])
        .ok()
        .map(str::to_owned)
}

/// The secret at `start`: [`SECRET_PREFIX`] and exactly [`SECRET_BODY`] characters of
/// `[A-Za-z0-9_-]`. The binary glues the next string right after it, so the length is what
/// ends it.
fn secret_at(hay: &[u8], start: usize) -> Option<String> {
    let end = start + SECRET_PREFIX.len() + SECRET_BODY;
    let body = hay.get(start + SECRET_PREFIX.len()..end)?;
    if !body
        .iter()
        .all(|b| b.is_ascii_alphanumeric() || *b == b'_' || *b == b'-')
    {
        return None;
    }
    std::str::from_utf8(&hay[start..end])
        .ok()
        .map(str::to_owned)
}

/// The changelist in `google_src/files/<digits>/depot` at `start`.
fn changelist_at(hay: &[u8], start: usize) -> Option<String> {
    let from = start + CL_PREFIX.len();
    let rest = hay.get(from..)?;
    let digits = rest
        .iter()
        .take(MAX_DIGITS + 1)
        .take_while(|b| b.is_ascii_digit())
        .count();
    if digits == 0 || digits > MAX_DIGITS || !rest[digits..].starts_with(CL_SUFFIX) {
        return None;
    }
    std::str::from_utf8(&rest[..digits]).ok().map(str::to_owned)
}

/// The version `agy --version` prints, such as `1.5.16`.
fn agy_version(agy: &Path) -> Result<String, String> {
    let out = Command::new(agy)
        .arg("--version")
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .map_err(|e| format!("cannot run `{} --version`: {e}", agy.display()))?;
    let text = String::from_utf8_lossy(&out.stdout);
    text.split_whitespace()
        .rev()
        .map(|w| w.trim_start_matches('v'))
        .find(|w| {
            !w.is_empty() && w.contains('.') && w.chars().all(|c| c.is_ascii_digit() || c == '.')
        })
        .map(str::to_owned)
        .ok_or_else(|| format!("`{} --version` printed no version", agy.display()))
}

/// Go's names for the operating system and architecture, which the CLI reports.
fn go_os_arch() -> (&'static str, &'static str) {
    let os = match std::env::consts::OS {
        "macos" => "darwin",
        other => other,
    };
    let arch = match std::env::consts::ARCH {
        "x86_64" => "amd64",
        "aarch64" => "arm64",
        "x86" => "386",
        other => other,
    };
    (os, arch)
}

/// The `User-Agent` of the Antigravity CLI for a version and changelist on this machine.
pub fn user_agent(version: &str, changelist: &str) -> String {
    let (os, arch) = go_os_arch();
    format!(
        "antigravity/cli/{version} (aidev_client; os_type={os}; arch={arch}; cl={changelist}; \
         auth_method=consumer)"
    )
}

/// The Antigravity CLI binary: `explicit` (the `--agy` flag) when given, else the first
/// `agy` on [`agy_search_paths`]. The error lists every place looked at.
pub fn find_agy(explicit: Option<&Path>) -> Result<PathBuf, Vec<PathBuf>> {
    let searched: Vec<PathBuf> = match explicit {
        Some(p) => vec![p.to_path_buf()],
        None => agy_search_paths(),
    };
    searched
        .iter()
        .find(|p| p.is_file())
        .cloned()
        .ok_or(searched)
}

/// Finds the Google client values for the Antigravity CLI binary at `agy`.
///
/// 1. The config file: `config` (the `--google-client-config` flag), else
///    [`default_config_path`] when that file exists. Fields it sets are used as they are.
/// 2. The binary for the fields the file leaves unset.
///
/// The error says what is missing and how to write the config file; it never holds a secret.
pub fn resolve(config: Option<&Path>, agy: &Path) -> Result<GoogleClient, String> {
    let config_path = config.map(Path::to_path_buf).or_else(default_config_path);
    let mut problems = Vec::new();
    let read = match (config, config_path.as_deref()) {
        (Some(p), _) => read_config(p),
        (None, Some(p)) if p.exists() => read_config(p),
        _ => Ok(ClientConfig::default()),
    };
    let cfg = match read {
        Ok(c) => c,
        Err(e) => {
            problems.push(e);
            return Err(missing_message(&problems, config_path.as_deref()));
        }
    };
    let cfg_origin = config_path
        .as_deref()
        .map_or_else(String::new, |p| format!("config file {}", p.display()));
    if cfg.complete() {
        return check_user_agent(
            GoogleClient {
                candidates: ClientCandidates::single(
                    OAuthClient {
                        client_id: cfg.client_id.unwrap_or_default(),
                        client_secret: cfg.client_secret.unwrap_or_default(),
                    },
                    cfg_origin,
                ),
                user_agent: cfg.user_agent.unwrap_or_default(),
            },
            config_path.as_deref(),
        );
    }

    let agy_origin = agy.display().to_string();
    let mut scan = BinaryScan::default();
    match scan_binary(agy) {
        Ok(s) => scan = s,
        Err(e) => problems.push(format!("cannot read {agy_origin}: {e}")),
    }
    let version = match cfg.user_agent {
        Some(_) => None,
        None => agy_version(agy).map_err(|e| problems.push(e)).ok(),
    };

    let (ids, id_from) = match cfg.client_id {
        Some(id) => (vec![id], cfg_origin.as_str()),
        None => (scan.ids, agy_origin.as_str()),
    };
    let (secrets, secret_from) = match cfg.client_secret {
        Some(s) => (vec![s], cfg_origin.as_str()),
        None => (scan.secrets, agy_origin.as_str()),
    };
    let user_agent = match (cfg.user_agent, version, scan.changelist) {
        (Some(ua), _, _) => Some(ua),
        (None, Some(v), Some(cl)) => Some(user_agent(&v, &cl)),
        (None, _, cl) => {
            if cl.is_none() {
                problems.push(format!(
                    "{agy_origin}: no build changelist found for the User-Agent"
                ));
            }
            None
        }
    };
    if ids.is_empty() {
        problems.push(format!("{agy_origin}: no OAuth client id found"));
    }
    if secrets.is_empty() {
        problems.push(format!("{agy_origin}: no OAuth client secret found"));
    }
    let origin = if id_from == secret_from {
        id_from.to_owned()
    } else {
        format!("{id_from} and {secret_from}")
    };
    match (ClientCandidates::new(ids, secrets, origin), user_agent) {
        (Some(candidates), Some(user_agent)) => check_user_agent(
            GoogleClient {
                candidates,
                user_agent,
            },
            config_path.as_deref(),
        ),
        _ => Err(missing_message(&problems, config_path.as_deref())),
    }
}

fn check_user_agent(c: GoogleClient, config: Option<&Path>) -> Result<GoogleClient, String> {
    if hyper::header::HeaderValue::from_str(&c.user_agent).is_ok() {
        return Ok(c);
    }
    let problem = "the user_agent holds characters an HTTP header cannot carry".to_owned();
    Err(missing_message(&[problem], config))
}

fn missing_message(problems: &[String], config: Option<&Path>) -> String {
    let mut m = String::from(
        "Google models are unavailable: the OAuth client and User-Agent could not be derived \
         from the Antigravity CLI. Other models work as usual.\n",
    );
    for p in problems {
        m.push_str(p);
        m.push('\n');
    }
    let config = config.map_or_else(
        || "$XDG_CONFIG_HOME/claude-mr/google.json".to_owned(),
        |p| p.display().to_string(),
    );
    m.push_str(&format!(
        "You may set the values yourself in {config} and restart claude-mr; each field is \
         optional, and a field set there replaces the derived value:\n\
         {{\"client_id\": \"<digits>-<hash>.apps.googleusercontent.com\", \
         \"client_secret\": \"GOCSPX-<28 characters>\", \
         \"user_agent\": \"antigravity/cli/<version> (aidev_client; os_type=linux; arch=amd64; cl=<changelist>; auth_method=consumer)\"}}\n\
         The Antigravity CLI binary holds the values. List the client id and secret \
         candidates with\n  \
         grep -aoE '[0-9]+-[a-z0-9]{{32}}\\.apps\\.googleusercontent\\.com|GOCSPX-[A-Za-z0-9_-]{{28}}' /path/to/agy | sort -u\n\
         (each id pairs with one secret; Google accepts only the right pair), the version with \
         `agy --version`, and the changelist with\n  \
         grep -aoE 'google_src/files/[0-9]+/depot' /path/to/agy | sort | uniq -c"
    ));
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    const ID_A: &str = "123456789012-abcdefghijklmnopqrstuvwxyz012345.apps.googleusercontent.com";
    const ID_B: &str = "42-zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz.apps.googleusercontent.com";
    const ID_LONG: &str =
        "12345678901234567890-yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy.apps.googleusercontent.com";
    const SEC_A: &str = "GOCSPX-FAKEfakeFAKEfake_0123-456789";
    const SEC_B: &str = "GOCSPX-anotherFakeSecretValue000000";

    /// Writes `bytes` to `name` inside `dir`; the directory goes when its guard drops.
    fn temp(dir: &TempDir, name: &str, bytes: &[u8]) -> PathBuf {
        let p = dir.path().join(name);
        std::fs::write(&p, bytes).unwrap();
        p
    }

    /// A stand-in for `agy`: a shell script that prints a version and carries the strings
    /// after its `exit`.
    fn fake_agy(dir: &TempDir, name: &str, payload: &str) -> PathBuf {
        let p = temp(
            dir,
            name,
            format!("#!/bin/sh\necho 9.8.7\nexit 0\n{payload}\n").as_bytes(),
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        p
    }

    /// A config path that does not exist, so the user's own file stays out of the tests.
    fn no_config(dir: &TempDir) -> PathBuf {
        dir.path().join("none.json")
    }

    #[test]
    fn fixtures_are_well_formed() {
        assert_eq!(SEC_A.len(), SECRET_PREFIX.len() + SECRET_BODY);
        assert_eq!(SEC_B.len(), SECRET_PREFIX.len() + SECRET_BODY);
    }

    #[test]
    fn scan_finds_glued_strings_across_pieces() {
        let d = tempfile::tempdir().unwrap();
        let mut b = vec![0x7fu8; 5 * CHUNK];
        let mut put = |at: usize, s: &[u8]| b[at..at + s.len()].copy_from_slice(s);
        // Glued to neighbouring text, as in the real binary.
        put(100, format!("xyz{ID_A}GetHub").as_bytes());
        put(
            5000,
            format!("authorize{SEC_A}https://cloudcode-pa").as_bytes(),
        );
        put(7000, b"/google_src/files/999/depot/google3");
        put(8000, b"/google_src/files/999/depot/google3");
        // Across piece boundaries (reads end at multiples of CHUNK): an id whose suffix is
        // cut, so its digits lie deep in the carried bytes; a secret and a changelist cut in
        // the middle. The changelist crossing a boundary counts once, so 999 wins.
        put(CHUNK - 64, format!("Q{ID_LONG}x").as_bytes());
        put(2 * CHUNK - 40, format!("Q{ID_B}x").as_bytes());
        put(3 * CHUNK - 20, format!("{SEC_B}field").as_bytes());
        put(4 * CHUNK - 10, b"google_src/files/1234/depot");
        // Repeats and near misses.
        put(2 * CHUNK + 4000, format!("{SEC_A}cloudcode").as_bytes());
        put(9000, b"GOCSPX-short!");
        put(
            9100,
            b"-abcdefghijklmnopqrstuvwxyz012345.apps.googleusercontent.com",
        );
        put(9200, b"google_src/files/x/depot");
        let p = temp(&d, "scan", &b);
        let s = scan_binary(&p).unwrap();
        assert_eq!(
            s.ids,
            vec![ID_A.to_owned(), ID_LONG.to_owned(), ID_B.to_owned()]
        );
        assert_eq!(s.secrets, vec![SEC_A.to_owned(), SEC_B.to_owned()]);
        assert_eq!(s.changelist.as_deref(), Some("999"));
    }

    #[test]
    fn find_agy_reports_where_it_looked() {
        let d = tempfile::tempdir().unwrap();
        let missing = Path::new("/nonexistent/agy");
        assert_eq!(find_agy(Some(missing)), Err(vec![missing.to_path_buf()]));
        let p = temp(&d, "agy-present", b"x");
        assert_eq!(find_agy(Some(&p)), Ok(p.clone()));
    }

    #[test]
    fn complete_config_skips_the_binary() {
        let d = tempfile::tempdir().unwrap();
        let p = temp(
            &d,
            "cfg.json",
            format!(r#"{{"client_id":"{ID_A}","client_secret":"{SEC_A}","user_agent":"ua/1"}}"#)
                .as_bytes(),
        );
        let c = resolve(Some(&p), Path::new("/nonexistent/agy")).unwrap();
        assert_eq!(c.candidates.counts(), (1, 1));
        assert_eq!(c.candidates.pair(0, 0).client_secret, SEC_A);
        assert_eq!(c.user_agent, "ua/1");
        assert!(!format!("{c:?}{:?}", c.candidates.pair(0, 0)).contains(SEC_A));

        std::fs::write(&p, br#"{"client_id":1}"#).unwrap();
        let e = resolve(Some(&p), Path::new("/nonexistent/agy")).unwrap_err();
        assert!(e.contains("client_secret"), "{e}");
    }

    #[test]
    fn binary_fills_what_the_config_leaves_unset() {
        let d = tempfile::tempdir().unwrap();
        let agy = fake_agy(
            &d,
            "agy-partial",
            &format!("..{SEC_B}https..{ID_A}..{SEC_A}cloudcode-pa google_src/files/55/depot"),
        );
        let c = resolve(Some(&no_config(&d)), &agy);
        // The flag points at a missing file: an error, not a silent fallback.
        assert!(c.unwrap_err().contains("cannot read"));

        let cfg = temp(&d, "cfg-partial.json", br#"{"user_agent":"mine/2"}"#);
        let c = resolve(Some(&cfg), &agy).unwrap();
        assert_eq!(c.candidates.counts(), (1, 2));
        assert_eq!(c.candidates.pair(0, 1).client_secret, SEC_A);
        assert_eq!(c.user_agent, "mine/2");

        std::fs::write(&cfg, b"{}").unwrap();
        let c = resolve(Some(&cfg), &agy).unwrap();
        let (os, arch) = go_os_arch();
        assert_eq!(
            c.user_agent,
            format!(
                "antigravity/cli/9.8.7 (aidev_client; os_type={os}; arch={arch}; cl=55; \
                 auth_method=consumer)"
            )
        );
    }

    #[test]
    fn missing_values_explain_without_secrets() {
        let d = tempfile::tempdir().unwrap();
        let cfg = temp(&d, "cfg-empty.json", b"{}");
        let agy = fake_agy(&d, "agy-noid", &format!("{SEC_A} but no id"));
        let e = resolve(Some(&cfg), &agy).unwrap_err();
        assert!(e.contains("no OAuth client id found"), "{e}");
        assert!(e.contains("no build changelist"), "{e}");
        assert!(
            e.contains("user_agent") && !e.contains("--no-google"),
            "{e}"
        );
        assert!(!e.contains(SEC_A), "{e}");

        let e = resolve(Some(&cfg), Path::new("/nonexistent/agy")).unwrap_err();
        assert!(
            e.contains("/nonexistent/agy") && e.contains("grep -aoE"),
            "{e}"
        );
        assert!(e.contains(&cfg.display().to_string()), "{e}");
    }
}
