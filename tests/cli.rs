//! The `claude-mr` binary at startup: Google is served only where the Antigravity CLI is
//! installed, and missing Google values never stop the router.

use std::path::Path;
use std::process::Stdio;

use tempfile::TempDir;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::time::timeout;

use super::LIMIT;

/// A fresh directory for one test, used as `HOME`; removed when the guard drops.
fn home() -> TempDir {
    let d = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(d.path().join("bin")).unwrap();
    d
}

/// An `agy` that prints a version and holds no OAuth client.
fn fake_agy(dir: &Path) {
    let p = dir.join("bin").join("agy");
    std::fs::write(&p, "#!/bin/sh\necho 1.0.0\n").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
}

/// Starts the router with `PATH` set to `dir/bin` only and returns it with every stderr
/// line up to and including the startup line.
async fn start(dir: &Path, args: &[&str]) -> (Child, Vec<String>) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_claude-mr"))
        .args(["--port", "0", "--usage-log"])
        .arg(dir.join("usage.jsonl"))
        .args(args)
        .env_clear()
        .env("PATH", dir.join("bin"))
        .env("HOME", dir)
        .env("XDG_CONFIG_HOME", dir.join("config"))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    let mut lines = BufReader::new(child.stderr.take().unwrap()).lines();
    let mut seen = Vec::new();
    timeout(LIMIT, async {
        while let Some(l) = lines.next_line().await.unwrap() {
            let done = l.starts_with("router: 127.0.0.1:");
            seen.push(l);
            if done {
                return;
            }
        }
        panic!("router exited: {seen:?}");
    })
    .await
    .unwrap();
    (child, seen)
}

#[tokio::test]
async fn without_agy_google_is_absent_and_silent() {
    let home = home();
    let dir = home.path();
    let (_child, lines) = start(dir, &[]).await;
    assert_eq!(lines.len(), 1, "{lines:?}");
    assert!(!lines[0].contains("gemini"), "{lines:?}");

    let (_child, lines) = start(dir, &["-v"]).await;
    assert_eq!(lines.len(), 2, "{lines:?}");
    assert!(
        lines[0].contains("Google models are not served"),
        "{lines:?}"
    );
    assert!(lines[0].contains("agy"), "{lines:?}");
    assert!(!lines[1].contains("gemini"), "{lines:?}");
}

#[tokio::test]
async fn agy_without_values_starts_and_explains_on_request() {
    let home = home();
    let dir = home.path();
    fake_agy(dir);
    let (_child, lines) = start(dir, &[]).await;
    assert_eq!(lines.len(), 1, "{lines:?}");
    assert!(
        lines[0].contains("gemini-4-neon-medium -> google"),
        "{lines:?}"
    );

    let (_child, lines) = start(dir, &["--verbose"]).await;
    let all = lines.join("\n");
    assert!(all.contains("Google models are unavailable"), "{all}");
    assert!(
        all.contains("\"client_id\"") && all.contains("\"user_agent\""),
        "{all}"
    );
    assert!(all.contains("config/claude-mr/google.json"), "{all}");
    let port: u16 = lines
        .last()
        .unwrap()
        .strip_prefix("router: 127.0.0.1:")
        .and_then(|r| r.split(';').next())
        .and_then(|p| p.parse().ok())
        .unwrap();

    let mut s = tokio::net::TcpStream::connect(("127.0.0.1", port))
        .await
        .unwrap();
    let body = r#"{"model":"gemini-4-neon-medium","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}"#;
    let req = format!(
        "POST /v1/messages HTTP/1.1\r\nhost: x\r\ncontent-type: application/json\r\n\
         content-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    );
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    s.write_all(req.as_bytes()).await.unwrap();
    let mut out = String::new();
    timeout(LIMIT, s.read_to_string(&mut out))
        .await
        .unwrap()
        .unwrap();
    assert!(out.starts_with("HTTP/1.1 400"), "{out}");
    assert!(out.contains("google.json"), "{out}");

    let (_child, lines) = start(dir, &["--no-google", "-v"]).await;
    assert_eq!(lines.len(), 1, "{lines:?}");
    assert!(!lines[0].contains("gemini"), "{lines:?}");
}
