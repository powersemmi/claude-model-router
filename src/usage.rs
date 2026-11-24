//! One JSON line per translated request, for measuring how much work a subscription gives.

use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::SystemTime;

use serde::Serialize;

/// The figures of one request.
#[derive(Serialize)]
pub struct UsageRecord<'a> {
    /// Provider name.
    pub provider: &'a str,
    /// Model the client asked for.
    pub model: &'a str,
    /// HTTP status returned to the client.
    pub status: u16,
    /// Prompt tokens, cached ones excluded.
    pub input_tokens: u64,
    /// Prompt tokens read from the cache.
    pub cache_read_tokens: u64,
    /// Generated tokens, thinking excluded.
    pub output_tokens: u64,
    /// Thinking tokens.
    pub thinking_tokens: u64,
    /// Milliseconds from receipt to the end of the response.
    pub latency_ms: u64,
    /// Milliseconds from receipt to the first generated token.
    pub ttft_ms: Option<u64>,
    /// The stop reason reported to the client.
    pub stop_reason: Option<&'a str>,
    /// An error message, when the request failed.
    pub error: Option<&'a str>,
}

#[derive(Serialize)]
struct Line<'a> {
    ts: &'a str,
    #[serde(flatten)]
    record: &'a UsageRecord<'a>,
}

/// An append-only usage log; a disabled log drops every record.
pub struct UsageLog {
    file: Option<Mutex<File>>,
    warned: AtomicBool,
}

impl UsageLog {
    /// Opens `path` for append, creating parent directories.
    pub fn open(path: &Path) -> io::Result<Self> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self {
            file: Some(Mutex::new(file)),
            warned: AtomicBool::new(false),
        })
    }

    /// A log that writes nothing.
    pub fn disabled() -> Self {
        Self {
            file: None,
            warned: AtomicBool::new(false),
        }
    }

    /// `<state dir>/claude-mr/usage.jsonl`, where the state dir is `$XDG_STATE_HOME` or
    /// `~/.local/state` on Linux (`dirs::state_dir`).
    pub fn default_path() -> Option<PathBuf> {
        Some(dirs::state_dir()?.join("claude-mr/usage.jsonl"))
    }

    /// Writes one line `{"ts":"2026-11-24T13:22:24.123Z", ...fields}`. Never panics; the first
    /// error goes to stderr and later ones are silent.
    pub fn record(&self, r: &UsageRecord<'_>) {
        let Some(file) = &self.file else { return };
        let ts = humantime::format_rfc3339_millis(SystemTime::now()).to_string();
        let mut buf = Vec::with_capacity(256);
        let line = Line { ts: &ts, record: r };
        if let Err(e) = serde_json::to_writer(&mut buf, &line) {
            self.warn(&e.to_string());
            return;
        }
        buf.push(b'\n');
        let result = match file.lock() {
            Ok(mut f) => f.write_all(&buf),
            Err(poisoned) => poisoned.into_inner().write_all(&buf),
        };
        if let Err(e) = result {
            self.warn(&e.to_string());
        }
    }

    fn warn(&self, msg: &str) {
        if !self.warned.swap(true, Ordering::Relaxed) {
            eprintln!("router: usage log write failed: {msg}");
        }
    }
}
