//! One JSON line per translated request, for measuring how much work a subscription gives.

use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

/// `t` in UTC as `2026-11-24T13:22:24.123Z`. A time before 1970 prints as the epoch.
fn rfc3339_millis(t: SystemTime) -> String {
    let ms = t.duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis());
    let secs = (ms / 1000) as i64;
    let (days, sod) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    let (y, m, d) = civil_from_days(days);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}.{:03}Z",
        sod / 3600,
        sod % 3600 / 60,
        sod % 60,
        ms % 1000
    )
}

/// The proleptic Gregorian date of a day count since 1970-01-01 (Howard Hinnant's
/// `civil_from_days`).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = yoe + era * 400 + i64::from(m <= 2);
    (y, m, d)
}

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
        let ts = rfc3339_millis(SystemTime::now());
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn rfc3339_matches_known_instants() {
        let at = |ms: u64| rfc3339_millis(UNIX_EPOCH + Duration::from_millis(ms));
        assert_eq!(at(0), "1970-01-01T00:00:00.000Z");
        assert_eq!(at(951_782_400_000), "2000-02-29T00:00:00.000Z");
        assert_eq!(at(1_795_526_544_123), "2026-11-24T13:22:24.123Z");
        assert_eq!(at(4_107_542_399_999), "2100-02-28T23:59:59.999Z");
        assert_eq!(at(4_107_542_400_000), "2100-03-01T00:00:00.000Z");
    }
}
