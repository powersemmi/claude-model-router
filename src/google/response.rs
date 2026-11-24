//! Translation of Gemini's `streamGenerateContent` server-sent events into Anthropic events.

use std::borrow::Cow;

use bytes::{Buf, BytesMut};
use serde::Deserialize;
use serde_json::value::RawValue;

use super::signatures::SignatureStore;
use crate::anthropic::stream::{MessageWriter, StopReason, Usage};

/// Splits an SSE byte stream into `data:` payloads. One buffer is kept and reused; events
/// split across chunks and several events in one chunk are handled, `\n` and `\r\n` both end
/// a line, and a blank line ends an event.
#[derive(Default)]
pub struct SseSplitter {
    line: BytesMut,
    event: BytesMut,
}

impl SseSplitter {
    /// An empty splitter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds `chunk` and calls `f` with the payload of every event it completes.
    pub fn feed(&mut self, chunk: &[u8], mut f: impl FnMut(&[u8])) {
        self.line.extend_from_slice(chunk);
        while let Some(pos) = self.line.iter().position(|&b| b == b'\n') {
            Self::process(&self.line[..pos], &mut self.event, &mut f);
            self.line.advance(pos + 1);
        }
    }

    /// Ends the stream: a trailing event without its blank line is delivered.
    pub fn finish(&mut self, mut f: impl FnMut(&[u8])) {
        if !self.line.is_empty() {
            Self::process(&self.line, &mut self.event, &mut f);
            self.line.clear();
        }
        if !self.event.is_empty() {
            f(&self.event);
            self.event.clear();
        }
    }

    fn process(line: &[u8], event: &mut BytesMut, f: &mut impl FnMut(&[u8])) {
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        if line.is_empty() {
            if !event.is_empty() {
                f(event);
                event.clear();
            }
        } else if let Some(rest) = line.strip_prefix(b"data:") {
            let rest = rest.strip_prefix(b" ").unwrap_or(rest);
            if !event.is_empty() {
                event.extend_from_slice(b"\n");
            }
            event.extend_from_slice(rest);
        }
    }
}

/// Token counts as Gemini reports them.
#[derive(Clone, Copy, Debug, Default)]
pub struct GeminiUsage {
    /// Prompt tokens, cached ones included.
    pub prompt: u64,
    /// Prompt tokens served from the cache.
    pub cached: u64,
    /// Answer tokens.
    pub candidates: u64,
    /// Thinking tokens.
    pub thoughts: u64,
}

impl GeminiUsage {
    fn anthropic(self, output: u64) -> Usage {
        Usage {
            input_tokens: self.prompt.saturating_sub(self.cached),
            cache_read_input_tokens: self.cached,
            output_tokens: output,
        }
    }
}

/// What an ended stream amounted to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// No content arrived before the end; nothing was written, the caller may retry.
    Empty,
    /// Gemini refused its own malformed tool call and nothing was emitted; nothing was
    /// written, the caller may retry.
    Malformed,
    /// The message was finished.
    Done,
}

#[derive(Deserialize)]
struct Chunk<'a> {
    #[serde(borrow)]
    response: Option<Body<'a>>,
    #[serde(borrow, flatten)]
    direct: Body<'a>,
}

#[derive(Deserialize, Default)]
struct Body<'a> {
    #[serde(borrow, default)]
    candidates: Vec<Candidate<'a>>,
    #[serde(default, rename = "usageMetadata")]
    usage: Option<UsageMeta>,
}

#[derive(Deserialize)]
struct Candidate<'a> {
    #[serde(borrow)]
    content: Option<Content<'a>>,
    #[serde(borrow, default, rename = "finishReason")]
    finish_reason: Option<Cow<'a, str>>,
}

#[derive(Deserialize)]
struct Content<'a> {
    #[serde(borrow, default)]
    parts: Vec<Part<'a>>,
}

#[derive(Deserialize)]
struct Part<'a> {
    #[serde(default)]
    thought: bool,
    #[serde(borrow, default)]
    text: Option<Cow<'a, str>>,
    #[serde(borrow, default, rename = "thoughtSignature")]
    signature: Option<Cow<'a, str>>,
    #[serde(borrow, default, rename = "functionCall")]
    call: Option<Call<'a>>,
}

#[derive(Deserialize)]
struct Call<'a> {
    #[serde(borrow)]
    name: Cow<'a, str>,
    #[serde(borrow, default)]
    args: Option<&'a RawValue>,
    #[serde(borrow, default)]
    id: Option<Cow<'a, str>>,
}

#[derive(Deserialize, Default)]
struct UsageMeta {
    #[serde(default, rename = "promptTokenCount")]
    prompt: Option<u64>,
    #[serde(default, rename = "cachedContentTokenCount")]
    cached: Option<u64>,
    #[serde(default, rename = "candidatesTokenCount")]
    candidates: Option<u64>,
    #[serde(default, rename = "thoughtsTokenCount")]
    thoughts: Option<u64>,
}

fn generated_id() -> String {
    format!("toolu_{}", uuid::Uuid::new_v4().simple())
}

/// Turns Gemini chunks into [`MessageWriter`] events.
pub struct GeminiTranslator<'r> {
    renamed: &'r [(String, String)],
    store: &'r SignatureStore,
    usage: GeminiUsage,
    emitted: bool,
    tool_called: bool,
    finish_reason: Option<String>,
    stop: Option<StopReason>,
}

impl<'r> GeminiTranslator<'r> {
    /// `renamed_tools` lists `(name sent to Gemini, original name)`; call names are mapped back.
    /// Signatures of function calls go to `store`, keyed by the call id.
    pub fn new(renamed_tools: &'r [(String, String)], store: &'r SignatureStore) -> Self {
        GeminiTranslator {
            renamed: renamed_tools,
            store,
            usage: GeminiUsage::default(),
            emitted: false,
            tool_called: false,
            finish_reason: None,
            stop: None,
        }
    }

    fn begin(&mut self, out: &mut MessageWriter) {
        if !self.emitted {
            self.emitted = true;
            out.set_usage(self.usage.anthropic(0));
        }
    }

    /// Handles one `data:` payload.
    pub fn on_event(
        &mut self,
        json: &[u8],
        out: &mut MessageWriter,
    ) -> Result<(), serde_json::Error> {
        let chunk: Chunk = serde_json::from_slice(json)?;
        let (candidates, usage) = match chunk.response {
            Some(b) => (b.candidates, b.usage),
            None => (chunk.direct.candidates, chunk.direct.usage),
        };
        if let Some(u) = usage {
            let g = &mut self.usage;
            g.prompt = u.prompt.unwrap_or(g.prompt);
            g.cached = u.cached.unwrap_or(g.cached);
            g.candidates = u.candidates.unwrap_or(g.candidates);
            g.thoughts = u.thoughts.unwrap_or(g.thoughts);
        }
        let Some(cand) = candidates.into_iter().next() else {
            return Ok(());
        };
        if let Some(content) = cand.content {
            for part in content.parts {
                self.part(part, out);
            }
        }
        if let Some(r) = cand.finish_reason {
            self.finish_reason = Some(r.into_owned());
        }
        Ok(())
    }

    fn part(&mut self, part: Part<'_>, out: &mut MessageWriter) {
        if let Some(call) = part.call {
            self.begin(out);
            let id: Cow<str> = call.id.unwrap_or_else(|| Cow::Owned(generated_id()));
            let name = self
                .renamed
                .iter()
                .find(|(sent, _)| *sent == call.name)
                .map_or(&*call.name, |(_, orig)| orig.as_str());
            if let Some(sig) = part.signature.as_deref().filter(|s| !s.is_empty()) {
                self.store.put(&id, sig);
            }
            out.tool_use(&id, name, call.args.map_or("{}", |a| a.get()));
            self.tool_called = true;
            return;
        }
        let text = part.text.as_deref().unwrap_or("");
        if text.is_empty() {
            return;
        }
        self.begin(out);
        if part.thought {
            out.thinking(text);
            if let Some(sig) = part.signature.as_deref().filter(|s| !s.is_empty()) {
                out.thinking_signature(sig);
            }
        } else {
            out.text(text);
        }
    }

    /// Ends the message. With [`Outcome::Done`] the writer holds the final event; with
    /// [`Outcome::Empty`] or [`Outcome::Malformed`] nothing is written.
    pub fn finish(&mut self, out: &mut MessageWriter) -> Outcome {
        let owned = self.finish_reason.clone();
        let reason = owned.as_deref();
        let refused = matches!(
            reason,
            Some(
                "SAFETY"
                    | "RECITATION"
                    | "PROHIBITED_CONTENT"
                    | "BLOCKLIST"
                    | "SPII"
                    | "IMAGE_SAFETY"
            )
        );
        let malformed = matches!(
            reason,
            Some("MALFORMED_FUNCTION_CALL" | "UNEXPECTED_TOOL_CALL")
        );
        if !self.emitted {
            if refused {
                // A refusal with no text is the answer: report it, do not retry.
                self.begin(out);
            } else if malformed {
                return Outcome::Malformed;
            } else {
                return Outcome::Empty;
            }
        }
        let stop = match reason {
            Some("MAX_TOKENS") => StopReason::MaxTokens,
            _ if refused => StopReason::Refusal,
            _ if self.tool_called => StopReason::ToolUse,
            _ => StopReason::EndTurn,
        };
        self.stop = Some(stop);
        let g = self.usage;
        out.finish(stop, g.anthropic(g.candidates + g.thoughts));
        Outcome::Done
    }

    /// The latest usage Gemini reported.
    pub fn usage(&self) -> GeminiUsage {
        self.usage
    }

    /// The stop reason sent by [`finish`](Self::finish).
    pub fn stop_reason(&self) -> Option<StopReason> {
        self.stop
    }

    /// Gemini's own `finishReason`, for logs.
    pub fn finish_reason(&self) -> Option<&str> {
        self.finish_reason.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const THOUGHT: &str = r#"data: {"response": {"candidates": [{"content": {"role": "model","parts": [{"thought": true,"text": "The request is to read a file\n\n\n"}]}}],"usageMetadata": {"promptTokenCount": 10408,"totalTokenCount": 10535,"thoughtsTokenCount": 127},"modelVersion": "gemini-4-neon","responseId": "sI8F"},"traceId": "eff9","metadata": {}}"#;
    const CALL: &str = r#"data: {"response": {"candidates": [{"content": {"role": "model","parts": [{"thoughtSignature": "ErsFCrgF","functionCall": {"name": "read_file","args": {"Path": "/x/note.txt"},"id": "call_134"}}]}}],"usageMetadata": {"promptTokenCount": 10408,"candidatesTokenCount": 78,"totalTokenCount": 10655,"thoughtsTokenCount": 169},"modelVersion": "gemini-4-neon","responseId": "sI8F"},"traceId": "eff9","metadata": {}}"#;
    const STOP: &str = r#"data: {"response": {"candidates": [{"content": {"role": "model","parts": [{"text": ""}]},"finishReason": "STOP"}],"usageMetadata": {"promptTokenCount": 10408,"candidatesTokenCount": 81,"totalTokenCount": 10658,"thoughtsTokenCount": 169},"modelVersion": "gemini-4-neon","responseId": "sI8F"},"traceId": "eff9","metadata": {}}"#;

    fn stream(events: &[&str], crlf: bool) -> Vec<u8> {
        let nl = if crlf { "\r\n" } else { "\n" };
        events
            .iter()
            .flat_map(|e| format!("{}{nl}{nl}", e.replace('\n', "\\n")).into_bytes())
            .collect()
    }

    fn parse_sse(b: &[u8]) -> Vec<(String, Value)> {
        let s = std::str::from_utf8(b).unwrap();
        s.split("\n\n")
            .filter(|e| !e.is_empty())
            .map(|e| {
                let (name, data) = e.split_once("\ndata: ").unwrap();
                let mut v: Value = serde_json::from_str(data).unwrap();
                if let Some(m) = v.get_mut("message") {
                    m["id"] = Value::Null;
                }
                (name.strip_prefix("event: ").unwrap().to_owned(), v)
            })
            .collect()
    }

    fn run(
        input: &[u8],
        step: usize,
        renamed: &[(String, String)],
        store: &SignatureStore,
    ) -> (Vec<(String, Value)>, Outcome, GeminiUsage) {
        let mut w = MessageWriter::sse("m").with_placeholder_signature("PH");
        let mut t = GeminiTranslator::new(renamed, store);
        let mut sp = SseSplitter::new();
        let mut events: Vec<Vec<u8>> = Vec::new();
        for c in input.chunks(step) {
            sp.feed(c, |e| events.push(e.to_vec()));
        }
        sp.finish(|e| events.push(e.to_vec()));
        for e in &events {
            t.on_event(e, &mut w).unwrap();
        }
        let o = t.finish(&mut w);
        let bytes = w.take().unwrap_or_default();
        (parse_sse(&bytes), o, t.usage())
    }

    #[test]
    fn captured_stream_any_chunking() {
        let input = stream(&[THOUGHT, CALL, STOP], false);
        let store = SignatureStore::new(8);
        let (whole, o, u) = run(&input, input.len(), &[], &store);
        assert_eq!(o, Outcome::Done);
        assert_eq!((u.prompt, u.thoughts, u.candidates), (10408, 169, 81));
        let (bytewise, _, _) = run(&input, 1, &[], &store);
        assert_eq!(whole, bytewise);
        let crlf = stream(&[THOUGHT, CALL, STOP], true);
        let (c, _, _) = run(&crlf, 7, &[], &store);
        assert_eq!(whole, c);
        assert_eq!(&*store.get("call_134").unwrap(), "ErsFCrgF");
        let s = serde_json::to_string(&whole).unwrap();
        assert!(s.contains("thinking_delta") && s.contains("input_json_delta"));
        assert!(s.contains("\"stop_reason\":\"tool_use\""));
        assert!(s.contains("\"output_tokens\":250"));
        assert!(s.contains("\"input_tokens\":10408"));
    }

    #[test]
    fn unwrapped_chunk() {
        let ev = r#"data: {"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":5,"cachedContentTokenCount":3,"candidatesTokenCount":1}}"#;
        let store = SignatureStore::new(2);
        let input = stream(&[ev], false);
        let (evs, o, u) = run(&input, 5, &[], &store);
        assert_eq!(o, Outcome::Done);
        assert_eq!(u.cached, 3);
        let s = serde_json::to_string(&evs).unwrap();
        assert!(s.contains("\"cache_read_input_tokens\":3"));
        assert!(s.contains("\"input_tokens\":2"));
        assert!(s.contains("\"text\":\"ok\"") && s.contains("end_turn"));
    }

    #[test]
    fn parallel_calls_and_renamed() {
        let ev = r#"data: {"response":{"candidates":[{"content":{"parts":[{"thoughtSignature":"S1","functionCall":{"name":"mcp_x","args":{"a":1},"id":"c1"}},{"functionCall":{"name":"b","args":{}}}]},"finishReason":"STOP"}]}}"#;
        let store = SignatureStore::new(8);
        let renamed = vec![("mcp_x".to_owned(), "mcp__x".to_owned())];
        let input = stream(&[ev], false);
        let (evs, o, _) = run(&input, 64, &renamed, &store);
        assert_eq!(o, Outcome::Done);
        let s = serde_json::to_string(&evs).unwrap();
        assert!(s.contains("\"name\":\"mcp__x\""));
        assert!(s.contains("\"name\":\"b\""));
        assert!(s.contains("toolu_"));
        assert_eq!(&*store.get("c1").unwrap(), "S1");
    }

    #[test]
    fn max_tokens_and_signature_on_empty_text() {
        let a = r#"data: {"response":{"candidates":[{"content":{"parts":[{"text":"cut"}]}}]}}"#;
        let b = r#"data: {"response":{"candidates":[{"content":{"parts":[{"thoughtSignature":"EV","text":""}]},"finishReason":"MAX_TOKENS"}]}}"#;
        let store = SignatureStore::new(2);
        let (evs, o, _) = run(&stream(&[a, b], false), 3, &[], &store);
        assert_eq!(o, Outcome::Done);
        let s = serde_json::to_string(&evs).unwrap();
        assert!(s.contains("max_tokens") && !s.contains("EV"));
    }

    #[test]
    fn empty_and_malformed() {
        let store = SignatureStore::new(2);
        let (e, o, _) = run(b"", 1, &[], &store);
        assert_eq!(o, Outcome::Empty);
        assert!(e.is_empty());
        let m = r#"data: {"response":{"candidates":[{"finishReason":"MALFORMED_FUNCTION_CALL"}]}}"#;
        let (e, o, _) = run(&stream(&[m], false), 9, &[], &store);
        assert_eq!(o, Outcome::Malformed);
        assert!(e.is_empty());
        let s = r#"data: {"response":{"candidates":[{"finishReason":"SAFETY"}]}}"#;
        let (e, o, _) = run(&stream(&[s], false), 9, &[], &store);
        assert_eq!(o, Outcome::Done);
        assert!(serde_json::to_string(&e).unwrap().contains("refusal"));
    }

    #[test]
    fn trailing_event_without_blank_line() {
        let store = SignatureStore::new(2);
        let ev = r#"data: {"response":{"candidates":[{"content":{"parts":[{"text":"x"}]},"finishReason":"STOP"}]}}"#;
        let (_, o, _) = run(ev.as_bytes(), 4, &[], &store);
        assert_eq!(o, Outcome::Done);
    }
}
