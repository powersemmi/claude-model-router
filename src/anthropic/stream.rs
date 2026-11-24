//! The Anthropic response writer: one object that turns semantic events (text, thinking, tool
//! calls, finish) into either an SSE byte stream or one collected Message JSON.
//!
//! The writer owns block indices, opening and closing of blocks, event order and serialization.
//! In SSE mode every event is serialized straight into one reused [`BytesMut`].

use bytes::{BufMut, Bytes, BytesMut};
use serde::Serialize;
use serde_json::value::RawValue;

/// Why the model stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    /// The model finished its turn.
    EndTurn,
    /// The output token limit was reached.
    MaxTokens,
    /// The model called a tool.
    ToolUse,
    /// A stop sequence matched.
    StopSequence,
    /// The model refused.
    Refusal,
}

impl StopReason {
    /// The wire name of the reason.
    pub fn as_str(self) -> &'static str {
        match self {
            StopReason::EndTurn => "end_turn",
            StopReason::MaxTokens => "max_tokens",
            StopReason::ToolUse => "tool_use",
            StopReason::StopSequence => "stop_sequence",
            StopReason::Refusal => "refusal",
        }
    }
}

/// Token counts of a message.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    /// Input tokens (cache reads excluded).
    pub input_tokens: u64,
    /// Input tokens served from the cache.
    pub cache_read_input_tokens: u64,
    /// Output tokens.
    pub output_tokens: u64,
}

#[derive(Serialize)]
struct UsageOut {
    input_tokens: u64,
    output_tokens: u64,
    cache_read_input_tokens: u64,
    cache_creation_input_tokens: u64,
}

impl UsageOut {
    fn new(u: Usage, output_tokens: u64) -> Self {
        UsageOut {
            input_tokens: u.input_tokens,
            output_tokens,
            cache_read_input_tokens: u.cache_read_input_tokens,
            cache_creation_input_tokens: 0,
        }
    }
}

#[derive(Serialize)]
struct Head<'a> {
    id: &'a str,
    #[serde(rename = "type")]
    kind: &'static str,
    role: &'static str,
    model: &'a str,
    content: [(); 0],
    stop_reason: Option<&'static str>,
    stop_sequence: Option<&'static str>,
    usage: UsageOut,
}

#[derive(Serialize)]
struct Empty {}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum StartBlock<'a> {
    Text {
        text: &'a str,
    },
    Thinking {
        thinking: &'a str,
        signature: &'a str,
    },
    ToolUse {
        id: &'a str,
        name: &'a str,
        input: Empty,
    },
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Delta<'a> {
    #[serde(rename = "text_delta")]
    Text { text: &'a str },
    #[serde(rename = "thinking_delta")]
    Thinking { thinking: &'a str },
    #[serde(rename = "signature_delta")]
    Signature { signature: &'a str },
    #[serde(rename = "input_json_delta")]
    InputJson { partial_json: &'a str },
}

#[derive(Serialize)]
struct MsgDelta {
    stop_reason: &'static str,
    stop_sequence: Option<&'static str>,
}

#[derive(Serialize)]
struct ErrBody<'a> {
    #[serde(rename = "type")]
    kind: &'a str,
    message: &'a str,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Event<'a> {
    MessageStart {
        message: Head<'a>,
    },
    ContentBlockStart {
        index: u32,
        content_block: StartBlock<'a>,
    },
    ContentBlockDelta {
        index: u32,
        delta: Delta<'a>,
    },
    ContentBlockStop {
        index: u32,
    },
    MessageDelta {
        delta: MsgDelta,
        usage: UsageOut,
    },
    MessageStop {},
    Error {
        error: ErrBody<'a>,
    },
}

impl Event<'_> {
    fn name(&self) -> &'static str {
        match self {
            Event::MessageStart { .. } => "message_start",
            Event::ContentBlockStart { .. } => "content_block_start",
            Event::ContentBlockDelta { .. } => "content_block_delta",
            Event::ContentBlockStop { .. } => "content_block_stop",
            Event::MessageDelta { .. } => "message_delta",
            Event::MessageStop {} => "message_stop",
            Event::Error { .. } => "error",
        }
    }
}

/// Serializes one SSE event into the buffer.
fn emit(buf: &mut BytesMut, ev: &Event<'_>) {
    buf.extend_from_slice(b"event: ");
    buf.extend_from_slice(ev.name().as_bytes());
    buf.extend_from_slice(b"\ndata: ");
    // Writing into a BytesMut cannot fail and these types always serialize.
    let _ = serde_json::to_writer(buf.writer(), ev);
    buf.extend_from_slice(b"\n\n");
}

/// The currently open content block.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Open {
    None,
    Text,
    Thinking { signed: bool },
}

/// A block of the collected (non-stream) message.
enum Collected {
    Text(String),
    Thinking {
        text: String,
        signature: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Box<RawValue>,
    },
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum CollectedOut<'a> {
    Text {
        text: &'a str,
    },
    Thinking {
        thinking: &'a str,
        signature: &'a str,
    },
    ToolUse {
        id: &'a str,
        name: &'a str,
        input: &'a RawValue,
    },
}

#[derive(Serialize)]
struct MessageOut<'a> {
    id: &'a str,
    #[serde(rename = "type")]
    kind: &'static str,
    role: &'static str,
    model: &'a str,
    content: Vec<CollectedOut<'a>>,
    stop_reason: Option<&'static str>,
    stop_sequence: Option<&'static str>,
    usage: UsageOut,
}

/// A random id `msg_` + 32 hex characters (one v4 UUID).
fn message_id() -> String {
    format!("msg_{}", uuid::Uuid::new_v4().simple())
}

/// Writes one Anthropic message as an SSE stream or as collected JSON.
pub struct MessageWriter {
    sse: bool,
    id: String,
    model: String,
    placeholder: Option<&'static str>,
    buf: BytesMut,
    started: bool,
    finished: bool,
    errored: bool,
    usage: Usage,
    final_usage: Usage,
    stop: Option<StopReason>,
    open: Open,
    /// Number of blocks opened so far; the open block's index is `count - 1`.
    count: u32,
    blocks: Vec<Collected>,
    error: Option<(String, String)>,
}

impl MessageWriter {
    fn new(model: &str, sse: bool) -> Self {
        MessageWriter {
            sse,
            id: message_id(),
            model: model.to_owned(),
            placeholder: None,
            buf: BytesMut::new(),
            started: false,
            finished: false,
            errored: false,
            usage: Usage::default(),
            final_usage: Usage::default(),
            stop: None,
            open: Open::None,
            count: 0,
            blocks: Vec::new(),
            error: None,
        }
    }

    /// A writer producing SSE bytes (streaming mode).
    pub fn sse(model: &str) -> Self {
        Self::new(model, true)
    }

    /// A writer collecting one Message JSON (non-stream mode).
    pub fn collect(model: &str) -> Self {
        Self::new(model, false)
    }

    /// A thinking block closed without a signature gets this signature first (Claude Code
    /// expects every thinking block to carry one).
    pub fn with_placeholder_signature(mut self, sig: &'static str) -> Self {
        self.placeholder = Some(sig);
        self
    }

    fn live(&self) -> bool {
        !self.finished && !self.errored
    }

    /// Emits `message_start` with the usage set so far. Idempotent; the first content event
    /// calls it implicitly.
    pub fn start(&mut self) {
        if self.started || !self.live() {
            return;
        }
        self.started = true;
        if self.sse {
            let ev = Event::MessageStart {
                message: Head {
                    id: &self.id,
                    kind: "message",
                    role: "assistant",
                    model: &self.model,
                    content: [],
                    stop_reason: None,
                    stop_sequence: None,
                    usage: UsageOut::new(self.usage, 0),
                },
            };
            emit(&mut self.buf, &ev);
        }
    }

    /// Sets the usage reported by `message_start`.
    pub fn set_usage(&mut self, u: Usage) {
        self.usage = u;
    }

    fn close_block(&mut self) {
        let open = std::mem::replace(&mut self.open, Open::None);
        if open == Open::None {
            return;
        }
        let index = self.count - 1;
        if let Open::Thinking { signed: false } = open {
            if let Some(sig) = self.placeholder {
                if self.sse {
                    let ev = Event::ContentBlockDelta {
                        index,
                        delta: Delta::Signature { signature: sig },
                    };
                    emit(&mut self.buf, &ev);
                } else if let Some(Collected::Thinking { signature, .. }) = self.blocks.last_mut() {
                    signature.push_str(sig);
                }
            }
        }
        if self.sse {
            emit(&mut self.buf, &Event::ContentBlockStop { index });
        }
    }

    fn begin_block(&mut self, open: Open, start: StartBlock<'_>, collected: Collected) {
        self.close_block();
        if self.sse {
            let ev = Event::ContentBlockStart {
                index: self.count,
                content_block: start,
            };
            emit(&mut self.buf, &ev);
        } else {
            self.blocks.push(collected);
        }
        self.count += 1;
        self.open = open;
    }

    /// Appends text, opening a text block if the open block is not one.
    pub fn text(&mut self, s: &str) {
        if s.is_empty() || !self.live() {
            return;
        }
        self.start();
        if self.open != Open::Text {
            self.begin_block(
                Open::Text,
                StartBlock::Text { text: "" },
                Collected::Text(String::new()),
            );
        }
        if self.sse {
            let ev = Event::ContentBlockDelta {
                index: self.count - 1,
                delta: Delta::Text { text: s },
            };
            emit(&mut self.buf, &ev);
        } else if let Some(Collected::Text(t)) = self.blocks.last_mut() {
            t.push_str(s);
        }
    }

    /// Appends thinking text, opening a thinking block if the open block is not one.
    pub fn thinking(&mut self, s: &str) {
        if s.is_empty() || !self.live() {
            return;
        }
        self.start();
        if !matches!(self.open, Open::Thinking { .. }) {
            self.begin_block(
                Open::Thinking { signed: false },
                StartBlock::Thinking {
                    thinking: "",
                    signature: "",
                },
                Collected::Thinking {
                    text: String::new(),
                    signature: String::new(),
                },
            );
        }
        if self.sse {
            let ev = Event::ContentBlockDelta {
                index: self.count - 1,
                delta: Delta::Thinking { thinking: s },
            };
            emit(&mut self.buf, &ev);
        } else if let Some(Collected::Thinking { text, .. }) = self.blocks.last_mut() {
            text.push_str(s);
        }
    }

    /// Sets the signature of the open thinking block; ignored when no thinking block is open.
    pub fn thinking_signature(&mut self, sig: &str) {
        if !self.live() || !matches!(self.open, Open::Thinking { .. }) {
            return;
        }
        self.open = Open::Thinking { signed: true };
        if self.sse {
            let ev = Event::ContentBlockDelta {
                index: self.count - 1,
                delta: Delta::Signature { signature: sig },
            };
            emit(&mut self.buf, &ev);
        } else if let Some(Collected::Thinking { signature, .. }) = self.blocks.last_mut() {
            signature.clear();
            signature.push_str(sig);
        }
    }

    /// Writes a whole tool call. `input_json` is already-valid JSON text.
    pub fn tool_use(&mut self, id: &str, name: &str, input_json: &str) {
        if !self.live() {
            return;
        }
        self.start();
        let collected = if self.sse {
            Collected::Text(String::new())
        } else {
            let input = serde_json::from_str::<Box<RawValue>>(input_json)
                .unwrap_or_else(|_| RawValue::from_string("{}".to_owned()).expect("valid"));
            Collected::ToolUse {
                id: id.to_owned(),
                name: name.to_owned(),
                input,
            }
        };
        self.begin_block(
            Open::None,
            StartBlock::ToolUse {
                id,
                name,
                input: Empty {},
            },
            collected,
        );
        if self.sse {
            let index = self.count - 1;
            let ev = Event::ContentBlockDelta {
                index,
                delta: Delta::InputJson {
                    partial_json: input_json,
                },
            };
            emit(&mut self.buf, &ev);
            emit(&mut self.buf, &Event::ContentBlockStop { index });
        }
    }

    /// Closes the open block and ends the message. A no-op after `finish` or `error`.
    pub fn finish(&mut self, stop: StopReason, usage: Usage) {
        if !self.live() {
            return;
        }
        self.start();
        self.close_block();
        self.finished = true;
        self.stop = Some(stop);
        self.final_usage = usage;
        if self.sse {
            let ev = Event::MessageDelta {
                delta: MsgDelta {
                    stop_reason: stop.as_str(),
                    stop_sequence: None,
                },
                usage: UsageOut::new(usage, usage.output_tokens),
            };
            emit(&mut self.buf, &ev);
            emit(&mut self.buf, &Event::MessageStop {});
        }
    }

    /// Reports an error: an `event: error` in SSE mode, remembered for [`into_json`](Self::into_json)
    /// in collect mode.
    pub fn error(&mut self, kind: &str, message: &str) {
        if self.finished || self.errored {
            return;
        }
        self.errored = true;
        if self.sse {
            let ev = Event::Error {
                error: ErrBody { kind, message },
            };
            emit(&mut self.buf, &ev);
        } else {
            self.error = Some((kind.to_owned(), message.to_owned()));
        }
    }

    /// Whether any content block was emitted.
    pub fn has_content(&self) -> bool {
        self.count > 0
    }

    /// Whether `finish` was called.
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// SSE mode: the bytes produced since the last take, `None` if there are none.
    pub fn take(&mut self) -> Option<Bytes> {
        if self.buf.is_empty() {
            None
        } else {
            Some(self.buf.split().freeze())
        }
    }

    /// Collect mode: the full Message JSON, or the error `(kind, message)` reported by [`error`](Self::error).
    pub fn into_json(self) -> Result<Bytes, (String, String)> {
        if let Some(e) = self.error {
            return Err(e);
        }
        let content = self
            .blocks
            .iter()
            .map(|b| match b {
                Collected::Text(t) => CollectedOut::Text { text: t },
                Collected::Thinking { text, signature } => CollectedOut::Thinking {
                    thinking: text,
                    signature,
                },
                Collected::ToolUse { id, name, input } => CollectedOut::ToolUse { id, name, input },
            })
            .collect();
        let usage = if self.finished {
            self.final_usage
        } else {
            self.usage
        };
        let msg = MessageOut {
            id: &self.id,
            kind: "message",
            role: "assistant",
            model: &self.model,
            content,
            stop_reason: self.stop.map(StopReason::as_str),
            stop_sequence: None,
            usage: UsageOut::new(usage, usage.output_tokens),
        };
        let mut out = BytesMut::with_capacity(512);
        match serde_json::to_writer((&mut out).writer(), &msg) {
            Ok(()) => Ok(out.freeze()),
            Err(e) => Err(("api_error".to_owned(), e.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn parse_sse(b: &[u8]) -> Vec<(String, Value)> {
        let s = std::str::from_utf8(b).unwrap();
        s.split("\n\n")
            .filter(|e| !e.is_empty())
            .map(|e| {
                let (name, data) = e.split_once("\ndata: ").unwrap();
                (
                    name.strip_prefix("event: ").unwrap().to_owned(),
                    serde_json::from_str(data).unwrap(),
                )
            })
            .collect()
    }

    const USAGE: Usage = Usage {
        input_tokens: 10,
        cache_read_input_tokens: 4,
        output_tokens: 7,
    };

    fn feed(w: &mut MessageWriter) {
        w.set_usage(Usage {
            input_tokens: 10,
            cache_read_input_tokens: 4,
            output_tokens: 0,
        });
        w.thinking("hmm ");
        w.thinking("ok");
        w.thinking_signature("SIG");
        w.text("");
        w.text("say \"hi\"\n\u{1F600} \u{e9}");
        w.tool_use("toolu_1", "Read", r#"{"path": "/a"}"#);
        w.finish(StopReason::ToolUse, USAGE);
    }

    #[test]
    fn sse_sequence() {
        let mut w = MessageWriter::sse("gemini-x");
        feed(&mut w);
        assert!(w.has_content() && w.is_finished());
        let raw = w.take().unwrap();
        assert!(w.take().is_none());
        assert!(raw.starts_with(b"event: message_start\ndata: {"));
        let evs = parse_sse(&raw);
        let names: Vec<&str> = evs.iter().map(|e| e.0.as_str()).collect();
        assert_eq!(
            names,
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        let m = &evs[0].1["message"];
        assert!(m["id"].as_str().unwrap().starts_with("msg_"));
        assert_eq!(m["id"].as_str().unwrap().len(), 36);
        assert_eq!(m["model"], "gemini-x");
        assert_eq!(m["usage"]["input_tokens"], 10);
        assert_eq!(m["usage"]["cache_read_input_tokens"], 4);
        assert_eq!(m["usage"]["output_tokens"], 0);
        assert_eq!(m["content"], json!([]));
        assert_eq!(evs[1].1["index"], 0);
        assert_eq!(
            evs[1].1["content_block"],
            json!({"type":"thinking","thinking":"","signature":""})
        );
        assert_eq!(
            evs[2].1["delta"],
            json!({"type":"thinking_delta","thinking":"hmm "})
        );
        assert_eq!(
            evs[4].1["delta"],
            json!({"type":"signature_delta","signature":"SIG"})
        );
        assert_eq!(evs[6].1["index"], 1);
        assert_eq!(evs[6].1["content_block"], json!({"type":"text","text":""}));
        assert_eq!(
            evs[7].1["delta"],
            json!({"type":"text_delta","text":"say \"hi\"\n\u{1F600} \u{e9}"})
        );
        assert_eq!(evs[9].1["index"], 2);
        assert_eq!(
            evs[9].1["content_block"],
            json!({"type":"tool_use","id":"toolu_1","name":"Read","input":{}})
        );
        assert_eq!(
            evs[10].1["delta"],
            json!({"type":"input_json_delta","partial_json":"{\"path\": \"/a\"}"})
        );
        assert_eq!(evs[11].1["index"], 2);
        assert_eq!(evs[12].1["delta"]["stop_reason"], "tool_use");
        assert_eq!(evs[12].1["delta"]["stop_sequence"], Value::Null);
        assert_eq!(
            evs[12].1["usage"],
            json!({"input_tokens":10,"output_tokens":7,"cache_read_input_tokens":4,"cache_creation_input_tokens":0})
        );
        assert_eq!(evs[13].1, json!({"type":"message_stop"}));
    }

    #[test]
    fn placeholder_signature_applied_once() {
        let mut w = MessageWriter::sse("m").with_placeholder_signature("PLACEHOLDER");
        w.thinking("a");
        w.text("b");
        w.thinking("c");
        w.thinking_signature("REAL");
        w.finish(StopReason::EndTurn, Usage::default());
        let evs = parse_sse(&w.take().unwrap());
        let sigs: Vec<(u64, &str)> = evs
            .iter()
            .filter(|e| e.1["delta"]["type"] == "signature_delta")
            .map(|e| {
                (
                    e.1["index"].as_u64().unwrap(),
                    e.1["delta"]["signature"].as_str().unwrap(),
                )
            })
            .collect();
        assert_eq!(sigs, [(0, "PLACEHOLDER"), (2, "REAL")]);
        // The placeholder comes before the stop of its block.
        let names: Vec<&str> = evs.iter().map(|e| e.0.as_str()).collect();
        assert_eq!(
            &names[..5],
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop"
            ]
        );
    }

    #[test]
    fn signature_without_open_thinking_is_ignored() {
        let mut w = MessageWriter::sse("m");
        w.text("x");
        w.thinking_signature("S");
        w.finish(StopReason::EndTurn, Usage::default());
        let raw = w.take().unwrap();
        assert!(!std::str::from_utf8(&raw)
            .unwrap()
            .contains("signature_delta"));
    }

    #[test]
    fn collect_json() {
        let mut w = MessageWriter::collect("gemini-x");
        feed(&mut w);
        let v: Value = serde_json::from_slice(&w.into_json().unwrap()).unwrap();
        assert_eq!(v["type"], "message");
        assert_eq!(v["role"], "assistant");
        assert_eq!(v["model"], "gemini-x");
        assert_eq!(v["stop_reason"], "tool_use");
        assert_eq!(v["stop_sequence"], Value::Null);
        assert_eq!(
            v["content"],
            json!([
                {"type":"thinking","thinking":"hmm ok","signature":"SIG"},
                {"type":"text","text":"say \"hi\"\n\u{1F600} \u{e9}"},
                {"type":"tool_use","id":"toolu_1","name":"Read","input":{"path":"/a"}},
            ])
        );
        assert_eq!(v["usage"]["output_tokens"], 7);
        assert_eq!(v["usage"]["cache_read_input_tokens"], 4);
    }

    #[test]
    fn collect_tool_input_verbatim_and_placeholder() {
        let mut w = MessageWriter::collect("m").with_placeholder_signature("P");
        w.thinking("t");
        w.tool_use("i", "n", r#"{"a":  [1, 2]}"#);
        w.finish(StopReason::ToolUse, Usage::default());
        let raw = w.into_json().unwrap();
        let s = std::str::from_utf8(&raw).unwrap();
        assert!(s.contains(r#""input":{"a":  [1, 2]}"#));
        assert!(s.contains(r#""signature":"P""#));
    }

    #[test]
    fn errors() {
        let mut w = MessageWriter::sse("m");
        w.error("api_error", "boom \"x\"");
        w.finish(StopReason::EndTurn, Usage::default());
        w.text("late");
        let evs = parse_sse(&w.take().unwrap());
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].0, "error");
        assert_eq!(
            evs[0].1,
            json!({"type":"error","error":{"type":"api_error","message":"boom \"x\""}})
        );

        let mut w = MessageWriter::collect("m");
        w.text("partial");
        w.error("overloaded_error", "busy");
        assert_eq!(
            w.into_json().unwrap_err(),
            ("overloaded_error".to_owned(), "busy".to_owned())
        );
    }

    #[test]
    fn finish_twice_and_take() {
        let mut w = MessageWriter::sse("m");
        assert!(w.take().is_none());
        w.start();
        w.start();
        w.text("a");
        w.finish(StopReason::EndTurn, Usage::default());
        w.finish(StopReason::MaxTokens, Usage::default());
        let evs = parse_sse(&w.take().unwrap());
        assert_eq!(evs.iter().filter(|e| e.0 == "message_start").count(), 1);
        assert_eq!(evs.iter().filter(|e| e.0 == "message_stop").count(), 1);
        assert!(w.take().is_none());
    }

    #[test]
    fn empty_text_opens_nothing() {
        let mut w = MessageWriter::sse("m");
        w.text("");
        w.thinking("");
        assert!(!w.has_content());
        assert!(w.take().is_none());
    }
}
