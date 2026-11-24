//! Borrowed Anthropic Messages request types.
//!
//! A request is parsed once with `serde_json::from_slice`; strings without escapes are borrowed
//! from the input and opaque JSON (tool input, schemas, tool results) stays as raw text.
//! Content blocks are one flat struct because internally tagged enums buffer their input and
//! would break borrowing and `RawValue`.

use std::borrow::Cow;
use std::fmt;

use serde::de::{self, Deserializer, SeqAccess, Visitor};
use serde::Deserialize;
use serde_json::value::RawValue;

/// Visitor producing a borrowed `Cow<str>` when the input allows it.
struct CowVisitor;

impl<'de> Visitor<'de> for CowVisitor {
    type Value = Cow<'de, str>;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a string")
    }

    fn visit_borrowed_str<E: de::Error>(self, v: &'de str) -> Result<Self::Value, E> {
        Ok(Cow::Borrowed(v))
    }

    fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
        Ok(Cow::Owned(v.to_owned()))
    }

    fn visit_string<E: de::Error>(self, v: String) -> Result<Self::Value, E> {
        Ok(Cow::Owned(v))
    }
}

/// A string that is borrowed when it carries no escapes.
struct CowStr<'a>(Cow<'a, str>);

impl<'de> Deserialize<'de> for CowStr<'de> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        d.deserialize_str(CowVisitor).map(CowStr)
    }
}

/// `deserialize_with` for `Option<Cow<str>>` that borrows (serde's derive only borrows a bare `Cow`).
fn opt_cow<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Cow<'de, str>>, D::Error> {
    Ok(Option::<CowStr<'de>>::deserialize(d)?.map(|c| c.0))
}

/// `deserialize_with` for `Vec<Cow<str>>` that borrows.
fn vec_cow<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<Cow<'de, str>>, D::Error> {
    Ok(Vec::<CowStr<'de>>::deserialize(d)?
        .into_iter()
        .map(|c| c.0)
        .collect())
}

/// Defines an enum `{ Text(Cow<str>), Blocks(Vec<Block>) }` deserialized from a string or an array.
macro_rules! text_or_blocks {
    ($(#[$meta:meta])* $name:ident, $visitor:ident) => {
        $(#[$meta])*
        #[derive(Debug, Clone)]
        pub enum $name<'a> {
            /// A plain string.
            Text(Cow<'a, str>),
            /// An array of content blocks.
            Blocks(Vec<Block<'a>>),
        }

        struct $visitor<'a>(std::marker::PhantomData<&'a ()>);

        impl<'de: 'a, 'a> Visitor<'de> for $visitor<'a> {
            type Value = $name<'a>;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a string or an array of content blocks")
            }

            fn visit_borrowed_str<E: de::Error>(self, v: &'de str) -> Result<Self::Value, E> {
                Ok($name::Text(Cow::Borrowed(v)))
            }

            fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
                Ok($name::Text(Cow::Owned(v.to_owned())))
            }

            fn visit_string<E: de::Error>(self, v: String) -> Result<Self::Value, E> {
                Ok($name::Text(Cow::Owned(v)))
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut blocks = Vec::with_capacity(seq.size_hint().unwrap_or(0).min(4096));
                while let Some(b) = seq.next_element::<Block<'a>>()? {
                    blocks.push(b);
                }
                Ok($name::Blocks(blocks))
            }
        }

        impl<'de: 'a, 'a> Deserialize<'de> for $name<'a> {
            fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                d.deserialize_any($visitor(std::marker::PhantomData))
            }
        }
    };
}

text_or_blocks!(
    /// The `system` field: a string or an array of blocks.
    System,
    SystemVisitor
);
text_or_blocks!(
    /// A message's `content`: a string or an array of blocks.
    Content,
    ContentVisitor
);
text_or_blocks!(
    /// A `tool_result` block's content parsed on demand: a string, or blocks (text / image).
    ToolResultContent,
    ToolResultVisitor
);

impl System<'_> {
    /// All text of the system prompt joined by `"\n\n"` (text blocks only).
    pub fn joined_text(&self) -> Cow<'_, str> {
        match self {
            System::Text(t) => Cow::Borrowed(t.as_ref()),
            System::Blocks(blocks) => {
                let mut parts = blocks
                    .iter()
                    .filter(|b| b.kind() == BlockKind::Text)
                    .filter_map(|b| b.text.as_deref());
                let Some(first) = parts.next() else {
                    return Cow::Borrowed("");
                };
                let Some(second) = parts.next() else {
                    return Cow::Borrowed(first);
                };
                let mut out = String::with_capacity(first.len() + second.len() + 2);
                out.push_str(first);
                out.push_str("\n\n");
                out.push_str(second);
                for p in parts {
                    out.push_str("\n\n");
                    out.push_str(p);
                }
                Cow::Owned(out)
            }
        }
    }
}

/// A Messages API request.
#[derive(Debug, Deserialize)]
pub struct MessagesRequest<'a> {
    /// The model name.
    #[serde(borrow)]
    pub model: Cow<'a, str>,
    /// The conversation.
    #[serde(borrow, default)]
    pub messages: Vec<Message<'a>>,
    /// The system prompt.
    #[serde(borrow, default)]
    pub system: Option<System<'a>>,
    /// Output token limit.
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// Sampling temperature.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Nucleus sampling.
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Top-k sampling.
    #[serde(default)]
    pub top_k: Option<u32>,
    /// Custom stop sequences.
    #[serde(borrow, default, deserialize_with = "vec_cow")]
    pub stop_sequences: Vec<Cow<'a, str>>,
    /// Whether the response is streamed.
    #[serde(default)]
    pub stream: bool,
    /// Tool definitions.
    #[serde(borrow, default)]
    pub tools: Vec<Tool<'a>>,
    /// Tool choice.
    #[serde(borrow, default)]
    pub tool_choice: Option<ToolChoice<'a>>,
    /// Extended thinking settings.
    #[serde(borrow, default)]
    pub thinking: Option<Thinking<'a>>,
    /// Output configuration (effort).
    #[serde(borrow, default)]
    pub output_config: Option<OutputConfig<'a>>,
}

/// A conversation turn.
#[derive(Debug, Deserialize)]
pub struct Message<'a> {
    /// Who speaks.
    pub role: Role,
    /// What was said.
    #[serde(borrow)]
    pub content: Content<'a>,
}

/// A message's author.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// The user.
    User,
    /// The assistant.
    Assistant,
    /// A system message placed inside `messages` (Claude Code sends these).
    System,
    /// Any other role string; such messages are ignored by the translators.
    #[serde(other)]
    Other,
}

/// A content block: one flat struct so nothing is buffered while parsing.
#[derive(Debug, Clone, Deserialize)]
pub struct Block<'a> {
    /// The block's `type`.
    #[serde(rename = "type", borrow)]
    pub kind: Cow<'a, str>,
    /// Text of a `text` block.
    #[serde(borrow, default, deserialize_with = "opt_cow")]
    pub text: Option<Cow<'a, str>>,
    /// Source of an `image` or `document` block.
    #[serde(borrow, default)]
    pub source: Option<Source<'a>>,
    /// Id of a `tool_use` block.
    #[serde(borrow, default, deserialize_with = "opt_cow")]
    pub id: Option<Cow<'a, str>>,
    /// Name of a `tool_use` block.
    #[serde(borrow, default, deserialize_with = "opt_cow")]
    pub name: Option<Cow<'a, str>>,
    /// Raw JSON input of a `tool_use` block.
    #[serde(borrow, default)]
    pub input: Option<&'a RawValue>,
    /// The call a `tool_result` answers.
    #[serde(borrow, default, deserialize_with = "opt_cow")]
    pub tool_use_id: Option<Cow<'a, str>>,
    /// Raw content of a `tool_result` block: a string or an array of blocks.
    #[serde(borrow, default)]
    pub content: Option<&'a RawValue>,
    /// Whether a `tool_result` reports an error.
    #[serde(default)]
    pub is_error: bool,
    /// Text of a `thinking` block.
    #[serde(borrow, default, deserialize_with = "opt_cow")]
    pub thinking: Option<Cow<'a, str>>,
    /// Signature of a `thinking` block.
    #[serde(borrow, default, deserialize_with = "opt_cow")]
    pub signature: Option<Cow<'a, str>>,
    /// Payload of a `redacted_thinking` block.
    #[serde(borrow, default, deserialize_with = "opt_cow")]
    pub data: Option<Cow<'a, str>>,
}

/// The known kinds of [`Block`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockKind {
    /// `text`
    Text,
    /// `image`
    Image,
    /// `document`
    Document,
    /// `tool_use`
    ToolUse,
    /// `tool_result`
    ToolResult,
    /// `thinking`
    Thinking,
    /// `redacted_thinking`
    RedactedThinking,
    /// Any other type.
    Other,
}

impl<'a> Block<'a> {
    /// The block's kind as an enum.
    pub fn kind(&self) -> BlockKind {
        match self.kind.as_ref() {
            "text" => BlockKind::Text,
            "image" => BlockKind::Image,
            "document" => BlockKind::Document,
            "tool_use" => BlockKind::ToolUse,
            "tool_result" => BlockKind::ToolResult,
            "thinking" => BlockKind::Thinking,
            "redacted_thinking" => BlockKind::RedactedThinking,
            _ => BlockKind::Other,
        }
    }

    /// Parses the `tool_result` content on demand. `None` when the block has no content.
    pub fn tool_result_content(&self) -> Result<Option<ToolResultContent<'a>>, serde_json::Error> {
        match self.content {
            None => Ok(None),
            Some(raw) => {
                let text: &'a str = raw.get();
                serde_json::from_str::<ToolResultContent<'a>>(text).map(Some)
            }
        }
    }
}

/// Source of an image or document.
#[derive(Debug, Clone, Deserialize)]
pub struct Source<'a> {
    /// `base64`, `url` or `text`.
    #[serde(rename = "type", borrow)]
    pub kind: Cow<'a, str>,
    /// Media type, e.g. `image/png`.
    #[serde(borrow, default, deserialize_with = "opt_cow")]
    pub media_type: Option<Cow<'a, str>>,
    /// Base64 data (or text).
    #[serde(borrow, default, deserialize_with = "opt_cow")]
    pub data: Option<Cow<'a, str>>,
    /// URL of a URL source.
    #[serde(borrow, default, deserialize_with = "opt_cow")]
    pub url: Option<Cow<'a, str>>,
}

/// A tool definition.
#[derive(Debug, Deserialize)]
pub struct Tool<'a> {
    /// The tool's name.
    #[serde(borrow)]
    pub name: Cow<'a, str>,
    /// What it does.
    #[serde(borrow, default, deserialize_with = "opt_cow")]
    pub description: Option<Cow<'a, str>>,
    /// Raw JSON schema of the input.
    #[serde(borrow, default)]
    pub input_schema: Option<&'a RawValue>,
    /// Set and different from `custom` for a server tool (e.g. `web_search_20250305`).
    #[serde(rename = "type", borrow, default, deserialize_with = "opt_cow")]
    pub kind: Option<Cow<'a, str>>,
}

/// How the model may pick tools.
#[derive(Debug, Deserialize)]
pub struct ToolChoice<'a> {
    /// `auto`, `any`, `tool` or `none`.
    #[serde(rename = "type", borrow)]
    pub kind: Cow<'a, str>,
    /// The tool to force when `kind` is `tool`.
    #[serde(borrow, default, deserialize_with = "opt_cow")]
    pub name: Option<Cow<'a, str>>,
    /// Whether parallel tool calls are off.
    #[serde(default)]
    pub disable_parallel_tool_use: Option<bool>,
}

/// Extended thinking settings.
#[derive(Debug, Deserialize)]
pub struct Thinking<'a> {
    /// `enabled`, `adaptive` or `disabled`.
    #[serde(rename = "type", borrow)]
    pub kind: Cow<'a, str>,
    /// Token budget when enabled.
    #[serde(default)]
    pub budget_tokens: Option<u32>,
}

/// Output configuration.
#[derive(Debug, Deserialize)]
pub struct OutputConfig<'a> {
    /// `low`, `medium`, `high` or `max`.
    #[serde(borrow, default, deserialize_with = "opt_cow")]
    pub effort: Option<Cow<'a, str>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    const REQUEST: &str = r#"{
      "model": "claude-opus-4-5",
      "max_tokens": 32000,
      "stream": true,
      "temperature": 1,
      "metadata": {"user_id": "u"},
      "context_management": {"edits": []},
      "stop_sequences": ["END", "a\nb"],
      "system": [
        {"type":"text","text":"You are helpful.","cache_control":{"type":"ephemeral"}},
        {"type":"text","text":"Second \"part\"\nline"}
      ],
      "messages": [
        {"role":"user","content":"hello"},
        {"role":"user","content":[
          {"type":"text","text":"look","cache_control":{"type":"ephemeral"}},
          {"type":"image","source":{"type":"base64","media_type":"image/png","data":"iVBOR"}}
        ]},
        {"role":"assistant","content":[
          {"type":"thinking","thinking":"hmm","signature":"sig123"},
          {"type":"text","text":"ok","citations":null},
          {"type":"tool_use","id":"toolu_1","name":"Read","input":{"path": "/a", "n":  1}}
        ]},
        {"role":"user","content":[
          {"type":"tool_result","tool_use_id":"toolu_1","content":"file text"},
          {"type":"tool_result","tool_use_id":"toolu_2","is_error":true,"content":[{"type":"text","text":"boom"}]}
        ]}
      ],
      "tools": [
        {"name":"Read","description":"Reads","input_schema":{"type":"object","properties":{ "path":{"type":"string"}}}},
        {"type":"web_search_20250305","name":"web_search"}
      ],
      "tool_choice": {"type":"auto","disable_parallel_tool_use":false},
      "thinking": {"type":"adaptive"},
      "output_config": {"effort":"high"}
    }"#;

    #[test]
    fn parses_system_and_unknown_roles() {
        let req: MessagesRequest = serde_json::from_str(
            r#"{"model":"m","messages":[
                {"role":"system","content":"be brief"},
                {"role":"system","content":[{"type":"text","text":"blocks"}]},
                {"role":"tool","content":"x"}]}"#,
        )
        .unwrap();
        assert_eq!(req.messages[0].role, Role::System);
        assert!(matches!(&req.messages[0].content, Content::Text(t) if t == "be brief"));
        assert_eq!(req.messages[1].role, Role::System);
        assert!(matches!(&req.messages[1].content, Content::Blocks(b) if b.len() == 1));
        assert_eq!(req.messages[2].role, Role::Other);
    }

    #[test]
    fn parses_realistic_request() {
        let req: MessagesRequest = serde_json::from_str(REQUEST).unwrap();
        assert_eq!(req.model, "claude-opus-4-5");
        assert!(matches!(req.model, Cow::Borrowed(_)));
        assert!(req.stream);
        assert_eq!(req.max_tokens, Some(32000));
        assert_eq!(req.stop_sequences, ["END", "a\nb"]);
        assert_eq!(
            req.system.as_ref().unwrap().joined_text(),
            "You are helpful.\n\nSecond \"part\"\nline"
        );
        assert_eq!(req.messages.len(), 4);
        assert_eq!(req.messages[0].role, Role::User);
        assert!(matches!(&req.messages[0].content, Content::Text(t) if t == "hello"));

        let Content::Blocks(b) = &req.messages[1].content else {
            panic!()
        };
        assert_eq!(b[0].kind(), BlockKind::Text);
        assert!(matches!(b[0].text, Some(Cow::Borrowed("look"))));
        assert_eq!(b[1].kind(), BlockKind::Image);
        let src = b[1].source.as_ref().unwrap();
        assert_eq!(src.kind, "base64");
        assert_eq!(src.media_type.as_deref(), Some("image/png"));
        assert_eq!(src.data.as_deref(), Some("iVBOR"));

        let Content::Blocks(b) = &req.messages[2].content else {
            panic!()
        };
        assert_eq!(b[0].kind(), BlockKind::Thinking);
        assert_eq!(b[0].signature.as_deref(), Some("sig123"));
        assert_eq!(b[2].kind(), BlockKind::ToolUse);
        assert_eq!(b[2].input.unwrap().get(), r#"{"path": "/a", "n":  1}"#);

        let Content::Blocks(b) = &req.messages[3].content else {
            panic!()
        };
        assert!(!b[0].is_error);
        assert!(matches!(
            b[0].tool_result_content().unwrap(),
            Some(ToolResultContent::Text(t)) if t == "file text"
        ));
        assert!(b[1].is_error);
        let Some(ToolResultContent::Blocks(inner)) = b[1].tool_result_content().unwrap() else {
            panic!()
        };
        assert_eq!(inner[0].text.as_deref(), Some("boom"));

        assert_eq!(req.tools.len(), 2);
        assert_eq!(
            req.tools[0].input_schema.unwrap().get(),
            r#"{"type":"object","properties":{ "path":{"type":"string"}}}"#
        );
        assert_eq!(req.tools[0].kind, None);
        assert_eq!(req.tools[1].kind.as_deref(), Some("web_search_20250305"));
        let tc = req.tool_choice.unwrap();
        assert_eq!(tc.kind, "auto");
        assert_eq!(tc.disable_parallel_tool_use, Some(false));
        assert_eq!(req.thinking.unwrap().kind, "adaptive");
        assert_eq!(req.output_config.unwrap().effort.as_deref(), Some("high"));
    }

    #[test]
    fn minimal_request_and_string_system() {
        let req: MessagesRequest =
            serde_json::from_slice(br#"{"model":"m","system":"sys"}"#).unwrap();
        assert!(req.messages.is_empty());
        assert!(!req.stream);
        assert!(req.tools.is_empty());
        assert!(matches!(&req.system, Some(System::Text(t)) if t == "sys"));
        assert_eq!(req.system.unwrap().joined_text(), "sys");
    }

    #[test]
    fn thinking_enabled_budget() {
        let req: MessagesRequest = serde_json::from_str(
            r#"{"model":"m","thinking":{"type":"enabled","budget_tokens":2048}}"#,
        )
        .unwrap();
        let t = req.thinking.unwrap();
        assert_eq!(t.kind, "enabled");
        assert_eq!(t.budget_tokens, Some(2048));
    }
}
