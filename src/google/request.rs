//! Converts an Anthropic Messages request into the Gemini `generateContent` request body that
//! Google's Code Assist backend takes.
//!
//! The body is written once with `#[derive(Serialize)]` structs that borrow from the parsed
//! request; tool `input` and `input_schema` pass through as raw JSON text.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use serde::{Serialize, Serializer};
use serde_json::value::RawValue;

use crate::anthropic::types::{
    Block, BlockKind, Content, MessagesRequest, Role, ToolResultContent,
};
use crate::google::schema;

/// The identity text the Antigravity client sends as its system instruction.
const AGY_IDENTITY: &str = "You are Antigravity, a powerful agentic AI assistant designed by the Google Deepmind team.\nYou are working with a USER to solve their task.";

/// Thought signature Gemini accepts when the real one is unavailable.
pub const SKIP_SIGNATURE: &str = "skip_thought_signature_validator";

const DEFAULT_SCHEMA: &str = r#"{"type":"object","properties":{}}"#;
const MAX_NAME_LEN: usize = 64;

/// How a tool's input schema is sent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchemaMode {
    /// Raw JSON Schema as `parametersJsonSchema`.
    JsonSchema,
    /// Sanitized OpenAPI-subset schema as `parameters` (see [`schema::clean`]).
    Cleaned,
}

/// Per-request settings of the translation.
#[derive(Debug, Clone)]
pub struct RequestOptions<'a> {
    /// Code Assist project id.
    pub project: &'a str,
    /// Backend model id, e.g. `gemini-4-neon-medium`.
    pub model: &'a str,
    /// `LOW`, `MEDIUM`, `HIGH` or `XHIGH`.
    pub thinking_level: Option<&'a str>,
    /// How tool schemas are sent.
    pub schema_mode: SchemaMode,
    /// Never use stored signatures; put [`SKIP_SIGNATURE`] on the first function call of each model turn.
    pub skip_signatures: bool,
    /// Upper bound for `maxOutputTokens`.
    pub max_output_tokens: u32,
}

/// A finished translation.
#[derive(Debug, Clone)]
pub struct Translated {
    /// The JSON body.
    pub body: Vec<u8>,
    /// Tool names that had to be changed: (name sent to Gemini, original Anthropic name).
    pub renamed_tools: Vec<(String, String)>,
}

/// The request could not be translated.
#[derive(Debug, Clone)]
pub struct TranslateError(pub String);

impl fmt::Display for TranslateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for TranslateError {}

/// Thinking level from a model id suffix: `-low`, `-medium`, `-high`, `-xhigh`.
pub fn thinking_level_for(model: &str) -> Option<&'static str> {
    [
        ("-xhigh", "XHIGH"),
        ("-high", "HIGH"),
        ("-medium", "MEDIUM"),
        ("-low", "LOW"),
    ]
    .into_iter()
    .find(|(suffix, _)| model.ends_with(suffix))
    .map(|(_, level)| level)
}

// ---- wire types ----

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Envelope<'r> {
    project: &'r str,
    model: &'r str,
    user_agent: &'static str,
    request_type: &'static str,
    request_id: String,
    request: Inner<'r>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Inner<'r> {
    contents: Vec<Turn<'r>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<Turn<'r>>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ToolGroup<'r>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_config: Option<ToolConfig<'r>>,
    generation_config: GenerationConfig<'r>,
    session_id: String,
}

#[derive(Serialize)]
struct Turn<'r> {
    role: &'static str,
    parts: Vec<Part<'r>>,
}

enum Signature {
    Skip,
    Stored(Arc<str>),
}

impl Serialize for Signature {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Signature::Skip => s.serialize_str(SKIP_SIGNATURE),
            Signature::Stored(a) => s.serialize_str(a),
        }
    }
}

#[derive(Serialize, Default)]
#[serde(rename_all = "camelCase")]
struct Part<'r> {
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<Cow<'r, str>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    inline_data: Option<InlineData<'r>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    file_data: Option<FileData<'r>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    function_call: Option<FunctionCall<'r>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    function_response: Option<FunctionResponse<'r>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thought_signature: Option<Signature>,
    /// Set on images that came out of a `tool_result`; they sort right after the responses.
    #[serde(skip)]
    from_result: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct InlineData<'r> {
    mime_type: Cow<'r, str>,
    data: Cow<'r, str>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct FileData<'r> {
    mime_type: Cow<'r, str>,
    file_uri: Cow<'r, str>,
}

#[derive(Serialize)]
struct FunctionCall<'r> {
    id: &'r str,
    name: Cow<'r, str>,
    args: &'r RawValue,
}

#[derive(Serialize)]
struct FunctionResponse<'r> {
    id: &'r str,
    name: Cow<'r, str>,
    response: Output<'r>,
}

#[derive(Serialize)]
struct Output<'r> {
    #[serde(skip_serializing_if = "Option::is_none")]
    output: Option<Cow<'r, str>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<Cow<'r, str>>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ToolGroup<'r> {
    function_declarations: Vec<Decl<'r>>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Decl<'r> {
    name: Cow<'r, str>,
    description: &'r str,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters_json_schema: Option<&'r RawValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<serde_json::Value>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ToolConfig<'r> {
    function_calling_config: CallingConfig<'r>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CallingConfig<'r> {
    mode: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    allowed_function_names: Option<Vec<Cow<'r, str>>>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GenerationConfig<'r> {
    max_output_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    stop_sequences: Vec<&'r str>,
    thinking_config: ThinkingConfig<'r>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ThinkingConfig<'r> {
    include_thoughts: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_level: Option<&'r str>,
}

// ---- tool names ----

/// Maps Anthropic tool names to names Gemini accepts.
struct Names<'r> {
    map: HashMap<&'r str, Cow<'r, str>>,
    used: HashSet<String>,
    renamed: Vec<(String, String)>,
}

impl<'r> Names<'r> {
    fn resolve(&mut self, orig: &'r str) -> Cow<'r, str> {
        if let Some(n) = self.map.get(orig) {
            return n.clone();
        }
        let base = sanitize(orig);
        let mut sent = base.clone();
        let mut n = 2u32;
        while self.used.contains(&sent) {
            let suffix = format!("_{n}");
            let keep = MAX_NAME_LEN - suffix.len();
            sent = format!("{}{suffix}", &base[..base.len().min(keep)]);
            n += 1;
        }
        self.used.insert(sent.clone());
        let cow = if sent == orig {
            Cow::Borrowed(orig)
        } else {
            self.renamed.push((sent.clone(), orig.to_owned()));
            Cow::Owned(sent)
        };
        self.map.insert(orig, cow.clone());
        cow
    }
}

/// Gemini's rule: `[A-Za-z0-9_.-]`, first char a letter or underscore, at most 64 chars.
fn sanitize(name: &str) -> String {
    let mut out = String::with_capacity(name.len().min(MAX_NAME_LEN) + 1);
    for c in name.chars() {
        out.push(
            if c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-') {
                c
            } else {
                '_'
            },
        );
    }
    if !out.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_') {
        out.insert(0, '_');
    }
    out.truncate(MAX_NAME_LEN);
    out
}

// ---- translation ----

fn raw(text: &str) -> Result<Box<RawValue>, TranslateError> {
    RawValue::from_string(text.to_owned()).map_err(|e| TranslateError(e.to_string()))
}

fn source_part<'a>(b: &Block<'a>, default_mime: &'static str) -> Option<Part<'a>> {
    let src = b.source.as_ref()?;
    let mime = src
        .media_type
        .clone()
        .unwrap_or(Cow::Borrowed(default_mime));
    match src.kind.as_ref() {
        "base64" => Some(Part {
            inline_data: Some(InlineData {
                mime_type: mime,
                data: src.data.clone()?,
            }),
            ..Part::default()
        }),
        "url" => Some(Part {
            file_data: Some(FileData {
                mime_type: mime,
                file_uri: src.url.clone()?,
            }),
            ..Part::default()
        }),
        "text" => text_part(src.data.clone()?),
        _ => None,
    }
}

fn text_part(t: Cow<'_, str>) -> Option<Part<'_>> {
    (!t.trim().is_empty()).then(|| Part {
        text: Some(t),
        ..Part::default()
    })
}

fn first_user_text(req: &MessagesRequest<'_>) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    if let Some(m) = req.messages.iter().find(|m| m.role == Role::User) {
        match &m.content {
            Content::Text(t) => t.hash(&mut h),
            Content::Blocks(bs) => {
                for b in bs.iter().filter(|b| b.kind() == BlockKind::Text) {
                    b.text.hash(&mut h);
                }
            }
        }
    }
    h.finish()
}

/// Translates `req` into a Gemini request body.
///
/// `signature_of(tool_use_id)` returns the thought signature stored when that call was produced.
pub fn build_request<'a, F>(
    req: &MessagesRequest<'a>,
    opts: &RequestOptions<'_>,
    signature_of: F,
) -> Result<Translated, TranslateError>
where
    F: Fn(&str) -> Option<Arc<str>>,
{
    let default_schema = raw(DEFAULT_SCHEMA)?;
    let empty_args = raw("{}")?;

    let mut names = Names {
        map: HashMap::with_capacity(req.tools.len()),
        used: HashSet::with_capacity(req.tools.len()),
        renamed: Vec::new(),
    };

    // Tools first, so declared names win over names that only appear in history.
    let mut decls: Vec<Decl<'_>> = Vec::with_capacity(req.tools.len());
    let mut declared: HashSet<&str> = HashSet::new();
    for t in &req.tools {
        if t.kind.as_deref().is_some_and(|k| k != "custom") {
            continue;
        }
        if !declared.insert(t.name.as_ref()) {
            continue;
        }
        let name = names.resolve(t.name.as_ref());
        let (json_schema, parameters) = match opts.schema_mode {
            SchemaMode::JsonSchema => (Some(t.input_schema.unwrap_or(&default_schema)), None),
            SchemaMode::Cleaned => {
                let parsed = t
                    .input_schema
                    .and_then(|r| serde_json::from_str::<serde_json::Value>(r.get()).ok())
                    .unwrap_or_else(|| serde_json::json!({"type":"object","properties":{}}));
                (None, Some(schema::clean(&parsed)))
            }
        };
        decls.push(Decl {
            name,
            description: t.description.as_deref().unwrap_or(""),
            parameters_json_schema: json_schema,
            parameters,
        });
    }

    let mut contents: Vec<Turn<'_>> = Vec::with_capacity(req.messages.len());
    let mut call_seen = false;
    let mut call_names: HashMap<&str, Cow<'_, str>> = HashMap::new();

    for msg in &req.messages {
        let role = match msg.role {
            Role::User | Role::System => "user",
            Role::Assistant => "model",
            Role::Other => continue,
        };
        let system_only = msg.role == Role::System;
        let same_role = contents.last().is_some_and(|t| t.role == role);
        let mut seen = same_role && call_seen;
        let mut parts: Vec<Part<'_>> = Vec::new();

        match &msg.content {
            Content::Text(t) => parts.extend(text_part(Cow::Borrowed(t.as_ref()))),
            Content::Blocks(blocks) => {
                for b in blocks {
                    if system_only && b.kind() != BlockKind::Text {
                        continue;
                    }
                    match b.kind() {
                        BlockKind::Text => parts.extend(b.text.clone().and_then(text_part)),
                        BlockKind::Image => parts.extend(source_part(b, "image/jpeg")),
                        BlockKind::Document => parts.extend(source_part(b, "application/pdf")),
                        BlockKind::ToolUse => {
                            let id = b.id.as_deref().unwrap_or("");
                            let name = names.resolve(b.name.as_deref().unwrap_or("unknown"));
                            call_names.insert(id, name.clone());
                            let stored = if opts.skip_signatures {
                                None
                            } else {
                                signature_of(id)
                            };
                            let sig = match stored {
                                Some(s) => Some(Signature::Stored(s)),
                                None if !seen => Some(Signature::Skip),
                                None => None,
                            };
                            seen = true;
                            parts.push(Part {
                                function_call: Some(FunctionCall {
                                    id,
                                    name,
                                    args: b.input.unwrap_or(&empty_args),
                                }),
                                thought_signature: sig,
                                ..Part::default()
                            });
                        }
                        BlockKind::ToolResult => {
                            let id = b.tool_use_id.as_deref().unwrap_or("");
                            let name = call_names
                                .get(id)
                                .cloned()
                                .unwrap_or(Cow::Borrowed("unknown"));
                            let mut images: Vec<Part<'_>> = Vec::new();
                            let text: Cow<'_, str> = match b.tool_result_content() {
                                Ok(Some(ToolResultContent::Text(t))) => t,
                                Ok(Some(ToolResultContent::Blocks(inner))) => {
                                    let mut texts: Vec<Cow<'_, str>> = Vec::new();
                                    for ib in inner {
                                        match ib.kind() {
                                            BlockKind::Text => texts.extend(ib.text),
                                            BlockKind::Image => images.extend(
                                                source_part(&ib, "image/jpeg").map(|p| Part {
                                                    from_result: true,
                                                    ..p
                                                }),
                                            ),
                                            _ => {}
                                        }
                                    }
                                    if texts.len() == 1 {
                                        texts.swap_remove(0)
                                    } else {
                                        Cow::Owned(texts.join("\n"))
                                    }
                                }
                                Ok(None) => Cow::Borrowed(""),
                                Err(_) => Cow::Borrowed(b.content.map_or("", |r| r.get())),
                            };
                            let response = if b.is_error {
                                Output {
                                    output: None,
                                    error: Some(text),
                                }
                            } else {
                                Output {
                                    output: Some(text),
                                    error: None,
                                }
                            };
                            parts.push(Part {
                                function_response: Some(FunctionResponse { id, name, response }),
                                ..Part::default()
                            });
                            parts.extend(images);
                        }
                        BlockKind::Thinking | BlockKind::RedactedThinking | BlockKind::Other => {}
                    }
                }
            }
        }

        if parts.is_empty() {
            continue;
        }
        call_seen = seen;
        if same_role {
            if let Some(last) = contents.last_mut() {
                last.parts.append(&mut parts);
            }
        } else {
            contents.push(Turn { role, parts });
        }
    }

    for t in &mut contents {
        if t.role == "user" && t.parts.iter().any(|p| p.function_response.is_some()) {
            t.parts
                .sort_by_key(|p| match (p.function_response.is_some(), p.from_result) {
                    (true, _) => 0u8,
                    (false, true) => 1,
                    _ => 2,
                });
        }
    }

    /// Swaps Claude Code's identity lines for the Antigravity client's own.
    fn rebrand(s: Cow<'_, str>) -> Cow<'_, str> {
        const AGY: &str = AGY_IDENTITY;
        const BILLING: &str = "x-anthropic-billing-header:";
        let hit = |l: &str| {
            let l = l.trim_start();
            l.starts_with(BILLING)
                || (l.starts_with("You are")
                    && (l.contains("Claude Code")
                        || l.contains("Claude Agent SDK")
                        || l.contains("Anthropic")))
        };
        if !s.lines().any(hit) {
            return s;
        }
        let mut out = String::with_capacity(s.len() + AGY.len());
        let mut placed = false;
        for line in s.lines() {
            if hit(line) {
                if !placed && !line.trim_start().starts_with(BILLING) {
                    out.push_str(AGY);
                    out.push('\n');
                    placed = true;
                }
                continue;
            }
            out.push_str(line);
            out.push('\n');
        }
        Cow::Owned(out)
    }
    // Google refuses Claude Code's prompt as a system instruction, so the prompt travels at the
    // head of the first user turn and the system instruction carries the client's own identity.
    let system = req.system.as_ref().map(|s| rebrand(s.joined_text()));
    if let Some(s) = system.filter(|s| !s.trim().is_empty()) {
        let head = format!("<system>\n{s}\n</system>");
        let first = contents.iter_mut().find(|t| t.role == "user");
        // One message: the prompt joins the first text part of the first user turn.
        match first.and_then(|t| t.parts.first_mut().filter(|p| p.text.is_some())) {
            Some(p) => {
                let rest = p.text.take().unwrap_or_default();
                p.text = Some(Cow::Owned(format!("{head}\n\n{rest}")));
            }
            None => {
                let wrapped = Part {
                    text: Some(Cow::Owned(head)),
                    ..Part::default()
                };
                match contents.iter_mut().find(|t| t.role == "user") {
                    Some(t) => t.parts.insert(0, wrapped),
                    None => contents.insert(
                        0,
                        Turn {
                            role: "user",
                            parts: vec![wrapped],
                        },
                    ),
                }
            }
        }
    }
    let system_instruction = Some(Turn {
        role: "user",
        parts: vec![Part {
            text: Some(Cow::Borrowed(AGY_IDENTITY)),
            ..Part::default()
        }],
    });

    let tool_config = match (&req.tool_choice, decls.is_empty()) {
        (Some(tc), false) => {
            let (mode, allowed) = match tc.kind.as_ref() {
                "any" => ("ANY", None),
                "tool" => match tc.name.as_deref() {
                    Some(n) => ("ANY", Some(vec![names.resolve(n)])),
                    None => ("ANY", None),
                },
                "none" => ("NONE", None),
                _ => ("AUTO", None),
            };
            Some(ToolConfig {
                function_calling_config: CallingConfig {
                    mode,
                    allowed_function_names: allowed,
                },
            })
        }
        _ => None,
    };

    let generation_config = GenerationConfig {
        max_output_tokens: req.max_tokens.unwrap_or(32000).min(opts.max_output_tokens),
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        stop_sequences: req.stop_sequences.iter().map(|s| s.as_ref()).collect(),
        thinking_config: ThinkingConfig {
            include_thoughts: true,
            thinking_level: opts.thinking_level,
        },
    };

    let envelope = Envelope {
        project: opts.project,
        model: opts.model,
        user_agent: "antigravity",
        request_type: "agent",
        request_id: format!("agent/{}", uuid::Uuid::new_v4()),
        request: Inner {
            contents,
            system_instruction,
            tools: if decls.is_empty() {
                Vec::new()
            } else {
                vec![ToolGroup {
                    function_declarations: decls,
                }]
            },
            tool_config,
            generation_config,
            session_id: (first_user_text(req) as i64).to_string(),
        },
    };
    let body = serde_json::to_vec(&envelope).map_err(|e| TranslateError(e.to_string()))?;
    Ok(Translated {
        body,
        renamed_tools: names.renamed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    const LONG: &str =
        "mcp__server__a_very_long_tool_name_that_goes_on_and_on_and_on_beyond_sixty_four";

    fn request_json() -> String {
        let text = json!({
          "model": "gemini-4-neon-medium",
          "max_tokens": 64000,
          "temperature": 0.5,
          "top_p": 0.9,
          "top_k": 40,
          "stop_sequences": ["END"],
          "metadata": {"user_id": "u"},
          "system": [
            {"type":"text","text":"You are helpful.","cache_control":{"type":"ephemeral"}},
            {"type":"text","text":"Second part"}
          ],
          "messages": [
            {"role":"user","content":"hello world"},
            {"role":"assistant","content":[
              {"type":"thinking","thinking":"hmm","signature":"s"},
              {"type":"text","text":"I will read"},
              {"type":"tool_use","id":"toolu_a","name":"Read","input":{"path":"/a"}},
              {"type":"tool_use","id":"toolu_b","name":"bad name!","input":{}}
            ]},
            {"role":"user","content":[
              {"type":"tool_result","tool_use_id":"toolu_a","content":"file contents"},
              {"type":"text","text":"thanks"},
              {"type":"tool_result","tool_use_id":"toolu_b","is_error":true,"content":[
                {"type":"text","text":"x"},{"type":"text","text":"y"},
                {"type":"image","source":{"type":"base64","media_type":"image/png","data":"iVBOR"}}]}
            ]},
            {"role":"assistant","content":[{"type":"thinking","thinking":"only","signature":"s"}]},
            {"role":"user","content":[
              {"type":"image","source":{"type":"url","url":"http://x/y.png"}},
              {"type":"text","text":"   "},
              {"type":"text","text":"more"}]}
          ],
          "tools": [
            {"name":"Read","description":"Reads","input_schema":{"type": "object",  "properties":{"path":{"type":"string"}}}},
            {"type":"web_search_20250305","name":"web_search"},
            {"name":"bad name!","input_schema":{"type":"object"}},
            {"name":LONG,"type":"custom","input_schema":{"type":"object"}}
          ],
          "tool_choice": {"type":"auto"}
        })
        .to_string();
        let compact = r#"{"properties":{"path":{"type":"string"}},"type":"object"}"#;
        assert!(text.contains(compact));
        text.replace(
            compact,
            r#"{"type": "object",  "properties":{"path":{"type":"string"}}}"#,
        )
    }

    fn opts(skip: bool, mode: SchemaMode) -> RequestOptions<'static> {
        RequestOptions {
            project: "proj-1",
            model: "gemini-4-neon-medium",
            thinking_level: Some("MEDIUM"),
            schema_mode: mode,
            skip_signatures: skip,
            max_output_tokens: 32000,
        }
    }

    fn translate(text: &str, o: &RequestOptions<'_>) -> (Value, Translated, String) {
        let req: MessagesRequest = serde_json::from_str(text).unwrap();
        let t = build_request(&req, o, |id| {
            (id == "toolu_a").then(|| Arc::<str>::from("SIG_A"))
        })
        .unwrap();
        let s = String::from_utf8(t.body.clone()).unwrap();
        (serde_json::from_str(&s).unwrap(), t, s)
    }

    #[test]
    fn full_request() {
        let (v, t, text) = translate(&request_json(), &opts(false, SchemaMode::JsonSchema));
        assert_eq!(v["project"], "proj-1");
        assert_eq!(v["model"], "gemini-4-neon-medium");
        assert_eq!(v["userAgent"], "antigravity");
        assert_eq!(v["requestType"], "agent");
        assert!(v["requestId"].as_str().unwrap().starts_with("agent/"));
        let r = &v["request"];
        assert_eq!(
            r["systemInstruction"],
            json!({"role":"user","parts":[{"text":AGY_IDENTITY}]})
        );
        let first = r["contents"][0]["parts"][0]["text"].as_str().unwrap();
        assert!(first.starts_with("<system>\nYou are helpful.\n\nSecond part\n</system>\n\n"));
        assert!(first.len() > "<system>\nYou are helpful.\n\nSecond part\n</system>\n\n".len());

        let c = r["contents"].as_array().unwrap();
        let roles: Vec<&str> = c.iter().map(|x| x["role"].as_str().unwrap()).collect();
        assert_eq!(roles, ["user", "model", "user"]);

        // model turn: thinking dropped, stored signature on the first call, none on the second.
        let m = c[1]["parts"].as_array().unwrap();
        assert_eq!(m.len(), 3);
        assert_eq!(m[0]["text"], "I will read");
        assert_eq!(m[1]["functionCall"]["name"], "Read");
        assert_eq!(m[1]["functionCall"]["args"], json!({"path":"/a"}));
        assert_eq!(m[1]["thoughtSignature"], "SIG_A");
        assert_eq!(m[2]["functionCall"]["name"], "bad_name_");
        assert_eq!(m[2]["functionCall"]["id"], "toolu_b");
        assert!(m[2].get("thoughtSignature").is_none());

        // user turn: responses first, then the image of the second response, then text, merged.
        let u = c[2]["parts"].as_array().unwrap();
        assert_eq!(u.len(), 6);
        assert_eq!(u[0]["functionResponse"]["name"], "Read");
        assert_eq!(u[0]["functionResponse"]["id"], "toolu_a");
        assert_eq!(
            u[0]["functionResponse"]["response"],
            json!({"output":"file contents"})
        );
        assert_eq!(u[1]["functionResponse"]["name"], "bad_name_");
        assert_eq!(
            u[1]["functionResponse"]["response"],
            json!({"error":"x\ny"})
        );
        assert_eq!(
            u[2]["inlineData"],
            json!({"mimeType":"image/png","data":"iVBOR"})
        );
        assert_eq!(u[3]["text"], "thanks");
        assert_eq!(
            u[4]["fileData"],
            json!({"mimeType":"image/jpeg","fileUri":"http://x/y.png"})
        );
        assert_eq!(u[5]["text"], "more");

        // tools
        let decls = r["tools"][0]["functionDeclarations"].as_array().unwrap();
        let names: Vec<&str> = decls.iter().map(|d| d["name"].as_str().unwrap()).collect();
        assert_eq!(names.len(), 3);
        assert_eq!(names[0], "Read");
        assert_eq!(names[1], "bad_name_");
        assert_eq!(names[2].len(), 64);
        assert_eq!(decls[0]["description"], "Reads");
        assert_eq!(decls[1]["description"], "");
        assert_eq!(decls[1]["parametersJsonSchema"], json!({"type":"object"}));
        assert!(text.contains(
            r#""parametersJsonSchema":{"type": "object",  "properties":{"path":{"type":"string"}}}"#
        ));
        assert_eq!(
            t.renamed_tools,
            vec![
                ("bad_name_".to_owned(), "bad name!".to_owned()),
                (LONG[..64].to_owned(), LONG.to_owned())
            ]
        );

        assert_eq!(
            r["toolConfig"],
            json!({"functionCallingConfig":{"mode":"AUTO"}})
        );
        assert_eq!(
            r["generationConfig"],
            json!({"maxOutputTokens":32000,"temperature":0.5,"topP":0.9,"topK":40,
                   "stopSequences":["END"],
                   "thinkingConfig":{"includeThoughts":true,"thinkingLevel":"MEDIUM"}})
        );
        assert!(r["sessionId"].as_str().unwrap().parse::<i64>().is_ok());
    }

    #[test]
    fn signature_rules() {
        let body = json!({
          "model":"m",
          "messages":[
            {"role":"user","content":"hi"},
            {"role":"assistant","content":[
              {"type":"tool_use","id":"toolu_x","name":"t","input":{}},
              {"type":"tool_use","id":"toolu_y","name":"t","input":{}}]},
            {"role":"user","content":[
              {"type":"tool_result","tool_use_id":"toolu_x","content":"1"},
              {"type":"tool_result","tool_use_id":"toolu_y","content":"2"}]},
            {"role":"assistant","content":[
              {"type":"tool_use","id":"toolu_a","name":"t","input":{}},
              {"type":"tool_use","id":"toolu_z","name":"t","input":{}}]}
          ],
          "tools":[{"name":"t","input_schema":{"type":"object"}}]
        })
        .to_string();
        let sigs = |skip| {
            let (v, _, _) = translate(&body, &opts(skip, SchemaMode::JsonSchema));
            let c = v["request"]["contents"].as_array().unwrap().clone();
            let sig = |i: usize, j: usize| c[i]["parts"][j].get("thoughtSignature").cloned();
            (sig(1, 0), sig(1, 1), sig(3, 0), sig(3, 1))
        };
        let sentinel = Some(json!(SKIP_SIGNATURE));
        // nothing stored for x, y, z: sentinel on the first call only; a stored one wins on a.
        let (a, b, c, d) = sigs(false);
        assert_eq!((a, b), (sentinel.clone(), None));
        assert_eq!((c, d), (Some(json!("SIG_A")), None));
        let (a, b, c, d) = sigs(true);
        assert_eq!((a, b, c, d), (sentinel.clone(), None, sentinel, None));
    }

    #[test]
    fn tool_choice_variants() {
        let mk = |tc: Value| {
            json!({"model":"m","messages":[{"role":"user","content":"hi"}],
              "tools":[{"name":"my tool","input_schema":{"type":"object"}}],
              "tool_choice":tc})
            .to_string()
        };
        let cfg = |tc: Value| {
            translate(&mk(tc), &opts(false, SchemaMode::JsonSchema)).0["request"]["toolConfig"]
                .clone()
        };
        assert_eq!(
            cfg(json!({"type":"any"}))["functionCallingConfig"]["mode"],
            "ANY"
        );
        assert_eq!(
            cfg(json!({"type":"none"}))["functionCallingConfig"]["mode"],
            "NONE"
        );
        assert_eq!(
            cfg(json!({"type":"tool","name":"my tool"})),
            json!({"functionCallingConfig":{"mode":"ANY","allowedFunctionNames":["my_tool"]}})
        );
        // no tools: no toolConfig
        let (v, _, _) = translate(
            &json!({"model":"m","messages":[{"role":"user","content":"hi"}],
                    "tool_choice":{"type":"auto"}})
            .to_string(),
            &opts(false, SchemaMode::JsonSchema),
        );
        assert!(v["request"].get("toolConfig").is_none());
        assert!(v["request"].get("tools").is_none());
        assert_eq!(
            v["request"]["systemInstruction"]["parts"][0]["text"],
            AGY_IDENTITY
        );
        assert_eq!(v["request"]["contents"][0]["parts"][0]["text"], "hi");
    }

    #[test]
    fn cleaned_mode_and_defaults() {
        let body = json!({"model":"m","messages":[{"role":"user","content":"hi"}],
            "tools":[{"name":"t","input_schema":{"type":"object","properties":{"a":{"type":["string","null"]}}}},
                     {"name":"u"}]})
        .to_string();
        let mut o = opts(false, SchemaMode::Cleaned);
        o.thinking_level = None;
        let (v, _, _) = translate(&body, &o);
        let d = &v["request"]["tools"][0]["functionDeclarations"];
        assert_eq!(d[0]["parameters"]["properties"]["a"]["nullable"], true);
        assert!(d[0].get("parametersJsonSchema").is_none());
        assert_eq!(d[1]["parameters"]["required"], json!(["reason"]));
        assert_eq!(
            v["request"]["generationConfig"],
            json!({"maxOutputTokens":32000,"thinkingConfig":{"includeThoughts":true}})
        );
    }

    #[test]
    fn session_id_follows_first_user_message() {
        let mk = |second: &str| {
            json!({"model":"m","messages":[
                {"role":"user","content":"same start"},
                {"role":"assistant","content":"a"},
                {"role":"user","content":second}]})
            .to_string()
        };
        let o = opts(false, SchemaMode::JsonSchema);
        let a = translate(&mk("one"), &o).0["request"]["sessionId"].clone();
        let b = translate(&mk("two"), &o).0["request"]["sessionId"].clone();
        assert_eq!(a, b);
        let other =
            json!({"model":"m","messages":[{"role":"user","content":"different"}]}).to_string();
        assert_ne!(a, translate(&other, &o).0["request"]["sessionId"]);
    }

    #[test]
    fn system_role_messages_become_user_text() {
        let text = json!({"model":"m","messages":[
            {"role":"user","content":"hi"},
            {"role":"system","content":"rule one"},
            {"role":"system","content":[{"type":"text","text":"rule two"}]},
            {"role":"tool","content":"dropped"},
            {"role":"assistant","content":"ok"}]})
        .to_string();
        let o = opts(false, SchemaMode::JsonSchema);
        let v = translate(&text, &o).0;
        let c = v["request"]["contents"].as_array().unwrap();
        assert_eq!(c.len(), 2);
        assert_eq!(c[0]["role"], "user");
        let texts: Vec<_> = c[0]["parts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p["text"].as_str().unwrap())
            .collect();
        assert_eq!(texts, ["hi", "rule one", "rule two"]);
        assert_eq!(c[1]["role"], "model");
        let only_system = json!({"model":"m","messages":[
            {"role":"system","content":"a"},{"role":"user","content":"b"}]})
        .to_string();
        let plain = json!({"model":"m","messages":[
            {"role":"user","content":"b"}]})
        .to_string();
        assert_eq!(
            translate(&only_system, &o).0["request"]["sessionId"],
            translate(&plain, &o).0["request"]["sessionId"]
        );
    }

    #[test]
    fn thinking_level_suffixes() {
        assert_eq!(thinking_level_for("gemini-4-neon-low"), Some("LOW"));
        assert_eq!(thinking_level_for("gemini-4-neon-medium"), Some("MEDIUM"));
        assert_eq!(thinking_level_for("gemini-4-neon-high"), Some("HIGH"));
        assert_eq!(thinking_level_for("gemini-4-neon-xhigh"), Some("XHIGH"));
        assert_eq!(thinking_level_for("gemini-4-neon"), None);
    }
}
