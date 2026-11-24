//! Fallback JSON schema sanitizer: turns a JSON Schema into the OpenAPI subset Gemini's
//! `parameters` field accepts. Used only after the backend rejected the raw schema, so it works on
//! `serde_json::Value` and does not try to be fast.

use serde_json::{Map, Value};

/// Deepest chain of `$ref` indirections that is followed.
const MAX_REF_DEPTH: u32 = 8;

/// Schema keywords whose value moves into the description as a hint.
const HINT_KEYS: [&str; 8] = [
    "minLength",
    "maxLength",
    "pattern",
    "minimum",
    "maximum",
    "minItems",
    "maxItems",
    "multipleOf",
];

/// Converts a JSON Schema into Gemini's OpenAPI-subset schema.
///
/// `$ref`s are resolved against the root's `$defs`/`definitions`, `allOf` is merged, `anyOf` and
/// `oneOf` collapse to their best option, types are upper-cased, unsupported constraints become
/// description hints and every object ends up with at least one property.
pub fn clean(schema: &Value) -> Value {
    let cx = Ctx { root: schema };
    let mut m = cx.build(schema, 0);
    finish(&mut m, "OBJECT");
    Value::Object(m)
}

struct Ctx<'a> {
    root: &'a Value,
}

impl Ctx<'_> {
    /// A cleaned and finished schema node.
    fn node(&self, v: &Value, depth: u32) -> Map<String, Value> {
        let mut m = self.build(v, depth);
        finish(&mut m, "STRING");
        m
    }

    fn resolve_ref(&self, r: &str) -> Option<&Value> {
        if !r.starts_with('#') {
            return None;
        }
        let name = r.rsplit('/').next()?;
        ["$defs", "definitions"]
            .iter()
            .find_map(|k| self.root.get(k).and_then(|d| d.get(name)))
    }

    /// A node with its keywords translated but defaults (type, placeholders) not yet applied.
    fn build(&self, v: &Value, depth: u32) -> Map<String, Value> {
        let Some(obj) = v.as_object() else {
            return Map::new();
        };
        if let Some(Value::String(r)) = obj.get("$ref") {
            let name = r.rsplit('/').next().unwrap_or(r);
            let target = if depth < MAX_REF_DEPTH {
                self.resolve_ref(r)
            } else {
                None
            };
            let mut m = match target {
                Some(t) => self.build(t, depth + 1),
                None => {
                    let mut m = Map::new();
                    m.insert("type".into(), "OBJECT".into());
                    m.insert("description".into(), format!("See: {name}").into());
                    return m;
                }
            };
            if let Some(Value::String(d)) = obj.get("description") {
                m.entry("description").or_insert_with(|| d.clone().into());
            }
            return m;
        }

        let mut out = Map::new();
        let mut hints: Vec<String> = Vec::new();
        let mut nullable = false;

        match obj.get("type") {
            Some(Value::String(t)) if t == "null" => {
                out.insert("type".into(), "STRING".into());
                nullable = true;
            }
            Some(Value::String(t)) => {
                out.insert("type".into(), t.to_uppercase().into());
            }
            Some(Value::Array(ts)) => {
                let mut first = None;
                for t in ts.iter().filter_map(Value::as_str) {
                    if t == "null" {
                        nullable = true;
                    } else if first.is_none() {
                        first = Some(t);
                    }
                }
                out.insert(
                    "type".into(),
                    first.unwrap_or("string").to_uppercase().into(),
                );
            }
            _ => {}
        }
        if let Some(Value::String(d)) = obj.get("description") {
            out.insert("description".into(), d.clone().into());
        }
        if let Some(Value::String(f)) = obj.get("format") {
            if f == "enum" || f == "date-time" {
                out.insert("format".into(), f.clone().into());
            } else {
                hints.push(format!("(format: {f})"));
            }
        }
        if let Some(c) = obj.get("const") {
            out.insert("enum".into(), Value::Array(vec![stringify(c).into()]));
        } else if let Some(Value::Array(e)) = obj.get("enum") {
            out.insert(
                "enum".into(),
                Value::Array(e.iter().map(|x| stringify(x).into()).collect()),
            );
        }
        if out.contains_key("enum") {
            out.insert("type".into(), "STRING".into());
        }

        if let Some(Value::Object(props)) = obj.get("properties") {
            let mut cleaned = Map::new();
            let mut nullable_props: Vec<&str> = Vec::new();
            for (k, pv) in props {
                let child = self.node(pv, depth);
                if child.get("nullable") == Some(&Value::Bool(true)) {
                    nullable_props.push(k);
                }
                cleaned.insert(k.clone(), Value::Object(child));
            }
            out.insert("properties".into(), Value::Object(cleaned));
            if let Some(Value::Array(req)) = obj.get("required") {
                let kept: Vec<Value> = req
                    .iter()
                    .filter_map(Value::as_str)
                    .filter(|r| !nullable_props.contains(r))
                    .map(|r| r.into())
                    .collect();
                out.insert("required".into(), Value::Array(kept));
            }
        } else if let Some(Value::Array(req)) = obj.get("required") {
            out.insert("required".into(), Value::Array(req.clone()));
        }

        match obj.get("items") {
            Some(v @ Value::Object(_)) => {
                out.insert("items".into(), Value::Object(self.node(v, depth)));
            }
            Some(Value::Array(entries)) => {
                let built: Vec<_> = entries.iter().map(|e| self.build(e, depth)).collect();
                let top = built.iter().map(rank).max();
                let best = built.into_iter().find(|m| Some(rank(m)) == top);
                if let Some(mut m) = best {
                    finish(&mut m, "STRING");
                    out.insert("items".into(), Value::Object(m));
                }
            }
            _ => {}
        }

        for key in HINT_KEYS {
            if let Some(v) = obj.get(key) {
                hints.push(format!("({key}: {})", stringify(v)));
            }
        }
        if obj.get("additionalProperties") == Some(&Value::Bool(false)) {
            hints.push("(additionalProperties: false)".into());
        }
        if let Some(d) = obj.get("default") {
            hints.push(format!("(default: {d})"));
        }

        if let Some(Value::Array(all)) = obj.get("allOf") {
            for sub in all {
                let m = self.build(sub, depth);
                merge_into(&mut out, m);
            }
        }
        for key in ["anyOf", "oneOf"] {
            if let Some(Value::Array(options)) = obj.get(key) {
                self.collapse(options, depth, &mut out, &mut nullable);
            }
        }

        if !hints.is_empty() {
            append_description(&mut out, &hints.join(" "));
        }
        if nullable {
            out.insert("nullable".into(), true.into());
        }
        out
    }

    /// Folds an `anyOf`/`oneOf` into `out`: the best option wins.
    fn collapse(
        &self,
        options: &[Value],
        depth: u32,
        out: &mut Map<String, Value>,
        nullable: &mut bool,
    ) {
        let mut built: Vec<Map<String, Value>> = Vec::new();
        for opt in options {
            if is_null_schema(opt) {
                *nullable = true;
            } else {
                built.push(self.build(opt, depth));
            }
        }
        let mut labels: Vec<String> = Vec::new();
        for m in &built {
            let label = type_label(m);
            if !labels.contains(&label) {
                labels.push(label);
            }
        }
        let Some(best_rank) = built.iter().map(rank).max() else {
            return;
        };
        let Some(pos) = built.iter().position(|m| rank(m) == best_rank) else {
            return;
        };
        let best = built.swap_remove(pos);
        merge_into(out, best);
        if labels.len() > 1 {
            append_description(out, &format!("Accepts: {}", labels.join(" | ")));
        }
    }
}

/// Merges `other` into `out`: properties and required unite, scalar keys keep the first value.
fn merge_into(out: &mut Map<String, Value>, other: Map<String, Value>) {
    for (k, v) in other {
        match (k.as_str(), v) {
            ("properties", Value::Object(p)) => {
                if let Some(Value::Object(existing)) = out.get_mut("properties") {
                    for (pk, pv) in p {
                        existing.entry(pk).or_insert(pv);
                    }
                } else {
                    out.insert(k, Value::Object(p));
                }
            }
            ("required", Value::Array(r)) => {
                if let Some(Value::Array(existing)) = out.get_mut("required") {
                    for x in r {
                        if !existing.contains(&x) {
                            existing.push(x);
                        }
                    }
                } else {
                    out.insert(k, Value::Array(r));
                }
            }
            (_, v) => {
                out.entry(k).or_insert(v);
            }
        }
    }
}

/// Appends `text` to the node's description, separated by a space.
fn append_description(m: &mut Map<String, Value>, text: &str) {
    let joined = match m.get("description") {
        Some(Value::String(d)) if !d.is_empty() => format!("{d} {text}"),
        _ => text.to_owned(),
    };
    m.insert("description".into(), joined.into());
}

/// Preference of a union option: object with properties, then array, then anything else.
fn rank(m: &Map<String, Value>) -> u8 {
    let has_props = matches!(m.get("properties"), Some(Value::Object(p)) if !p.is_empty());
    let ty = m.get("type").and_then(Value::as_str);
    if has_props {
        3
    } else if ty == Some("ARRAY") || m.contains_key("items") {
        2
    } else {
        1
    }
}

fn type_label(m: &Map<String, Value>) -> String {
    match m.get("type").and_then(Value::as_str) {
        Some(t) => t.to_lowercase(),
        None if m.contains_key("properties") => "object".into(),
        None if m.contains_key("items") => "array".into(),
        None => "string".into(),
    }
}

fn is_null_schema(v: &Value) -> bool {
    match v.get("type") {
        Some(Value::String(t)) => t == "null",
        Some(Value::Array(ts)) => ts.iter().all(|t| t == "null") && !ts.is_empty(),
        _ => v.get("const") == Some(&Value::Null),
    }
}

/// A scalar as plain text (strings without quotes).
fn stringify(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Applies defaults: type, `items`, the object placeholder; drops what the type cannot carry.
fn finish(m: &mut Map<String, Value>, default_type: &str) {
    let ty = match m.get("type").and_then(Value::as_str) {
        Some(t) => t.to_owned(),
        None if m.contains_key("properties") => "OBJECT".into(),
        None if m.contains_key("items") => "ARRAY".into(),
        None => default_type.to_owned(),
    };
    m.insert("type".into(), ty.clone().into());
    if ty != "OBJECT" {
        m.remove("properties");
        m.remove("required");
    }
    if ty != "ARRAY" {
        m.remove("items");
    } else if !matches!(m.get("items"), Some(Value::Object(_))) {
        let mut item = Map::new();
        item.insert("type".into(), "STRING".into());
        m.insert("items".into(), Value::Object(item));
    }
    if ty == "OBJECT" {
        let empty = !matches!(m.get("properties"), Some(Value::Object(p)) if !p.is_empty());
        if empty {
            let mut reason = Map::new();
            reason.insert("type".into(), "STRING".into());
            reason.insert("description".into(), "Reason for calling this tool".into());
            let mut props = Map::new();
            props.insert("reason".into(), Value::Object(reason));
            m.insert("properties".into(), Value::Object(props));
            m.insert("required".into(), Value::Array(vec!["reason".into()]));
        } else {
            let existing: Vec<String> = match m.get("properties") {
                Some(Value::Object(p)) => p.keys().cloned().collect(),
                _ => Vec::new(),
            };
            let kept: Vec<Value> = match m.get("required") {
                Some(Value::Array(r)) => r
                    .iter()
                    .filter(|x| x.as_str().is_some_and(|s| existing.iter().any(|e| e == s)))
                    .cloned()
                    .collect(),
                _ => Vec::new(),
            };
            if kept.is_empty() {
                m.remove("required");
            } else {
                m.insert("required".into(), Value::Array(kept));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn resolves_refs_and_unresolved_become_placeholders() {
        let s = json!({
            "type":"object",
            "properties":{"a":{"$ref":"#/$defs/Item"},"b":{"$ref":"#/$defs/Missing"}},
            "$defs":{"Item":{"type":"object","properties":{"n":{"type":"integer"}},"required":["n"]}}
        });
        let c = clean(&s);
        assert_eq!(c["properties"]["a"]["properties"]["n"]["type"], "INTEGER");
        assert_eq!(c["properties"]["a"]["required"], json!(["n"]));
        assert_eq!(c["properties"]["b"]["description"], "See: Missing");
        assert_eq!(c["properties"]["b"]["type"], "OBJECT");
    }

    #[test]
    fn recursive_ref_stops() {
        let s = json!({
            "$ref":"#/$defs/Node",
            "$defs":{"Node":{"type":"object","properties":{"next":{"$ref":"#/$defs/Node"}}}}
        });
        let c = clean(&s);
        let mut cur = &c;
        let mut depth = 0;
        while let Some(n) = cur["properties"].get("next") {
            cur = n;
            depth += 1;
            assert!(depth < 20);
        }
        assert_eq!(cur["type"], "OBJECT");
    }

    #[test]
    fn any_of_with_null_picks_object_and_marks_nullable() {
        let s = json!({"type":"object","properties":{
            "x":{"anyOf":[{"type":"null"},{"type":"string"},
                {"type":"object","properties":{"k":{"type":"string"}}}],"description":"d"}},
            "required":["x"]});
        let c = clean(&s);
        let x = &c["properties"]["x"];
        assert_eq!(x["type"], "OBJECT");
        assert_eq!(x["nullable"], true);
        assert_eq!(x["properties"]["k"]["type"], "STRING");
        assert_eq!(x["description"], "d Accepts: string | object");
        assert!(c.get("required").is_none());
    }

    #[test]
    fn type_array_becomes_nullable_and_leaves_required() {
        let s = json!({"type":"object","properties":{
            "a":{"type":["string","null"]},"b":{"type":"number"}},"required":["a","b","ghost"]});
        let c = clean(&s);
        assert_eq!(c["properties"]["a"]["type"], "STRING");
        assert_eq!(c["properties"]["a"]["nullable"], true);
        assert_eq!(c["required"], json!(["b"]));
    }

    #[test]
    fn const_and_enum_are_strings() {
        let s = json!({"type":"object","properties":{
            "c":{"const":5},"e":{"type":"integer","enum":[1,2,"x"]}}});
        let c = clean(&s);
        assert_eq!(c["properties"]["c"]["enum"], json!(["5"]));
        assert_eq!(c["properties"]["e"]["enum"], json!(["1", "2", "x"]));
        assert_eq!(c["properties"]["e"]["type"], "STRING");
    }

    #[test]
    fn constraints_move_into_description() {
        let s = json!({"type":"object","properties":{
            "s":{"type":"string","minLength":1,"maxLength":9,"description":"name","format":"uri"},
            "t":{"type":"string","format":"date-time"},
            "o":{"type":"object","additionalProperties":false,"properties":{"q":{"type":"string"}}}}});
        let c = clean(&s);
        let sd = c["properties"]["s"]["description"].as_str().unwrap();
        assert!(sd.starts_with("name "));
        assert!(sd.contains("(minLength: 1)"));
        assert!(sd.contains("(maxLength: 9)"));
        assert!(sd.contains("(format: uri)"));
        assert!(c["properties"]["s"].get("format").is_none());
        assert!(c["properties"]["s"].get("minLength").is_none());
        assert_eq!(c["properties"]["t"]["format"], "date-time");
        assert!(c["properties"]["o"]["description"]
            .as_str()
            .unwrap()
            .contains("additionalProperties: false"));
        assert!(c["properties"]["o"].get("additionalProperties").is_none());
    }

    #[test]
    fn empty_object_gets_reason_placeholder() {
        let c = clean(&json!({"type":"object","properties":{}}));
        assert_eq!(c["properties"]["reason"]["type"], "STRING");
        assert_eq!(c["required"], json!(["reason"]));
        let c = clean(&json!({}));
        assert_eq!(c["type"], "OBJECT");
        assert_eq!(c["required"], json!(["reason"]));
    }

    #[test]
    fn array_without_items_gets_string_items_and_tuples_pick_best() {
        let s = json!({"type":"object","properties":{
            "a":{"type":"array"},
            "t":{"type":"array","items":[{"type":"string"},{"type":"object","properties":{"z":{"type":"number"}}}]}}});
        let c = clean(&s);
        assert_eq!(c["properties"]["a"]["items"], json!({"type":"STRING"}));
        assert_eq!(c["properties"]["t"]["items"]["type"], "OBJECT");
        assert_eq!(
            c["properties"]["t"]["items"]["properties"]["z"]["type"],
            "NUMBER"
        );
    }

    #[test]
    fn all_of_merges_and_non_objects_lose_properties() {
        let s = json!({"allOf":[
            {"type":"object","properties":{"a":{"type":"string"}},"required":["a"]},
            {"properties":{"b":{"type":"string"}},"required":["b"]}],
            "properties":{"s":{"type":"string","properties":{"x":{"type":"string"}}}}});
        let c = clean(&s);
        assert_eq!(c["type"], "OBJECT");
        assert!(c["properties"].get("a").is_some());
        assert!(c["properties"].get("b").is_some());
        assert!(c["properties"]["s"].get("properties").is_none());
        let mut req: Vec<&str> = c["required"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(Value::as_str)
            .collect();
        req.sort();
        assert_eq!(req, ["a", "b"]);
    }
}
