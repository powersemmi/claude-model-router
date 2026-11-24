//! The Google provider's building blocks: the OAuth client and `User-Agent` of the
//! Antigravity CLI, the user's login with it, the
//! converter from an Anthropic Messages request to the Gemini `generateContent` body used by
//! Google's Code Assist backend, the translation of Gemini's server-sent events into Anthropic
//! events, and the store of thought signatures that Claude Code does not carry between turns.

pub mod auth;
pub mod client;
pub mod provider;
pub mod request;
pub mod response;
pub mod schema;
pub mod signatures;
