//! The Anthropic side of the router, shared by every provider that translates another API:
//! the Messages request types and the response writer (SSE stream and plain JSON).

pub mod stream;
pub mod types;
