# Contributing to claude-mr

`claude-mr` is one binary crate: a local proxy that routes Claude Code's API calls by model name
to a local model server, to Google Gemini through the Antigravity CLI login, or to
api.anthropic.com. This page covers the environment, the checks a change passes before review,
and the rules for tests.

## Environment

- **Rust** through rustup. `rust-toolchain.toml` selects stable with rustfmt and clippy.
- **Linux.** The router and its tests are built and run on Linux; CI runs on Ubuntu.
- **cargo-deny** for the dependency gate: `cargo install cargo-deny --locked`.
- For trying Google models by hand: the Antigravity CLI (`agy`), logged in once, and
  `secret-tool` (libsecret) for the keyring. The test suite needs neither.

## Checking a change

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo deny check
```

CI runs the same four, plus CodeQL, and reports them as one check, `CI result`.

`claude-mr -v` prints the diagnostic lines: where the Google values came from, why Google models
are not served, streams a client closed early. Run the router with it when a change touches
startup or routing, and check the startup line lists the routes you expect.

## Tests

- Tests run against in-process fakes: a fake Anthropic upstream, a fake Google (token endpoint,
  project lookup, `streamGenerateContent`), a fake `agy`. They reach no real service and need no
  login.
- Fixtures hold fake values only. A real token, API key, OAuth client id or client secret never
  enters the repository, a test or a log line in a test assertion.
- Every file a test writes lives in a `tempfile` guard, so it is removed even when the test
  panics.
- Error messages are tested for the secrets they must not contain.

## Pull requests

- One logical change per pull request.
- Open it as a draft. CI runs when it is marked ready for review.
- A pull request merges as one squashed commit after the `CI result` check passes.
- Documentation changes with the code: the README for setup and behaviour, the `--help` text for
  flags, rustdoc for items. Each says what exists and what it does.
