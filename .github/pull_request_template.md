# Description

Please include a summary of the change and specify which issue is being addressed. Additionally,
provide relevant motivation and context.

Fixes # (issue number)

## Type of change

Please delete options that are not relevant.

- [ ] Documentation (README, rustdoc, CLI help)
- [ ] Bug fix (a non-breaking change that resolves an issue)
- [ ] New feature (a non-breaking change that adds functionality)
- [ ] Breaking change (a flag, default, config file field or routing rule that behaves
      differently for an existing setup)

## Checklist

- [ ] `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test` and
      `cargo deny check` pass locally
- [ ] I have performed a self-review of my own code
- [ ] I have updated the README and the `--help` text where the behaviour changed
- [ ] I have added tests that validate my fix or new feature
- [ ] Tests and fixtures hold no real token, key, client id or client secret
