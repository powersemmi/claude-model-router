# Security policy

## Supported versions

Security fixes ship in a new release of `claude-mr` from `main`. Only the newest release
receives fixes; upgrading to it is the fix.

## Reporting a vulnerability

Report it privately through GitHub:
[open a security advisory](https://github.com/powersemmi/claude-model-router/security/advisories/new).
Do not open a public issue, pull request or discussion for it.

A useful report names the affected version or commit, the backend involved (Google, local,
Anthropic), the impact, and the shortest command line or request that reproduces it. Replace any
token, key or client secret in it with `<redacted>`.

## What happens next

- The report is acknowledged within 7 days.
- The fix is prepared in a private fork of the advisory and released.
- The advisory is published on GitHub with the fixed version.
- The reporter is credited in the advisory unless they ask otherwise.

## Scope

The `claude-mr` binary and its source in this repository. The router listens on 127.0.0.1 and
handles credentials: the Claude login it forwards to api.anthropic.com, the Antigravity refresh
and access tokens, and the OAuth client values it derives from the `agy` binary. A path that
exposes any of them (in a log line, an error message, a response, or to another host) is in
scope.
