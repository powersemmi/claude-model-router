# claude-mr

`claude-mr` is a local proxy for Claude Code. It reads the `model` field of each API request
and sends the request to one of three backends:

- **a local model server** that speaks the Anthropic Messages API (llama.cpp's server, for
  example), for the models you list with `--local-model`;
- **Google Gemini** through Google's Cloud Code Assist API, with the login of the Antigravity
  CLI (`agy`), for the `gemini-4-neon-*` models. The router translates the request to Gemini and
  the answer back;
- **api.anthropic.com** for every other model, unchanged, with your own Claude login.

It listens on `127.0.0.1` only. Claude Code reaches it through `ANTHROPIC_BASE_URL`.

```sh
cargo install --path .
claude-mr
ANTHROPIC_BASE_URL=http://127.0.0.1:8787 claude --model gemini-4-neon-medium
```

The step-by-step setup below is written so that a person or a coding agent can follow it
command by command. Every step is safe to run again.

## Setup, step by step

### 1. Check the prerequisites

Run each check from the repository root.

| Check | Command | What the output means |
|---|---|---|
| Rust toolchain | `cargo --version` | Prints `cargo 1.x.y …`. "command not found": install Rust from <https://rustup.rs> first. |
| Claude Code | `claude --version` | Prints `2.x.y (Claude Code)`. |
| Antigravity CLI | `command -v agy && agy --version` | Prints a path and a version such as `1.5.16`: Google models can be served. No output: Google models are not served, and the router runs without them. |
| Antigravity login | `secret-tool lookup service gemini username antigravity >/dev/null && echo logged-in` | Prints `logged-in`: the router can use the login. No output: run `agy` once and log in. Needed only for Google models. |
| Local model server | `curl -s -m 3 -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8080/v1/models` | `000`: nothing listens there. Any other code: a server answers. Use the server's own URL if it is elsewhere. |
| Port is free | `curl -s -m 3 -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8787/` | `000`: the port is free (or a router is not running yet). Any other code: something already listens; it may be a `claude-mr` from an earlier setup (see step 4). |

### 2. Ask the person these questions

Ask only these; everything else has a working default.

1. **Local models.** Which model ids does the local server serve, and at which URL? Default:
   none needed. Without a local server, skip `--local` and `--local-model`; the default local
   model id `qwen3.8-flash-next-coder-iq1_m` then answers with an error when picked.
2. **Gemini.** Do they want Gemini models through their Antigravity login? Default: yes when
   `agy` is installed (step 1), and nothing to configure. No: add `--no-google`.
3. **Port.** Default `8787`. Change it only when step 1 shows the port taken by something else.

### 3. Install

```sh
cargo install --path .
command -v claude-mr
```

The first command builds a release binary and puts it into `~/.cargo/bin` (about 20 seconds);
run again, it replaces the binary. The second prints the installed path; `~/.cargo/bin` must be
on `PATH`.

Check that it starts and see what it serves, with the flags chosen in step 2:

```sh
timeout 3 claude-mr -v --port 8787
```

Expected output, then the command ends after 3 seconds:

```text
router: Google OAuth client from /home/you/.local/bin/agy (3 client ids, 3 secrets); User-Agent antigravity/cli/1.5.16 (…)
router: 127.0.0.1:8787; gemini-4-neon-high -> google, gemini-4-neon-low -> google, gemini-4-neon-medium -> google, gemini-4-neon-xhigh -> google, qwen3.8-flash-next-coder-iq1_m -> local; rest -> api.anthropic.com
```

Without `agy` the first line reads `router: Google models are not served: …` and the table has
no `gemini-*` entries. `cannot bind 127.0.0.1:8787: Address already in use` means the router
already runs (step 4) or another program holds the port.

### 4. Run it persistently (Linux, systemd user service)

Write the unit. Put the path from `command -v claude-mr` into `ExecStart`, and add the flags
from step 2 to that line (for example `--local http://127.0.0.1:8080 --local-model my-model`):

```sh
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/claude-mr.service <<'EOF'
[Unit]
Description=claude-mr, the model router for Claude Code

[Service]
ExecStart=%h/.cargo/bin/claude-mr --port 8787
Restart=on-failure
RestartSec=2

[Install]
WantedBy=default.target
EOF
systemd-analyze --user verify ~/.config/systemd/user/claude-mr.service
```

`systemd-analyze` prints nothing when the unit is valid. Then enable and start it:

```sh
systemctl --user daemon-reload
systemctl --user enable --now claude-mr
systemctl --user restart claude-mr
systemctl --user status claude-mr --no-pager
journalctl --user -u claude-mr -n 20 --no-pager
```

`status` shows `active (running)`; the journal holds the startup line from step 3 and one line
per request. After a change to the unit, run `daemon-reload` and `restart` again. The service
runs while the person is logged in, which is also when the keyring with the Antigravity login is
unlocked.

### 5. Point Claude Code at the router

Set `ANTHROPIC_BASE_URL` in the `env` block of `~/.claude/settings.json`. That reaches every
Claude Code session, including ones not started from a shell, and wins over a shell export of
the same variable. The command merges the key into an existing file and creates the file when
it is missing (it needs `jq`):

```sh
f=~/.claude/settings.json
mkdir -p ~/.claude
[ -s "$f" ] || echo '{}' > "$f"
jq '.env.ANTHROPIC_BASE_URL = "http://127.0.0.1:8787"' "$f" > "$f.tmp" && mv "$f.tmp" "$f"
jq -r '.env.ANTHROPIC_BASE_URL' "$f"
```

The last command prints `http://127.0.0.1:8787`. The Claude login stays as it is: requests for
Claude models go through the router to api.anthropic.com with the same credentials. For one
shell only, `export ANTHROPIC_BASE_URL=http://127.0.0.1:8787` does the same. Inside Claude Code,
`/status` shows an `Anthropic base URL` line with the router's address.

### 6. Pick a routed model and check that routing works

Start Claude Code on a routed model with `claude --model <id>`, or switch inside a session with
`/model <id>`. The ids are the ones in the router's startup line, such as `gemini-4-neon-medium`
or a local model id. Any other id, such as the built-in Claude models, goes to Anthropic.

Smoke test, one request per backend in use:

```sh
claude -p --model gemini-4-neon-low "Reply with the single word OK"
claude -p --model <local-model-id> "Reply with the single word OK"
journalctl --user -u claude-mr -n 5 --no-pager
```

Each `claude -p` prints `OK` (a preceding `[claude-code:unrecognized_model]` line is Claude
Code noting a model id it does not know, and is harmless). The journal shows one line per
request with the backend that served it:

```text
router: POST /v1/messages model=gemini-4-neon-low side=google status=200 in=… out=2 …
```

`side=local` and `side=anthropic` mark the other two backends.

## How the Google values are found

Talking to Google takes three values of the Antigravity CLI: its OAuth client id, its client
secret and its `User-Agent`. The router derives all three from the installed `agy` binary at
startup, once, with nothing to configure:

- It reads the binary once (about 0.1 s) and collects every OAuth client id and `GOCSPX-`
  secret in it. The binary holds several OAuth clients, so the first token refresh tries the id
  and secret pairs until Google accepts one, and keeps that pair.
- It builds the `User-Agent` the way the CLI builds it: the version from `agy --version`, the
  build changelist from the binary, and this machine's operating system and architecture.

`claude-mr -v` prints where the values came from.

### Overriding the Google values

A person who wants to replace a derived value, or who runs an `agy` build the derivation does
not understand, sets it in `$XDG_CONFIG_HOME/claude-mr/google.json` (usually
`~/.config/claude-mr/google.json`), or in the file given with `--google-client-config`. Each
field is optional; a field set there replaces the derived value, and the others are still
derived:

```json
{
  "client_id": "<digits>-<hash>.apps.googleusercontent.com",
  "client_secret": "GOCSPX-<28 characters>",
  "user_agent": "antigravity/cli/<version> (aidev_client; os_type=linux; arch=amd64; cl=<changelist>; auth_method=consumer)"
}
```

The values can be read from the binary by hand:

```sh
grep -aoE '[0-9]+-[a-z0-9]{32}\.apps\.googleusercontent\.com|GOCSPX-[A-Za-z0-9_-]{28}' "$(command -v agy)" | sort -u
agy --version
grep -aoE 'google_src/files/[0-9]+/depot' "$(command -v agy)" | sort | uniq -c
```

Each client id pairs with one of the secrets; Google accepts only the right pair.

## Options

| Flag | Meaning |
|---|---|
| `--port <PORT>` | Port on 127.0.0.1; default 8787; 0 picks a free one. |
| `--local <URL>` | Base URL of the local model server; default `http://127.0.0.1:8080`. |
| `--local-model <NAME>` | Model served locally; repeatable. Default `qwen3.8-flash-next-coder-iq1_m`. |
| `--usage-log <PATH>` | Usage log file; default `$XDG_STATE_HOME/claude-mr/usage.jsonl`. |
| `-v`, `--verbose` | Print diagnostic lines: why Google models are not served or cannot answer, where the Google client values came from, streams the client closed early. |
| `--google-model <NAME>` | Model served by Google; repeatable. Default the four `gemini-4-neon-*` levels (`low`, `medium`, `high`, `xhigh`). |
| `--google-credentials <FILE>` | JSON file with the Antigravity login, in the keyring entry's shape; default the keyring. |
| `--agy <FILE>` | The Antigravity CLI binary (also `CLAUDE_MR_AGY`); default the first `agy` on `PATH`, then `~/.local/bin/agy`. |
| `--google-client-config <FILE>` | Config file for the Google client values; default `$XDG_CONFIG_HOME/claude-mr/google.json`. |
| `--no-google` | Serve no Google models and skip looking for the Antigravity CLI. |

## Usage log

Every request served by Google adds one JSON line to the usage log: time, provider, model,
status, input, cached, output and thinking tokens, latency, time to first token, stop reason and
error. The default file is `$XDG_STATE_HOME/claude-mr/usage.jsonl` (usually
`~/.local/state/claude-mr/usage.jsonl`); `--usage-log` sets another.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| No `gemini-*` models in the startup line | `agy` is not found. Install the Antigravity CLI, or pass its path with `--agy`. `claude-mr -v` lists the places it looked. |
| A Gemini model answers with a 400 error that starts "Google models are unavailable" | `agy` is installed, but a value could not be derived from it. The error names what is missing, the config file path and a JSON template. Set the missing fields in that file (see "Overriding the Google values") and restart the router. |
| A Gemini model answers "google credentials: … not found" | The Antigravity login is missing from the keyring. Run `agy` and log in. |
| `cannot bind 127.0.0.1:8787: Address already in use` | A router already runs there (`systemctl --user status claude-mr`), or another program holds the port: pick another `--port` and use it in `ANTHROPIC_BASE_URL` too. |
| Every Claude Code session fails with "Connection refused" | `ANTHROPIC_BASE_URL` points at the router, and the router is not running. Start it (`systemctl --user start claude-mr`), or remove the key (see Uninstall). |
| A local model answers `router upstream failed: client error (Connect)` | Nothing listens at the `--local` URL. Start the local server, or fix the URL in the unit. |

## Uninstall

Every step is safe to run when the part is already gone.

```sh
systemctl --user disable --now claude-mr
rm -f ~/.config/systemd/user/claude-mr.service
systemctl --user daemon-reload
f=~/.claude/settings.json
[ -s "$f" ] && jq 'del(.env.ANTHROPIC_BASE_URL)' "$f" > "$f.tmp" && mv "$f.tmp" "$f"
cargo uninstall claude-model-router
```

The config file `~/.config/claude-mr/google.json` and the usage log
`~/.local/state/claude-mr/usage.jsonl` stay; delete them by hand when they are no longer wanted.

## License

MIT, see [LICENSE](LICENSE).
