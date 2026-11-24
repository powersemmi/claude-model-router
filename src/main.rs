//! `claude-mr` command line.
//!
//! ```text
//! claude-mr [--port 8787] [--local http://127.0.0.1:8080] [--local-model NAME ...] [-v]
//! ANTHROPIC_BASE_URL=http://127.0.0.1:8787 claude
//! ```
//!
//! Google models are served when the Antigravity CLI (`agy`) is installed; see
//! [`claude_model_router::google::client`] for how its OAuth client and `User-Agent` are found.
#![forbid(unsafe_code)]

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use clap::Parser;
use claude_model_router::google::auth::{CredentialSource, GoogleEndpoints};
use claude_model_router::google::client;
use claude_model_router::google::provider::{GoogleConfig, GoogleProvider};
use claude_model_router::passthrough::{PassAuth, Passthrough};
use claude_model_router::provider::{upstream_client, UpstreamClient};
use claude_model_router::routes::Routes;
use claude_model_router::usage::UsageLog;
use claude_model_router::{serve, set_verbose, verbose};
use hyper::Uri;
use tokio::net::TcpListener;

const DEFAULT_MODEL: &str = "qwen3.8-flash-next-coder-iq1_m";
const GOOGLE_MODELS: [&str; 4] = [
    "gemini-4-neon-low",
    "gemini-4-neon-medium",
    "gemini-4-neon-high",
    "gemini-4-neon-xhigh",
];

#[derive(Parser)]
#[command(about = "Route Claude Code's API calls by model name")]
struct Cli {
    /// Port to listen on (127.0.0.1 only); 0 picks a free one.
    #[arg(long, default_value_t = 8787)]
    port: u16,
    /// Base URL of the local model server.
    #[arg(long, default_value = "http://127.0.0.1:8080")]
    local: String,
    /// Model id served locally (repeatable).
    #[arg(long = "local-model", value_name = "NAME")]
    local_model: Vec<String>,
    /// Usage log file (JSON lines); default `$XDG_STATE_HOME/claude-mr/usage.jsonl`.
    #[arg(long, value_name = "PATH")]
    usage_log: Option<PathBuf>,
    /// Base URL of the Anthropic API.
    #[arg(
        long,
        hide = true,
        env = "CLAUDE_ROUTER_ANTHROPIC_URL",
        default_value = "https://api.anthropic.com"
    )]
    anthropic_url: String,
    /// Print diagnostic lines: why Google models are not served or cannot answer, and
    /// streams the client closed early.
    #[arg(short, long)]
    verbose: bool,
    /// Model id served by Google's Gemini (repeatable); default the four `gemini-4-neon-*`
    /// levels. Google models are served when the Antigravity CLI `agy` is found.
    #[arg(long = "google-model", value_name = "NAME")]
    google_model: Vec<String>,
    /// JSON file with the Antigravity login (the keyring entry's shape); default the keyring.
    #[arg(long, value_name = "FILE")]
    google_credentials: Option<PathBuf>,
    /// The Antigravity CLI binary; default the first `agy` on PATH, then `~/.local/bin/agy`.
    /// Its OAuth client id, secret and User-Agent are read from it.
    #[arg(long, value_name = "FILE", env = "CLAUDE_MR_AGY")]
    agy: Option<PathBuf>,
    /// JSON file whose `client_id`, `client_secret`, `user_agent` win over the values read
    /// from `agy`; default `$XDG_CONFIG_HOME/claude-mr/google.json` when it exists.
    #[arg(long, value_name = "FILE")]
    google_client_config: Option<PathBuf>,
    /// Serve no model with Google and skip looking for the Antigravity CLI.
    #[arg(long)]
    no_google: bool,
    /// Base URL of Google's Cloud Code Assist API.
    #[arg(
        long,
        hide = true,
        env = "CLAUDE_ROUTER_GOOGLE_CODE_ASSIST_URL",
        default_value = "https://daily-cloudcode-pa.googleapis.com"
    )]
    google_code_assist_url: String,
    /// Google's OAuth token endpoint.
    #[arg(
        long,
        hide = true,
        env = "CLAUDE_ROUTER_GOOGLE_TOKEN_URL",
        default_value = "https://oauth2.googleapis.com/token"
    )]
    google_token_url: String,
}

fn parse_uri(s: &str) -> Result<Uri, String> {
    s.parse::<Uri>().map_err(|e| format!("bad URL {s}: {e}"))
}

/// Builds the routing table: Anthropic as the default, one provider per further backend.
/// Adding a provider is a few lines here.
fn build_routes(
    cli: &Cli,
    client: &UpstreamClient,
    usage: &Arc<UsageLog>,
) -> Result<Routes, String> {
    let anthropic = Passthrough::new(
        "anthropic",
        parse_uri(&cli.anthropic_url)?,
        PassAuth::Forward,
        client.clone(),
    )?;
    let mut routes = Routes::new(Arc::new(anthropic));

    let local = Arc::new(Passthrough::new(
        "local",
        parse_uri(&cli.local)?,
        PassAuth::LocalKey,
        client.clone(),
    )?);
    let mut models = cli.local_model.clone();
    if models.is_empty() {
        models.push(DEFAULT_MODEL.to_owned());
    }
    for model in models {
        routes.add(model, local.clone());
    }

    if let Some(google_client) = google_client(cli) {
        let google = Arc::new(GoogleProvider::new(
            GoogleConfig {
                credentials: cli
                    .google_credentials
                    .clone()
                    .map_or_else(CredentialSource::antigravity, CredentialSource::File),
                endpoints: GoogleEndpoints {
                    code_assist: parse_uri(&cli.google_code_assist_url)?,
                    token: parse_uri(&cli.google_token_url)?,
                },
                ..GoogleConfig::new(google_client)
            },
            client.clone(),
            Arc::clone(usage),
        ));
        let mut models = cli.google_model.clone();
        if models.is_empty() {
            models.extend(GOOGLE_MODELS.map(str::to_owned));
        }
        for model in models {
            routes.add(model, google.clone());
        }
    }
    Ok(routes)
}

/// The Google client values, or why they are missing; `None` when Google is not served: with
/// `--no-google`, or when the Antigravity CLI is not installed. Problems are printed with
/// `--verbose`; a request to a Google model answers with them too.
fn google_client(cli: &Cli) -> Option<Result<client::GoogleClient, String>> {
    if cli.no_google {
        return None;
    }
    let agy = match client::find_agy(cli.agy.as_deref()) {
        Ok(p) => p,
        Err(searched) => {
            if verbose() {
                let at: Vec<String> = searched.iter().map(|p| p.display().to_string()).collect();
                eprintln!(
                    "router: Google models are not served: the Antigravity CLI `agy` was not \
                     found at {}",
                    at.join(", ")
                );
            }
            return None;
        }
    };
    let resolved = client::resolve(cli.google_client_config.as_deref(), &agy);
    if verbose() {
        match &resolved {
            Ok(c) => {
                let (ids, secrets) = c.candidates.counts();
                eprintln!(
                    "router: Google OAuth client from {} ({ids} client ids, {secrets} \
                     secrets); User-Agent {}",
                    c.candidates.origin(),
                    c.user_agent
                );
            }
            Err(why) => eprintln!("router: {why}"),
        }
    }
    Some(resolved)
}

/// Opens the usage log; a log that cannot be opened is replaced by a disabled one.
fn open_usage(cli: &Cli) -> UsageLog {
    let Some(path) = cli.usage_log.clone().or_else(UsageLog::default_path) else {
        return UsageLog::disabled();
    };
    UsageLog::open(&path).unwrap_or_else(|e| {
        eprintln!("router: usage log {} not opened: {e}", path.display());
        UsageLog::disabled()
    })
}

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    set_verbose(cli.verbose);
    let usage = Arc::new(open_usage(&cli));
    let routes = match build_routes(&cli, &upstream_client(), &usage) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("router: {e}");
            return ExitCode::from(2);
        }
    };
    let listener = match TcpListener::bind(("127.0.0.1", cli.port)).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("router: cannot bind 127.0.0.1:{}: {e}", cli.port);
            return ExitCode::FAILURE;
        }
    };
    let rest = cli
        .anthropic_url
        .strip_prefix("https://")
        .unwrap_or(&cli.anthropic_url);
    let listed: Vec<String> = routes
        .models()
        .map(|(m, p)| format!("{m} -> {p}"))
        .collect();
    let port = listener.local_addr().map_or(cli.port, |a| a.port());
    eprintln!(
        "router: 127.0.0.1:{}; {}; rest -> {}",
        port,
        listed.join(", "),
        rest
    );
    match serve(listener, routes).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("router: {e}");
            ExitCode::FAILURE
        }
    }
}
