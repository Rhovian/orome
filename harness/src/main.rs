use std::path::PathBuf;

use clap::{Parser, Subcommand};

use orome_harness::cli;
use orome_harness::config::HarnessConfig;
use orome_harness::dispatch::ProviderTarget;
use orome_harness::providers::claude::ClaudeProvider;
use orome_harness::providers::codex::CodexProvider;
use orome_harness::providers::orome::OromeProvider;
use orome_harness::providers::ModelId;

#[derive(Parser)]
#[command(name = "orome", about = "Orome coding harness")]
struct Cli {
    /// Path to orome.yaml config file.
    #[arg(long, default_value = "orome.yaml")]
    config: PathBuf,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Interactive planning session — decompose work into tasks.
    Plan {
        /// Use Claude CLI for planning.
        #[arg(long, group = "planner")]
        claude: bool,

        /// Use Codex CLI for planning.
        #[arg(long, group = "planner")]
        codex: bool,

        /// Model to use (e.g. "opus", "sonnet").
        #[arg(long)]
        model: Option<String>,
    },

    /// Execute the plan at .orome/plan.yaml.
    Run {
        /// Use Claude CLI for workers and reviewers.
        #[arg(long, group = "provider")]
        claude: bool,

        /// Use Codex CLI for workers and reviewers.
        #[arg(long, group = "provider")]
        codex: bool,

        /// Use orome local inference for workers and reviewers.
        #[arg(long, group = "provider")]
        local: bool,

        /// Model to use (e.g. "opus", "sonnet", "qwen3.5-27b").
        #[arg(long)]
        model: Option<String>,

        /// Path to plan file.
        #[arg(long, default_value = ".orome/plan.yaml")]
        plan: PathBuf,

        /// Validate plan without executing.
        #[arg(long)]
        dry_run: bool,

        /// Maximum retries per task on reviewer rejection.
        #[arg(long, default_value = "2")]
        max_retries: u32,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Load config. Fall back to defaults if file doesn't exist.
    let config = load_config(&cli.config)?;

    let project_root = std::env::current_dir()?;

    match cli.command {
        Command::Plan {
            claude,
            codex,
            model,
        } => {
            let provider = if claude {
                "claude"
            } else if codex {
                "codex"
            } else {
                // Default to claude if neither specified.
                "claude"
            };

            cli::plan::run_plan(&config, provider, model.as_deref(), &project_root).await?;
        }

        Command::Run {
            claude,
            codex,
            local,
            model,
            plan,
            dry_run,
            max_retries,
        } => {
            let target = build_provider_target(claude, codex, local, model.as_deref(), &config)?;

            // Use same provider for worker and reviewer in v1.
            // In the future, these could be different.
            cli::run::run_plan(&plan, &target, &target, max_retries, dry_run).await?;
        }
    }

    Ok(())
}

fn load_config(path: &std::path::Path) -> anyhow::Result<HarnessConfig> {
    if path.exists() {
        let contents = std::fs::read_to_string(path)?;
        let config: HarnessConfig = serde_yaml::from_str(&contents)?;
        Ok(config)
    } else {
        // Minimal default config.
        Ok(HarnessConfig {
            providers: Default::default(),
            defaults: orome_harness::config::Defaults {
                scratchpad_dir: PathBuf::from(".orome/scratchpad"),
                trace_dir: PathBuf::from(".orome/traces"),
                plan_dir: PathBuf::from(".orome"),
                review: orome_harness::config::ReviewDefaults {
                    reviewers_per_task: 1,
                    max_retries: 2,
                },
            },
        })
    }
}

fn build_provider_target(
    claude: bool,
    codex: bool,
    local: bool,
    model: Option<&str>,
    config: &HarnessConfig,
) -> anyhow::Result<ProviderTarget> {
    if claude {
        Ok(ProviderTarget::Claude {
            provider: ClaudeProvider::new("claude".into()),
            model: model.map(String::from),
        })
    } else if codex {
        Ok(ProviderTarget::Codex {
            provider: CodexProvider::new("codex".into()),
            model: model.map(String::from),
        })
    } else if local {
        // Find the local provider's server URL from config.
        let server_url = config
            .providers
            .iter()
            .find_map(|(_, p)| match p {
                orome_harness::config::ProviderConfig::Local { server_url, .. } => {
                    Some(server_url.clone())
                }
                _ => None,
            })
            .unwrap_or_else(|| "http://localhost:8080".into());

        let model_id = model.unwrap_or("qwen3.5-27b");

        Ok(ProviderTarget::Local {
            provider: OromeProvider::new(server_url),
            model: ModelId(model_id.into()),
        })
    } else {
        // Default to claude.
        Ok(ProviderTarget::Claude {
            provider: ClaudeProvider::new("claude".into()),
            model: model.map(String::from),
        })
    }
}
