use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Top-level harness configuration, deserialized from `orome.yaml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessConfig {
    pub providers: HashMap<String, ProviderConfig>,
    pub defaults: Defaults,
}

/// Provider configuration. Two kinds:
/// - `local`: orome inference server, models hardcoded, full telemetry.
/// - `cli`: wraps an external CLI (claude, codex). Capabilities come from
///   reading the CLI source (pinned submodule), not from runtime probing.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ProviderConfig {
    Local {
        server_url: String,
        models: HashMap<String, LocalModelConfig>,
    },
    Cli {
        command: String,
    },
}

/// Hardcoded local model config. We built the server — we know these values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalModelConfig {
    pub weight_memory_gb: u64,
    pub tokens_per_sec: u64,
    pub architecture: Architecture,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Architecture {
    Dense,
    Moe,
}

/// Default settings, overridable per-plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Defaults {
    pub scratchpad_dir: PathBuf,
    pub trace_dir: PathBuf,
    pub plan_dir: PathBuf,
    pub review: ReviewDefaults,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewDefaults {
    pub reviewers_per_task: u32,
    pub max_retries: u32,
}
