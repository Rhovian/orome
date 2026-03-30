use serde::{Deserialize, Serialize};

use crate::providers::{ModelId, ProviderId};
use crate::roles::Role;

/// Top-level harness configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessConfig {
    pub local: LocalConfig,
    pub providers: Vec<ProviderConfig>,
    pub role_bindings: Vec<RoleBinding>,
    pub scratchpad_dir: String,
}

/// Local inference capacity\ configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalConfig {
    /// Total unified memory in bytes.
    pub total_memory: u64,
    /// OS headroom to reserve, in bytes.
    pub os_headroom: u64,
    /// Orome inference server base URL.
    pub server_url: String,
}

/// Configuration for a single provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub provider_id: ProviderId,
    pub kind: ProviderKind,
    pub api_key_env: Option<String>,
    pub base_url: Option<String>,
    pub max_concurrent: u32,
    pub tokens_per_minute: u64,
    pub cost_budget: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    OromeLocal,
    Anthropic,
    OpenAi,
}

/// Default role → model binding policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleBinding {
    pub role: Role,
    /// Ordered preference list — scheduler tries first, falls back.
    pub candidates: Vec<BindingCandidate>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindingCandidate {
    pub provider_id: ProviderId,
    pub model_id: ModelId,
}
