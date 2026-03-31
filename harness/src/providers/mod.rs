pub mod claude;
pub mod codex;
pub mod orome;
pub mod types;

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};

use types::{ChatRequest, ChatResponse, StreamEvent};

/// Unique provider identifier (e.g. "orome-local", "claude", "codex").
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProviderId(pub String);

/// Unique model identifier within a provider (e.g. "qwen3.5-35b-a3b").
/// For CLI providers, this may be an alias like "opus" or "sonnet".
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelId(pub String);

// ---------------------------------------------------------------------------
// Two provider interfaces for two fundamentally different relationships:
//
// InferenceProvider (orome-local):
//   The harness drives the tool loop. Calls chat() per turn, intercepts
//   tool calls, executes them, feeds results back. Full visibility into
//   every turn, tool call, token, and timing.
//
// CliProvider (claude, codex):
//   The harness hands off a complete task. The CLI runs autonomously with
//   its own tool loop. The harness sees input and output, not the middle.
//   Telemetry comes from the CLI's structured output (JSON/JSONL).
// ---------------------------------------------------------------------------

/// Turn-level inference — the harness drives the tool loop.
///
/// Used for orome-local where we have full control: the harness constructs
/// each message, sends it, inspects the response for tool calls, executes
/// tools, appends results, and loops until done.
///
/// This gives us full instrumentation: per-turn token counts, tool call
/// latency, prefill/decode timing, KV cache state — everything.
#[async_trait]
pub trait InferenceProvider: Send + Sync {
    /// Single inference call. The harness manages the conversation loop.
    async fn chat(
        &self,
        model: &ModelId,
        request: ChatRequest,
    ) -> Result<ChatResponse, ProviderError>;

    /// Streaming inference. Same as chat() but token-by-token.
    async fn chat_stream(
        &self,
        model: &ModelId,
        request: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = StreamEvent> + Send>>, ProviderError>;

    /// Load model weights into memory. Blocks until ready.
    async fn load_model(&self, model: &ModelId) -> Result<(), ProviderError>;

    /// Unload model weights from memory.
    async fn unload_model(&self, model: &ModelId) -> Result<(), ProviderError>;

    /// Check if the inference server is up and healthy.
    async fn health_check(&self) -> Result<(), ProviderError>;
}

/// Task-level dispatch — the CLI drives its own tool loop.
///
/// Used for CLI providers (claude, codex) where we hand off a complete
/// task and get back a result. The CLI handles its own multi-turn
/// conversation, tool execution, and context management.
///
/// The harness captures:
/// - The CLI's structured output (usage, cost, duration)
/// - Git diff of what changed on disk
/// - The final result text
///
/// What the harness does NOT see:
/// - Individual turns within the CLI session
/// - Per-tool-call latency inside the CLI
/// - Internal context/prompt construction
#[async_trait]
pub trait CliProvider: Send + Sync {
    /// Dispatch a task to the CLI. The prompt is the fully assembled
    /// task description (objective, criteria, hints, retry context).
    /// The CLI runs autonomously and returns when done.
    async fn dispatch(
        &self,
        prompt: &str,
        config: CliDispatchConfig,
    ) -> Result<CliDispatchResult, ProviderError>;

    /// Check if the CLI is installed and functional.
    async fn health_check(&self) -> Result<(), ProviderError>;
}

/// Configuration for a CLI dispatch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliDispatchConfig {
    /// Model alias (e.g. "opus", "sonnet" for claude; model name for codex).
    pub model: Option<String>,
    /// JSON schema the CLI should conform its output to.
    /// Used to get structured TaskResult/TaskGrade back.
    pub output_schema: Option<serde_json::Value>,
    /// Maximum cost in USD for this dispatch (if the CLI supports it).
    pub max_budget_usd: Option<f64>,
}

/// What comes back from a CLI dispatch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliDispatchResult {
    /// The CLI's text output.
    pub output: String,
    /// Structured output if a schema was provided and the CLI honored it.
    pub structured_output: Option<serde_json::Value>,
    /// Token usage reported by the CLI.
    pub usage: Option<CliUsage>,
    /// Total wall-clock duration of the dispatch.
    pub duration_ms: u64,
    /// Cost reported by the CLI (if available).
    pub cost_usd: Option<f64>,
    /// CLI session ID for potential resumption.
    pub session_id: Option<String>,
    /// Whether the CLI reported an error.
    pub is_error: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("model not found: {0}")]
    ModelNotFound(String),
    #[error("model not loaded: {0}")]
    ModelNotLoaded(String),
    #[error("rate limited, retry after {retry_after_ms}ms")]
    RateLimited { retry_after_ms: u64 },
    #[error("context length exceeded: {used} > {limit}")]
    ContextLengthExceeded { used: u64, limit: u64 },
    #[error("provider error: {message}")]
    Internal { message: String },
    #[error("network error: {0}")]
    Network(String),
    #[error("cli error: {message}")]
    Cli { message: String, exit_code: Option<i32> },
}
