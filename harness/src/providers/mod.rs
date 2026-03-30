pub mod types;

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};

use types::{ChatRequest, ChatResponse, StreamEvent};

/// Unique provider identifier (e.g. "orome-local", "anthropic", "openai-codex").
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProviderId(pub String);

/// Unique model identifier within a provider (e.g. "qwen3.5-35b-a3b", "claude-opus-4-6").
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelId(pub String);

/// What a provider declares about itself.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderCapabilities {
    pub provider_id: ProviderId,
    pub models: Vec<ModelCapabilities>,
}

/// What a provider declares about a specific model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCapabilities {
    pub model_id: ModelId,
    pub context_length: u64,
    pub max_output_tokens: u64,
    pub supports_tools: bool,
    pub supports_streaming: bool,
    pub supports_thinking: bool,
    /// Cost per million input tokens (0.0 for local).
    pub cost_per_m_input: f64,
    /// Cost per million output tokens (0.0 for local).
    pub cost_per_m_output: f64,
    /// Weight memory in bytes (0 for cloud).
    pub weight_memory: u64,
    /// Estimated per-request overhead in bytes (KV cache + buffers).
    pub per_request_memory: u64,
    /// Estimated tokens/sec for local models (0.0 for cloud).
    pub tokens_per_sec: f64,
}

/// The core provider abstraction. All backends implement this.
/// Normalized interface — the harness never sees provider-specific types.
#[async_trait]
pub trait Provider: Send + Sync {
    fn capabilities(&self) -> &ProviderCapabilities;

    /// Non-streaming chat completion.
    async fn chat(
        &self,
        model: &ModelId,
        request: ChatRequest,
    ) -> Result<ChatResponse, ProviderError>;

    /// Streaming chat completion.
    async fn chat_stream(
        &self,
        model: &ModelId,
        request: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = StreamEvent> + Send>>, ProviderError>;

    /// Load a model into memory. No-op for cloud providers.
    async fn load_model(&self, model: &ModelId) -> Result<(), ProviderError>;

    /// Unload a model from memory. No-op for cloud providers.
    async fn unload_model(&self, model: &ModelId) -> Result<(), ProviderError>;

    /// Check if the provider is healthy and reachable.
    async fn health_check(&self) -> Result<(), ProviderError>;
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
}
