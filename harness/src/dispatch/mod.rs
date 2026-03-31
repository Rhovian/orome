//! Dispatcher — the unified layer between the task loop and providers.
//!
//! Takes a role payload and a provider assignment. Routes to either:
//! - CLI provider: serialize prompt, hand off, capture result + git diff
//! - Local inference: drive the tool loop (chat → tool calls → execute → repeat)
//!
//! Both paths produce a uniform `DispatchResult`.

use std::time::Instant;

use crate::plan::{ReviewerPayload, TiebreakPayload, WorkerPayload};
use crate::prompts::{reviewer, worker};
use crate::providers::claude::ClaudeProvider;
use crate::providers::codex::CodexProvider;
use crate::providers::orome::OromeProvider;
use crate::providers::types::{
    ChatMessage, ChatRequest, ChatResponse, ChatRole, FinishReason, ToolDefinition,
};
use crate::providers::{CliDispatchConfig, CliProvider, InferenceProvider, ModelId, ProviderError};
use crate::tools::ToolResult;

/// Uniform result from any dispatch, regardless of provider type.
#[derive(Debug, Clone)]
pub struct DispatchResult {
    /// The model's final text output.
    pub output: String,
    /// Git diff of filesystem changes (captured by harness before/after).
    pub diff: String,
    /// Token usage if available.
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    /// Wall-clock duration of the dispatch.
    pub duration_ms: u64,
    /// Cost in USD if reported by CLI provider.
    pub cost_usd: Option<f64>,
    /// Whether the dispatch ended in error.
    pub is_error: bool,
}

/// Which provider to dispatch to.
pub enum ProviderTarget {
    Claude {
        provider: ClaudeProvider,
        model: Option<String>,
    },
    Codex {
        provider: CodexProvider,
        model: Option<String>,
    },
    Local {
        provider: OromeProvider,
        model: ModelId,
    },
}

/// Tool executor trait — the dispatcher calls this when a local model
/// emits tool calls. The harness implements this with actual file I/O
/// and shell execution.
#[async_trait::async_trait]
pub trait ToolExecutor: Send + Sync {
    async fn execute(
        &self,
        tool_call_id: &str,
        name: &str,
        arguments: &str,
    ) -> ToolResult;
}

/// Dispatch a worker task.
pub async fn dispatch_worker(
    target: &ProviderTarget,
    payload: &WorkerPayload,
    tool_executor: Option<&dyn ToolExecutor>,
    tools: &[ToolDefinition],
) -> Result<DispatchResult, ProviderError> {
    let prompt = worker::build_worker_prompt(payload);

    match target {
        ProviderTarget::Claude { provider, model } => {
            dispatch_cli(provider, &prompt, model.as_deref()).await
        }
        ProviderTarget::Codex { provider, model } => {
            dispatch_cli(provider, &prompt, model.as_deref()).await
        }
        ProviderTarget::Local {
            provider,
            model,
        } => {
            let executor = tool_executor.ok_or_else(|| ProviderError::Internal {
                message: "local dispatch requires a tool executor".into(),
            })?;
            dispatch_local(provider, model, &prompt, tools, executor).await
        }
    }
}

/// Dispatch a review task.
pub async fn dispatch_review(
    target: &ProviderTarget,
    payload: &ReviewerPayload,
) -> Result<DispatchResult, ProviderError> {
    let prompt = reviewer::build_reviewer_prompt(payload);

    match target {
        ProviderTarget::Claude { provider, model } => {
            dispatch_cli(provider, &prompt, model.as_deref()).await
        }
        ProviderTarget::Codex { provider, model } => {
            dispatch_cli(provider, &prompt, model.as_deref()).await
        }
        ProviderTarget::Local { provider, model } => {
            // Reviewers don't use tools — single turn.
            let request = ChatRequest {
                messages: vec![
                    ChatMessage {
                        role: ChatRole::User,
                        content: Some(prompt),
                        tool_calls: None,
                        tool_call_id: None,
                    },
                ],
                max_tokens: 4096,
                temperature: 0.3,
                tools: vec![],
            };
            let start = Instant::now();
            let resp = provider.chat(model, request).await?;
            let duration_ms = start.elapsed().as_millis() as u64;

            Ok(DispatchResult {
                output: resp.content.unwrap_or_default(),
                diff: String::new(),
                input_tokens: Some(resp.usage.prompt_tokens),
                output_tokens: Some(resp.usage.completion_tokens),
                duration_ms,
                cost_usd: None,
                is_error: false,
            })
        }
    }
}

/// Dispatch a tiebreak request.
pub async fn dispatch_tiebreak(
    target: &ProviderTarget,
    payload: &TiebreakPayload,
) -> Result<DispatchResult, ProviderError> {
    let prompt = reviewer::build_tiebreak_prompt(payload);

    match target {
        ProviderTarget::Claude { provider, model } => {
            dispatch_cli(provider, &prompt, model.as_deref()).await
        }
        ProviderTarget::Codex { provider, model } => {
            dispatch_cli(provider, &prompt, model.as_deref()).await
        }
        ProviderTarget::Local { provider, model } => {
            let request = ChatRequest {
                messages: vec![
                    ChatMessage {
                        role: ChatRole::User,
                        content: Some(prompt),
                        tool_calls: None,
                        tool_call_id: None,
                    },
                ],
                max_tokens: 4096,
                temperature: 0.3,
                tools: vec![],
            };
            let start = Instant::now();
            let resp = provider.chat(model, request).await?;
            let duration_ms = start.elapsed().as_millis() as u64;

            Ok(DispatchResult {
                output: resp.content.unwrap_or_default(),
                diff: String::new(),
                input_tokens: Some(resp.usage.prompt_tokens),
                output_tokens: Some(resp.usage.completion_tokens),
                duration_ms,
                cost_usd: None,
                is_error: false,
            })
        }
    }
}

// ---------------------------------------------------------------------------
// CLI dispatch — hand off prompt to CLI, capture result
// ---------------------------------------------------------------------------

async fn dispatch_cli(
    provider: &(dyn CliProvider + Send + Sync),
    prompt: &str,
    model: Option<&str>,
) -> Result<DispatchResult, ProviderError> {
    let config = CliDispatchConfig {
        model: model.map(|s| s.to_string()),
        output_schema: None,
        max_budget_usd: None,
    };

    let result = provider.dispatch(prompt, config).await?;

    Ok(DispatchResult {
        output: result.output,
        diff: String::new(), // Caller captures git diff before/after.
        input_tokens: result.usage.as_ref().map(|u| u.input_tokens),
        output_tokens: result.usage.as_ref().map(|u| u.output_tokens),
        duration_ms: result.duration_ms,
        cost_usd: result.cost_usd,
        is_error: result.is_error,
    })
}

// ---------------------------------------------------------------------------
// Local dispatch — drive the agentic tool loop
// ---------------------------------------------------------------------------

/// Maximum turns in the tool loop before giving up.
const MAX_TOOL_LOOP_TURNS: usize = 50;

async fn dispatch_local(
    provider: &OromeProvider,
    model: &ModelId,
    initial_prompt: &str,
    tools: &[ToolDefinition],
    executor: &dyn ToolExecutor,
) -> Result<DispatchResult, ProviderError> {
    let start = Instant::now();

    let mut messages = vec![ChatMessage {
        role: ChatRole::User,
        content: Some(initial_prompt.to_string()),
        tool_calls: None,
        tool_call_id: None,
    }];

    let mut total_input_tokens: u64 = 0;
    let mut total_output_tokens: u64 = 0;
    let mut final_output = String::new();

    for _turn in 0..MAX_TOOL_LOOP_TURNS {
        let request = ChatRequest {
            messages: messages.clone(),
            max_tokens: 8192,
            temperature: 0.6,
            tools: tools.to_vec(),
        };

        let resp: ChatResponse = provider.chat(model, request).await?;

        total_input_tokens += resp.usage.prompt_tokens;
        total_output_tokens += resp.usage.completion_tokens;

        // Accumulate any text content.
        if let Some(content) = &resp.content {
            final_output.push_str(content);
        }

        // If no tool calls, we're done.
        if resp.tool_calls.is_empty() || resp.finish_reason != FinishReason::ToolCalls {
            break;
        }

        // Append the assistant's response (with tool calls) to the conversation.
        messages.push(ChatMessage {
            role: ChatRole::Assistant,
            content: resp.content.clone(),
            tool_calls: Some(resp.tool_calls.clone()),
            tool_call_id: None,
        });

        // Execute each tool call and append the results.
        for tc in &resp.tool_calls {
            let tool_result = executor.execute(&tc.id, &tc.name, &tc.arguments).await;
            messages.push(ChatMessage {
                role: ChatRole::Tool,
                content: Some(tool_result.output),
                tool_calls: None,
                tool_call_id: Some(tc.id.clone()),
            });
        }
    }

    let duration_ms = start.elapsed().as_millis() as u64;

    Ok(DispatchResult {
        output: final_output,
        diff: String::new(), // Caller captures git diff.
        input_tokens: Some(total_input_tokens),
        output_tokens: Some(total_output_tokens),
        duration_ms,
        cost_usd: None,
        is_error: false,
    })
}
