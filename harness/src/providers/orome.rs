//! Orome local inference provider.
//!
//! Talks to the orome inference server over HTTP/SSE. The server exposes
//! an OpenAI-compatible API at `/v1/chat/completions` with an `x_orome`
//! extension for timing data (prefill_ms, decode_ms, tokens_per_sec).
//!
//! The harness drives the tool loop: it calls chat(), inspects for tool
//! calls, executes tools, feeds results back, and loops until done.
//! This gives full visibility into every turn.

use std::pin::Pin;

use async_trait::async_trait;
use futures::stream::Stream;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use super::types::{
    ChatMessage, ChatRequest, ChatResponse, ChatRole, FinishReason, StreamEvent, Timing,
    ToolCall, ToolCallDelta, ToolDefinition, Usage,
};
use super::{InferenceProvider, ModelId, ProviderError};

pub struct OromeProvider {
    client: Client,
    base_url: String,
}

impl OromeProvider {
    pub fn new(base_url: String) -> Self {
        Self {
            client: Client::new(),
            base_url,
        }
    }
}

#[async_trait]
impl InferenceProvider for OromeProvider {
    async fn chat(
        &self,
        _model: &ModelId,
        request: ChatRequest,
    ) -> Result<ChatResponse, ProviderError> {
        let wire_req = to_wire_request(&request);

        let resp = self
            .client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .json(&wire_req)
            .send()
            .await
            .map_err(|e| ProviderError::Network(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(ProviderError::Internal {
                message: format!("server returned {status}: {body}"),
            });
        }

        let wire_resp: WireResponse =
            resp.json().await.map_err(|e| ProviderError::Internal {
                message: format!("failed to parse response: {e}"),
            })?;

        Ok(from_wire_response(wire_resp))
    }

    async fn chat_stream(
        &self,
        _model: &ModelId,
        request: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = StreamEvent> + Send>>, ProviderError> {
        let mut wire_req = to_wire_request(&request);
        wire_req.stream = true;

        let resp = self
            .client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .json(&wire_req)
            .send()
            .await
            .map_err(|e| ProviderError::Network(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(ProviderError::Internal {
                message: format!("server returned {status}: {body}"),
            });
        }

        let byte_stream = resp.bytes_stream();

        let event_stream = futures::stream::unfold(
            SseState {
                inner: Box::pin(byte_stream),
                buffer: String::new(),
            },
            |mut state| async move {
                loop {
                    // Try to extract a complete SSE event from the buffer.
                    if let Some(event) = state.try_parse_event() {
                        return Some((event, state));
                    }

                    // Read more data from the byte stream.
                    match state.inner.next().await {
                        Some(Ok(bytes)) => {
                            state
                                .buffer
                                .push_str(&String::from_utf8_lossy(&bytes));
                        }
                        Some(Err(e)) => {
                            return Some((
                                StreamEvent::Error {
                                    message: e.to_string(),
                                },
                                state,
                            ));
                        }
                        None => return None, // Stream ended.
                    }
                }
            },
        );

        Ok(Box::pin(event_stream))
    }

    async fn load_model(&self, model: &ModelId) -> Result<(), ProviderError> {
        // The server loads one model at startup. We can only verify
        // that the requested model matches what's loaded.
        let health = self.get_health().await?;
        if health.model != model.0 {
            return Err(ProviderError::ModelNotLoaded(format!(
                "requested '{}' but server has '{}' loaded",
                model.0, health.model
            )));
        }
        Ok(())
    }

    async fn unload_model(&self, _model: &ModelId) -> Result<(), ProviderError> {
        // Server doesn't support unloading via HTTP.
        Ok(())
    }

    async fn health_check(&self) -> Result<(), ProviderError> {
        self.get_health().await?;
        Ok(())
    }
}

impl OromeProvider {
    async fn get_health(&self) -> Result<HealthResponse, ProviderError> {
        let resp = self
            .client
            .get(format!("{}/health", self.base_url))
            .send()
            .await
            .map_err(|e| ProviderError::Network(e.to_string()))?;

        if !resp.status().is_success() {
            return Err(ProviderError::Internal {
                message: "health check failed".into(),
            });
        }

        resp.json::<HealthResponse>()
            .await
            .map_err(|e| ProviderError::Internal {
                message: format!("failed to parse health response: {e}"),
            })
    }
}

// ---------------------------------------------------------------------------
// Wire format types — OpenAI-compatible JSON on the wire.
// These are private to this module; the public interface uses harness types.
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct WireRequest {
    messages: Vec<WireMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<WireTool>,
    max_tokens: u64,
    temperature: f32,
    stream: bool,
}

#[derive(Serialize)]
struct WireMessage {
    role: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<WireToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Serialize)]
struct WireTool {
    r#type: &'static str,
    function: WireToolFunction,
}

#[derive(Serialize)]
struct WireToolFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Serialize, Deserialize)]
struct WireToolCall {
    id: String,
    r#type: String,
    function: WireToolCallFunction,
}

#[derive(Serialize, Deserialize)]
struct WireToolCallFunction {
    name: String,
    arguments: String,
}

#[derive(Deserialize)]
struct WireResponse {
    choices: Vec<WireChoice>,
    #[serde(default)]
    usage: Option<WireUsage>,
    #[serde(default)]
    x_orome: Option<WireTiming>,
}

#[derive(Deserialize)]
struct WireChoice {
    message: WireResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct WireResponseMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<WireToolCall>>,
}

#[derive(Deserialize)]
struct WireUsage {
    prompt_tokens: u64,
    completion_tokens: u64,
}

#[derive(Deserialize)]
struct WireTiming {
    prefill_ms: f64,
    decode_ms: f64,
    tokens_per_sec: f64,
}

#[derive(Deserialize)]
struct WireStreamChunk {
    choices: Vec<WireStreamChoice>,
    #[serde(default)]
    usage: Option<WireUsage>,
    #[serde(default)]
    x_orome: Option<WireTiming>,
}

#[derive(Deserialize)]
struct WireStreamChoice {
    delta: WireStreamDelta,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct WireStreamDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<WireStreamToolCall>>,
}

#[derive(Deserialize)]
struct WireStreamToolCall {
    #[serde(default)]
    index: u32,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<WireStreamToolCallFunction>,
}

#[derive(Deserialize)]
struct WireStreamToolCallFunction {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Deserialize)]
struct HealthResponse {
    model: String,
}

// ---------------------------------------------------------------------------
// Translation: harness types <-> wire types
// ---------------------------------------------------------------------------

fn to_wire_request(req: &ChatRequest) -> WireRequest {
    WireRequest {
        messages: req.messages.iter().map(to_wire_message).collect(),
        tools: req.tools.iter().map(to_wire_tool).collect(),
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        stream: false,
    }
}

fn to_wire_message(msg: &ChatMessage) -> WireMessage {
    WireMessage {
        role: match msg.role {
            ChatRole::System => "system",
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
            ChatRole::Tool => "tool",
        },
        content: msg.content.clone(),
        tool_calls: msg.tool_calls.as_ref().map(|calls| {
            calls
                .iter()
                .map(|tc| WireToolCall {
                    id: tc.id.clone(),
                    r#type: "function".into(),
                    function: WireToolCallFunction {
                        name: tc.name.clone(),
                        arguments: tc.arguments.clone(),
                    },
                })
                .collect()
        }),
        tool_call_id: msg.tool_call_id.clone(),
    }
}

fn to_wire_tool(tool: &ToolDefinition) -> WireTool {
    WireTool {
        r#type: "function",
        function: WireToolFunction {
            name: tool.name.clone(),
            description: tool.description.clone(),
            parameters: tool.parameters.clone(),
        },
    }
}

fn from_wire_response(resp: WireResponse) -> ChatResponse {
    let choice = &resp.choices[0];

    let tool_calls = choice
        .message
        .tool_calls
        .as_ref()
        .map(|calls| {
            calls
                .iter()
                .map(|tc| ToolCall {
                    id: tc.id.clone(),
                    name: tc.function.name.clone(),
                    arguments: tc.function.arguments.clone(),
                })
                .collect()
        })
        .unwrap_or_default();

    let finish_reason = match choice.finish_reason.as_deref() {
        Some("tool_calls") => FinishReason::ToolCalls,
        Some("length") => FinishReason::Length,
        _ => FinishReason::Stop,
    };

    ChatResponse {
        content: choice.message.content.clone(),
        tool_calls,
        finish_reason,
        usage: resp
            .usage
            .map(|u| Usage {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
            })
            .unwrap_or(Usage {
                prompt_tokens: 0,
                completion_tokens: 0,
            }),
        timing: resp.x_orome.map(|t| Timing {
            prefill_ms: t.prefill_ms,
            decode_ms: t.decode_ms,
            tokens_per_sec: t.tokens_per_sec,
        }),
    }
}

fn from_wire_stream_chunk(chunk: WireStreamChunk) -> StreamEvent {
    let choice = &chunk.choices[0];

    // Final chunk: has finish_reason set.
    if let Some(reason) = &choice.finish_reason {
        let finish_reason = match reason.as_str() {
            "tool_calls" => FinishReason::ToolCalls,
            "length" => FinishReason::Length,
            _ => FinishReason::Stop,
        };

        return StreamEvent::Done {
            finish_reason,
            usage: chunk
                .usage
                .map(|u| Usage {
                    prompt_tokens: u.prompt_tokens,
                    completion_tokens: u.completion_tokens,
                })
                .unwrap_or(Usage {
                    prompt_tokens: 0,
                    completion_tokens: 0,
                }),
            timing: chunk.x_orome.map(|t| Timing {
                prefill_ms: t.prefill_ms,
                decode_ms: t.decode_ms,
                tokens_per_sec: t.tokens_per_sec,
            }),
        };
    }

    // Delta chunk: content or tool call deltas.
    StreamEvent::Delta {
        content: choice.delta.content.clone(),
        tool_calls: choice.delta.tool_calls.as_ref().map(|calls| {
            calls
                .iter()
                .map(|tc| {
                    let func = tc.function.as_ref();
                    ToolCallDelta {
                        index: tc.index,
                        id: tc.id.clone(),
                        name: func.and_then(|f| f.name.clone()),
                        arguments: func.and_then(|f| f.arguments.clone()),
                    }
                })
                .collect()
        }),
    }
}

// ---------------------------------------------------------------------------
// SSE parser — processes the raw byte stream into StreamEvents.
// ---------------------------------------------------------------------------

struct SseState {
    inner: Pin<Box<dyn Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send>>,
    buffer: String,
}

impl SseState {
    /// Try to extract and parse a complete SSE event from the buffer.
    /// Returns None if no complete event is available yet.
    fn try_parse_event(&mut self) -> Option<StreamEvent> {
        loop {
            // SSE events are separated by double newlines.
            let sep = self.buffer.find("\n\n")?;
            let event_str = self.buffer[..sep].to_string();
            self.buffer = self.buffer[sep + 2..].to_string();

            // Each line in the event starts with "data: ".
            let data = event_str.strip_prefix("data: ").unwrap_or(&event_str);

            // Terminal event.
            if data == "[DONE]" {
                continue; // Skip — the final data chunk before [DONE] has the Done event.
            }

            match serde_json::from_str::<WireStreamChunk>(data) {
                Ok(chunk) => return Some(from_wire_stream_chunk(chunk)),
                Err(e) => {
                    return Some(StreamEvent::Error {
                        message: format!("failed to parse SSE chunk: {e}"),
                    });
                }
            }
        }
    }
}
