//! Codex CLI provider adapter.
//!
//! Wraps the `codex` CLI in `codex exec --json` mode.
//! The CLI handles its own multi-turn tool loop; we parse the JSONL output.

use async_trait::async_trait;
use tokio::process::Command;

use super::{CliDispatchConfig, CliDispatchResult, CliProvider, CliUsage, ProviderError};

pub struct CodexProvider {
    /// Path or name of the codex CLI binary.
    command: String,
}

impl CodexProvider {
    pub fn new(command: String) -> Self {
        Self { command }
    }
}

#[async_trait]
impl CliProvider for CodexProvider {
    async fn dispatch(
        &self,
        prompt: &str,
        config: CliDispatchConfig,
    ) -> Result<CliDispatchResult, ProviderError> {
        let mut cmd = Command::new(&self.command);

        cmd.arg("exec").arg("--json");

        if let Some(model) = &config.model {
            cmd.arg("--model").arg(model);
        }

        // codex --output-schema takes a file path, not inline JSON.
        // Write a temp file if a schema is provided.
        let _schema_tempfile = if let Some(schema) = &config.output_schema {
            let tmp = tempfile::NamedTempFile::new().map_err(|e| ProviderError::Cli {
                message: format!("failed to create temp schema file: {e}"),
                exit_code: None,
            })?;
            std::fs::write(tmp.path(), schema.to_string()).map_err(|e| ProviderError::Cli {
                message: format!("failed to write temp schema file: {e}"),
                exit_code: None,
            })?;
            cmd.arg("--output-schema").arg(tmp.path());
            Some(tmp) // keep alive until command completes
        } else {
            None
        };

        // Prompt as positional argument.
        cmd.arg(prompt);

        let start = std::time::Instant::now();
        let output = cmd.output().await.map_err(|e| ProviderError::Cli {
            message: format!("failed to spawn codex: {e}"),
            exit_code: None,
        })?;
        let duration_ms = start.elapsed().as_millis() as u64;

        let stdout = String::from_utf8_lossy(&output.stdout);

        // codex exec --json returns JSONL — one JSON object per line.
        // Key event types:
        //   {"type":"item.completed","item":{"type":"agent_message","text":"..."}}
        //   {"type":"turn.completed","usage":{"input_tokens":N,"output_tokens":N}}
        let mut result_text = String::new();
        let mut usage: Option<CliUsage> = None;
        let is_error = !output.status.success();

        for line in stdout.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let Ok(event) = serde_json::from_str::<serde_json::Value>(line) else {
                continue;
            };

            match event["type"].as_str() {
                Some("item.completed") => {
                    if let Some(item) = event.get("item") {
                        if item["type"].as_str() == Some("agent_message") {
                            if let Some(text) = item["text"].as_str() {
                                if !result_text.is_empty() {
                                    result_text.push('\n');
                                }
                                result_text.push_str(text);
                            }
                        }
                    }
                }
                Some("turn.completed") => {
                    if let Some(u) = event.get("usage") {
                        usage = Some(CliUsage {
                            input_tokens: u["input_tokens"].as_u64().unwrap_or(0),
                            output_tokens: u["output_tokens"].as_u64().unwrap_or(0),
                        });
                    }
                }
                _ => {}
            }
        }

        // Codex doesn't report cost or session ID in the same way as Claude.
        Ok(CliDispatchResult {
            output: result_text,
            structured_output: None, // TODO: parse from last agent_message if schema was used
            usage,
            duration_ms,
            cost_usd: None,
            session_id: None,
            is_error,
        })
    }

    async fn health_check(&self) -> Result<(), ProviderError> {
        let output = Command::new(&self.command)
            .arg("--version")
            .output()
            .await
            .map_err(|e| ProviderError::Cli {
                message: format!("codex not found or not executable: {e}"),
                exit_code: None,
            })?;

        if !output.status.success() {
            return Err(ProviderError::Cli {
                message: "codex --version returned non-zero".into(),
                exit_code: output.status.code(),
            });
        }

        Ok(())
    }
}
