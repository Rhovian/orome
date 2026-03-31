//! Claude CLI provider adapter.
//!
//! Wraps the `claude` CLI in `--print --output-format json` mode.
//! The CLI handles its own multi-turn tool loop; we see the final result.

use async_trait::async_trait;
use tokio::process::Command;

use super::{CliDispatchConfig, CliDispatchResult, CliProvider, CliUsage, ProviderError};

pub struct ClaudeProvider {
    /// Path or name of the claude CLI binary.
    command: String,
}

impl ClaudeProvider {
    pub fn new(command: String) -> Self {
        Self { command }
    }

    fn build_command(&self, config: &CliDispatchConfig) -> Command {
        let mut cmd = Command::new(&self.command);

        // Non-interactive, structured JSON output, no session persistence.
        cmd.arg("--print")
            .arg("--output-format")
            .arg("json")
            .arg("--no-session-persistence");

        if let Some(model) = &config.model {
            cmd.arg("--model").arg(model);
        }

        if let Some(schema) = &config.output_schema {
            // --json-schema takes inline JSON.
            cmd.arg("--json-schema").arg(schema.to_string());
        }

        if let Some(budget) = config.max_budget_usd {
            cmd.arg("--max-budget-usd").arg(budget.to_string());
        }

        cmd
    }
}

#[async_trait]
impl CliProvider for ClaudeProvider {
    async fn dispatch(
        &self,
        prompt: &str,
        config: CliDispatchConfig,
    ) -> Result<CliDispatchResult, ProviderError> {
        let mut cmd = self.build_command(&config);

        // Pass prompt as the positional argument.
        // For very long prompts, stdin would be safer, but claude -p
        // accepts the prompt as an argument.
        cmd.arg(prompt);

        let start = std::time::Instant::now();
        let output = cmd.output().await.map_err(|e| ProviderError::Cli {
            message: format!("failed to spawn claude: {e}"),
            exit_code: None,
        })?;
        let duration_ms = start.elapsed().as_millis() as u64;

        let stdout = String::from_utf8_lossy(&output.stdout);

        // claude --print --output-format json returns a single JSON object:
        // {
        //   "type": "result",
        //   "subtype": "success" | "error",
        //   "is_error": false,
        //   "result": "...",
        //   "total_cost_usd": 0.032,
        //   "duration_ms": 2774,
        //   "usage": { "input_tokens": N, "output_tokens": N },
        //   "session_id": "uuid",
        //   "structured_output": { ... }  // if --json-schema was used
        // }
        let parsed: serde_json::Value =
            serde_json::from_str(&stdout).map_err(|e| ProviderError::Cli {
                message: format!(
                    "failed to parse claude output: {e}\nraw: {}",
                    &stdout[..stdout.len().min(500)]
                ),
                exit_code: output.status.code(),
            })?;

        let is_error = parsed["is_error"].as_bool().unwrap_or(false);

        let result_text = parsed["result"]
            .as_str()
            .unwrap_or("")
            .to_string();

        let structured_output = parsed.get("structured_output").and_then(|v| {
            if v.is_null() {
                None
            } else {
                Some(v.clone())
            }
        });

        let usage = parsed.get("usage").and_then(|u| {
            Some(CliUsage {
                input_tokens: u["input_tokens"].as_u64()?,
                output_tokens: u["output_tokens"].as_u64()?,
            })
        });

        let cost_usd = parsed["total_cost_usd"].as_f64();

        let session_id = parsed["session_id"].as_str().map(|s| s.to_string());

        // Use duration from the CLI response if available, fall back to wall clock.
        let cli_duration_ms = parsed["duration_ms"].as_u64().unwrap_or(duration_ms);

        Ok(CliDispatchResult {
            output: result_text,
            structured_output,
            usage,
            duration_ms: cli_duration_ms,
            cost_usd,
            session_id,
            is_error,
        })
    }

    async fn health_check(&self) -> Result<(), ProviderError> {
        let output = Command::new(&self.command)
            .arg("--version")
            .output()
            .await
            .map_err(|e| ProviderError::Cli {
                message: format!("claude not found or not executable: {e}"),
                exit_code: None,
            })?;

        if !output.status.success() {
            return Err(ProviderError::Cli {
                message: "claude --version returned non-zero".into(),
                exit_code: output.status.code(),
            });
        }

        Ok(())
    }
}
