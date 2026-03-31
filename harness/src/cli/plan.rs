//! `orome plan` — interactive planning session.
//!
//! Launches an interactive CLI session (claude or codex) with the planner
//! system prompt injected. The user iterates on the plan; the model writes
//! `.orome/plan.yaml`.

use std::path::Path;
use std::process::Stdio;

use crate::config::HarnessConfig;
use crate::prompts::planner::{
    build_planner_system_prompt, ModelEntry, ProjectContext, ProviderEntry, ProviderRoster,
};

/// Run the interactive planning session.
pub async fn run_plan(
    config: &HarnessConfig,
    provider: &str,
    model: Option<&str>,
    project_root: &Path,
) -> anyhow::Result<()> {
    let project_context = gather_project_context(project_root).await?;
    let roster = build_roster(config);
    let system_prompt = build_planner_system_prompt(&project_context, &roster);

    // Write the system prompt to a temp file for inspection/debugging.
    let plan_dir = project_root.join(
        config
            .defaults
            .plan_dir
            .to_str()
            .unwrap_or(".orome"),
    );
    std::fs::create_dir_all(&plan_dir)?;
    let context_file = plan_dir.join("planner-context.md");
    std::fs::write(&context_file, &system_prompt)?;

    eprintln!(
        "Planner context written to {}",
        context_file.display()
    );
    eprintln!("Launching interactive {} session...", provider);
    eprintln!(
        "The model will write the plan to {}/plan.yaml",
        plan_dir.display()
    );
    eprintln!();

    match provider {
        "claude" => launch_claude(&system_prompt, model).await,
        "codex" => launch_codex(&system_prompt, model).await,
        other => anyhow::bail!("unknown planner provider: {other}"),
    }
}

async fn launch_claude(system_prompt: &str, model: Option<&str>) -> anyhow::Result<()> {
    let mut cmd = tokio::process::Command::new("claude");

    // --append-system-prompt injects into the system prompt without
    // replacing Claude Code's built-in system prompt.
    cmd.arg("--append-system-prompt").arg(system_prompt);

    if let Some(m) = model {
        cmd.arg("--model").arg(m);
    }

    // Inherit stdio so the user gets a normal interactive session.
    cmd.stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());

    let status = cmd.status().await?;

    if !status.success() {
        anyhow::bail!("claude exited with status {}", status);
    }

    Ok(())
}

async fn launch_codex(system_prompt: &str, model: Option<&str>) -> anyhow::Result<()> {
    let mut cmd = tokio::process::Command::new("codex");

    // Codex uses --instructions for system prompt injection.
    cmd.arg("--instructions").arg(system_prompt);

    if let Some(m) = model {
        cmd.arg("--model").arg(m);
    }

    cmd.stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());

    let status = cmd.status().await?;

    if !status.success() {
        anyhow::bail!("codex exited with status {}", status);
    }

    Ok(())
}

async fn gather_project_context(root: &Path) -> anyhow::Result<ProjectContext> {
    // File tree (condensed).
    let tree_output = tokio::process::Command::new("find")
        .args([
            ".", "-type", "f",
            "-not", "-path", "./.git/*",
            "-not", "-path", "./.orome/*",
            "-not", "-path", "./target/*",
            "-not", "-path", "./.venv/*",
            "-not", "-name", "*.o",
            "-not", "-name", "*.dylib",
        ])
        .current_dir(root)
        .output()
        .await?;
    let tree_digest = String::from_utf8_lossy(&tree_output.stdout)
        .lines()
        .take(200) // Cap at 200 files to keep prompt manageable.
        .collect::<Vec<_>>()
        .join("\n");

    // Recent commits.
    let log_output = tokio::process::Command::new("git")
        .args(["log", "--oneline", "-15"])
        .current_dir(root)
        .output()
        .await?;
    let recent_commits = String::from_utf8_lossy(&log_output.stdout)
        .lines()
        .map(|s| s.to_string())
        .collect();

    Ok(ProjectContext {
        root: root.to_string_lossy().into_owned(),
        tree_digest,
        recent_commits,
    })
}

fn build_roster(config: &HarnessConfig) -> ProviderRoster {
    let mut entries = Vec::new();

    for (name, provider_config) in &config.providers {
        match provider_config {
            crate::config::ProviderConfig::Local { models, .. } => {
                let model_entries: Vec<ModelEntry> = models
                    .iter()
                    .map(|(model_name, model_config)| ModelEntry {
                        name: model_name.clone(),
                        notes: format!(
                            "{} tok/s, {:?}, {}GB weights",
                            model_config.tokens_per_sec,
                            model_config.architecture,
                            model_config.weight_memory_gb,
                        ),
                    })
                    .collect();
                entries.push(ProviderEntry {
                    name: name.clone(),
                    kind: "local".into(),
                    models: model_entries,
                });
            }
            crate::config::ProviderConfig::Cli { command } => {
                entries.push(ProviderEntry {
                    name: name.clone(),
                    kind: format!("cli ({})", command),
                    models: vec![], // Capabilities come from the CLI itself.
                });
            }
        }
    }

    ProviderRoster { entries }
}
