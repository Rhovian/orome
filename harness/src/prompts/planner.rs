//! Planner prompt construction.
//!
//! The planner system prompt is injected into the interactive `orome plan`
//! session. It gives the planner model everything it needs to decompose
//! user requests into structured task plans.

use std::fmt::Write;

/// Context about the project, gathered by the harness at session start.
pub struct ProjectContext {
    /// Absolute path to the project root.
    pub root: String,
    /// Condensed file tree (e.g. output of `tree -L 3` or similar).
    pub tree_digest: String,
    /// Recent commit summaries (e.g. last 10 `git log --oneline`).
    pub recent_commits: Vec<String>,
}

/// Context about available providers, read from orome.yaml + runtime state.
pub struct ProviderRoster {
    pub entries: Vec<ProviderEntry>,
}

pub struct ProviderEntry {
    pub name: String,
    pub kind: String,
    pub models: Vec<ModelEntry>,
}

pub struct ModelEntry {
    pub name: String,
    /// Free-form notes (e.g. "68 tok/s, MoE 3B active", "dense 27B").
    pub notes: String,
}

/// Build the system prompt for an interactive planner session.
///
/// This is injected at the start of `orome plan --claude` (or `--codex`).
/// The user then converses normally; the model writes `.orome/plan.yaml`.
pub fn build_planner_system_prompt(
    project: &ProjectContext,
    roster: &ProviderRoster,
) -> String {
    let mut p = String::with_capacity(4096);

    // -- Role identity --
    writeln!(p, "You are a planner for the Orome coding harness.").unwrap();
    writeln!(p).unwrap();
    writeln!(
        p,
        "Your job is to collaborate with the user to decompose their request into \
         a structured task plan. You do NOT execute tasks — you produce a plan that \
         the harness will execute by dispatching tasks to workers and reviewers."
    )
    .unwrap();
    writeln!(p).unwrap();

    // -- Plan file schema --
    writeln!(p, "## Plan File Format").unwrap();
    writeln!(p).unwrap();
    writeln!(
        p,
        "Write the plan to `.orome/plan.yaml`. Update it as the user iterates. \
         The file must conform to this schema:"
    )
    .unwrap();
    writeln!(p).unwrap();
    writeln!(p, "```yaml").unwrap();
    writeln!(p, "{}", PLAN_SCHEMA).unwrap();
    writeln!(p, "```").unwrap();
    writeln!(p).unwrap();

    // -- Task design guidance --
    writeln!(p, "## Task Design Principles").unwrap();
    writeln!(p).unwrap();
    writeln!(
        p,
        "- Each task should be independently executable by a single worker.\n\
         - Acceptance criteria must be concrete and verifiable — a reviewer will \
           evaluate the work against them.\n\
         - Use dependencies to express ordering. Tasks without dependencies can \
           run in parallel.\n\
         - Prefer smaller, focused tasks over large monolithic ones.\n\
         - The `hints` section is advisory — use `relevant_files` to point the \
           worker to the right code and `guidance` for anything non-obvious."
    )
    .unwrap();
    writeln!(p).unwrap();

    // -- Available providers --
    writeln!(p, "## Available Providers").unwrap();
    writeln!(p).unwrap();
    for entry in &roster.entries {
        writeln!(p, "### {} ({})", entry.name, entry.kind).unwrap();
        for model in &entry.models {
            if model.notes.is_empty() {
                writeln!(p, "- {}", model.name).unwrap();
            } else {
                writeln!(p, "- {} — {}", model.name, model.notes).unwrap();
            }
        }
        writeln!(p).unwrap();
    }

    // -- Project context --
    writeln!(p, "## Project").unwrap();
    writeln!(p).unwrap();
    writeln!(p, "Root: `{}`", project.root).unwrap();
    writeln!(p).unwrap();

    if !project.tree_digest.is_empty() {
        writeln!(p, "### File Tree").unwrap();
        writeln!(p, "```").unwrap();
        writeln!(p, "{}", project.tree_digest).unwrap();
        writeln!(p, "```").unwrap();
        writeln!(p).unwrap();
    }

    if !project.recent_commits.is_empty() {
        writeln!(p, "### Recent Commits").unwrap();
        for commit in &project.recent_commits {
            writeln!(p, "- {commit}").unwrap();
        }
        writeln!(p).unwrap();
    }

    // -- Workflow instructions --
    writeln!(p, "## Workflow").unwrap();
    writeln!(p).unwrap();
    writeln!(
        p,
        "1. Discuss the user's request and clarify scope.\n\
         2. Propose a task decomposition.\n\
         3. Write the plan to `.orome/plan.yaml`.\n\
         4. Iterate based on user feedback — update the file each time.\n\
         5. When the user is satisfied, they will run `orome run` to execute."
    )
    .unwrap();

    p
}

/// Compact plan schema embedded in the planner prompt.
/// Shows required structure without the full annotated reference.
const PLAN_SCHEMA: &str = r#"plan:
  id: <unique-id>
  summary: <one-line summary>
  reasoning: |
    <why this decomposition>
  tasks:
    - id: <task-id>
      objective: <what the worker must accomplish>
      acceptance_criteria:
        - <verifiable criterion>
      dependencies: []          # task IDs that must complete first
      hints:
        relevant_files:
          - <path>
        guidance: <optional free-form guidance>
  # Optional — overrides defaults from orome.yaml
  review:
    reviewers_per_task: 1
    max_retries: 2"#;
