//! Plan file types — the contract between planning and execution.
//!
//! The planner model writes a plan to `.orome/plan.yaml` during
//! `orome plan --claude`. The harness reads it on `orome run`,
//! validates it, and dispatches tasks respecting the dependency graph.

use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::tasks::TaskId;

/// A complete plan, as emitted by the planner and consumed by the harness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plan {
    pub id: String,
    pub summary: String,
    /// Planner's reasoning for this decomposition. Auditable.
    pub reasoning: String,
    pub tasks: Vec<PlanTask>,
    /// Optional overrides for review policy (from orome.yaml defaults).
    pub review: Option<ReviewPolicy>,
}

/// A single task within a plan. This is what the planner produces —
/// it describes *what* to do, not *how* or *where* to run it.
///
/// The harness maps this to a TaskSpec at dispatch time, filling in
/// budget and tool details based on provider capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanTask {
    pub id: TaskId,
    pub objective: String,
    pub acceptance_criteria: Vec<String>,
    /// Task IDs that must complete before this task starts.
    #[serde(default)]
    pub dependencies: Vec<TaskId>,
    /// Advisory hints from the planner. Not binding.
    #[serde(default)]
    pub hints: TaskHints,
}

/// Advisory hints from the planner to the harness.
/// The harness decides what to act on — these are suggestions, not commands.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TaskHints {
    /// Files the worker should look at.
    #[serde(default)]
    pub relevant_files: Vec<PathBuf>,
    /// Free-form guidance injected into the worker's context.
    pub guidance: Option<String>,
}

/// Review policy overrides for a plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewPolicy {
    pub reviewers_per_task: Option<u32>,
    pub max_retries: Option<u32>,
}

// ---------------------------------------------------------------------------
// Payloads — what the harness assembles before dispatching to a provider.
//
// For CLI providers (claude, codex): serialized into the prompt string.
// For orome-local: used to construct the ChatRequest and drive the tool loop.
// ---------------------------------------------------------------------------

/// What the harness sends to a worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerPayload {
    pub task_id: TaskId,
    pub objective: String,
    pub acceptance_criteria: Vec<String>,
    pub hints: TaskHints,
    /// Present on retry — reviewer feedback from the prior attempt.
    pub prior_attempt: Option<PriorAttempt>,
}

/// Context from a previous failed attempt, forwarded to the worker on retry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorAttempt {
    pub attempt: u32,
    pub result_summary: String,
    /// Reviewer's actionable direction — what to fix.
    pub reviewer_direction: Vec<String>,
}

/// What the harness sends to a reviewer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewerPayload {
    pub task_id: TaskId,
    pub objective: String,
    pub acceptance_criteria: Vec<String>,
    /// What the worker produced.
    pub result: WorkerEvidence,
}

/// Evidence bundle for the reviewer — the work product and its context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerEvidence {
    pub outcome: String,
    pub self_assessment: String,
    /// File diffs produced by the worker.
    pub diffs: String,
    /// Condensed scratchpad — what the worker did, not the raw manifest.
    #[serde(default)]
    pub scratchpad_digest: Vec<String>,
    /// Artifact summaries.
    #[serde(default)]
    pub artifacts: Vec<ArtifactSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactSummary {
    pub kind: String,
    pub path: Option<PathBuf>,
    pub summary: String,
}

/// What the harness sends when escalating a split verdict to the planner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TiebreakPayload {
    pub task_id: TaskId,
    pub objective: String,
    pub acceptance_criteria: Vec<String>,
    pub result: WorkerEvidence,
    pub reviews: Vec<ReviewSummary>,
}

/// Condensed review verdict for tiebreak context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewSummary {
    pub reviewer: String,
    pub scores: HashMap<String, f32>,
    pub confidence: f32,
    pub rationale: String,
    pub direction: Vec<String>,
}
