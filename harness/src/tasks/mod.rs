pub mod scratchpad;

use serde::{Deserialize, Serialize};

use crate::roles::Role;
use crate::tools::ToolName;
use crate::types::{DurationMs, TimestampMs};

/// Unique task identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TaskId(pub String);

/// What the planner wants done.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskSpec {
    pub id: TaskId,
    pub parent: Option<TaskId>,
    pub objective: String,
    pub acceptance_criteria: Vec<String>,
    pub budget: TaskBudget,
    pub tool_allowlist: Option<Vec<ToolName>>,
}

/// Resource constraints for a task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskBudget {
    pub max_tokens: u64,
    pub max_wall_time: DurationMs,
    /// Shared with subtasks — prevents runaway delegation.
    pub iteration_limit: u32,
}

/// What a worker produces.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    pub task_id: TaskId,
    pub outcome: TaskOutcome,
    pub artifacts: Vec<Artifact>,
    pub self_assessment: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskOutcome {
    Success,
    Failure,
    Partial,
}

/// A concrete output from a task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifact {
    pub kind: ArtifactKind,
    pub path: Option<std::path::PathBuf>,
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    FileDiff,
    TestResult,
    SearchResult,
    Summary,
    Other(String),
}

/// What a reviewer produces.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskGrade {
    pub task_id: TaskId,
    pub reviewer_model: String,
    pub reviewer_role: Role,
    pub scores: GradeScores,
    pub confidence: f32,
    pub rationale: String,
}

/// Dimensional scores, each 0.0–1.0.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradeScores {
    pub correctness: f32,
    pub completeness: f32,
    pub efficiency: f32,
    pub code_quality: f32,
}

/// Per-task telemetry, attached to every result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskTelemetry {
    pub task_id: TaskId,
    pub model_id: String,
    pub backend: String,
    pub role: Role,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub wall_time: DurationMs,
    pub prefill_time: DurationMs,
    pub decode_time: DurationMs,
    pub tool_calls: Vec<ToolCallTelemetry>,
    pub retries: u32,
    pub escalations: u32,
    pub preemptions: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallTelemetry {
    pub tool: ToolName,
    pub latency: DurationMs,
    pub output_bytes: u64,
    pub success: bool,
}

/// Lifecycle state machine for a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskState {
    /// Created by planner, waiting for worker assignment.
    Pending,
    /// Worker assigned, executing.
    Running,
    /// Worker preempted, can resume from scratchpad.
    Preempted,
    /// Worker finished, awaiting review.
    AwaitingReview,
    /// Review in progress.
    UnderReview,
    /// Review complete, accepted.
    Completed,
    /// Review complete, rejected — may retry.
    Rejected,
    /// Escalated to planner.
    Escalated,
}

/// Aggregate tracking a task through its lifecycle.
pub struct Task {
    pub spec: TaskSpec,
    pub state: TaskState,
    pub assigned_to: Option<crate::scheduler::ModelAssignment>,
    pub scratchpad: scratchpad::Scratchpad,
    pub result: Option<TaskResult>,
    pub grade: Option<TaskGrade>,
    pub telemetry: Option<TaskTelemetry>,
    pub created_at: TimestampMs,
}
