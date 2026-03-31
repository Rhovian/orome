use serde::{Deserialize, Serialize};

use crate::roles::Role;
use crate::tasks::{TaskGrade, TaskId, TaskResult, TaskSpec};
use crate::types::TimestampMs;

/// Every message routed through the harness hub.
/// Models never communicate directly — all messages flow through here.
///
/// `task_id` is the single source of truth for which task this message
/// belongs to. Message payloads do not repeat it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Envelope {
    pub id: String,
    pub timestamp: TimestampMs,
    pub from: Participant,
    pub to: Participant,
    pub task_id: TaskId,
    /// The role the sender is serving for this message.
    pub from_role: Option<Role>,
    /// The role the receiver should serve for this message.
    pub to_role: Option<Role>,
    pub payload: Message,
}

/// Who sent or receives a message. Identity only — role context
/// is on the envelope, not here.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Participant {
    /// The harness itself (scheduler, task manager logic).
    Harness,
    /// A specific model instance.
    Model { model_id: String },
}

/// Structured message types — the communication protocol.
///
/// Task identity comes from the enclosing `Envelope.task_id`.
/// Payloads carry only data specific to the message type.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Message {
    /// Planner/task-manager → worker.
    TaskAssignment { spec: TaskSpec },
    /// Worker → task-manager (optional, periodic).
    ProgressUpdate {
        status: String,
        tokens_so_far: u64,
    },
    /// Worker → task-manager.
    ResultSubmission { result: TaskResult },
    /// Task-manager → reviewer.
    ReviewRequest {},
    /// Reviewer → task-manager.
    ReviewVerdict { grade: TaskGrade },
    /// Task-manager → planner (when stuck).
    Escalation { reason: String },
    /// Any role → harness (request a subtask).
    Delegation {
        parent_task_id: TaskId,
        spec: TaskSpec,
    },
    /// Harness → tiebreaker (reviewer disagreement).
    TiebreakerRequest { reviews: Vec<TaskGrade> },
    /// Tiebreaker → harness.
    TiebreakerVerdict { grade: TaskGrade },
}
