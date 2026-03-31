use serde::{Deserialize, Serialize};

/// Canonical roles in the harness. Any model can serve any role.
/// Bindings are dynamic and determined at dispatch time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// Decomposes user intent into bounded tasks with acceptance criteria.
    Planner,
    /// Owns a task through completion. Primarily deterministic harness logic
    /// with an escape hatch to a model for ambiguous cases.
    TaskManager,
    /// Executes a task. Writes to scratchpad, produces artifacts, calls tools.
    Worker,
    /// Evaluates completed work against acceptance criteria.
    /// Produces dimensional grades, not just pass/fail.
    Reviewer,
    /// Evaluates process quality. Was the task fairly scoped?
    /// Did the worker loop? Is the reviewer calibrated?
    Referee,
    /// Resolves disagreements between reviewers only.
    Tiebreaker,
    /// Compresses context, produces handoff summaries.
    Summarizer,
    /// Retrieves information from codebase, docs, external sources.
    Searcher,
}

impl Role {
    /// Roles that require model-driven judgment (not deterministic harness logic).
    pub fn requires_model(&self) -> bool {
        !matches!(self, Role::TaskManager)
    }

    /// What scratchpad access this role gets for a given task.
    pub fn scratchpad_access(&self) -> ScratchpadAccess {
        match self {
            Role::Worker => ScratchpadAccess::ReadWriteOwn,
            Role::Reviewer => ScratchpadAccess::ReadOnly,
            Role::Referee => ScratchpadAccess::ReadOnly,
            Role::Tiebreaker => ScratchpadAccess::ReadOnly,
            Role::Summarizer => ScratchpadAccess::ReadOnly,
            Role::Searcher => ScratchpadAccess::None,
            Role::Planner => ScratchpadAccess::None,
            Role::TaskManager => ScratchpadAccess::ReadOnly,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScratchpadAccess {
    /// Read/write own task scratchpad, read-only parent.
    ReadWriteOwn,
    /// Read-only on completed scratchpad.
    ReadOnly,
    /// No scratchpad access.
    None,
}
