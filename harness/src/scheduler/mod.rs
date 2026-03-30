use serde::{Deserialize, Serialize};

use crate::providers::{ModelId, ProviderId};
use crate::roles::Role;
use crate::tasks::TaskId;

/// Memory budget for local inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalCapacity {
    /// Total unified memory in bytes.
    pub total_memory: u64,
    /// Reserved for OS and system overhead.
    pub os_headroom: u64,
    /// Currently consumed by loaded models and active requests.
    pub used: u64,
}

impl LocalCapacity {
    pub fn available(&self) -> u64 {
        self.total_memory
            .saturating_sub(self.os_headroom)
            .saturating_sub(self.used)
    }
}

/// Cloud rate/cost limits for a provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudCapacity {
    pub provider_id: ProviderId,
    pub max_concurrent_requests: u32,
    pub tokens_per_minute: u64,
    pub cost_budget_remaining: f64,
}

/// A model currently loaded and available for work.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadedModel {
    pub provider_id: ProviderId,
    pub model_id: ModelId,
    /// Memory consumed by this model's weights + buffers.
    pub memory_used: u64,
    /// What this model is currently doing, if anything.
    pub assignment: Option<ModelAssignment>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelAssignment {
    pub task_id: TaskId,
    pub role: Role,
}

/// Scheduler decision for a pending task.
#[derive(Debug, Clone)]
pub enum SchedulerDecision {
    /// Dispatch to this model immediately.
    Dispatch {
        provider_id: ProviderId,
        model_id: ModelId,
    },
    /// Wait — insufficient capacity, will retry.
    Wait,
    /// Preempt this assignment to free capacity.
    Preempt {
        evict: ModelAssignment,
        then_dispatch: ProviderId,
        model_id: ModelId,
    },
    /// Reject — task exceeds maximum possible capacity.
    Reject { reason: String },
}
