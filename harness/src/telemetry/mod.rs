use serde::{Deserialize, Serialize};

use crate::providers::ModelId;
use crate::roles::Role;

/// Aggregate performance record for a model on a specific task type.
/// Feeds back into planner routing decisions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformanceRecord {
    pub model_id: ModelId,
    pub task_type: String,
    pub role: Role,
    pub samples: u32,
    pub avg_correctness: f32,
    pub avg_completeness: f32,
    pub avg_efficiency: f32,
    pub avg_code_quality: f32,
    pub avg_tokens_per_task: f64,
    pub avg_wall_time_ms: f64,
}
