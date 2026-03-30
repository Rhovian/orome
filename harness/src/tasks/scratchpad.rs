use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::TaskId;
use crate::types::TimestampMs;

/// Manifest entry for a scratchpad item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScratchpadEntry {
    pub timestamp: TimestampMs,
    pub kind: ScratchpadEntryKind,
    pub summary: String,
    /// Relative path within the scratchpad directory, if file-backed.
    pub path: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScratchpadEntryKind {
    ToolOutput,
    FileWrite,
    FileDiff,
    ReasoningStep,
    TestResult,
    SearchResult,
    Note,
}

/// Handle to a task's scratchpad on disk.
///
/// Layout:
///   <base>/<task_id>/
///     manifest.jsonl   — append-only entry log
///     files/           — file artifacts
#[derive(Debug)]
pub struct Scratchpad {
    pub task_id: TaskId,
    pub root: PathBuf,
}

impl Scratchpad {
    pub fn new(base_dir: &std::path::Path, task_id: &TaskId) -> Self {
        let root = base_dir.join(&task_id.0);
        Self {
            task_id: task_id.clone(),
            root,
        }
    }

    pub fn manifest_path(&self) -> PathBuf {
        self.root.join("manifest.jsonl")
    }

    pub fn files_dir(&self) -> PathBuf {
        self.root.join("files")
    }
}
