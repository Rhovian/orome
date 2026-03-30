use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Duration in milliseconds. Serializes as a plain u64.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct DurationMs(pub u64);

impl DurationMs {
    pub fn from_duration(d: Duration) -> Self {
        Self(d.as_millis() as u64)
    }

    pub fn to_duration(self) -> Duration {
        Duration::from_millis(self.0)
    }
}

/// Timestamp in milliseconds since Unix epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TimestampMs(pub u64);
