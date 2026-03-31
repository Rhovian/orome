//! Worker prompt construction.
//!
//! For CLI dispatch: the entire string is passed to `claude -p` or `codex exec`.
//! For local dispatch: used as the user message in the first ChatRequest turn.

use std::fmt::Write;

use crate::plan::{PriorAttempt, WorkerPayload};

/// Build the task prompt for a worker.
///
/// This is a self-contained prompt that gives the worker everything it needs:
/// the task, the criteria, relevant context, and retry feedback if applicable.
pub fn build_worker_prompt(payload: &WorkerPayload) -> String {
    let mut p = String::with_capacity(2048);

    // -- Role + task --
    writeln!(p, "You are a worker in the Orome coding harness.").unwrap();
    writeln!(p).unwrap();
    writeln!(p, "## Task: {}", payload.task_id.0).unwrap();
    writeln!(p).unwrap();
    writeln!(p, "**Objective:** {}", payload.objective).unwrap();
    writeln!(p).unwrap();

    // -- Acceptance criteria --
    writeln!(p, "## Acceptance Criteria").unwrap();
    writeln!(p).unwrap();
    for criterion in &payload.acceptance_criteria {
        writeln!(p, "- {criterion}").unwrap();
    }
    writeln!(p).unwrap();

    // -- Hints --
    if !payload.hints.relevant_files.is_empty() {
        writeln!(p, "## Relevant Files").unwrap();
        writeln!(p).unwrap();
        for file in &payload.hints.relevant_files {
            writeln!(p, "- {}", file.display()).unwrap();
        }
        writeln!(p).unwrap();
    }

    if let Some(guidance) = &payload.hints.guidance {
        writeln!(p, "## Guidance").unwrap();
        writeln!(p).unwrap();
        writeln!(p, "{guidance}").unwrap();
        writeln!(p).unwrap();
    }

    // -- Prior attempt (retry context) --
    if let Some(prior) = &payload.prior_attempt {
        build_retry_context(&mut p, prior);
    }

    // -- Output instructions --
    writeln!(p, "## Instructions").unwrap();
    writeln!(p).unwrap();
    writeln!(
        p,
        "Complete the task. Use the tools available to you to read, write, \
         and test code. When done, provide a brief self-assessment of whether \
         all acceptance criteria are met."
    )
    .unwrap();

    p
}

fn build_retry_context(p: &mut String, prior: &PriorAttempt) {
    writeln!(p, "## Prior Attempt (attempt {})", prior.attempt).unwrap();
    writeln!(p).unwrap();
    writeln!(p, "**Previous result:** {}", prior.result_summary).unwrap();
    writeln!(p).unwrap();
    if !prior.reviewer_direction.is_empty() {
        writeln!(p, "**Reviewer feedback — address these issues:**").unwrap();
        writeln!(p).unwrap();
        for direction in &prior.reviewer_direction {
            writeln!(p, "- {direction}").unwrap();
        }
        writeln!(p).unwrap();
    }
}
