//! Reviewer and tiebreaker prompt construction.
//!
//! Reviewers evaluate completed work against acceptance criteria.
//! They produce dimensional grades, not just pass/fail.

use std::fmt::Write;

use crate::plan::{ReviewerPayload, TiebreakPayload};

/// Build the review prompt for a reviewer.
pub fn build_reviewer_prompt(payload: &ReviewerPayload) -> String {
    let mut p = String::with_capacity(4096);

    // -- Role --
    writeln!(p, "You are a reviewer in the Orome coding harness.").unwrap();
    writeln!(p).unwrap();
    writeln!(
        p,
        "Evaluate the worker's output against the acceptance criteria. \
         Be rigorous but fair — grade what was actually asked for."
    )
    .unwrap();
    writeln!(p).unwrap();

    // -- Task spec --
    writeln!(p, "## Task: {}", payload.task_id.0).unwrap();
    writeln!(p).unwrap();
    writeln!(p, "**Objective:** {}", payload.objective).unwrap();
    writeln!(p).unwrap();
    writeln!(p, "### Acceptance Criteria").unwrap();
    writeln!(p).unwrap();
    for criterion in &payload.acceptance_criteria {
        writeln!(p, "- {criterion}").unwrap();
    }
    writeln!(p).unwrap();

    // -- Worker evidence --
    writeln!(p, "## Worker's Result").unwrap();
    writeln!(p).unwrap();
    writeln!(p, "**Outcome:** {}", payload.result.outcome).unwrap();
    writeln!(
        p,
        "**Self-assessment:** {}",
        payload.result.self_assessment
    )
    .unwrap();
    writeln!(p).unwrap();

    if !payload.result.artifacts.is_empty() {
        writeln!(p, "### Artifacts").unwrap();
        writeln!(p).unwrap();
        for artifact in &payload.result.artifacts {
            match &artifact.path {
                Some(path) => writeln!(p, "- [{}] {}: {}", artifact.kind, path.display(), artifact.summary).unwrap(),
                None => writeln!(p, "- [{}] {}", artifact.kind, artifact.summary).unwrap(),
            }
        }
        writeln!(p).unwrap();
    }

    if !payload.result.scratchpad_digest.is_empty() {
        writeln!(p, "### What the Worker Did").unwrap();
        writeln!(p).unwrap();
        for entry in &payload.result.scratchpad_digest {
            writeln!(p, "- {entry}").unwrap();
        }
        writeln!(p).unwrap();
    }

    if !payload.result.diffs.is_empty() {
        writeln!(p, "### Diffs").unwrap();
        writeln!(p).unwrap();
        writeln!(p, "```diff").unwrap();
        writeln!(p, "{}", payload.result.diffs).unwrap();
        writeln!(p, "```").unwrap();
        writeln!(p).unwrap();
    }

    // -- Grading rubric --
    writeln!(p, "## Grading").unwrap();
    writeln!(p).unwrap();
    writeln!(
        p,
        "Score each dimension from 0.0 to 1.0:\n\
         - **correctness**: Does the code do what was asked? Are there bugs?\n\
         - **completeness**: Are all acceptance criteria met?\n\
         - **efficiency**: Is the approach reasonable? No unnecessary complexity?\n\
         - **code_quality**: Clean, idiomatic, well-structured code?"
    )
    .unwrap();
    writeln!(p).unwrap();
    writeln!(
        p,
        "Respond with JSON:\n\
         ```json\n\
         {{\n\
         \x20 \"verdict\": \"accept\" | \"reject\",\n\
         \x20 \"scores\": {{ \"correctness\": 0.0, \"completeness\": 0.0, \"efficiency\": 0.0, \"code_quality\": 0.0 }},\n\
         \x20 \"confidence\": 0.0,\n\
         \x20 \"rationale\": \"...\",\n\
         \x20 \"direction\": [\"actionable feedback if rejecting\"]\n\
         }}\n\
         ```"
    )
    .unwrap();
    writeln!(p).unwrap();
    writeln!(
        p,
        "If rejecting, `direction` must contain specific, actionable feedback \
         the worker can act on. Do not reject without telling the worker how to fix it."
    )
    .unwrap();

    p
}

/// Build the tiebreak prompt for a planner resolving a split verdict.
pub fn build_tiebreak_prompt(payload: &TiebreakPayload) -> String {
    let mut p = String::with_capacity(4096);

    // -- Role --
    writeln!(p, "You are resolving a split reviewer verdict as tiebreaker.").unwrap();
    writeln!(p).unwrap();
    writeln!(
        p,
        "You authored the original task. Reviewers disagreed on the result. \
         Review the evidence and all reviewer verdicts, then make a final decision."
    )
    .unwrap();
    writeln!(p).unwrap();

    // -- Task spec --
    writeln!(p, "## Task: {}", payload.task_id.0).unwrap();
    writeln!(p).unwrap();
    writeln!(p, "**Objective:** {}", payload.objective).unwrap();
    writeln!(p).unwrap();
    writeln!(p, "### Acceptance Criteria").unwrap();
    writeln!(p).unwrap();
    for criterion in &payload.acceptance_criteria {
        writeln!(p, "- {criterion}").unwrap();
    }
    writeln!(p).unwrap();

    // -- Worker evidence --
    writeln!(p, "## Worker's Result").unwrap();
    writeln!(p).unwrap();
    writeln!(p, "**Outcome:** {}", payload.result.outcome).unwrap();
    writeln!(
        p,
        "**Self-assessment:** {}",
        payload.result.self_assessment
    )
    .unwrap();
    writeln!(p).unwrap();

    if !payload.result.diffs.is_empty() {
        writeln!(p, "### Diffs").unwrap();
        writeln!(p).unwrap();
        writeln!(p, "```diff").unwrap();
        writeln!(p, "{}", payload.result.diffs).unwrap();
        writeln!(p, "```").unwrap();
        writeln!(p).unwrap();
    }

    // -- Reviewer verdicts --
    writeln!(p, "## Reviewer Verdicts").unwrap();
    writeln!(p).unwrap();
    for (i, review) in payload.reviews.iter().enumerate() {
        writeln!(p, "### Reviewer {} ({})", i + 1, review.reviewer).unwrap();
        writeln!(p).unwrap();
        writeln!(p, "**Confidence:** {:.1}", review.confidence).unwrap();

        let scores_str: Vec<String> = review
            .scores
            .iter()
            .map(|(k, v)| format!("{k}: {v:.2}"))
            .collect();
        writeln!(p, "**Scores:** {}", scores_str.join(", ")).unwrap();
        writeln!(p).unwrap();
        writeln!(p, "**Rationale:** {}", review.rationale).unwrap();
        writeln!(p).unwrap();

        if !review.direction.is_empty() {
            writeln!(p, "**Direction:**").unwrap();
            for d in &review.direction {
                writeln!(p, "- {d}").unwrap();
            }
            writeln!(p).unwrap();
        }
    }

    // -- Decision format --
    writeln!(p, "## Your Verdict").unwrap();
    writeln!(p).unwrap();
    writeln!(
        p,
        "Respond with the same JSON format as a reviewer verdict. \
         Your decision is final for this task."
    )
    .unwrap();

    p
}
