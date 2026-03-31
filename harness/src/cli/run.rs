//! `orome run` — execute a plan.
//!
//! Reads `.orome/plan.yaml`, validates the dependency DAG, and dispatches
//! tasks sequentially through the worker → reviewer → retry loop.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::dispatch::{dispatch_review, dispatch_worker, DispatchResult, ProviderTarget};
use crate::plan::{
    Plan, PlanTask, PriorAttempt, ReviewerPayload, WorkerEvidence, WorkerPayload,
};
use crate::tasks::TaskId;

/// Run result for a single task.
struct TaskRunResult {
    task_id: TaskId,
    outcome: TaskOutcome,
    worker_result: DispatchResult,
    review_result: Option<DispatchResult>,
}

enum TaskOutcome {
    Accepted,
    Failed { reason: String },
}

/// Execute the plan.
pub async fn run_plan(
    plan_path: &Path,
    worker_target: &ProviderTarget,
    reviewer_target: &ProviderTarget,
    max_retries: u32,
    dry_run: bool,
) -> anyhow::Result<()> {
    // 1. Read and parse plan.
    let plan_contents = std::fs::read_to_string(plan_path)
        .map_err(|e| anyhow::anyhow!("failed to read plan: {e}"))?;

    let plan_file: PlanFile = serde_yaml::from_str(&plan_contents)
        .map_err(|e| anyhow::anyhow!("failed to parse plan: {e}"))?;
    let plan = plan_file.plan;

    eprintln!("Plan: {} ({})", plan.summary, plan.id);
    eprintln!("Tasks: {}", plan.tasks.len());
    eprintln!();

    // 2. Validate dependency DAG.
    validate_dag(&plan.tasks)?;

    if dry_run {
        eprintln!("Dry run — plan is valid. Execution order:");
        let order = topological_sort(&plan.tasks)?;
        for (i, task_id) in order.iter().enumerate() {
            let task = plan.tasks.iter().find(|t| &t.id == task_id).unwrap();
            eprintln!("  {}. [{}] {}", i + 1, task.id.0, task.objective);
        }
        return Ok(());
    }

    // 3. Execute tasks in dependency order.
    let order = topological_sort(&plan.tasks)?;
    let task_map: HashMap<&TaskId, &PlanTask> =
        plan.tasks.iter().map(|t| (&t.id, t)).collect();

    let mut completed: HashSet<TaskId> = HashSet::new();
    let mut failed: HashSet<TaskId> = HashSet::new();
    let mut results: Vec<TaskRunResult> = Vec::new();

    for task_id in &order {
        let task = task_map[task_id];

        // Skip if a dependency failed.
        let blocked_by: Vec<_> = task
            .dependencies
            .iter()
            .filter(|dep| failed.contains(dep))
            .collect();
        if !blocked_by.is_empty() {
            eprintln!(
                "SKIP [{}] — blocked by failed: {:?}",
                task.id.0,
                blocked_by.iter().map(|d| &d.0).collect::<Vec<_>>()
            );
            failed.insert(task.id.clone());
            continue;
        }

        eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        eprintln!("TASK [{}] {}", task.id.0, task.objective);
        eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        let result =
            run_task_with_review(task, worker_target, reviewer_target, max_retries).await?;

        match &result.outcome {
            TaskOutcome::Accepted => {
                eprintln!("  ✓ ACCEPTED");
                completed.insert(task.id.clone());
            }
            TaskOutcome::Failed { reason } => {
                eprintln!("  ✗ FAILED: {reason}");
                failed.insert(task.id.clone());
            }
        }
        eprintln!();

        results.push(result);
    }

    // 4. Summary.
    eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    eprintln!("PLAN COMPLETE");
    eprintln!(
        "  Completed: {}/{}",
        completed.len(),
        plan.tasks.len()
    );
    if !failed.is_empty() {
        eprintln!(
            "  Failed: {} ({:?})",
            failed.len(),
            failed.iter().map(|f| &f.0).collect::<Vec<_>>()
        );
    }

    let total_duration: u64 = results.iter().map(|r| r.worker_result.duration_ms).sum();
    let total_input: u64 = results
        .iter()
        .filter_map(|r| r.worker_result.input_tokens)
        .sum();
    let total_output: u64 = results
        .iter()
        .filter_map(|r| r.worker_result.output_tokens)
        .sum();
    let total_cost: f64 = results
        .iter()
        .filter_map(|r| r.worker_result.cost_usd)
        .sum();

    eprintln!("  Duration: {:.1}s", total_duration as f64 / 1000.0);
    eprintln!("  Tokens: {} in / {} out", total_input, total_output);
    if total_cost > 0.0 {
        eprintln!("  Cost: ${:.4}", total_cost);
    }

    if !failed.is_empty() {
        anyhow::bail!("{} task(s) failed", failed.len());
    }

    Ok(())
}

/// Execute a single task with the worker → reviewer → retry loop.
async fn run_task_with_review(
    task: &PlanTask,
    worker_target: &ProviderTarget,
    reviewer_target: &ProviderTarget,
    max_retries: u32,
) -> anyhow::Result<TaskRunResult> {
    let mut prior_attempt: Option<PriorAttempt> = None;

    for attempt in 0..=max_retries {
        if attempt > 0 {
            eprintln!("  Retry {attempt}/{max_retries}...");
        }

        // -- Worker dispatch --
        let worker_payload = WorkerPayload {
            task_id: task.id.clone(),
            objective: task.objective.clone(),
            acceptance_criteria: task.acceptance_criteria.clone(),
            hints: task.hints.clone(),
            prior_attempt: prior_attempt.clone(),
        };

        eprintln!("  Worker dispatching...");
        let worker_result = dispatch_worker(worker_target, &worker_payload, None, &[])
            .await
            .map_err(|e| anyhow::anyhow!("worker dispatch failed: {e}"))?;

        eprintln!(
            "  Worker done ({:.1}s, {} tokens out)",
            worker_result.duration_ms as f64 / 1000.0,
            worker_result.output_tokens.unwrap_or(0),
        );

        if worker_result.is_error {
            return Ok(TaskRunResult {
                task_id: task.id.clone(),
                outcome: TaskOutcome::Failed {
                    reason: "worker reported error".into(),
                },
                worker_result,
                review_result: None,
            });
        }

        // -- Capture git diff --
        let diff = capture_git_diff().await.unwrap_or_default();

        // -- Reviewer dispatch --
        let review_payload = ReviewerPayload {
            task_id: task.id.clone(),
            objective: task.objective.clone(),
            acceptance_criteria: task.acceptance_criteria.clone(),
            result: WorkerEvidence {
                outcome: "completed".into(),
                self_assessment: worker_result.output.clone(),
                diffs: diff,
                scratchpad_digest: vec![],
                artifacts: vec![],
            },
        };

        eprintln!("  Reviewer evaluating...");
        let review_result = dispatch_review(reviewer_target, &review_payload)
            .await
            .map_err(|e| anyhow::anyhow!("review dispatch failed: {e}"))?;

        eprintln!(
            "  Review done ({:.1}s)",
            review_result.duration_ms as f64 / 1000.0,
        );

        // -- Parse verdict --
        let verdict = parse_verdict(&review_result.output);

        match verdict {
            Verdict::Accept => {
                return Ok(TaskRunResult {
                    task_id: task.id.clone(),
                    outcome: TaskOutcome::Accepted,
                    worker_result,
                    review_result: Some(review_result),
                });
            }
            Verdict::Reject { direction } => {
                eprintln!("  Reviewer rejected: {:?}", direction);
                prior_attempt = Some(PriorAttempt {
                    attempt: attempt + 1,
                    result_summary: worker_result.output.clone(),
                    reviewer_direction: direction,
                });
                // Continue to next retry.
            }
        }
    }

    Ok(TaskRunResult {
        task_id: task.id.clone(),
        outcome: TaskOutcome::Failed {
            reason: format!("rejected after {} retries", max_retries),
        },
        worker_result: DispatchResult {
            output: String::new(),
            diff: String::new(),
            input_tokens: None,
            output_tokens: None,
            duration_ms: 0,
            cost_usd: None,
            is_error: true,
        },
        review_result: None,
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Wrapper for serde — plan.yaml has `plan:` as the top-level key.
#[derive(serde::Deserialize)]
struct PlanFile {
    plan: Plan,
}

enum Verdict {
    Accept,
    Reject { direction: Vec<String> },
}

/// Best-effort parse of reviewer output. Looks for "accept"/"reject" in JSON
/// or falls back to keyword matching.
fn parse_verdict(output: &str) -> Verdict {
    // Try to find JSON in the output.
    if let Some(start) = output.find('{') {
        if let Some(end) = output.rfind('}') {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&output[start..=end]) {
                if let Some(verdict) = v["verdict"].as_str() {
                    if verdict == "accept" {
                        return Verdict::Accept;
                    }
                    let direction = v["direction"]
                        .as_array()
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_str().map(String::from))
                                .collect()
                        })
                        .unwrap_or_default();
                    return Verdict::Reject { direction };
                }
            }
        }
    }

    // Fallback: keyword matching.
    let lower = output.to_lowercase();
    if lower.contains("accept") && !lower.contains("reject") {
        Verdict::Accept
    } else {
        Verdict::Reject {
            direction: vec![output.to_string()],
        }
    }
}

/// Validate that task dependencies form a DAG (no cycles, all deps exist).
fn validate_dag(tasks: &[PlanTask]) -> anyhow::Result<()> {
    let ids: HashSet<&TaskId> = tasks.iter().map(|t| &t.id).collect();

    // Check all dependencies reference existing tasks.
    for task in tasks {
        for dep in &task.dependencies {
            if !ids.contains(dep) {
                anyhow::bail!(
                    "task '{}' depends on '{}' which doesn't exist",
                    task.id.0,
                    dep.0
                );
            }
        }
    }

    // Check for cycles via topological sort.
    topological_sort(tasks)?;
    Ok(())
}

/// Kahn's algorithm — returns tasks in execution order.
fn topological_sort(tasks: &[PlanTask]) -> anyhow::Result<Vec<TaskId>> {
    let mut in_degree: HashMap<&TaskId, usize> = HashMap::new();
    let mut dependents: HashMap<&TaskId, Vec<&TaskId>> = HashMap::new();

    for task in tasks {
        in_degree.entry(&task.id).or_insert(0);
        for dep in &task.dependencies {
            *in_degree.entry(&task.id).or_insert(0) += 1;
            dependents.entry(dep).or_default().push(&task.id);
        }
    }

    let mut queue: Vec<&TaskId> = in_degree
        .iter()
        .filter(|&(_, deg)| *deg == 0)
        .map(|(&id, _)| id)
        .collect();
    // Sort for deterministic order.
    queue.sort_by(|a, b| a.0.cmp(&b.0));

    let mut order = Vec::new();
    while let Some(id) = queue.pop() {
        order.push(id.clone());
        if let Some(deps) = dependents.get(id) {
            for dep_id in deps {
                let deg = in_degree.get_mut(dep_id).unwrap();
                *deg -= 1;
                if *deg == 0 {
                    queue.push(dep_id);
                    queue.sort_by(|a, b| a.0.cmp(&b.0));
                }
            }
        }
    }

    if order.len() != tasks.len() {
        anyhow::bail!("dependency cycle detected in plan");
    }

    Ok(order)
}

async fn capture_git_diff() -> anyhow::Result<String> {
    let output = tokio::process::Command::new("git")
        .args(["diff", "HEAD"])
        .output()
        .await?;
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}
