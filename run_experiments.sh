#!/bin/bash
# run_experiments.sh — Autonomous experiment runner for orome
#
# Launches autonomous agent sessions in a loop. Each session:
#   1. Reads status.md for handoff context
#   2. Runs experiments (modify code, build, benchmark, log)
#   3. Writes status.md before exiting
#   4. This script launches the next session
#
# Recovery files (check these in the morning):
#   experiments/<model>/errors.log   — Runner-level errors
#   experiments/<model>/logs/        — Full session logs
#   experiments/<model>/results.tsv  — All experiment results
#   experiments/<model>/status.md    — Last session's handoff note
#
# Usage:
#   ./run_experiments.sh <model>                               # run forever with Claude
#   ./run_experiments.sh <model> --agent codex                # run forever with Codex
#   ./run_experiments.sh <model> --sessions 5                 # run 5 Claude sessions then stop
#   ./run_experiments.sh <model> --agent codex --sessions 5   # run 5 Codex sessions then stop
#
# Experiment targets are discovered from experiments/*/program.md.
# Current repo target:
#   qwen35-35B   — Qwen3.5-35B-A3B GGUF optimization
#
# To stop: Ctrl+C or kill this script. Current experiment will finish.

set -euo pipefail

# Ensure PATH includes common install locations
export PATH="$HOME/.local/bin:/opt/homebrew/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---- Parse arguments ----
MODEL="${1:-}"
if [[ -z "$MODEL" ]]; then
    echo "Usage: $0 <model> [--agent claude|codex] [--sessions N]"
    echo ""
    echo "Available models:"
    for d in experiments/*/; do
        [[ -f "$d/program.md" ]] && echo "  $(basename "$d")"
    done
    exit 1
fi

EXPERIMENT_DIR="$SCRIPT_DIR/experiments/$MODEL"
if [[ ! -d "$EXPERIMENT_DIR" ]]; then
    echo "ERROR: No experiment directory at $EXPERIMENT_DIR"
    echo "Available models:"
    for d in experiments/*/; do
        [[ -f "$d/program.md" ]] && echo "  $(basename "$d")"
    done
    exit 1
fi

if [[ ! -f "$EXPERIMENT_DIR/program.md" ]]; then
    echo "ERROR: No program.md in $EXPERIMENT_DIR"
    exit 1
fi

shift  # consume model arg
MAX_SESSIONS=0
AGENT="claude"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --agent)
            [[ $# -ge 2 ]] || { echo "ERROR: --agent requires a value"; exit 1; }
            AGENT="$2"
            shift 2
            ;;
        --sessions)
            [[ $# -ge 2 ]] || { echo "ERROR: --sessions requires a value"; exit 1; }
            MAX_SESSIONS="$2"
            shift 2
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

case "$AGENT" in
    claude|codex) ;;
    *)
        echo "ERROR: --agent must be 'claude' or 'codex' (got: $AGENT)"
        exit 1
        ;;
esac

if ! command -v "$AGENT" >/dev/null 2>&1; then
    echo "ERROR: '$AGENT' is not installed or not on PATH"
    exit 1
fi

# ---- Determine branch name ----
BRANCH_NAME="autoresearch/orome"
if [[ "$MODEL" != "qwen35-35B" ]]; then
    BRANCH_NAME="autoresearch/orome-${MODEL#qwen35-}"
fi

# ---- Setup paths ----
MAX_CONSECUTIVE_FAILURES=3
SESSION_NUM=0
CONSECUTIVE_FAILURES=0
LOG_DIR="$EXPERIMENT_DIR/logs"
ERROR_LOG="$EXPERIMENT_DIR/errors.log"
RESULTS_SCOPE_FILE="$EXPERIMENT_DIR/results_scope.json"
mkdir -p "$LOG_DIR"

# Trap Ctrl+C gracefully
trap 'echo ""; echo "[runner] Caught interrupt. Waiting for current session to finish..."; STOP=1' INT
STOP=0

echo "============================================"
echo "  orome autonomous experiment runner"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Model: $MODEL"
echo "  Agent: $AGENT"
echo "  Experiment dir: $EXPERIMENT_DIR"
echo "  Working dir: $SCRIPT_DIR"
echo "  Branch: $BRANCH_NAME"
echo "  Max sessions: ${MAX_SESSIONS:-unlimited}"
echo "  Error log: $ERROR_LOG"
echo "============================================"
echo "" | tee -a "$ERROR_LOG"
echo "=== Runner started $(date '+%Y-%m-%d %H:%M:%S') model=$MODEL ===" >> "$ERROR_LOG"

# Ensure we're on the right branch
if git rev-parse --git-dir > /dev/null 2>&1; then
    BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")
    if [[ "$BRANCH" != "$BRANCH_NAME" ]]; then
        echo "[runner] Switching to branch $BRANCH_NAME"
        git checkout -b "$BRANCH_NAME" 2>/dev/null || git checkout "$BRANCH_NAME"
    fi
    echo "[runner] On branch: $(git branch --show-current)"
fi

# Safety check: verify the codebase builds before starting
echo "[runner] Pre-flight build check..."
if ! make clean && make 2>/dev/null; then
    echo "[runner] FATAL: Initial build failed. Fix before running." | tee -a "$ERROR_LOG"
    exit 1
fi
echo "[runner] Build OK."

# ---- Collect cross-model regression checks ----
# Each OTHER experiment's cross_check.json defines a quick smoke benchmark.
# After each successful session, we run all of them to catch regressions.
CROSS_CHECKS=()
for d in experiments/*/; do
    other="$(basename "$d")"
    [[ "$other" == "$MODEL" ]] && continue
    cc="$d/cross_check.json"
    [[ -f "$cc" ]] && CROSS_CHECKS+=("$cc")
done
if [[ ${#CROSS_CHECKS[@]} -gt 0 ]]; then
    echo "[runner] Cross-model regression checks: ${CROSS_CHECKS[*]}"
else
    echo "[runner] No cross-model regression checks configured"
fi

CURRENT_BEST_LABEL=""
CURRENT_BEST_END_BEFORE_COMMIT=""
if [[ -f "$RESULTS_SCOPE_FILE" ]]; then
    CURRENT_BEST_LABEL=$(python3 -c "import json; print(json.load(open('$RESULTS_SCOPE_FILE')).get('current_best_label', ''))")
    CURRENT_BEST_END_BEFORE_COMMIT=$(python3 -c "import json; print(json.load(open('$RESULTS_SCOPE_FILE')).get('current_best_end_before_commit', ''))")
fi

best_results_row() {
    local results_file="$1"
    local scope_mode="${2:-all}"
    python3 - "$results_file" "$scope_mode" "$CURRENT_BEST_END_BEFORE_COMMIT" <<'PY'
import csv
import sys

path, scope_mode, end_before = sys.argv[1:]

with open(path, newline="") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

if scope_mode == "scoped" and end_before:
    scoped = []
    found = False
    for row in rows:
        if row.get("commit") == end_before:
            found = True
            break
        scoped.append(row)
    if not found:
        sys.exit(2)
    rows = scoped

best = None
for row in rows:
    if row.get("status") != "keep":
        continue
    try:
        tok = float(row.get("tok_sec", ""))
    except ValueError:
        continue
    if best is None or tok > best[0]:
        best = (tok, row)

if best is not None:
    row = best[1]
    print("\t".join(row.get(k, "") for k in (
        "commit", "tok_sec", "ttft_ms", "proj_avg_ms", "status", "description"
    )))
PY
}

run_cross_checks() {
    # Run each cross-model check. Returns 0 if all pass, 1 if any regress.
    local any_fail=0
    for cc in "${CROSS_CHECKS[@]}"; do
        local cc_model cc_min_tok cc_tokens cc_k cc_desc
        cc_model=$(python3 -c "import json; d=json.load(open('$cc')); print(d.get('model') or d.get('model_dir') or '')")
        cc_min_tok=$(python3 -c "import json; print(json.load(open('$cc'))['min_tok_sec'])")
        cc_tokens=$(python3 -c "import json; print(json.load(open('$cc')).get('tokens', 5))")
        cc_k=$(python3 -c "import json; print(json.load(open('$cc')).get('k', 8))")
        cc_desc=$(python3 -c "import json; print(json.load(open('$cc'))['description'])")

        if [[ -z "$cc_model" ]]; then
            echo "[cross-check] SKIP: $cc_desc (no model configured: $cc)"
            continue
        fi

        if [[ ! -e "$cc_model" ]]; then
            echo "[cross-check] SKIP: $cc_desc (model not found: $cc_model)"
            continue
        fi

        echo "[cross-check] Running: $cc_desc"
        local output
        output=$(./orome --model "$cc_model" --prompt "Hello" --tokens "$cc_tokens" --k "$cc_k" 2>&1)
        local tok_sec
        tok_sec=$(echo "$output" | grep -o '[0-9.]* tok/s' | head -1 | awk '{print $1}')

        if [[ -z "$tok_sec" ]]; then
            echo "[cross-check] FAIL: $cc_desc — could not parse tok/s"
            echo "  Output: $(echo "$output" | tail -3)"
            any_fail=1
        else
            local passed
            passed=$(python3 -c "print('yes' if $tok_sec >= $cc_min_tok else 'no')")
            if [[ "$passed" == "yes" ]]; then
                echo "[cross-check] PASS: $cc_desc — ${tok_sec} tok/s (min: ${cc_min_tok})"
            else
                echo "[cross-check] FAIL: $cc_desc — ${tok_sec} tok/s < ${cc_min_tok} min"
                any_fail=1
            fi
        fi
    done
    return $any_fail
}

run_agent_session() {
    local prompt="$1"

    case "$AGENT" in
        claude)
            # --output-format stream-json: streams output incrementally (visible in real-time)
            # --dangerously-skip-permissions: no prompts (we trust the agent)
            # --max-turns: limit turns per session to avoid runaway
            claude --output-format stream-json \
                --verbose \
                --dangerously-skip-permissions \
                --max-turns 80 \
                -p "$prompt"
            ;;
        codex)
            # --json: streams events incrementally (visible in real-time)
            # --dangerously-bypass-approvals-and-sandbox: no prompts (we trust the agent)
            codex exec \
                --json \
                --dangerously-bypass-approvals-and-sandbox \
                -C "$SCRIPT_DIR" \
                "$prompt"
            ;;
    esac
}

while true; do
    if [[ "$STOP" -eq 1 ]]; then
        echo "[runner] Stopping (interrupt received)."
        break
    fi

    if [[ "$MAX_SESSIONS" -gt 0 && "$SESSION_NUM" -ge "$MAX_SESSIONS" ]]; then
        echo "[runner] Reached max sessions ($MAX_SESSIONS). Stopping."
        break
    fi

    SESSION_NUM=$((SESSION_NUM + 1))
    TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
    SESSION_LOG="$LOG_DIR/session_${TIMESTAMP}.log"

    echo ""
    echo "[runner] === Session $SESSION_NUM ($MODEL) starting at $(date '+%H:%M:%S') ==="
    echo "[runner] Log: $SESSION_LOG"

    # Build the prompt: program.md + current status + results history
    PROMPT="$(cat "$EXPERIMENT_DIR/program.md")"
    if [[ -f "$EXPERIMENT_DIR/status.md" ]]; then
        PROMPT="$PROMPT

---

## Current Status (from previous session)

$(cat "$EXPERIMENT_DIR/status.md")"
    fi

    if [[ -f "$EXPERIMENT_DIR/results.tsv" ]]; then
        PROMPT="$PROMPT

---

## Experiment History

\`\`\`
$(cat "$EXPERIMENT_DIR/results.tsv")
\`\`\`"
    fi

    # Tell the agent where its experiment files live
    PROMPT="$PROMPT

---

## Experiment Paths

- Results: \`experiments/$MODEL/results.tsv\`
- Status: \`experiments/$MODEL/status.md\`
- Bench errors: \`experiments/$MODEL/bench_err.txt\`"

    # Snapshot the codebase state before the session (for recovery)
    GIT_HEAD_BEFORE=$(git rev-parse HEAD 2>/dev/null || echo "unknown")

    # Launch agent session and stream output to the log in real time.
    run_agent_session "$PROMPT" 2>&1 | tee "$SESSION_LOG"

    EXIT_CODE=${PIPESTATUS[0]}
    GIT_HEAD_AFTER=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
    echo ""
    echo "[runner] Session $SESSION_NUM finished (exit code: $EXIT_CODE) at $(date '+%H:%M:%S')"

    # Log errors and track consecutive failures
    if [[ "$EXIT_CODE" -ne 0 ]]; then
        CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
        {
            echo ""
            echo "--- Session $SESSION_NUM FAILED (exit=$EXIT_CODE) at $(date '+%Y-%m-%d %H:%M:%S') ---"
            echo "  Model: $MODEL"
            echo "  Consecutive failures: $CONSECUTIVE_FAILURES / $MAX_CONSECUTIVE_FAILURES"
            echo "  Log: $SESSION_LOG"
            echo "  Git before: $GIT_HEAD_BEFORE"
            echo "  Git after:  $GIT_HEAD_AFTER"
            echo "  Tail of session log:"
            tail -20 "$SESSION_LOG" 2>/dev/null | sed 's/^/    /'
            echo "---"
        } >> "$ERROR_LOG"
        echo "[runner] ERROR logged to $ERROR_LOG (consecutive: $CONSECUTIVE_FAILURES/$MAX_CONSECUTIVE_FAILURES)"

        # Safety: verify the build still works after a failed session
        if ! make 2>/dev/null; then
            echo "[runner] WARNING: Build broken after failed session. Reverting to $GIT_HEAD_BEFORE" | tee -a "$ERROR_LOG"
            git checkout -- src/ include/ Makefile shaders.metal 2>/dev/null || true
            echo "  Reverted source files to clean state" >> "$ERROR_LOG"
        fi

        # Circuit breaker
        if [[ "$CONSECUTIVE_FAILURES" -ge "$MAX_CONSECUTIVE_FAILURES" ]]; then
            echo "[runner] CIRCUIT BREAKER: $MAX_CONSECUTIVE_FAILURES consecutive failures. Stopping." | tee -a "$ERROR_LOG"
            break
        fi
    else
        CONSECUTIVE_FAILURES=0  # reset on success
        echo "Session $SESSION_NUM OK (exit=0) at $(date '+%H:%M:%S') git=$GIT_HEAD_AFTER" >> "$ERROR_LOG"

        # Cross-model regression check (only after successful sessions that changed code)
        if [[ ${#CROSS_CHECKS[@]} -gt 0 && "$GIT_HEAD_BEFORE" != "$GIT_HEAD_AFTER" ]]; then
            echo "[runner] Running cross-model regression checks..."
            if ! run_cross_checks; then
                {
                    echo ""
                    echo "--- Session $SESSION_NUM REGRESSION at $(date '+%Y-%m-%d %H:%M:%S') ---"
                    echo "  Model: $MODEL"
                    echo "  Git before: $GIT_HEAD_BEFORE"
                    echo "  Git after:  $GIT_HEAD_AFTER"
                    echo "  Cross-model regression detected. Reverting to $GIT_HEAD_BEFORE."
                    echo "---"
                } >> "$ERROR_LOG"
                echo "[runner] REGRESSION detected. Reverting session commits..."
                git reset --hard "$GIT_HEAD_BEFORE" 2>/dev/null || true
                echo "[runner] Reverted to $GIT_HEAD_BEFORE"
            fi
        fi
    fi

    # Show current best if results exist
    if [[ -f "$EXPERIMENT_DIR/results.tsv" ]]; then
        BEST=""
        if [[ -n "$CURRENT_BEST_END_BEFORE_COMMIT" ]]; then
            if BEST=$(best_results_row "$EXPERIMENT_DIR/results.tsv" scoped); then
                if [[ -n "$CURRENT_BEST_LABEL" ]]; then
                    echo "[runner] Current best (${CURRENT_BEST_LABEL}): $BEST"
                else
                    echo "[runner] Current best: $BEST"
                fi
            else
                echo "[runner] WARNING: Could not find scoped results boundary $CURRENT_BEST_END_BEFORE_COMMIT; falling back to all-time history"
            fi
        fi

        if [[ -z "$BEST" ]]; then
            BEST=$(best_results_row "$EXPERIMENT_DIR/results.tsv" all || echo "no results yet")
            echo "[runner] Current best: $BEST"
        fi

        if [[ -n "$CURRENT_BEST_END_BEFORE_COMMIT" ]]; then
            HISTORICAL_BEST=$(best_results_row "$EXPERIMENT_DIR/results.tsv" all || true)
            if [[ -n "$HISTORICAL_BEST" && "$HISTORICAL_BEST" != "$BEST" ]]; then
                echo "[runner] Historical context: $HISTORICAL_BEST"
            fi
        fi
    fi

    # Brief pause between sessions
    if [[ "$STOP" -eq 0 ]]; then
        echo "[runner] Next session in 10 seconds..."
        sleep 10
    fi
done

echo ""
echo "[runner] === Runner finished ==="
echo "[runner] Model: $MODEL"
echo "[runner] Total sessions: $SESSION_NUM"
echo "[runner] Results in: $EXPERIMENT_DIR/results.tsv"
echo "[runner] Errors in: $ERROR_LOG"
echo "[runner] Logs in: $LOG_DIR/"
echo "=== Runner finished $(date '+%Y-%m-%d %H:%M:%S'), $SESSION_NUM sessions, model=$MODEL ===" >> "$ERROR_LOG"
if [[ -f "$EXPERIMENT_DIR/status.md" ]]; then
    echo ""
    echo "[runner] Final status:"
    cat "$EXPERIMENT_DIR/status.md"
fi
