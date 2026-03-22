#!/bin/bash
# run_experiments.sh — Autonomous experiment runner for orome
#
# Launches Claude Code sessions in a loop. Each session:
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
#   ./run_experiments.sh <model>                # run forever
#   ./run_experiments.sh <model> --sessions 5   # run 5 sessions then stop
#
# Models:
#   qwen35-35B   — Qwen3.5-35B-A3B (fits in RAM, mlock path)
#   qwen35-397B  — Qwen3.5-397B-A17B (pread path, adaptive memory)
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
    echo "Usage: $0 <model> [--sessions N]"
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
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sessions) MAX_SESSIONS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

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
mkdir -p "$LOG_DIR"

# Trap Ctrl+C gracefully
trap 'echo ""; echo "[runner] Caught interrupt. Waiting for current session to finish..."; STOP=1' INT
STOP=0

echo "============================================"
echo "  orome autonomous experiment runner"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Model: $MODEL"
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

run_cross_checks() {
    # Run each cross-model check. Returns 0 if all pass, 1 if any regress.
    local any_fail=0
    for cc in "${CROSS_CHECKS[@]}"; do
        local cc_model_dir cc_min_tok cc_tokens cc_k cc_desc
        cc_model_dir=$(python3 -c "import json; print(json.load(open('$cc'))['model_dir'])")
        cc_min_tok=$(python3 -c "import json; print(json.load(open('$cc'))['min_tok_sec'])")
        cc_tokens=$(python3 -c "import json; print(json.load(open('$cc')).get('tokens', 5))")
        cc_k=$(python3 -c "import json; print(json.load(open('$cc')).get('k', 8))")
        cc_desc=$(python3 -c "import json; print(json.load(open('$cc'))['description'])")

        if [[ ! -d "$cc_model_dir" ]]; then
            echo "[cross-check] SKIP: $cc_desc (model dir not found: $cc_model_dir)"
            continue
        fi

        echo "[cross-check] Running: $cc_desc"
        local output
        output=$(./orome --model "$cc_model_dir" --prompt "Hello" --tokens "$cc_tokens" --k "$cc_k" 2>&1)
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

    # Launch Claude Code session
    # --output-format stream-json: streams output incrementally (visible in real-time)
    # --dangerously-skip-permissions: no prompts (we trust the agent)
    # --max-turns: limit turns per session to avoid runaway
    claude --output-format stream-json \
        --verbose \
        --dangerously-skip-permissions \
        --max-turns 80 \
        -p "$PROMPT" \
        2>&1 | tee "$SESSION_LOG"

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
        BEST=$(grep "keep" "$EXPERIMENT_DIR/results.tsv" 2>/dev/null | sort -t$'\t' -k2 -rn | head -1 || echo "no results yet")
        echo "[runner] Current best: $BEST"
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
