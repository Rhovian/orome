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
#   errors.log       — Runner-level errors (session crashes, non-zero exits)
#   logs/             — Full session logs (one per session)
#   results.tsv       — All experiment results (kept, discarded, crashed)
#   status.md         — Last session's handoff note
#
# Usage:
#   ./run_experiments.sh              # run forever
#   ./run_experiments.sh --sessions 5 # run 5 sessions then stop
#
# To stop: Ctrl+C or kill this script. Current experiment will finish.

set -euo pipefail

# Ensure PATH includes common install locations
export PATH="$HOME/.local/bin:/opt/homebrew/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MAX_SESSIONS="${1:-0}"  # 0 = unlimited
MAX_CONSECUTIVE_FAILURES=3  # circuit breaker: stop if N sessions fail in a row
SESSION_NUM=0
CONSECUTIVE_FAILURES=0
LOG_DIR="$SCRIPT_DIR/logs"
ERROR_LOG="$SCRIPT_DIR/errors.log"
mkdir -p "$LOG_DIR"

# Trap Ctrl+C gracefully
trap 'echo ""; echo "[runner] Caught interrupt. Waiting for current session to finish..."; STOP=1' INT
STOP=0

echo "============================================"
echo "  orome autonomous experiment runner"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Working dir: $SCRIPT_DIR"
echo "  Max sessions: ${MAX_SESSIONS:-unlimited}"
echo "  Error log: $ERROR_LOG"
echo "============================================"
echo "" | tee -a "$ERROR_LOG"
echo "=== Runner started $(date '+%Y-%m-%d %H:%M:%S') ===" >> "$ERROR_LOG"

# Ensure we're on the right branch
if git rev-parse --git-dir > /dev/null 2>&1; then
    BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")
    if [[ "$BRANCH" != autoresearch/orome* ]]; then
        echo "[runner] Creating branch autoresearch/orome from current HEAD"
        git checkout -b "autoresearch/orome" 2>/dev/null || git checkout "autoresearch/orome"
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
    echo "[runner] === Session $SESSION_NUM starting at $(date '+%H:%M:%S') ==="
    echo "[runner] Log: $SESSION_LOG"

    # Build the prompt: program.md + current status
    PROMPT="$(cat program.md)"
    if [[ -f status.md ]]; then
        PROMPT="$PROMPT

---

## Current Status (from previous session)

$(cat status.md)"
    fi

    if [[ -f results.tsv ]]; then
        PROMPT="$PROMPT

---

## Experiment History

\`\`\`
$(cat results.tsv)
\`\`\`"
    fi

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
            git checkout -- src/ include/ Makefile 2>/dev/null || true
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
    fi

    # Show current best if results exist
    if [[ -f results.tsv ]]; then
        BEST=$(grep "keep" results.tsv 2>/dev/null | sort -t$'\t' -k2 -rn | head -1 || echo "no results yet")
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
echo "[runner] Total sessions: $SESSION_NUM"
echo "[runner] Results in: $SCRIPT_DIR/results.tsv"
echo "[runner] Errors in: $ERROR_LOG"
echo "[runner] Logs in: $LOG_DIR/"
echo "=== Runner finished $(date '+%Y-%m-%d %H:%M:%S'), $SESSION_NUM sessions ===" >> "$ERROR_LOG"
if [[ -f status.md ]]; then
    echo ""
    echo "[runner] Final status:"
    cat status.md
fi
