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
#   experiments/<model>/errors.log            — Runner-level errors
#   experiments/<model>/logs/                 — Full session logs
#   experiments/<model>/results.tsv           — Current campaign results
#   experiments/<model>/results.historical.tsv — Older historical context
#   experiments/<model>/status.md             — Last session's handoff note
#
# Usage:
#   ./run_experiments.sh <model>                               # run forever with Claude
#   ./run_experiments.sh <model> --agent codex                # run forever with Codex
#   ./run_experiments.sh <model> --sessions 5                 # run 5 Claude sessions then stop
#   ./run_experiments.sh <model> --agent codex --sessions 5   # run 5 Codex sessions then stop
#
# Experiment targets are discovered from experiments/*/program.md.
# Current repo targets:
#   qwen35-9B    — Qwen3.5-9B dense GGUF optimization
#   qwen35-27B   — Qwen3.5-27B dense GGUF optimization
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
RESULTS_HISTORICAL_FILE="$EXPERIMENT_DIR/results.historical.tsv"
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

tracked_worktree_dirty() {
    ! git diff --quiet || ! git diff --cached --quiet
}

# Ensure we're on the right branch
if git rev-parse --git-dir > /dev/null 2>&1; then
    BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")
    if [[ "$BRANCH" != "$BRANCH_NAME" ]]; then
        echo "[runner] Switching to branch $BRANCH_NAME"
        git checkout -b "$BRANCH_NAME" 2>/dev/null || git checkout "$BRANCH_NAME"
    fi
    echo "[runner] On branch: $(git branch --show-current)"
fi

if tracked_worktree_dirty; then
    echo "[runner] FATAL: tracked worktree is dirty before starting. Commit, stash, or revert tracked changes first." | tee -a "$ERROR_LOG"
    exit 1
fi

# Safety check: verify the codebase builds before starting
echo "[runner] Pre-flight build check..."
if ! make clean && make 2>/dev/null; then
    echo "[runner] FATAL: Initial build failed. Fix before running." | tee -a "$ERROR_LOG"
    exit 1
fi
echo "[runner] Build OK."

validate_cross_check_config() {
    local cc="$1"
    python3 - "$cc" <<'PY'
import json
import sys
from pathlib import Path

path = sys.argv[1]

with open(path) as f:
    d = json.load(f)

missing = [k for k in ("model", "min_tok_sec", "description") if not d.get(k)]
if missing:
    print(f"missing required keys: {', '.join(missing)}")
    sys.exit(1)

if d.get("quality_check", True) is False:
    print("quality_check is disabled")
    sys.exit(1)

llama = d.get("llama_compare")
if llama is not None:
    if not isinstance(llama, dict):
        print("llama_compare must be an object")
        sys.exit(1)
    if llama.get("enabled", False):
        model_alias = str(llama.get("model_alias", "")).strip()
        if not model_alias:
            print("llama_compare.enabled requires model_alias")
            sys.exit(1)
        repo = str(llama.get("repo", "")).strip()
        if not repo:
            print("llama_compare.enabled requires repo")
            sys.exit(1)
        if not Path(repo).exists():
            print(f"llama_compare repo not found: {repo}")
            sys.exit(1)

        throughput = llama.get("throughput", {})
        if throughput is None:
            throughput = {}
        if not isinstance(throughput, dict):
            print("llama_compare.throughput must be an object")
            sys.exit(1)
        if int(throughput.get("tokens", 100)) <= 0:
            print("llama_compare.throughput.tokens must be > 0")
            sys.exit(1)
        if int(throughput.get("trials", 1)) <= 0:
            print("llama_compare.throughput.trials must be > 0")
            sys.exit(1)
        if int(throughput.get("warmup_runs", 0)) < 0:
            print("llama_compare.throughput.warmup_runs must be >= 0")
            sys.exit(1)
        if float(throughput.get("cooldown_sec", 0.0)) < 0:
            print("llama_compare.throughput.cooldown_sec must be >= 0")
            sys.exit(1)

        quality = llama.get("quality", {})
        if quality is None:
            quality = {}
        if not isinstance(quality, dict):
            print("llama_compare.quality must be an object")
            sys.exit(1)
        if int(quality.get("runs", 1)) <= 0:
            print("llama_compare.quality.runs must be > 0")
            sys.exit(1)
        if int(quality.get("min_orome_passes", 1)) < 0:
            print("llama_compare.quality.min_orome_passes must be >= 0")
            sys.exit(1)
        if int(quality.get("min_runs_passing", 1)) <= 0:
            print("llama_compare.quality.min_runs_passing must be > 0")
            sys.exit(1)

        source_hints = llama.get("source_hints", [])
        if source_hints is None:
            source_hints = []
        if not isinstance(source_hints, list):
            print("llama_compare.source_hints must be an array")
            sys.exit(1)

        cases_file = str(quality.get("cases_file", "")).strip()
        if cases_file and not Path(cases_file).exists():
            print(f"llama_compare.quality.cases_file not found: {cases_file}")
            sys.exit(1)
PY
}

llama_compare_enabled() {
    local cc="$1"
    python3 - "$cc" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    d = json.load(f)

cfg = d.get("llama_compare") or {}
print("yes" if cfg.get("enabled", False) else "no")
PY
}

build_llama_reference_prompt_section() {
    local cc="$1"
    python3 - "$cc" <<'PY'
import json
import sys
from pathlib import Path

with open(sys.argv[1]) as f:
    d = json.load(f)

cfg = d.get("llama_compare") or {}
if not cfg.get("enabled", False):
    raise SystemExit(0)

repo = Path(str(cfg.get("repo", ""))).resolve()
alias = str(cfg.get("model_alias", "")).strip()
hints = cfg.get("source_hints") or []

print("## Local Reference Implementation")
print()
print("- Use local `llama.cpp` as the parity target for this campaign.")
print(f"- Local repo: `{repo}`")
print(f"- Throughput compare: `python3 tools/compare_orome_llama.py --models {alias} --json`")
print(f"- Quality compare: `python3 tools/compare_orome_llama_quality.py --models {alias} --json`")
if hints:
    print("- Relevant local llama.cpp files:")
    for hint in hints:
        print(f"  - `{repo / hint}`")
PY
}

validate_cross_check_suite() {
    local suite_fail=0
    local count=0
    local d name cc reason

    for d in experiments/*/; do
        [[ -f "$d/program.md" ]] || continue
        count=$((count + 1))
        name="$(basename "$d")"
        cc="$d/cross_check.json"

        if [[ ! -f "$cc" ]]; then
            echo "[runner] FATAL: $name is missing cross_check.json" | tee -a "$ERROR_LOG"
            suite_fail=1
            continue
        fi

        if ! reason="$(validate_cross_check_config "$cc" 2>&1)"; then
            echo "[runner] FATAL: invalid cross_check.json for $name: $reason" | tee -a "$ERROR_LOG"
            suite_fail=1
        fi
    done

    if [[ "$count" -eq 0 ]]; then
        echo "[runner] FATAL: no experiment targets found under experiments/" | tee -a "$ERROR_LOG"
        exit 1
    fi

    if [[ "$suite_fail" -ne 0 ]]; then
        exit 1
    fi

    echo "[runner] Cross-check suite validated for $count experiment targets."
}

validate_cross_check_suite

session_log_failure_reason() {
    local log_file="$1"

    if rg -q '"subtype":"error_max_turns"' "$log_file"; then
        echo "agent hit max turns"
        return 0
    fi

    if rg -q '"is_error":true' "$log_file"; then
        echo "agent reported an error"
        return 0
    fi

    if rg -q '"type":"error"' "$log_file"; then
        echo "agent emitted an error event"
        return 0
    fi

    return 1
}

sanitize_summary_field() {
    printf '%s' "$1" | tr '\t\r\n' ' ' | sed 's/[[:space:]]\+/ /g; s/^ //; s/ $//'
}

declare -a CHECK_SUMMARIES=()

reset_check_summaries() {
    CHECK_SUMMARIES=()
}

record_check_summary() {
    local label status desc tok min_tok reply reasons
    label="$(sanitize_summary_field "${1:-}")"
    status="$(sanitize_summary_field "${2:-}")"
    desc="$(sanitize_summary_field "${3:-}")"
    tok="$(sanitize_summary_field "${4:-}")"
    min_tok="$(sanitize_summary_field "${5:-}")"
    reply="$(sanitize_summary_field "${6:-}")"
    reasons="$(sanitize_summary_field "${7:-}")"
    CHECK_SUMMARIES+=("${label}"$'\t'"${status}"$'\t'"${desc}"$'\t'"${tok}"$'\t'"${min_tok}"$'\t'"${reply}"$'\t'"${reasons}")
}

actualize_results_keep_commit() {
    local results_file="$1"
    local commit_sha="$2"
    [[ -f "$results_file" ]] || return 0

    python3 - "$results_file" "$commit_sha" <<'PY'
import csv
import io
import sys
from pathlib import Path

path = Path(sys.argv[1])
commit_sha = sys.argv[2]
short = commit_sha[:7]

with path.open(newline="") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

target_idx = None
for idx, row in enumerate(rows):
    if row.get("status") == "keep" and row.get("commit", "").endswith("+dirty"):
        target_idx = idx

if target_idx is None:
    raise SystemExit(0)

rows[target_idx]["commit"] = short

fieldnames = rows[0].keys() if rows else ["commit", "tok_sec", "ttft_ms", "proj_avg_ms", "status", "description"]
buf = io.StringIO()
writer = csv.DictWriter(buf, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
writer.writeheader()
writer.writerows(rows)
path.write_text(buf.getvalue())
PY
}

actualize_status_file() {
    local status_file="$1"
    local branch_name="$2"
    local commit_sha="$3"
    local session_note="${4:-}"
    local worktree_state="clean"
    local section

    if ! git diff --quiet || ! git diff --cached --quiet; then
        worktree_state="dirty"
    fi

    section="## Runner Validation

- Session commit: \`${commit_sha:0:7}\`
- Branch: \`${branch_name}\`
- Working tree after runner checks: \`${worktree_state}\`"

    if [[ -n "$session_note" ]]; then
        section="$section
- Runner note: ${session_note}"
    fi

    if [[ ${#CHECK_SUMMARIES[@]} -gt 0 ]]; then
        local entry label status desc tok min_tok reply reasons
        for entry in "${CHECK_SUMMARIES[@]}"; do
            IFS=$'\t' read -r label status desc tok min_tok reply reasons <<<"$entry"
            if [[ "$status" == "PASS" ]]; then
                section="$section
- ${label}: pass — ${desc}"
                [[ -n "$tok" && -n "$min_tok" ]] && section="$section (${tok} tok/s, min ${min_tok})"
                [[ -n "$reply" ]] && section="$section; reply: ${reply}"
            elif [[ "$status" == "SKIP" ]]; then
                section="$section
- ${label}: skip — ${desc}"
                [[ -n "$reasons" ]] && section="$section; ${reasons}"
            else
                section="$section
- ${label}: fail — ${desc}"
                [[ -n "$reasons" ]] && section="$section; reason: ${reasons}"
                [[ -n "$reply" ]] && section="$section; reply: ${reply}"
            fi
        done
    fi

    STATUS_SECTION="$section" python3 - "$status_file" <<'PY'
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
section = os.environ["STATUS_SECTION"].rstrip() + "\n"

text = path.read_text() if path.exists() else ""
marker = "## Runner Validation"
if marker in text:
    text = text.split(marker, 1)[0].rstrip() + "\n\n" + section
else:
    text = (text.rstrip() + "\n\n" + section) if text.strip() else section

path.write_text(text)
PY
}

# ---- Collect regression checks ----
# The current experiment can define a self-check in cross_check.json.
# Each OTHER experiment's cross_check.json defines a quick cross-model smoke check.
# After each successful session, we run the self-check first, then all cross-model checks.
SELF_CHECK="$EXPERIMENT_DIR/cross_check.json"
if [[ ! -f "$SELF_CHECK" ]]; then
    SELF_CHECK=""
fi

CROSS_CHECKS=()
for d in experiments/*/; do
    other="$(basename "$d")"
    [[ "$other" == "$MODEL" ]] && continue
    cc="$d/cross_check.json"
    [[ -f "$cc" ]] && CROSS_CHECKS+=("$cc")
done
if [[ -n "$SELF_CHECK" ]]; then
    echo "[runner] Self-check configured: $SELF_CHECK"
else
    echo "[runner] No self-check configured"
fi
LLAMA_COMPARE_CHECK=""
if [[ -n "$SELF_CHECK" && "$(llama_compare_enabled "$SELF_CHECK")" == "yes" ]]; then
    LLAMA_COMPARE_CHECK="$SELF_CHECK"
fi
if [[ -n "$LLAMA_COMPARE_CHECK" ]]; then
    echo "[runner] llama.cpp parity checks configured: $LLAMA_COMPARE_CHECK"
else
    echo "[runner] No llama.cpp parity checks configured"
fi
if [[ ${#CROSS_CHECKS[@]} -gt 0 ]]; then
    echo "[runner] Cross-model regression checks: ${CROSS_CHECKS[*]}"
else
    echo "[runner] No cross-model regression checks configured"
fi

best_results_row() {
    local results_file="$1"
    python3 - "$results_file" <<'PY'
import csv
import sys

path = sys.argv[1]

with open(path, newline="") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

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

run_check_config() {
    local cc="$1"
    local label="$2"
    local cc_model cc_min_tok cc_tokens cc_k cc_desc cc_prompt
    cc_model=$(python3 -c "import json; d=json.load(open('$cc')); print(d.get('model') or d.get('model_dir') or '')")
    cc_min_tok=$(python3 -c "import json; print(json.load(open('$cc'))['min_tok_sec'])")
    cc_tokens=$(python3 -c "import json; print(json.load(open('$cc')).get('tokens', 5))")
    cc_k=$(python3 -c "import json; print(json.load(open('$cc')).get('k', 8))")
    cc_desc=$(python3 -c "import json; print(json.load(open('$cc'))['description'])")
    cc_prompt=$(python3 -c "import json; print(json.load(open('$cc')).get('prompt', ''))")

    if [[ -z "$cc_model" ]]; then
        echo "[$label] SKIP: $cc_desc (no model configured: $cc)"
        record_check_summary "$label" "SKIP" "$cc_desc" "" "" "" "no model configured"
        return 0
    fi

    if [[ ! -e "$cc_model" ]]; then
        echo "[$label] SKIP: $cc_desc (model not found: $cc_model)"
        record_check_summary "$label" "SKIP" "$cc_desc" "" "" "" "model not found: $cc_model"
        return 0
    fi

    echo "[$label] Running: $cc_desc"
    local tmp_json tmp_err rc
    tmp_json=$(mktemp)
    tmp_err=$(mktemp)
    rc=0
    if [[ -n "$cc_prompt" ]]; then
        python3 tools/benchmark.py \
            --infer ./orome \
            --model "$cc_model" \
            --prompt "$cc_prompt" \
            --tokens "$cc_tokens" \
            --k "$cc_k" \
            --trials 1 \
            --warmup-runs 0 \
            --cooldown-sec 0 \
            --json \
            --quality-config "$cc" \
            >"$tmp_json" 2>"$tmp_err" || rc=$?
    else
        python3 tools/benchmark.py \
            --infer ./orome \
            --model "$cc_model" \
            --tokens "$cc_tokens" \
            --k "$cc_k" \
            --trials 1 \
            --warmup-runs 0 \
            --cooldown-sec 0 \
            --json \
            --quality-config "$cc" \
            >"$tmp_json" 2>"$tmp_err" || rc=$?
    fi

    if [[ ! -s "$tmp_json" ]]; then
        echo "[$label] FAIL: $cc_desc — benchmark produced no JSON (exit $rc)"
        echo "  stderr: $(tail -3 "$tmp_err" | tr '\n' ' ')"
        record_check_summary "$label" "FAIL" "$cc_desc" "" "$cc_min_tok" "" "benchmark produced no JSON (exit $rc)"
        rm -f "$tmp_json" "$tmp_err"
        return 1
    fi

    local parsed tok_sec quality_pass quality_reply quality_reasons
    parsed=$(python3 - "$tmp_json" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    d = json.load(f)

print(d.get("tok_sec_median", ""))
print("yes" if d.get("quality_pass") else "no")
print((d.get("quality_reply") or "").replace("\n", "\\n"))
print("; ".join(d.get("quality_reasons") or []))
PY
)
    tok_sec=$(printf '%s\n' "$parsed" | sed -n '1p')
    quality_pass=$(printf '%s\n' "$parsed" | sed -n '2p')
    quality_reply=$(printf '%s\n' "$parsed" | sed -n '3p')
    quality_reasons=$(printf '%s\n' "$parsed" | sed -n '4p')

    if [[ "$quality_pass" != "yes" ]]; then
        echo "[$label] FAIL: $cc_desc — quality gate failed"
        [[ -n "$quality_reasons" ]] && echo "  Reason: $quality_reasons"
        [[ -n "$quality_reply" ]] && echo "  Reply: $quality_reply"
        [[ -s "$tmp_err" ]] && echo "  stderr: $(tail -3 "$tmp_err" | tr '\n' ' ')"
        record_check_summary "$label" "FAIL" "$cc_desc" "$tok_sec" "$cc_min_tok" "$quality_reply" "$quality_reasons"
        rm -f "$tmp_json" "$tmp_err"
        return 1
    fi

    if [[ -z "$tok_sec" ]]; then
        echo "[$label] FAIL: $cc_desc — could not parse tok/s"
        [[ -s "$tmp_err" ]] && echo "  stderr: $(tail -3 "$tmp_err" | tr '\n' ' ')"
        record_check_summary "$label" "FAIL" "$cc_desc" "" "$cc_min_tok" "$quality_reply" "could not parse tok/s"
        rm -f "$tmp_json" "$tmp_err"
        return 1
    fi

    local passed
    passed=$(python3 -c "print('yes' if $tok_sec >= $cc_min_tok else 'no')")
    if [[ "$passed" == "yes" ]]; then
        echo "[$label] PASS: $cc_desc — ${tok_sec} tok/s (min: ${cc_min_tok}); reply: ${quality_reply}"
        record_check_summary "$label" "PASS" "$cc_desc" "$tok_sec" "$cc_min_tok" "$quality_reply" ""
        rm -f "$tmp_json" "$tmp_err"
        return 0
    fi

    echo "[$label] FAIL: $cc_desc — ${tok_sec} tok/s < ${cc_min_tok} min"
    [[ -n "$quality_reply" ]] && echo "  Reply: $quality_reply"
    record_check_summary "$label" "FAIL" "$cc_desc" "$tok_sec" "$cc_min_tok" "$quality_reply" "${tok_sec} tok/s < ${cc_min_tok} min"
    rm -f "$tmp_json" "$tmp_err"
    return 1
}

run_self_check() {
    [[ -n "$SELF_CHECK" ]] || return 0
    run_check_config "$SELF_CHECK" "self-check"
}

run_llama_compare_check() {
    local cc="$1"
    local values=()
    mapfile -t values < <(python3 - "$cc" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    d = json.load(f)

cfg = d.get("llama_compare") or {}
if not cfg.get("enabled", False):
    raise SystemExit(0)

throughput = cfg.get("throughput") or {}
quality = cfg.get("quality") or {}

print(str(cfg.get("model_alias", "")).strip())
print(str(cfg.get("repo", "")).strip())
print(int(throughput.get("tokens", 100)))
print(int(throughput.get("trials", 1)))
print(int(throughput.get("warmup_runs", 0)))
print(float(throughput.get("cooldown_sec", 0.0)))
print(str(throughput.get("min_ratio_orome_over_llama", "")).strip())
print(int(quality.get("runs", 1)))
print(int(quality.get("min_orome_passes", 1)))
print(int(quality.get("min_runs_passing", 1)))
print(str(quality.get("cases_file", "")).strip())
PY
)

    [[ ${#values[@]} -gt 0 ]] || return 0

    local model_alias="${values[0]}"
    local llama_repo="${values[1]}"
    local throughput_tokens="${values[2]}"
    local throughput_trials="${values[3]}"
    local throughput_warmup="${values[4]}"
    local throughput_cooldown="${values[5]}"
    local throughput_min_ratio="${values[6]}"
    local quality_runs="${values[7]}"
    local quality_min_orome_passes="${values[8]}"
    local quality_min_runs_passing="${values[9]}"
    local quality_cases_file="${values[10]}"

    local any_fail=0
    local tmp_json tmp_err rc parsed orome_tok llama_tok ratio desc reason

    echo "[llama-compare] Running throughput parity check..."
    tmp_json=$(mktemp)
    tmp_err=$(mktemp)
    rc=0
    python3 tools/compare_orome_llama.py \
        --models "$model_alias" \
        --tokens "$throughput_tokens" \
        --trials "$throughput_trials" \
        --warmup-runs "$throughput_warmup" \
        --cooldown-sec "$throughput_cooldown" \
        --llama-repo "$llama_repo" \
        --json \
        >"$tmp_json" 2>"$tmp_err" || rc=$?

    if [[ "$rc" -ne 0 || ! -s "$tmp_json" ]]; then
        echo "[llama-compare] FAIL: throughput compare could not complete"
        [[ -s "$tmp_err" ]] && echo "  stderr: $(tail -3 "$tmp_err" | tr '\n' ' ')"
        record_check_summary "llama-throughput" "FAIL" "llama.cpp throughput parity" "" "" "" "compare command failed"
        any_fail=1
    else
        parsed=$(python3 - "$tmp_json" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    d = json.load(f)
r = d["results"][0]
print(r["orome"]["tok_sec"])
print(r["llama"]["tok_sec"])
print(r.get("ratio_orome_over_llama", ""))
PY
)
        orome_tok=$(printf '%s\n' "$parsed" | sed -n '1p')
        llama_tok=$(printf '%s\n' "$parsed" | sed -n '2p')
        ratio=$(printf '%s\n' "$parsed" | sed -n '3p')
        desc="Orome ${orome_tok} tok/s vs llama.cpp ${llama_tok} tok/s (ratio ${ratio})"

        local throughput_ok="yes"
        if [[ -n "$throughput_min_ratio" ]]; then
            throughput_ok=$(python3 - "$ratio" "$throughput_min_ratio" <<'PY'
import sys
ratio = float(sys.argv[1] or 0.0)
floor = float(sys.argv[2])
print("yes" if ratio >= floor else "no")
PY
)
            desc="${desc}, min ratio ${throughput_min_ratio}"
        fi

        if [[ "$throughput_ok" == "yes" ]]; then
            echo "[llama-compare] PASS: $desc"
            record_check_summary "llama-throughput" "PASS" "$desc" "" "" "" ""
        else
            echo "[llama-compare] FAIL: $desc"
            reason="ratio ${ratio} < ${throughput_min_ratio}"
            record_check_summary "llama-throughput" "FAIL" "llama.cpp throughput parity" "" "" "" "$reason"
            any_fail=1
        fi
    fi
    rm -f "$tmp_json" "$tmp_err"

    echo "[llama-compare] Running quality parity check..."
    local pass_runs=0
    local run_idx=0
    local orome_scores=()
    local llama_scores=()
    for ((run_idx=1; run_idx<=quality_runs; run_idx++)); do
        tmp_json=$(mktemp)
        tmp_err=$(mktemp)
        rc=0
        if [[ -n "$quality_cases_file" ]]; then
            python3 tools/compare_orome_llama_quality.py \
                --models "$model_alias" \
                --llama-repo "$llama_repo" \
                --cases-file "$quality_cases_file" \
                --json \
                >"$tmp_json" 2>"$tmp_err" || rc=$?
        else
            python3 tools/compare_orome_llama_quality.py \
                --models "$model_alias" \
                --llama-repo "$llama_repo" \
                --json \
                >"$tmp_json" 2>"$tmp_err" || rc=$?
        fi

        if [[ "$rc" -ne 0 || ! -s "$tmp_json" ]]; then
            echo "[llama-compare] FAIL: quality compare run ${run_idx} could not complete"
            [[ -s "$tmp_err" ]] && echo "  stderr: $(tail -3 "$tmp_err" | tr '\n' ' ')"
            record_check_summary "llama-quality" "FAIL" "llama.cpp quality parity" "" "" "" "quality compare command failed"
            rm -f "$tmp_json" "$tmp_err"
            return 1
        fi

        parsed=$(python3 - "$tmp_json" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    d = json.load(f)
r = d["results"][0]
print(r["orome_passes"])
print(r["llama_passes"])
print(r["case_count"])
PY
)
        local orome_passes llama_passes case_count
        orome_passes=$(printf '%s\n' "$parsed" | sed -n '1p')
        llama_passes=$(printf '%s\n' "$parsed" | sed -n '2p')
        case_count=$(printf '%s\n' "$parsed" | sed -n '3p')
        orome_scores+=("${orome_passes}/${case_count}")
        llama_scores+=("${llama_passes}/${case_count}")
        local quality_run_ok
        quality_run_ok=$(python3 - "$orome_passes" "$quality_min_orome_passes" <<'PY'
import sys
print("yes" if int(sys.argv[1]) >= int(sys.argv[2]) else "no")
PY
)
        if [[ "$quality_run_ok" == "yes" ]]; then
            pass_runs=$((pass_runs + 1))
        fi
        rm -f "$tmp_json" "$tmp_err"
    done

    desc="Orome per-run cases: $(IFS=', '; echo "${orome_scores[*]}"); llama.cpp per-run cases: $(IFS=', '; echo "${llama_scores[*]}")"
    if [[ "$pass_runs" -ge "$quality_min_runs_passing" ]]; then
        echo "[llama-compare] PASS: ${pass_runs}/${quality_runs} runs met the quality floor; ${desc}"
        record_check_summary "llama-quality" "PASS" "${pass_runs}/${quality_runs} runs met the quality floor; ${desc}" "" "" "" ""
    else
        echo "[llama-compare] FAIL: ${pass_runs}/${quality_runs} runs met the quality floor (need ${quality_min_runs_passing}); ${desc}"
        reason="${pass_runs}/${quality_runs} runs met the quality floor (need ${quality_min_runs_passing})"
        record_check_summary "llama-quality" "FAIL" "llama.cpp quality parity" "" "" "" "${reason}; ${desc}"
        any_fail=1
    fi

    return "$any_fail"
}

run_cross_checks() {
    # Run each cross-model check. Returns 0 if all pass, 1 if any regress.
    local any_fail=0
    for cc in "${CROSS_CHECKS[@]}"; do
        if ! run_check_config "$cc" "cross-check"; then
            any_fail=1
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

    # Build the prompt: program.md + current status + current campaign results
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

## Current Campaign Results

\`\`\`
$(cat "$EXPERIMENT_DIR/results.tsv")
\`\`\`"
    fi

    if [[ -f "$RESULTS_HISTORICAL_FILE" ]]; then
        PROMPT="$PROMPT

---

## Historical Context

Older 35B packed-format results live in \`experiments/$MODEL/results.historical.tsv\`.
Use them for hypotheses and prior art, not as the live GGUF baseline."
    fi

    # Tell the agent where its experiment files live
    PROMPT="$PROMPT

---

## Experiment Paths

- Results: \`experiments/$MODEL/results.tsv\`
- Historical results: \`experiments/$MODEL/results.historical.tsv\`
- Status: \`experiments/$MODEL/status.md\`
- Bench errors: \`experiments/$MODEL/bench_err.txt\`

## Runner Post-Processing

- After a successful session, the runner normalizes the retained \`keep\` row in \`results.tsv\` to the final commit hash.
- The runner also refreshes a \`## Runner Validation\` section in \`status.md\` with authoritative self-check and cross-check outcomes.
- Do not guess whether runner-managed validation did or did not run; leave that section to the runner."

    if [[ -n "$LLAMA_COMPARE_CHECK" ]]; then
        LLAMA_PROMPT_SECTION="$(build_llama_reference_prompt_section "$LLAMA_COMPARE_CHECK")"
        if [[ -n "$LLAMA_PROMPT_SECTION" ]]; then
            PROMPT="$PROMPT

---

$LLAMA_PROMPT_SECTION"
        fi
    fi

    # Snapshot the codebase state before the session (for recovery)
    GIT_HEAD_BEFORE=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
    reset_check_summaries

    # Launch agent session and stream output to the log in real time.
    run_agent_session "$PROMPT" 2>&1 | tee "$SESSION_LOG"

    EXIT_CODE=${PIPESTATUS[0]}
    GIT_HEAD_AFTER=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
    SESSION_FAILURE_REASON=""
    if [[ "$EXIT_CODE" -eq 0 ]]; then
        if SESSION_FAILURE_REASON="$(session_log_failure_reason "$SESSION_LOG")"; then
            EXIT_CODE=86
        else
            SESSION_FAILURE_REASON=""
        fi
    fi
    echo ""
    echo "[runner] Session $SESSION_NUM finished (exit code: $EXIT_CODE) at $(date '+%H:%M:%S')"
    if [[ -n "$SESSION_FAILURE_REASON" ]]; then
        echo "[runner] Session log indicates failure: $SESSION_FAILURE_REASON"
    fi

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
            [[ -n "$SESSION_FAILURE_REASON" ]] && echo "  Failure reason: $SESSION_FAILURE_REASON"
            echo "  Tail of session log:"
            tail -20 "$SESSION_LOG" 2>/dev/null | sed 's/^/    /'
            echo "---"
        } >> "$ERROR_LOG"
        echo "[runner] ERROR logged to $ERROR_LOG (consecutive: $CONSECUTIVE_FAILURES/$MAX_CONSECUTIVE_FAILURES)"

        if [[ "$GIT_HEAD_BEFORE" != "$GIT_HEAD_AFTER" ]] || tracked_worktree_dirty; then
            echo "[runner] Reverting failed session to $GIT_HEAD_BEFORE" | tee -a "$ERROR_LOG"
            git reset --hard "$GIT_HEAD_BEFORE" 2>/dev/null || true
            actualize_status_file "$EXPERIMENT_DIR/status.md" "$(git branch --show-current)" "$GIT_HEAD_BEFORE" "Session failed and runner reverted session changes."
        fi

        # Circuit breaker
        if [[ "$CONSECUTIVE_FAILURES" -ge "$MAX_CONSECUTIVE_FAILURES" ]]; then
            echo "[runner] CIRCUIT BREAKER: $MAX_CONSECUTIVE_FAILURES consecutive failures. Stopping." | tee -a "$ERROR_LOG"
            break
        fi
    else
        CONSECUTIVE_FAILURES=0  # reset on success
        echo "Session $SESSION_NUM OK (exit=0) at $(date '+%H:%M:%S') git=$GIT_HEAD_AFTER" >> "$ERROR_LOG"

        # Self-check and cross-model regression checks (only after successful sessions that changed code)
        if [[ "$GIT_HEAD_BEFORE" != "$GIT_HEAD_AFTER" ]]; then
            regression_fail=0
            if [[ -n "$SELF_CHECK" ]]; then
                echo "[runner] Running self-check..."
                if ! run_self_check; then
                    regression_fail=1
                fi
            fi
            if [[ "$regression_fail" -eq 0 && -n "$LLAMA_COMPARE_CHECK" ]]; then
                echo "[runner] Running llama.cpp parity checks..."
                if ! run_llama_compare_check "$LLAMA_COMPARE_CHECK"; then
                    regression_fail=1
                fi
            fi
            if [[ "$regression_fail" -eq 0 && ${#CROSS_CHECKS[@]} -gt 0 ]]; then
                echo "[runner] Running cross-model regression checks..."
                if ! run_cross_checks; then
                    regression_fail=1
                fi
            fi
            if [[ "$regression_fail" -ne 0 ]]; then
                {
                    echo ""
                    echo "--- Session $SESSION_NUM REGRESSION at $(date '+%Y-%m-%d %H:%M:%S') ---"
                    echo "  Model: $MODEL"
                    echo "  Git before: $GIT_HEAD_BEFORE"
                    echo "  Git after:  $GIT_HEAD_AFTER"
                    echo "  Regression or quality gate failure detected. Reverting to $GIT_HEAD_BEFORE."
                    echo "---"
                } >> "$ERROR_LOG"
                echo "[runner] REGRESSION detected. Reverting session commits..."
                git reset --hard "$GIT_HEAD_BEFORE" 2>/dev/null || true
                echo "[runner] Reverted to $GIT_HEAD_BEFORE"
                actualize_status_file "$EXPERIMENT_DIR/status.md" "$(git branch --show-current)" "$GIT_HEAD_BEFORE" "Session changes were reverted after runner validation failed."
            else
                actualize_results_keep_commit "$EXPERIMENT_DIR/results.tsv" "$GIT_HEAD_AFTER"
                actualize_status_file "$EXPERIMENT_DIR/status.md" "$(git branch --show-current)" "$GIT_HEAD_AFTER" ""
            fi
        else
            if tracked_worktree_dirty; then
                echo "[runner] Session made no new commit but left tracked changes. Reverting to $GIT_HEAD_BEFORE."
                git reset --hard "$GIT_HEAD_BEFORE" 2>/dev/null || true
                actualize_status_file "$EXPERIMENT_DIR/status.md" "$(git branch --show-current)" "$GIT_HEAD_BEFORE" "Session made no new commit; runner reverted leftover tracked changes."
            else
                actualize_status_file "$EXPERIMENT_DIR/status.md" "$(git branch --show-current)" "$GIT_HEAD_AFTER" "Session made no new commit; runner validation did not need to run."
            fi
        fi
    fi

    # Show current best if results exist
    if [[ -f "$EXPERIMENT_DIR/results.tsv" ]]; then
        BEST=$(best_results_row "$EXPERIMENT_DIR/results.tsv" || echo "no results yet")
        echo "[runner] Current best: $BEST"
        if [[ -f "$RESULTS_HISTORICAL_FILE" ]]; then
            HISTORICAL_BEST=$(best_results_row "$RESULTS_HISTORICAL_FILE" || true)
            if [[ -n "$HISTORICAL_BEST" ]]; then
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
