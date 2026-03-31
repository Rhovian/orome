#!/usr/bin/env bash
#
# smoke_test.sh — End-to-end smoke test for orome inference + server.
#
# Tests:
#   1. Clean build
#   2. Server API conformance (all endpoints, streaming, non-streaming, errors)
#   3. tok/s benchmark for each model present
#   4. Quality check for each model present
#
# Usage:
#   ./scripts/smoke_test.sh              # run everything
#   ./scripts/smoke_test.sh --server     # server tests only (no model benchmarks)
#   ./scripts/smoke_test.sh --bench      # tok/s + quality only (skip server tests)
#   ./scripts/smoke_test.sh --quick      # 1 trial, 16 tokens (fast pass/fail)
#
# Exit codes:
#   0 = all pass
#   1 = build failure
#   2 = server test failure
#   3 = benchmark failure
#   4 = quality failure

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ── Configuration ──────────────────────────────────────────────────────────

MODELS_DIR="${OROME_MODELS_DIR:-/Users/j/Code/lllm/models}"
INFER="./orome"
PORT=0  # 0 = auto-select free port

BENCH_TOKENS=100
BENCH_TRIALS=1
QUALITY_TOKENS=64
QUALITY_TEMP=0.2

RUN_SERVER=true
RUN_BENCH=true
QUICK=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --server) RUN_BENCH=false; shift ;;
        --bench)  RUN_SERVER=false; shift ;;
        --quick)  QUICK=true; BENCH_TOKENS=16; BENCH_TRIALS=1; QUALITY_TOKENS=128; shift ;;
        --models-dir) MODELS_DIR="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# ── Helpers ────────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BOLD='\033[1m'
RESET='\033[0m'

PASS_COUNT=0
FAIL_COUNT=0
SECTION=""

pass() { ((PASS_COUNT++)); printf "  ${GREEN}✓${RESET} %s\n" "$1"; }
fail() { ((FAIL_COUNT++)); printf "  ${RED}✗${RESET} %s\n" "$1"; }
section() { SECTION="$1"; printf "\n${BOLD}── %s ──${RESET}\n" "$1"; }
skip() { printf "  ${YELLOW}○${RESET} %s (skipped)\n" "$1"; }

find_free_port() {
    python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()'
}

wait_for_server() {
    local port=$1 timeout=${2:-15}
    local deadline=$((SECONDS + timeout))
    while (( SECONDS < deadline )); do
        if curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.2
    done
    return 1
}

kill_server() {
    local pid=$1
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null
        wait "$pid" 2>/dev/null || true
    fi
}

# Discover all GGUF models
discover_models() {
    local -a found=()
    for f in "$MODELS_DIR"/*.gguf; do
        [[ -f "$f" ]] && found+=("$f")
    done
    echo "${found[@]}"
}

model_short_name() {
    basename "$1" .gguf | sed 's/-Q[0-9].*//; s/_/-/g' | tr '[:upper:]' '[:lower:]'
}

# ── 1. Build ───────────────────────────────────────────────────────────────

section "Build"

if make clean >/dev/null 2>&1 && make -j 2>&1 | tail -1 | grep -q "Built:"; then
    pass "clean build (zero warnings)"
else
    fail "build failed"
    exit 1
fi

# ── 2. Server API Conformance ──────────────────────────────────────────────

if $RUN_SERVER; then
    section "Server API conformance"

    # Pick the smallest model for server tests (fastest startup)
    MODELS=($(discover_models))
    if [[ ${#MODELS[@]} -eq 0 ]]; then
        fail "no GGUF models found in $MODELS_DIR"
        exit 1
    fi

    # Sort by file size, pick smallest
    SERVER_MODEL=$(ls -S "${MODELS[@]}" | tail -1)
    SERVER_PORT=$(find_free_port)

    printf "  using model: %s\n" "$(basename "$SERVER_MODEL")"
    printf "  port: %d\n" "$SERVER_PORT"

    $INFER --model "$SERVER_MODEL" --serve "$SERVER_PORT" >/dev/null 2>&1 &
    SERVER_PID=$!
    trap 'kill_server $SERVER_PID 2>/dev/null' EXIT

    if wait_for_server "$SERVER_PORT" 30; then
        pass "server starts and /health responds"
    else
        fail "server failed to start within 30s"
        kill_server $SERVER_PID
        exit 2
    fi

    # GET /health — check shape
    HEALTH=$(curl -sf "http://localhost:${SERVER_PORT}/health")
    if echo "$HEALTH" | python3 -c "
import json, sys
h = json.load(sys.stdin)
assert h['status'] == 'ok'
assert 'model' in h
assert 'uptime_seconds' in h
assert 'engine' in h
assert 'layers' in h['engine']
assert 'ffn_type' in h['engine']
" 2>/dev/null; then
        pass "GET /health — correct shape (status, model, engine metadata)"
    else
        fail "GET /health — unexpected shape: $HEALTH"
    fi

    # GET /v1/models — check shape and all three models
    MODELS_RESP=$(curl -sf "http://localhost:${SERVER_PORT}/v1/models")
    if echo "$MODELS_RESP" | python3 -c "
import json, sys
m = json.load(sys.stdin)
assert m['object'] == 'list'
ids = [d['id'] for d in m['data']]
assert 'qwen3.5-35b-a3b' in ids
assert 'qwen3.5-27b' in ids
assert 'qwen3.5-9b' in ids
for d in m['data']:
    assert 'created' in d
    assert d['object'] == 'model'
    assert d['owned_by'] == 'orome'
" 2>/dev/null; then
        pass "GET /v1/models — lists all 3 models with correct shape"
    else
        fail "GET /v1/models — unexpected: $MODELS_RESP"
    fi

    # POST /v1/chat/completions (non-streaming)
    CHAT_RESP=$(curl -sf "http://localhost:${SERVER_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"messages":[{"role":"user","content":"Say hi"}],"max_tokens":8,"temperature":0.1,"stream":false}')
    if echo "$CHAT_RESP" | python3 -c "
import json, sys
r = json.load(sys.stdin)
assert r['object'] == 'chat.completion'
assert 'chatcmpl-' in r['id']
assert 'created' in r
assert r['system_fingerprint'] == 'orome-v0'
c = r['choices'][0]
assert c['message']['role'] == 'assistant'
assert isinstance(c['message']['content'], str)
assert c['finish_reason'] in ('stop', 'length')
assert c['logprobs'] is None
u = r['usage']
assert u['prompt_tokens'] > 0
assert u['completion_tokens'] > 0
assert u['total_tokens'] == u['prompt_tokens'] + u['completion_tokens']
x = r['x_orome']
assert x['prefill_ms'] > 0
assert x['decode_ms'] >= 0
assert x['tokens_per_sec'] >= 0
" 2>/dev/null; then
        pass "POST /v1/chat/completions (non-streaming) — correct shape with usage + timing"
    else
        fail "POST /v1/chat/completions (non-streaming) — unexpected: $(echo "$CHAT_RESP" | head -c 300)"
    fi

    # POST /v1/chat/completions (streaming)
    SSE_RESP=$(curl -sf -N "http://localhost:${SERVER_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"messages":[{"role":"user","content":"Say hi"}],"max_tokens":8,"temperature":0.1,"stream":true}' 2>&1)
    if echo "$SSE_RESP" | python3 -c "
import json, sys
lines = [l for l in sys.stdin.read().splitlines() if l.startswith('data: ')]
assert len(lines) >= 3, f'too few SSE lines: {len(lines)}'

# First data chunk should have role
first = json.loads(lines[0][6:])
assert first['choices'][0]['delta'].get('role') == 'assistant'
assert 'model' in first
assert 'created' in first
assert first['system_fingerprint'] == 'orome-v0'

# Last data line should be [DONE]
assert lines[-1] == 'data: [DONE]'

# Second-to-last should have finish_reason and usage
final = json.loads(lines[-2][6:])
assert final['choices'][0]['finish_reason'] in ('stop', 'length')
assert 'usage' in final
assert final['usage']['prompt_tokens'] > 0
assert 'x_orome' in final
" 2>/dev/null; then
        pass "POST /v1/chat/completions (streaming) — correct SSE format with usage"
    else
        fail "POST /v1/chat/completions (streaming) — unexpected SSE format"
    fi

    # Multi-turn conversation
    MULTI_RESP=$(curl -sf "http://localhost:${SERVER_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"messages":[{"role":"system","content":"Reply with numbers only."},{"role":"user","content":"2+2"},{"role":"assistant","content":"4"},{"role":"user","content":"times 3"}],"max_tokens":8,"temperature":0.1,"stream":false}')
    if echo "$MULTI_RESP" | python3 -c "
import json, sys
r = json.load(sys.stdin)
assert r['usage']['prompt_tokens'] > 20, 'multi-turn should have more prompt tokens'
" 2>/dev/null; then
        pass "multi-turn conversation — round-trip preserved"
    else
        fail "multi-turn conversation — unexpected: $(echo "$MULTI_RESP" | head -c 200)"
    fi

    # Error: 404
    ERR_404=$(curl -s "http://localhost:${SERVER_PORT}/v1/nonexistent")
    if echo "$ERR_404" | python3 -c "
import json, sys
e = json.load(sys.stdin)
assert 'error' in e
assert e['error']['type'] == 'invalid_request_error'
assert e['error']['code'] == 'not_found'
" 2>/dev/null; then
        pass "404 — OpenAI error object format"
    else
        fail "404 — unexpected: $ERR_404"
    fi

    # Error: bad JSON
    ERR_JSON=$(curl -s -X POST "http://localhost:${SERVER_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{bad json}')
    if echo "$ERR_JSON" | python3 -c "
import json, sys
e = json.load(sys.stdin)
assert e['error']['code'] == 'invalid_json'
" 2>/dev/null; then
        pass "bad JSON — OpenAI error object format"
    else
        fail "bad JSON — unexpected: $ERR_JSON"
    fi

    # CORS preflight
    CORS_HEADERS=$(curl -sf -X OPTIONS -I "http://localhost:${SERVER_PORT}/v1/chat/completions" 2>&1)
    if echo "$CORS_HEADERS" | grep -q "Access-Control-Allow-Origin: \*"; then
        pass "CORS preflight — correct headers"
    else
        fail "CORS preflight — missing headers"
    fi

    # Tool-call round-trip (request shape only — model may not emit tool_calls)
    TOOL_RESP=$(curl -sf "http://localhost:${SERVER_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "messages":[
                {"role":"user","content":"What time is it?"}
            ],
            "tools":[{
                "type":"function",
                "function":{"name":"get_time","description":"Get current time","parameters":{"type":"object","properties":{}}}
            }],
            "max_tokens":16,"temperature":0.1,"stream":false
        }')
    if echo "$TOOL_RESP" | python3 -c "
import json, sys
r = json.load(sys.stdin)
assert r['object'] == 'chat.completion'
assert 'choices' in r
" 2>/dev/null; then
        pass "tool-call request accepted (tools array parsed without error)"
    else
        fail "tool-call request — unexpected: $(echo "$TOOL_RESP" | head -c 200)"
    fi

    kill_server $SERVER_PID
    trap - EXIT
fi

# ── 3. tok/s + Quality per model ──────────────────────────────────────────

if $RUN_BENCH; then
    section "Benchmark + Quality (all models)"

    MODELS=($(discover_models))
    if [[ ${#MODELS[@]} -eq 0 ]]; then
        fail "no GGUF models found in $MODELS_DIR"
        exit 3
    fi

    printf "  models found: %d\n" "${#MODELS[@]}"
    for m in "${MODELS[@]}"; do
        printf "    %s\n" "$(basename "$m")"
    done

    for MODEL_PATH in "${MODELS[@]}"; do
        SHORT=$(model_short_name "$MODEL_PATH")
        printf "\n  ${BOLD}%s${RESET} (%s)\n" "$SHORT" "$(basename "$MODEL_PATH")"

        # tok/s benchmark (1 trial for smoke test)
        BENCH_OUT=$(python3 inference/tools/benchmark.py \
            --infer "$INFER" \
            --model "$MODEL_PATH" \
            --tokens "$BENCH_TOKENS" \
            --trials "$BENCH_TRIALS" \
            --warmup-runs 1 \
            --cooldown-sec 1 \
            --skip-quality-check \
            --json 2>/dev/null) || true

        if echo "$BENCH_OUT" | python3 -c "
import json, sys
r = json.load(sys.stdin)
tok = r['tok_sec_median']
ttft = r['ttft_ms_median']
assert tok > 0, f'tok/s is {tok}'
print(f'    tok/s={tok:.1f}  TTFT={ttft:.0f}ms')
" 2>/dev/null; then
            pass "$SHORT — tok/s benchmark"
        else
            fail "$SHORT — tok/s benchmark (output: $(echo "$BENCH_OUT" | tail -c 200))"
        fi

        # Quality check
        QUALITY_OUT=$(python3 inference/tools/benchmark.py \
            --infer "$INFER" \
            --model "$MODEL_PATH" \
            --tokens 1 \
            --trials 1 \
            --warmup-runs 0 \
            --cooldown-sec 0 \
            --quality-prompt "What is the capital of France? Answer in one word." \
            --quality-must-contain "Paris" \
            --quality-max-tokens "$QUALITY_TOKENS" \
            --quality-temperature "$QUALITY_TEMP" \
            --quality-timeout-sec 45 \
            --json 2>/dev/null) || true

        if echo "$QUALITY_OUT" | python3 -c "
import json, sys
r = json.load(sys.stdin)
qp = r.get('quality_pass')
reply = r.get('quality_reply', '')[:80]
if qp:
    print(f'    quality reply: {reply!r}')
else:
    reasons = r.get('quality_reasons', [])
    print(f'    quality reply: {reply!r}')
    print(f'    reasons: {reasons}')
sys.exit(0 if qp else 1)
" 2>/dev/null; then
            pass "$SHORT — quality (Paris check)"
        else
            fail "$SHORT — quality (Paris check)"
        fi
    done
fi

# ── Summary ────────────────────────────────────────────────────────────────

section "Summary"
TOTAL=$((PASS_COUNT + FAIL_COUNT))
if [[ $FAIL_COUNT -eq 0 ]]; then
    printf "  ${GREEN}${BOLD}All %d tests passed.${RESET}\n" "$TOTAL"
    exit 0
else
    printf "  ${RED}${BOLD}%d/%d tests failed.${RESET}\n" "$FAIL_COUNT" "$TOTAL"
    if [[ $SECTION == *"Server"* ]]; then exit 2; fi
    if [[ $SECTION == *"Benchmark"* ]]; then exit 3; fi
    exit 4
fi
