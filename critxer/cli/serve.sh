#!/usr/bin/env bash
# Serve pilot auditor models via vLLM as OpenAI-compatible endpoints.
#
# Assumes a single node with the venvs laid out as $ROOT/.venv/{research,vllm}; set CRITXER_ROOT
# if yours differ. GPU choice is a parameter because node availability shifts between runs:
#   QWEN_GPU / GEMMA_GPU  (default 0 and 1)
# If the node is shared, check what holds a GPU before taking it.
#
# Usage:
#   critxer/cli/serve.sh up        # serve both models
#   critxer/cli/serve.sh qwen27b   # the dense 27B
#   critxer/cli/serve.sh injector  # the fault-injector model
#   critxer/cli/serve.sh down      # stop servers
set -euo pipefail

ROOT=${CRITXER_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
VLLM_PY="$ROOT/.venv/vllm/bin"
LOGS="$ROOT/temp/logs/vllm"
# PIDs of servers *this script* started, so `down` never touches anyone else's.
SERVERS_PIDFILE="$ROOT/temp/cvf_vllm_servers.pid"

QWEN_GPU="${QWEN_GPU:-0}"
# Third model, used ONLY as the fault injector: neither auditor may write the faults it audits,
# or self-attribution contaminates the effect under study.
INJECTOR_GPU="${INJECTOR_GPU:-0}"
INJECTOR_PORT="${INJECTOR_PORT:-9023}"
GEMMA_GPU="${GEMMA_GPU:-1}"
QWEN_PORT="${QWEN_PORT:-9021}"
GEMMA_PORT="${GEMMA_PORT:-9022}"

# Thinking is disabled deliberately -- variable-length reasoning traces would break the
# identical-output/identical-budget invariant the whole study rests on. Both families enable
# reasoning by DEFAULT, so these flags are required, not optional.
COMMON=(--async-scheduling --max-model-len 16384 --gpu-memory-utilization 0.90
        --language-model-only --default-chat-template-kwargs '{"enable_thinking": false}')

mkdir -p "$LOGS"

case "${1:?usage: serve.sh {up|qwen27b|injector|down}}" in
down)
    # Kill ONLY the servers this script started. `pkill -f "vllm serve"` was used here
    # before and it killed a concurrent agent's server on another GPU -- it matches every
    # vLLM process on a shared node, not just ours. Never pattern-kill on this box.
    if [[ ! -f "$SERVERS_PIDFILE" ]]; then
        echo "no $SERVERS_PIDFILE -- nothing this script started is recorded; refusing to guess" >&2
        exit 1
    fi
    while read -r pid; do
        [[ -n "$pid" ]] || continue
        if kill "$pid" 2>/dev/null; then echo "stopped $pid"; else echo "$pid already gone"; fi
    done < "$SERVERS_PIDFILE"
    rm -f "$SERVERS_PIDFILE"
    ;;
up)
    # Qwen3.6 MoE -- triton backends + capped seqs per the known-good config.
    CUDA_VISIBLE_DEVICES="$QWEN_GPU" VLLM_USE_DEEP_GEMM=0 VLLM_USE_FLASHINFER_SAMPLER=0 \
    nohup "$VLLM_PY/vllm" serve Qwen/Qwen3.6-35B-A3B \
        --enable-expert-parallel --port "$QWEN_PORT" \
        --moe-backend triton --gdn-prefill-backend triton --max-num-seqs 32 \
        --reasoning-parser qwen3 "${COMMON[@]}" \
        > "$LOGS/qwen36_serve.log" 2>&1 &
    echo $! > "$SERVERS_PIDFILE"

    # Gemma 4 MoE.
    CUDA_VISIBLE_DEVICES="$GEMMA_GPU" VLLM_USE_DEEP_GEMM=0 VLLM_USE_FLASHINFER_SAMPLER=0 \
    nohup "$VLLM_PY/vllm" serve google/gemma-4-26B-A4B-it \
        --enable-expert-parallel --port "$GEMMA_PORT" \
        --reasoning-parser gemma4 "${COMMON[@]}" \
        > "$LOGS/gemma4_serve.log" 2>&1 &
    echo $! >> "$SERVERS_PIDFILE"

    echo "serving: qwen3.6 :$QWEN_PORT (gpu $QWEN_GPU), gemma4 :$GEMMA_PORT (gpu $GEMMA_GPU)"
    ;;
qwen27b)
    # Dense 27B. Two flags are load-bearing and were both found the hard way:
    # VLLM_USE_FLASHINFER_SAMPLER=0 because the dense/hybrid path JITs through FlashInfer and
    # needs nvcc, which this image lacks; --max-num-seqs 256 because the dense model has KV
    # headroom the MoE does not and 32 leaves the GPU idle at this batch size.
    CUDA_VISIBLE_DEVICES="${QWEN27_GPU:-1}" VLLM_USE_DEEP_GEMM=0 VLLM_USE_FLASHINFER_SAMPLER=0 \
    nohup "$VLLM_PY/vllm" serve Qwen/Qwen3.6-27B \
        --port "${QWEN27_PORT:-9024}" --max-num-seqs 256 \
        --gdn-prefill-backend triton --reasoning-parser qwen3 "${COMMON[@]}" \
        > "$LOGS/qwen27b_serve.log" 2>&1 &
    echo $! >> "$SERVERS_PIDFILE"
    echo "serving: qwen3.6-27B :${QWEN27_PORT:-9024} (gpu ${QWEN27_GPU:-1})"
    ;;
injector)
    # Mistral needs its own tokenizer/config/load format. Not a reasoning model, so no
    # reasoning parser and no enable_thinking kwarg -- passing one errors on this family.
    CUDA_VISIBLE_DEVICES="$INJECTOR_GPU" VLLM_USE_DEEP_GEMM=0 VLLM_USE_FLASHINFER_SAMPLER=0 \
    nohup "$VLLM_PY/vllm" serve mistralai/Ministral-3-14B-Instruct-2512 \
        --tokenizer_mode mistral --config_format mistral --load_format mistral \
        --async-scheduling --port "$INJECTOR_PORT" \
        --max-model-len 16384 --gpu-memory-utilization 0.90 --language-model-only \
        > "$LOGS/ministral_serve.log" 2>&1 &
    echo $! >> "$SERVERS_PIDFILE"
    echo "serving: ministral :$INJECTOR_PORT (gpu $INJECTOR_GPU)"
    echo "logs: $LOGS/"
    ;;
*) echo "usage: serve.sh {up|qwen27b|injector|down}" >&2; exit 1 ;;
esac
