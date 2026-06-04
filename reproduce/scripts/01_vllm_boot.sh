#!/usr/bin/env bash
# 01_vllm_boot.sh — Boot a vLLM OpenAI-compatible server in the background.
# Idempotent: noop if already serving the same model on the same port.
#
# Usage:
#   ./01_vllm_boot.sh [MODEL] [PORT]
#   default MODEL=google/gemma-4-31B-it, PORT=8010
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
[ -f .env ] && source .env

MODEL="${1:-google/gemma-4-31B-it}"
PORT="${2:-8010}"
LOG="${ROOT}/logs/vllm-${PORT}.log"
mkdir -p "${ROOT}/logs"

# Already up?
if curl -s --max-time 2 "http://localhost:${PORT}/v1/models" 2>/dev/null | grep -q "object"; then
  echo "vLLM already serving on :${PORT}"
  curl -s "http://localhost:${PORT}/v1/models" | head -c 200; echo
  exit 0
fi

PY="${VLLM_PY:-$(which python3)}"
echo "Booting vLLM (${MODEL}) on :${PORT} via ${PY}…"

# Qwen3.6 needs thinking-off override; download the override template lazily.
EXTRA_ARGS=()
if [[ "${MODEL}" == *Qwen3.6* || "${MODEL}" == *qwen3.6* ]]; then
  TPL="${HOME}/.cache/citation-guard/qwen35b-no-think.jinja"
  if [ ! -f "${TPL}" ]; then
    mkdir -p "$(dirname "${TPL}")"
    # Fetch the model's default chat_template, flip the enable_thinking default to OFF.
    "${PY}" - <<'PYEOF' "${MODEL}" "${TPL}"
import sys, re
from transformers import AutoTokenizer
model, out = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
ct = tok.chat_template
# Reverse default: enable_thinking=False is now the default branch.
ct = re.sub(
    r"\{%-\s*if enable_thinking is defined and enable_thinking is false\s*%}\s*"
    r"\{\{-\s*'<think>\\n\\n</think>\\n\\n'\s*\}\}\s*"
    r"\{%-\s*else\s*%}\s*\{\{-\s*'<think>\\n'\s*\}\}\s*\{%-\s*endif\s*%}",
    "{%- if enable_thinking is defined and enable_thinking is true %}\n"
    "    {{- '<think>\\n' }}\n"
    "{%- else %}\n"
    "    {{- '<think>\\n\\n</think>\\n\\n' }}\n"
    "{%- endif %}",
    ct, count=1, flags=re.DOTALL,
)
open(out, "w").write(ct)
print(f"wrote {out}")
PYEOF
  fi
  EXTRA_ARGS+=(--chat-template "${TPL}")
fi

# Pick reasonable defaults; user can override via env.
TP="${TP_SIZE:-1}"
MAX_LEN="${MAX_LEN:-16384}"
GPU_FRAC="${GPU_FRAC:-0.85}"
TOOL_PARSER="${TOOL_PARSER:-hermes}"   # gemma4 / qwen3_coder if needed
if [[ "${MODEL}" == *gemma-4* ]]; then TOOL_PARSER=gemma4; fi
if [[ "${MODEL}" == *Qwen3.6* || "${MODEL}" == *qwen3.6* ]]; then TOOL_PARSER=qwen3_coder; fi

nohup "${PY}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --tensor-parallel-size "${TP}" \
  --gpu-memory-utilization "${GPU_FRAC}" \
  --max-model-len "${MAX_LEN}" \
  --max-num-seqs 4 \
  --dtype bfloat16 \
  --trust-remote-code \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_PARSER}" \
  "${EXTRA_ARGS[@]}" \
  --port "${PORT}" > "${LOG}" 2>&1 &

PID=$!
echo "  pid=${PID}  log=${LOG}"
echo "Waiting for :${PORT} ready…"
for i in $(seq 1 180); do
  if curl -s --max-time 3 "http://localhost:${PORT}/v1/models" 2>/dev/null | grep -q "object"; then
    echo "vLLM ready on :${PORT}"
    exit 0
  fi
  sleep 5
done
echo "ERROR: vLLM did not become ready in 15 minutes. See ${LOG}" >&2
exit 1
