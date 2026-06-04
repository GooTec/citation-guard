#!/usr/bin/env bash
# reproduce_minimal.sh — end-to-end minimal viable reproduction.
# 1 model (Gemma-4-31B) × 4 conditions (bare, /lit, OS, PQA) × BioASQ n=200
# Target: ~2 hours on one H100 80 GB.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
START=$(date +%s)

MODEL="${MODEL:-google/gemma-4-31B-it}"
PORT="${PORT:-8010}"
VLLM_BASE="http://localhost:${PORT}/v1"

log "== 1. Setup =="
./scripts/00_setup.sh
source .env

log "== 1b. Build BioASQ slice from public source (not redistributed) =="
# Rebuilds data/bioasq_validation_n200.json from the public rag-mini-bioasq dataset via the
# structural manifest, then checksum-verifies it against the paper's slice. Skips if present.
if [ ! -f data/bioasq_validation_n200.json ]; then
  "${VLLM_PY:-$(which python3)}" data/make_bioasq_slice.py
fi

log "== 2. Boot vLLM (${MODEL}) on :${PORT} =="
./scripts/01_vllm_boot.sh "${MODEL}" "${PORT}"

log "== 3. Generate (bare, /lit, OS, PQA) =="
# Use the os/pqa venvs for those, system python (or vllmvenv) for bare/lit.
PY="${VLLM_PY:-$(which python3)}"

# bare + lit via httpx (no extra deps; runs in any venv with httpx + numpy)
if ! "${PY}" -c "import httpx" 2>/dev/null; then
  log "Installing httpx into VLLM_PY…"
  "${PY}" -m pip install -q httpx numpy
fi

"${PY}" scripts/02_generate.py --condition b2  --model "${MODEL}" --vllm-base "${VLLM_BASE}"
"${PY}" scripts/02_generate.py --condition lit --model "${MODEL}" --vllm-base "${VLLM_BASE}"

# OpenScholar (uses osvenv internally)
"${PY}" scripts/02_generate.py --condition os  --model "${MODEL}" --vllm-base "${VLLM_BASE}"

# PaperQA2 (uses pqavenv internally)
"${PY}" scripts/02_generate.py --condition pqa --model "${MODEL}" --vllm-base "${VLLM_BASE}"

log "== 4. Tear down vLLM (free GPU for guard) =="
pkill -f "vllm.entrypoints.openai.api_server.*${PORT}" 2>/dev/null || true
sleep 5

log "== 5. Guard (AttrScore-3B, 4 cells) =="
"${PY}" scripts/03_guard.py --model "${MODEL}"

log "== 6. Bootstrap item-level 95% CIs =="
"${PY}" scripts/04_bootstrap.py --model "${MODEL}"

log "== 7. Conformal cross-domain sanity =="
"${PY}" scripts/05_conformal_sanity.py --model "${MODEL}"

END=$(date +%s)
log "== DONE in $(( (END-START)/60 )) min =="
echo
echo "Results: ${ROOT}/results/summary.json  (and summary.txt)"
echo "Diff vs paper:  diff <(jq -S . results/summary.json) <(jq -S . expected_results/summary.json)"
