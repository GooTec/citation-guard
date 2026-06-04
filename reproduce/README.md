# Reproducibility kit — `citation-guard`

**Target**: reproduce the paper's headline numbers on one H100 80 GB in ≈ 2 hours.

```
reproduce/
├── data/
│   ├── bioasq_slice_manifest.json  # row IDs + PubMed IDs + question hashes (no passage text)
│   ├── make_bioasq_slice.py        # rebuilds the n=200 slice from public rag-mini-bioasq + verifies md5
│   └── qasa_calibration_scores.json
│       # bioasq_validation_n200.json is built here on first run (not shipped as raw text)
├── scripts/
│   ├── 00_setup.sh                # check GPU, create venvs, set HF cache
│   ├── 01_vllm_boot.sh            # boot Gemma-4-31B vLLM on port 8010
│   ├── 02_generate.py             # bare + /lit + OS + PQA generation
│   ├── 03_guard.py                # AttrScore verifier, 4 cells
│   ├── 04_bootstrap.py            # item-bootstrap 95% CIs
│   ├── 05_conformal_sanity.py     # QASA τ → BioASQ cross-domain check
│   └── reproduce_minimal.sh       # ★ end-to-end, ≈ 2 h
├── expected_results/
│   ├── summary.json               # expected citF1 / unsup% / CIs per cell
│   └── README.md                  # tolerance bands
├── docs/
└── requirements.txt               # vllm 0.21, paper-qa, openai, etc.
```

## Hardware requirements

- **GPU**: 1× NVIDIA H100 80 GB (or equivalent ≥ 80 GB HBM; A100 80 GB also works).
- **RAM**: ≥ 64 GB.
- **Disk**: ~80 GB free (model weights ~62 GB for Gemma-4-31B + ~3 GB for AttrScore + caches).
- **Network**: HuggingFace download access on first run.

CPU-only reproduction is **not** supported for `reproduce_minimal.sh` (vLLM requires GPU). For
CPU users, please consume the released `expected_results/` directly.

## Software requirements (auto-installed by `00_setup.sh`)

- Python 3.12 (or ≥ 3.10).
- `vllm == 0.21.0` (for Qwen3.6 / FP8 / Mamba-hybrid support).
- `paper-qa >= 5` (for PaperQA2 cell).
- `transformers >= 4.40`, `torch >= 2.0` (for AttrScore verifier).
- `nltk`, `spacy` punkt (for OpenScholar cell).

## One-line minimal run

```bash
cd reproduce
./scripts/reproduce_minimal.sh         # ≈ 2 h on one H100 80 GB
```

This:
1. Sets up two isolated venvs (`pqavenv` for PaperQA2, `osvenv` for OpenScholar) — pyenv/uv based, no conda.
2. Boots Gemma-4-31B-it vLLM on `localhost:8010`.
3. Runs four cells: `bare`, `/lit`, `OpenScholar`, `PaperQA2` × Gemma-4-31B × BioASQ-200.
4. Tears down vLLM. Loads AttrScore-3B; runs the guard on all 4 cells.
5. Bootstraps 95 % CIs and the cross-domain conformal sanity check.
6. Writes `results/summary.json`. Diff against `expected_results/summary.json`.

Expected outputs (per cell, n = 200, see `expected_results/README.md` for full table):

| Cell | citation-F1 | unsupported %  [95 % CI] |
|---|--:|--:|
| Gemma-4-31B / bare | 0.828 | 1.1 [0.2, 2.2] |
| Gemma-4-31B / /lit | 0.673 | 3.3 [1.7, 5.1] |
| Gemma-4-31B / OS   | 0.982 | 7.5 [6.0, 9.1] |
| Gemma-4-31B / PQA  | 0.762 | 10.4 [8.2, 12.6] |

Tolerances (single H100, vLLM 0.21 + transformers 4.x at temperature 0):
- citation-F1 mean: ±0.01
- unsupported % point estimate: ±0.5 pp
- CI bounds: ±0.5 pp (bootstrap variance)

## Why these tolerances are non-zero

vLLM batching can produce minor token-level differences across runs even at temperature 0 (kernel
scheduling, KV-cache placement). The AttrScore verifier is deterministic given fixed input. The
combined effect is < 0.5 pp on aggregated metrics; if you see > 1 pp drift, please open an issue with
your `nvidia-smi`, `pip freeze`, and run log.

## Full 2-model reproduction (≈ 4 h)

Re-running with `MODEL=<Qwen3.6-35B-A3B-FP8 id>` adds the second model (with the `chat_template` thinking-off override).
Outputs match Table~tab:bioasq in the paper.

## Reproducibility notes

- **Resume-by-id**: every generation script resumes from `results.jsonl` on re-run. Killing and
  restarting `reproduce_minimal.sh` mid-run is safe.
- **Qwen3.6 thinking-off**: required by the paper. The full-run script installs the override
  template under `~/.cache/citation-guard/qwen35b-no-think.jinja` and passes it to vLLM via
  `--chat-template`. Without this override, `<think>` traces contaminate the generation.
- **OpenScholar S2_API_KEY**: a dummy key suffices when `--ss_retriever` is off (we use only the
  provided BioASQ `ctxs`). The setup script writes `S2_API_KEY=dummy` to `reproduce/.env`.
- **No openclaude required**. The reproducibility kit only uses `vllm`, `paper-qa`, the OpenScholar
  CLI, and the `citation-guard` pip package.
