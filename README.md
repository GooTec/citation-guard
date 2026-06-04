# citation-guard

**A local, validated citation-faithfulness guard for cited scientific synthesis.**

[![PyPI status](https://img.shields.io/badge/PyPI-pre--release-yellow)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python ≥3.9](https://img.shields.io/badge/python-%E2%89%A53.9-blue.svg)](#)
[![No openclaude dep](https://img.shields.io/badge/openclaude-not%20required-brightgreen.svg)](#dependencies)

When a large language model writes a literature synthesis that cites a provided passage set by `[N]`,
`citation-guard` checks each cited sentence against its passage with a deterministic, **gold-validated
attribution model** (AttrScore, 3B) and applies a three-step policy:

- **verify** — the cited passage supports the claim → keep.
- **re-attribute** — it does not, but another *provided* passage does → re-point the citation (keep the claim).
- **flag** — no provided passage supports it → mark `[N UNVERIFIED]` (default; `--remove` to drop).

Unlike asking the LLM to self-verify its own citations — which our experiments show is unreliable
(quality-judges barely separate answers with five times more unsupported citations, even when
explicitly asked) — the decision is made by an **external, validated verifier**. You then re-check
only the **flagged** citations instead of re-verifying every one (≈ 80–96 % less manual checking in
our evaluation).

Runs **locally** on a single GPU (or CPU) with a 3 B model — no frontier API, no cluster.

> The `citation-guard` Python package has **zero dependency on openclaude**. The openclaude plugin
> (`/sci-cite-guard`) is a *separate, optional* component under `openclaude-plugin/` that consumes the pip
> package as its backend.

## Install

```bash
# from PyPI (once released)
pip install citation-guard

# from source
git clone https://github.com/GooTec/citation-guard.git
cd citation-guard
pip install -e .
citation-guard --selftest         # warms the model + runs a bundled example
```
First run downloads `osunlp/attrscore-flan-t5-xl` (~3 GB) from HuggingFace. CPU works out of the
box; a GPU just makes it faster (`CUDA_VISIBLE_DEVICES=0`). No frontier API, no cluster.

## Quickstart — Python

```python
from citation_guard import guard

answer = "LNPs form a protein corona [1]. They bake bread [2]."
ctxs   = [
    {"text": "The corona on lipid nanoparticles modulates biodistribution..."},
    {"text": "Quantum dots are semiconductor nanocrystals..."},
]

verified_answer, report = guard(answer, ctxs)  # default: verify -> re-attribute -> flag
# report keys: n_cited, verified, re_attributed, flagged, manual_check_reduction, audit
```

## Quickstart — CLI

```bash
echo '{"answer":"LNPs form a protein corona [1]. They bake bread [2].",
       "ctxs":[{"text":"The corona on lipid nanoparticles modulates biodistribution..."},
               {"text":"Quantum dots are semiconductor nanocrystals..."}]}' | citation-guard
# or
citation-guard --input answer.json --no-reattribute --out result.json
```
`stderr`: `cited=2 verified=1 re-attributed=0 flagged=1 -> manual checks reduced 50%`
`stdout` / `--out`: `{"verified_answer": "...[2 UNVERIFIED]...", "report": {...}}`

## Reproduce the paper (1 H100, ≈ 2 hours)

A minimal reproducibility kit ships under `reproduce/`. It re-runs **one model × four pipelines** on
the public BioASQ validation slice (n = 200), then applies the guard + bootstrap CIs + cross-domain
sanity check.

```bash
cd reproduce
./scripts/reproduce_minimal.sh             # ~2 h on one H100 80 GB
diff <(jq -S . results/summary.json) <(jq -S . expected_results/summary.json)
```
The BioASQ slice is **not** shipped as raw text; `reproduce/data/make_bioasq_slice.py` rebuilds it
from the public `rag-datasets/rag-mini-bioasq` dataset (via a structural manifest of IDs + PubMed IDs
+ question hashes) and checksum-verifies the result, so the slice is byte-identical to the paper's.
The scripts build it automatically on first run. `reproduce/data/` also ships the QASA calibration
scores (derived scores only, 50 KB). The second model (Qwen3.6-35B-A3B-FP8) is reproduced by
re-running with `MODEL` set to it. See `reproduce/README.md` for step-by-step instructions and
acceptable tolerance bands.

## Quickstart — openclaude (optional plugin)

```text
/plugin marketplace add  https://github.com/GooTec/citation-guard   # or a local clone dir
/plugin install          citation-guard@bionexus-citation-guard
```
Then call **`/sci-cite-guard`** after any cited synthesis. The skill auto-installs the pip package
on first use if it is not found. An opt-in auto-run-after-synthesis hook is documented in
`openclaude-plugin/install.md`. **The plugin is optional — uninstalling or skipping it does
not affect the Python package.**

## Scope & honest limits

- Checks **attribution locality** (does the cited passage support the claim) — **not** conclusion
  correctness (e.g. in-vitro → clinical over-extrapolation) and **not** whether a reference exists
  in the world.
- The verifier is **moderate** (gold Cohen's κ ≈ 0.46–0.52) and **prompt-sensitive**; the bundled
  prompt is the gold-validated configuration from the paper. Treat output as a **triage**
  (flag-mode default = no silent deletion), not a guarantee. Residual missed-unsupported is bounded
  by the verifier's recall (≈ 0.90 on gold).
- A distribution-free **conformal guarantee** (split-conformal calibrated on QASA gold) is provided
  in `reproduce/scripts/05_conformal_sanity.py` and discussed in the paper §4.
- **Not a sole gate in high-stakes settings.** In patient-facing or otherwise safety-critical
  biomedical contexts the guard must not be the only check; human expert review of every flagged
  **and** verified sentence remains required.

## How it was validated (paper)

AttrScore chosen over strict NLI (DeBERTa / TRUE NLI, which over-flag 5–7 ×) and frontier GPT-4o on
SciFact gold (Cohen's κ = 0.46, recall = 0.90); re-validated in-domain on QASA (κ ≈ 0.52); checked
across four open 27–35 B models (Gemma-4, Qwen3.6) and replicated on the BioASQ biomedical
benchmark (n = 200, eight cells). See the accompanying Patterns paper for full
methodology and results.

## Dependencies

- **Runtime**: `transformers >= 4.40`, `torch >= 2.0`. *No openclaude dependency.*
- **Reproduce-only** (declared in `reproduce/requirements.txt`): `vllm == 0.21.0`, `paper-qa`,
  `openai` (for vLLM-served chat completions).

## License

MIT — see [`LICENSE`](LICENSE). Model weights for `osunlp/attrscore-flan-t5-xl` are governed by
the upstream model card on HuggingFace.

## Cite

If you use `citation-guard` in academic work, please cite the accompanying Patterns paper. A Zenodo DOI for the code
repository will be issued on first public release.
