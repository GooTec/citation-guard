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
- **re-attribute** — it does not, but another *provided* passage does → re-point the citation (keep the
  claim). Re-attribution is a swappable slot: a deterministic **BM25** ranker (default, no extra model)
  proposes a candidate and the verifier re-checks support before the pointer is moved; pass
  `--reattribute-by verifier` to rank by the attribution score instead.
- **flag** — no provided passage supports it → mark `[N UNVERIFIED]` (default; `--remove` to drop).

The support decision is made by an **external, gold-validated verifier**, not by asking the generator to
self-check. You then re-check only the **flagged** citations instead of re-verifying every one: a few cited
sentences per hundred at the validated operating point in our evaluation.

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

## Reproduce — minimal cross-domain demo (1 H100, ≈ 2 hours)

A minimal kit ships under `reproduce/`. It is a **cross-domain demonstration**, not the paper's headline:
it re-runs **one model × four pipelines** on the public BioASQ biomedical slice (n = 200), applies the
guard with bootstrap CIs, and checks that the QASA-calibrated conformal threshold transfers (a stability
sanity check). The paper's headline results (verifier matched-catch on SciFact, re-attribution and
conformal on the full QASA test set, n = 1375) are produced by the experiment scripts described in the
paper, not by this minimal kit.

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
/plugin install          citation-guard@gootec-citation-guard
```
Then call **`/sci-cite-guard`** after any cited synthesis. The skill auto-installs the pip package
on first use if it is not found. An opt-in auto-run-after-synthesis hook is documented in
`openclaude-plugin/install.md`. **The plugin is optional — uninstalling or skipping it does
not affect the Python package.**

## Scope & honest limits

- Checks **attribution locality** (does the cited passage support the claim) — **not** conclusion
  correctness (e.g. in-vitro → clinical over-extrapolation) and **not** whether a reference exists
  in the world.
- The verifier is an **imperfect instrument** and **prompt-sensitive**; the bundled prompt is the
  gold-validated configuration from the paper (supported-class recall ≈ 0.90 on SciFact, ≈ 0.94 on a
  held-out split). At a *matched catch rate* it separates supported from unsupported about as well as a
  strict frontier judge, so it is adopted on cost, not strictness. Treat output as a **triage**
  (flag-mode default = no silent deletion), not a guarantee.
- A distribution-free **conformal guarantee** (split-conformal calibrated on QASA gold) turns the
  verifier's score into a finite-sample bound on the unsupported citations that slip through unflagged;
  see `reproduce/scripts/05_conformal_sanity.py` and the paper.
- **Not a sole gate in high-stakes settings.** In patient-facing or otherwise safety-critical
  biomedical contexts the guard must not be the only check; human expert review of every flagged
  **and** verified sentence remains required.

## How it was validated (paper)

On SciFact gold the candidate verifiers (AttrScore-3B, DeBERTa-NLI, GPT-4o, RAGAS, OpenScholar post-hoc)
separate supported from unsupported comparably at a matched catch rate, so AttrScore-3B is adopted on cost
(it runs locally) at a high-recall operating point (0.90 on SciFact, 0.94 on a held-out split).
Re-attribution, the swappable-slot comparison, and conformal calibration are evaluated on the full QASA test
set (n = 1375): a deterministic BM25 (0.69 recall@1) matches the best open generator's self-attribution far
more cheaply than the verifier score (0.58). The unsupported-citation rate is reported across four open
27–35 B models (Gemma-4, Qwen3.6) and three pipelines. See the accompanying paper for full methodology and
results.

## Dependencies

- **Runtime**: `transformers >= 4.40`, `torch >= 2.0`. *No openclaude dependency.*
- **Reproduce-only** (declared in `reproduce/requirements.txt`): `vllm == 0.21.0`, `paper-qa`,
  `openai` (for vLLM-served chat completions).

## License

MIT — see [`LICENSE`](LICENSE). Model weights for `osunlp/attrscore-flan-t5-xl` are governed by
the upstream model card on HuggingFace.

## Cite

If you use `citation-guard` in academic work, please cite the accompanying paper (under review at *ACM
Transactions on Intelligent Systems and Technology*, Special Issue on LLM-Driven Agentic AI). A Zenodo DOI
for this repository will be issued on the tagged release.
