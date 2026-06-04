# Expected results — `reproduce_minimal.sh`

`summary.json` is the paper's headline numbers for **Gemma-4-31B on BioASQ n=200** (paper v0.8,
Table `tab:bioasq`). The reproducibility script computes the same metrics; diff against this file.

## Tolerance bands

Single H100 80 GB, vLLM 0.21, transformers 4.x, AttrScore at temperature 0:

| Metric | Tolerance |
|---|---|
| `citation_f1_mean` per cell | ±0.01 |
| `unsupported_pct` point estimate | ±0.5 pp |
| `unsupported_ci95` bounds | ±0.5 pp (bootstrap variance, B=2000) |
| `n_cited` per cell | exact (model temperature=0, deterministic) |
| `paired_lit_minus_bare_pp.point` | ±0.5 pp |

If you see drift beyond these bands, please open an issue with:
- `nvidia-smi` output
- `pip freeze` of the venv that ran each cell
- `results/<cell>/results.jsonl` (first 5 lines)
- `results/<cell>/guard_peritem.jsonl` (first 5 lines)

## Why these tolerances are non-zero

vLLM batching can produce minor token-level differences across runs even at temperature 0 (CUDA
kernel scheduling, KV-cache placement, FP8 quantisation rounding for Qwen). The AttrScore verifier
itself is deterministic given fixed input. The combined effect is < 0.5 pp on aggregated metrics.

## Diff command

```bash
diff <(jq -S . results/summary.json) <(jq -S . expected_results/summary.json)
```

Cell-by-cell with numerical tolerance:

```bash
python -c "
import json
exp = json.load(open('expected_results/summary.json'))
got = json.load(open('results/summary.json'))
for cond in exp['cells']:
    e, g = exp['cells'][cond], got['cells'][cond]
    df1 = abs(g['citation_f1_mean'] - e['citation_f1_mean'])
    dpct = abs(g['unsupported_pct'] - e['unsupported_pct'])
    flag = 'OK' if df1 <= 0.01 and dpct <= 0.5 else 'OUT OF TOLERANCE'
    print(f'{cond:10}  citF1 Δ={df1:.3f}  unsup% Δ={dpct:.1f}pp  {flag}')
"
```

## Notes on `cond_pqa` (PaperQA2)

PaperQA2 sometimes emits multi-citation `(ctx1, ctx2)` instead of single `(ctxN)`. The regex in
`_pqa_run.py` only converts single-parens form to `[N]`; expect ~5% of citations to remain in
parens form and not be counted by `compute_citation_f1`. This matches what the paper's PQA cells
report (citation_f1 is computed identically in both pipelines).
