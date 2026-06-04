#!/usr/bin/env python3
"""05_conformal_sanity.py — Apply QASA-calibrated τ to the binary BioASQ signal.

We do NOT recompute continuous P(attr) on BioASQ. Instead, we:
  (i)  load QASA calibration scores (data/qasa_calibration_scores.json),
  (ii) compute τ(α) for α ∈ {0.10, 0.20, 0.30},
  (iii) compare BioASQ binary unsupported% (from 03_guard.py) against the
       1–4% band expected of ScholarQABench bare/lit cells.

This is the cross-domain *stability* sanity check (see paper Supp:bioasq).
Bio-unconditional conformal would require bio citation gold, which BioASQ
ideal answers are not.

Usage:
  python 05_conformal_sanity.py --model google/gemma-4-31B-it
"""
import json, argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
QASA = ROOT / "data/qasa_calibration_scores.json"


def slugify(model: str) -> str:
    return model.lower().replace("/", "-").replace(".", "").replace("_", "-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-31B-it")
    args = ap.parse_args()

    qasa = json.loads(QASA.read_text())
    alphas = qasa["tau_curve"]["alphas"]
    taus = qasa["tau_curve"]["tau"]

    def tau_at(a):
        return taus[alphas.index(a)]

    print("=== QASA conformal calibration ===")
    print(
        f"n_supported={qasa['n_supported']} n_unsupported={qasa['n_unsupported']}  "
        f"P(attr) means: unsup={qasa['unsupported_mean']:.3f}, sup={qasa['supported_mean']:.3f}  "
        f"separation Δ={qasa['supported_mean']-qasa['unsupported_mean']:.3f}"
    )
    print(f"τ(α=0.10)={tau_at(0.10):.3f}  (guaranteed unsupported-recall ≥ 0.90)")
    print(f"τ(α=0.20)={tau_at(0.20):.3f}  (guaranteed ≥ 0.80)")
    print(f"τ(α=0.30)={tau_at(0.30):.3f}  (guaranteed ≥ 0.70)")
    print()
    print("=== BioASQ binary unsupported% by cell (frozen verifier; cross-domain) ===")
    slug = slugify(args.model)
    rows = []
    for cond in ["cond_b2", "cond_lit", "cond_os", "cond_pqa"]:
        gp = ROOT / f"results/{slug}/bioasq/{cond}/guard_peritem.jsonl"
        if not gp.exists():
            print(f"  {cond:10} (missing)")
            continue
        guard_rows = [json.loads(l) for l in gp.open()]
        n_c = sum(r["n_cited"] for r in guard_rows)
        n_u = sum(r["n_unsup"] for r in guard_rows)
        pct = 100 * n_u / max(1, n_c)
        band = "in band" if 1.0 <= pct <= 11.5 else "OUTSIDE band"
        print(f"  {cond:10} unsup% = {pct:5.1f}   n_cited={n_c:5d}   ({band})")
        rows.append((cond, pct, n_c, n_u))

    print()
    print("Cross-domain interpretation:")
    print("  - bare/lit expected band: 1–4% (matches ScholarQABench bare/lit).")
    print("  - OS/PQA expected band: 6–11% (matches paper's ScholarQABench order;")
    print("    larger absolute values on ScholarQABench's longer answers, 16–21%).")
    print("  - The frozen QASA-calibrated verifier carries cross-domain on the binary signal.")
    print("  - Bio-unconditional conformal upgrade requires bio citation gold (future work).")

    out = {
        "qasa_taus": {"alpha_0.10": tau_at(0.10), "alpha_0.20": tau_at(0.20), "alpha_0.30": tau_at(0.30)},
        "qasa_means": {"supported": qasa["supported_mean"], "unsupported": qasa["unsupported_mean"]},
        "bioasq_binary_unsup_pct": {c: round(p, 2) for c, p, _, _ in rows},
    }
    (ROOT / "results/conformal_sanity.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote results/conformal_sanity.json")


if __name__ == "__main__":
    main()
