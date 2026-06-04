#!/usr/bin/env python3
"""04_bootstrap.py — Item-bootstrap 95% CIs for unsupported rate on BioASQ cells.

Consumes results/{model_slug}/bioasq/{cond}/guard_peritem.jsonl
Writes results/summary.json (machine-checkable) + summary.txt (human-readable).

Usage:
  python 04_bootstrap.py --model google/gemma-4-31B-it
"""
import json, argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
CONDS = ["cond_b2", "cond_lit", "cond_os", "cond_pqa"]


def slugify(model: str) -> str:
    return model.lower().replace("/", "-").replace(".", "").replace("_", "-")


def rate_ci(rows, rng, B=2000):
    u = np.array([r["n_unsup"] for r in rows])
    c = np.array([r["n_cited"] for r in rows])
    n = len(rows)
    pt = u.sum() / max(1, c.sum())
    bs = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        cc = c[idx].sum()
        bs.append(u[idx].sum() / cc if cc else 0)
    return 100 * pt, 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)


def f1_mean(rows):
    return float(np.mean([r["citation_f1"] for r in rows]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-31B-it")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    slug = slugify(args.model)
    summary = {"model": args.model, "model_slug": slug, "cells": {}}
    lines = []

    lines.append("=== BioASQ headline (item-bootstrap 95% CI) ===")
    lines.append(f"model={args.model}")
    lines.append(f"{'cond':10}{'citF1':>9}{'unsup%':>9}{'95% CI':>16}{'n_items':>10}{'n_cited':>10}")
    lines.append("-" * 64)
    for cond in CONDS:
        gp = ROOT / f"results/{slug}/bioasq/{cond}/guard_peritem.jsonl"
        rp = ROOT / f"results/{slug}/bioasq/{cond}/results.jsonl"
        if not gp.exists() or not rp.exists():
            lines.append(f"{cond:10}{'(missing)':>9}")
            summary["cells"][cond] = {"status": "missing"}
            continue
        guard_rows = [json.loads(l) for l in gp.open()]
        gen_rows = [json.loads(l) for l in rp.open()]
        pt, lo, hi = rate_ci(guard_rows, rng)
        f1 = f1_mean(gen_rows)
        n_cited = sum(r["n_cited"] for r in guard_rows)
        summary["cells"][cond] = {
            "citation_f1_mean": round(f1, 3),
            "unsupported_pct": round(pt, 1),
            "unsupported_ci95": [round(lo, 1), round(hi, 1)],
            "n_items": len(guard_rows),
            "n_cited": n_cited,
        }
        lines.append(
            f"{cond:10}{f1:>9.3f}{pt:>9.1f}{f'[{lo:.1f},{hi:.1f}]':>16}"
            f"{len(guard_rows):>10}{n_cited:>10}"
        )

    # paired Δ on (cond_b2, cond_lit)
    if "cond_b2" in summary["cells"] and "cond_lit" in summary["cells"]:
        rb = [json.loads(l) for l in (ROOT / f"results/{slug}/bioasq/cond_b2/guard_peritem.jsonl").open()]
        rc = [json.loads(l) for l in (ROOT / f"results/{slug}/bioasq/cond_lit/guard_peritem.jsonl").open()]
        bb = {r["id"]: r for r in rb}
        cc = {r["id"]: r for r in rc}
        ids = [i for i in cc if i in bb]
        ub = np.array([bb[i]["n_unsup"] for i in ids])
        cb = np.array([bb[i]["n_cited"] for i in ids])
        uc = np.array([cc[i]["n_unsup"] for i in ids])
        ccnt = np.array([cc[i]["n_cited"] for i in ids])
        n = len(ids)

        def delta(idx):
            rate_c = uc[idx].sum() / max(1, ccnt[idx].sum())
            rate_b = ub[idx].sum() / max(1, cb[idx].sum())
            return 100 * (rate_c - rate_b)

        pt = delta(np.arange(n))
        bs = [delta(rng.integers(0, n, n)) for _ in range(2000)]
        lo, hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
        summary["paired_lit_minus_bare_pp"] = {
            "point": round(float(pt), 2),
            "ci95": [round(float(lo), 2), round(float(hi), 2)],
            "significant_positive": bool(lo > 0),
        }
        lines.append("")
        lines.append(f"=== paired /lit - bare unsupported delta ===")
        lines.append(
            f"  {pt:+.2f} pp  [{lo:+.2f}, {hi:+.2f}]  sig+ = {bool(lo > 0)}"
        )

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out_dir/'summary.json'} and {out_dir/'summary.txt'}")


if __name__ == "__main__":
    main()
