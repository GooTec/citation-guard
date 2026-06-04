#!/usr/bin/env python3
"""03_guard.py — Run the AttrScore guard on all generated BioASQ cells.

Resume-by-id, per-item torch.cuda.empty_cache (T5 KV cache safety).
Writes results/{model_slug}/bioasq/{cond}/guard_peritem.jsonl.

Usage:
  python 03_guard.py --model google/gemma-4-31B-it
                     [--cells cond_b2,cond_lit,cond_os,cond_pqa]
"""
import json, re, random, argparse, sys, subprocess
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data/bioasq_validation_n200.json"


def _ensure_data():
    """The BioASQ slice is not redistributed; rebuild it from the public source on first use."""
    if DATA.exists():
        return
    gen = ROOT / "data/make_bioasq_slice.py"
    print(f"[03_guard] {DATA.name} missing; rebuilding from public rag-mini-bioasq...", file=sys.stderr)
    subprocess.run([sys.executable, str(gen), "--out", str(DATA)], check=True)

PROMPT = (
    "As an Attribution Validator, verify whether the given reference can support the claim. "
    "Answer with Attributable, Extrapolatory, or Contradictory.\nClaim: {c}\nReference: {r}"
)
SENT = re.compile(r"(?<=[.!?])\s+")
CITE = re.compile(r"\[(\d+)\]")


def slugify(model: str) -> str:
    return model.lower().replace("/", "-").replace(".", "").replace("_", "-")


def words(t):
    return re.findall(r"[a-z0-9]+", t.lower())


def rougeL_recall(cand, ref):
    a, b = words(cand)[:3500], words(ref)[:400]
    if not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for ai in a:
        cur = [0] * (len(b) + 1)
        for j, bj in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if ai == bj else (prev[j] if prev[j] >= cur[j - 1] else cur[j - 1])
        prev = cur
    return prev[len(b)] / len(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-31B-it")
    ap.add_argument("--cells", default="cond_b2,cond_lit,cond_os,cond_pqa")
    args = ap.parse_args()
    _ensure_data()

    random.seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("osunlp/attrscore-flan-t5-xl")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "osunlp/attrscore-flan-t5-xl", dtype=torch.bfloat16
    ).to(dev).eval()
    model.config.use_cache = False
    print(f"Loaded AttrScore-3B on {dev}", flush=True)

    @torch.inference_mode()
    def supported(claim, ref):
        if not str(ref).strip():
            return False
        enc = tok(
            [PROMPT.format(c=str(claim)[:600], r=str(ref)[:1500])],
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(dev)
        out = model.generate(**enc, max_new_tokens=8, use_cache=False, do_sample=False)
        txt = tok.decode(out[0], skip_special_tokens=True)
        del out, enc
        return txt.strip().lower().startswith("attribut")

    def analyze(answer, ctxs, reference):
        n = len(ctxs)
        texts = [str(c.get("text", ""))[:600] for c in ctxs]
        sents = SENT.split(answer)
        cited_idx, unsup_idx = [], []
        for k, s in enumerate(sents):
            cs = [int(m) for m in CITE.findall(s)]
            cs = [c for c in cs if 1 <= c <= n]
            if not cs:
                continue
            cited_idx.append(k)
            claim = CITE.sub("", s).strip()
            if not supported(claim, " ".join(texts[c - 1] for c in cs)):
                unsup_idx.append(k)
        cov_orig = rougeL_recall(answer, reference)
        keep_guard = [s for k, s in enumerate(sents) if k not in unsup_idx]
        cov_guard = rougeL_recall(" ".join(keep_guard), reference)
        rm = (
            set(random.sample(cited_idx, min(len(unsup_idx), len(cited_idx))))
            if unsup_idx
            else set()
        )
        keep_rand = [s for k, s in enumerate(sents) if k not in rm]
        cov_rand = rougeL_recall(" ".join(keep_rand), reference)
        return dict(
            n_cited=len(cited_idx),
            n_unsup=len(unsup_idx),
            cov_orig=cov_orig,
            cov_guard=cov_guard,
            cov_rand=cov_rand,
        )

    bench = {str(s["id"]): s for s in json.loads(DATA.read_text())}
    slug = slugify(args.model)

    for cond in args.cells.split(","):
        p = ROOT / f"results/{slug}/bioasq/{cond}/results.jsonl"
        if not p.exists():
            print(f"SKIP {cond}: {p} not found", flush=True)
            continue
        outp = ROOT / f"results/{slug}/bioasq/{cond}/guard_peritem.jsonl"
        rows = [json.loads(l) for l in p.open()]
        done_ids = set()
        if outp.exists():
            for l in outp.open():
                try:
                    done_ids.add(json.loads(l)["id"])
                except Exception:
                    pass
        mode = "a" if done_ids else "w"
        with outp.open(mode) as fo:
            for r in rows:
                if r["id"] in done_ids:
                    continue
                it = bench.get(str(r["id"]), {})
                res = analyze(r.get("answer", ""), it.get("ctxs", []), str(it.get("output", "")))
                fo.write(json.dumps({"id": r["id"], **res}) + "\n")
                fo.flush()
                torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        print(f"{slug}/{cond}: {len(rows)} items -> guard_peritem.jsonl", flush=True)
    print("done")


if __name__ == "__main__":
    main()
