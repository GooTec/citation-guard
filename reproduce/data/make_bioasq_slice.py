#!/usr/bin/env python3
"""Rebuild the BioASQ n=200 validation slice used in the paper.

We do NOT redistribute the raw BioASQ passages (they are published PubMed abstracts
governed by BioASQ's challenge terms, and contain author-contact metadata). Instead this
script reconstructs the exact slice from the public `rag-datasets/rag-mini-bioasq` dataset
using a small structural manifest (`bioasq_slice_manifest.json`: row indices + PubMed IDs +
question hashes, no passage text). The rebuilt file is checksum-verified against the manifest's
`expected_md5`, so the reproduction is bit-for-bit identical to what the paper used.

Usage
-----
    python make_bioasq_slice.py                 # download from HuggingFace (needs network)
    python make_bioasq_slice.py --from-parquet DIR   # offline: DIR/qa.parquet + DIR/corpus.parquet
    python make_bioasq_slice.py --out PATH      # default: ./bioasq_validation_n200.json

The manifest pins the upstream dataset/config/split; if the upstream row order ever changes,
the per-question hash check fails loudly instead of producing a silently different slice.
"""
import argparse
import ast
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "bioasq_slice_manifest.json")


def _load_sources(from_parquet):
    """Return (qa_rows, pmid2passage). qa_rows: list of {question, answer, relevant_passage_ids}."""
    if from_parquet:
        import pandas as pd
        qa = pd.read_parquet(os.path.join(from_parquet, "qa.parquet"))
        co = pd.read_parquet(os.path.join(from_parquet, "corpus.parquet"))
        qa_rows = qa.to_dict("records")
        # corpus.parquet may be PMID-indexed or carry an explicit id column
        if "id" in co.columns:
            pmid2passage = {int(i): p for i, p in zip(co["id"], co["passage"])}
        else:
            pmid2passage = {int(i): p for i, p in zip(co.index, co["passage"])}
        return qa_rows, pmid2passage
    from datasets import load_dataset
    qa = load_dataset("rag-datasets/rag-mini-bioasq", "question-answer-passages", split="test")
    co = load_dataset("rag-datasets/rag-mini-bioasq", "text-corpus", split="passages")
    qa_rows = [{"question": r["question"], "answer": r["answer"],
                "relevant_passage_ids": r["relevant_passage_ids"]} for r in qa]
    pmid2passage = {int(r["id"]): r["passage"] for r in co}
    return qa_rows, pmid2passage


def _rel_ids(val):
    return val if isinstance(val, (list, tuple)) else ast.literal_eval(val)


def build_slice(from_parquet=None):
    meta = json.load(open(MANIFEST, encoding="utf-8"))
    entries = meta["entries"]
    qa_rows, pmid2passage = _load_sources(from_parquet)

    records = []
    for m in entries:
        row = m["qa_row"]
        if row >= len(qa_rows):
            sys.exit(f"ERROR: upstream qa has {len(qa_rows)} rows; manifest expects row {row}. "
                     "The upstream dataset changed; regenerate the manifest.")
        q = qa_rows[row]
        got = hashlib.sha256(q["question"].encode()).hexdigest()[:16]
        if got != m["q_sha16"]:
            sys.exit(f"ERROR: row {row} question hash mismatch ({got} != {m['q_sha16']}). "
                     "Upstream row order changed; cannot reproduce the exact slice.")
        ctxs = [{"title": f"PMID:{p}", "text": pmid2passage[p]} for p in m["pmids"]]
        records.append({"input": q["question"], "ctxs": ctxs, "id": m["id"],
                        "subject": "bio", "output": q["answer"].strip()})

    text = json.dumps(records, ensure_ascii=False, indent=2)
    md5 = hashlib.md5(text.encode()).hexdigest()
    if md5 != meta["expected_md5"]:
        sys.exit(f"ERROR: rebuilt slice md5 {md5} != expected {meta['expected_md5']}. "
                 "Reconstruction does not match the paper's slice.")
    return text, md5, len(records)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-parquet", metavar="DIR",
                    help="offline source dir with qa.parquet + corpus.parquet")
    ap.add_argument("--out", default=os.path.join(HERE, "bioasq_validation_n200.json"))
    a = ap.parse_args()

    text, md5, n = build_slice(a.from_parquet)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"OK: wrote {n} records to {a.out} (md5 {md5}, verified against manifest)")


if __name__ == "__main__":
    main()
