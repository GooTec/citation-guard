#!/usr/bin/env python3
"""_pqa_run.py — PaperQA2 on fixed BioASQ ctxs with a local vLLM endpoint.
Invoked by 02_generate.py via the pqavenv. Resume-safe.

Env vars: BIOASQ_INPUT, OUT_DIR
Argv:     [model_slug] [vllm_base]
"""
import json, asyncio, re, os, sys
from pathlib import Path
from paperqa import Docs, Text, Doc, Settings
from paperqa.settings import AnswerSettings

MODEL_SLUG = sys.argv[1]
VLLM = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8010/v1"
MODEL_API = f"openai/{MODEL_SLUG}"
INPUT = Path(os.environ["BIOASQ_INPUT"])
OUTDIR = Path(os.environ["OUT_DIR"])
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTF = OUTDIR / "results.jsonl"

cfg = {"model_list": [{"model_name": MODEL_API, "litellm_params": {
    "model": MODEL_API, "api_base": VLLM, "api_key": "dummy"}}]}
settings = Settings(
    llm=MODEL_API, summary_llm=MODEL_API, llm_config=cfg, summary_llm_config=cfg,
    embedding="sparse",
    answer=AnswerSettings(evidence_retrieval=False, evidence_k=20,
                          answer_max_sources=20, answer_length="about 500 words"),
)

CW = re.compile(
    r"\d+[\.\d]*\s*%?|\b(increase[ds]?|decrease[ds]?|higher|lower|greater|less|found|"
    r"showed?|reported|demonstrated|suggest[s]?|associated|correlat|significant|"
    r"improv|reduc|enhanc)\b",
    re.I,
)


def cit_f1(ans, ctxs):
    n = len(ctxs)
    ci = [int(m) for m in re.findall(r"\[(\d+)\]", ans)]
    vc = {i for i in ci if 1 <= i <= n}
    prec = len(vc) / len(set(ci)) if ci else 1.0
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", ans) if len(s.strip()) > 20]
    cw = [s for s in sents if CW.search(s)]
    cwc = [s for s in cw if re.search(r"\[\d+\]", s)]
    rec = len(cwc) / len(cw) if cw else 0.0
    d = prec + rec
    return {
        "citation_precision": round(prec, 3),
        "citation_recall": round(rec, 3),
        "citation_f1": round(2 * prec * rec / d, 3) if d > 0 else 0.0,
        "n_citations_in_answer": len(ci),
    }


async def main():
    data = json.load(open(INPUT))
    done = {json.loads(l)["id"] for l in OUTF.open()} if OUTF.exists() else set()
    print(f"total={len(data)} done={len(done)} model={MODEL_SLUG}", flush=True)
    with OUTF.open("a") as fout:
        for k, item in enumerate(data):
            if item["id"] in done:
                continue
            try:
                docs = Docs()
                for i, ctx in enumerate(item["ctxs"], 1):
                    t = str(ctx.get("text", "")).strip()
                    if not t:
                        continue
                    doc = Doc(docname=f"ctx{i}", citation=f"ctx{i}", dockey=f"ctx{i}")
                    await docs.aadd_texts([Text(text=t, name=f"ctx{i}", doc=doc)], doc, settings=settings)
                s = await docs.aquery(item["input"], settings=settings)
                ans = re.sub(r"\(ctx(\d+)\)", r"[\1]", s.answer or "")
            except Exception as e:
                print(f"  [{k}] {item['id']} ERROR: {str(e)[:120]}", flush=True)
                ans = ""
            cf = cit_f1(ans, item["ctxs"])
            rec = {
                "id": item["id"], "benchmark": "bioasq", "model": MODEL_SLUG, "variant": "cond_pqa",
                "question": item["input"], "answer": ans,
                "reference": item.get("output", ""), "tokens_out": len(ans.split()),
                **cf,
            }
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            if (k + 1) % 10 == 0:
                print(f"  {k+1}/{len(data)}", flush=True)
    print("DONE", flush=True)


asyncio.run(main())
