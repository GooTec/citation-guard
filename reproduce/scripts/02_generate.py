#!/usr/bin/env python3
"""02_generate.py — Generate BioASQ answers for a single model under 4 conditions.

This is a slimmed, single-model port of the paper's `run_benchmark.py`. It assumes a vLLM
OpenAI-compatible endpoint is already running on `--vllm-base` and writes results.jsonl per cell.

Conditions:
  bare  — minimal output-control prompt, httpx call (no harness)
  /lit  — same prompt + a /lit-style "follow citation discipline" preamble (no openclaude required)
  OS    — OpenScholar pipeline on the same vLLM endpoint (external/OpenScholar)
  PQA   — PaperQA2 on the same vLLM endpoint (fixed ctxs, no retrieval)

Resume-by-id: re-running skips items already in results.jsonl.

Usage:
  python 02_generate.py --condition b2    --model google/gemma-4-31B-it
  python 02_generate.py --condition lit   --model google/gemma-4-31B-it
  python 02_generate.py --condition os    --model google/gemma-4-31B-it
  python 02_generate.py --condition pqa   --model google/gemma-4-31B-it

Outputs: results/{model_slug}/bioasq/{cond}/results.jsonl
"""
import json, argparse, os, re, sys, subprocess
from pathlib import Path
import httpx

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data/bioasq_validation_n200.json"


def _ensure_data():
    """The BioASQ slice is not redistributed; rebuild it from the public source on first use."""
    if DATA.exists():
        return
    gen = ROOT / "data/make_bioasq_slice.py"
    print(f"[02_generate] {DATA.name} missing; rebuilding from public rag-mini-bioasq...", file=sys.stderr)
    subprocess.run([sys.executable, str(gen), "--out", str(DATA)], check=True)


SYS_BARE = (
    "You answer scientific questions using ONLY the provided passages. "
    "Cite passages by their 1-indexed number in square brackets, e.g., [1], [2]. "
    "Do not invent citations. Keep the answer focused and ≤ 250 words."
)
SYS_LIT = SYS_BARE + (
    " Follow strict citation discipline: every claim that draws on a passage must end with [N]. "
    "If a passage contradicts another, note the disagreement and cite both."
)


def slugify(model: str) -> str:
    return model.lower().replace("/", "-").replace(".", "").replace("_", "-")


def build_prompt(sample, sys_prompt):
    ctxs_text = "\n\n".join(
        f"[{i+1}] {c.get('title','')}\n{c.get('text','')}"
        for i, c in enumerate(sample["ctxs"])
    )
    user = f"Question: {sample['input']}\n\nPassages:\n{ctxs_text}\n\nWrite a cited answer."
    return sys_prompt, user


def compute_citation_f1(ans: str, ctxs: list) -> dict:
    n = len(ctxs)
    ci = [int(m) for m in re.findall(r"\[(\d+)\]", ans)]
    vc = {i for i in ci if 1 <= i <= n}
    prec = len(vc) / len(set(ci)) if ci else 1.0
    cw_re = re.compile(
        r"\d+[\.\d]*\s*%?|\b(increase[ds]?|decrease[ds]?|higher|lower|greater|less|found|"
        r"showed?|reported|demonstrated|suggest[s]?|associated|correlat|significant|"
        r"improv|reduc|enhanc)\b",
        re.I,
    )
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", ans) if len(s.strip()) > 20]
    cw = [s for s in sents if cw_re.search(s)]
    cwc = [s for s in cw if re.search(r"\[\d+\]", s)]
    rec = len(cwc) / len(cw) if cw else 0.0
    d = prec + rec
    return {
        "citation_precision": round(prec, 3),
        "citation_recall": round(rec, 3),
        "citation_f1": round(2 * prec * rec / d, 3) if d > 0 else 0.0,
        "n_citations_in_answer": len(ci),
    }


def run_httpx(vllm_base, model, sys_prompt, user, max_tokens, qwen_no_think=False):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if qwen_no_think:
        # Belt-and-braces: chat_template override (preferred) + extra_body fallback.
        payload["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    r = httpx.post(
        f"{vllm_base}/chat/completions",
        json=payload,
        headers={"Authorization": "Bearer dummy"},
        timeout=600,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def gen_bare_or_lit(args, condition):
    samples = json.load(open(DATA))
    out_dir = ROOT / f"results/{slugify(args.model)}/bioasq/cond_{condition}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_f = out_dir / "results.jsonl"
    done = {json.loads(l)["id"] for l in out_f.open()} if out_f.exists() else set()
    print(f"[{condition}] total={len(samples)}  done={len(done)}", flush=True)

    sys_prompt = SYS_BARE if condition == "b2" else SYS_LIT
    qwen = "qwen" in args.model.lower()

    with out_f.open("a") as fo:
        for k, s in enumerate(samples):
            if s["id"] in done:
                continue
            sysp, user = build_prompt(s, sys_prompt)
            try:
                ans = run_httpx(args.vllm_base, args.model, sysp, user, args.max_tokens, qwen)
            except Exception as e:
                print(f"  [{k}] {s['id']} ERROR: {str(e)[:120]}", flush=True)
                ans = ""
            cf = compute_citation_f1(ans, s["ctxs"])
            rec = {
                "id": s["id"],
                "benchmark": "bioasq",
                "model": args.model,
                "variant": f"cond_{condition}",
                "question": s["input"],
                "answer": ans,
                "reference": s.get("output", ""),
                "tokens_out": len(ans.split()),
                **cf,
            }
            fo.write(json.dumps(rec) + "\n")
            fo.flush()
            if (k + 1) % 10 == 0:
                print(f"  {k+1}/{len(samples)}", flush=True)
    print(f"[{condition}] DONE", flush=True)


def gen_os(args):
    # Calls external/OpenScholar/run.py with our fixed ctxs.
    os_repo = ROOT / "external/OpenScholar"
    osvenv_py = ROOT / ".venvs/osvenv/bin/python"
    if not os_repo.exists() or not osvenv_py.exists():
        print("ERROR: external/OpenScholar or .venvs/osvenv missing. Run 00_setup.sh first.", file=sys.stderr)
        sys.exit(1)
    out_path = ROOT / f"external/os_io/bioasq_{slugify(args.model)}_os.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["S2_API_KEY"] = "dummy"
    subprocess.check_call(
        [
            str(osvenv_py),
            str(os_repo / "run.py"),
            "--input_file", str(DATA),
            "--output_file", str(out_path),
            "--task_name", "default",
            "--use_contexts",
            "--model_name", args.model,
            "--api", args.vllm_base,
            "--api_key_fp", str(ROOT / "external/dummy_key.txt"),
            "--max_tokens", str(args.max_tokens),
        ],
        env=env,
    )
    # Convert OS output → our results.jsonl
    samples = {str(s["id"]): s for s in json.load(open(DATA))}
    data = json.load(open(out_path)).get("data", [])
    out_dir = ROOT / f"results/{slugify(args.model)}/bioasq/cond_os"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_f = out_dir / "results.jsonl"
    with out_f.open("w") as fo:
        for r in data:
            rid = r.get("id")
            ans = r.get("output") or ""
            if isinstance(ans, (list, dict)):
                ans = str(ans)
            # OS uses 0-indexed [N]; shift +1 to our 1-indexed convention.
            ans = re.sub(r"\[(\d+)\]", lambda m: f"[{int(m.group(1))+1}]", ans)
            s = samples.get(rid, {})
            cf = compute_citation_f1(ans, s.get("ctxs", []))
            fo.write(json.dumps({
                "id": rid, "benchmark": "bioasq", "model": args.model, "variant": "cond_os",
                "question": s.get("input", ""), "answer": ans,
                "reference": s.get("output", ""), "tokens_out": len(ans.split()), **cf,
            }) + "\n")
    print(f"[os] DONE → {out_f}", flush=True)


def gen_pqa(args):
    pqavenv_py = ROOT / ".venvs/pqavenv/bin/python"
    if not pqavenv_py.exists():
        print("ERROR: .venvs/pqavenv missing. Run 00_setup.sh first.", file=sys.stderr)
        sys.exit(1)
    # Inline pqa runner: minimal, fixed-ctxs, model_slug.
    runner = ROOT / "scripts/_pqa_run.py"
    subprocess.check_call(
        [str(pqavenv_py), str(runner), args.model, args.vllm_base],
        env={**os.environ, "BIOASQ_INPUT": str(DATA), "OUT_DIR": str(ROOT / f"results/{slugify(args.model)}/bioasq/cond_pqa")},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=["b2", "lit", "os", "pqa"])
    ap.add_argument("--model", default="google/gemma-4-31B-it")
    ap.add_argument("--vllm-base", default="http://localhost:8010/v1")
    ap.add_argument("--max-tokens", type=int, default=1500)
    args = ap.parse_args()
    _ensure_data()

    if args.condition in ("b2", "lit"):
        gen_bare_or_lit(args, args.condition)
    elif args.condition == "os":
        gen_os(args)
    else:
        gen_pqa(args)


if __name__ == "__main__":
    main()
