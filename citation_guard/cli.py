"""citation-guard CLI.

  citation-guard --input answer.json [--no-reattribute] [--remove] [--out report.json]
  echo '{"answer":"...[1]...","ctxs":[{"text":"..."}]}' | citation-guard

Input JSON: {"answer": "<text with [N] citations>", "ctxs": [{"text": "..."}, ...], "question": "(optional)"}
Output JSON: {"verified_answer": "...", "report": {n_cited, verified, re_attributed, flagged,
              manual_check_reduction, audit: [...]}}
"""
import sys
import json
import argparse
from pathlib import Path

from .core import guard

# Bundled example: exercises all three tiers (verify / re-attribute / flag) end-to-end.
_SELFTEST = {
    "answer": ("Aspirin reduces inflammation [1]. Compound X binds the receptor with high affinity [1]. "
               "The drug cured every cancer in all patients [2, 1]."),
    "ctxs": [{"text": "Aspirin is a nonsteroidal anti-inflammatory drug that reduces inflammation."},
             {"text": "Compound X binds the target receptor with high affinity in vitro."}],
}


def selftest():
    """One-command check: confirms install + warms the model + verifies the 3-tier guard works."""
    sys.stderr.write("[citation-guard] self-test: loading model (first run downloads ~3 GB from "
                     "HuggingFace, one time) and running the bundled example...\n")
    verified, r = guard(_SELFTEST["answer"], _SELFTEST["ctxs"])
    ok = (r["n_cited"] == 3 and r["verified"] >= 1 and r["re_attributed"] >= 1 and r["flagged"] >= 1)
    print(json.dumps({"verified_answer": verified, "report": r}, indent=2, ensure_ascii=False))
    sys.stderr.write(
        f"[citation-guard] self-test: cited={r['n_cited']} verified={r['verified']} "
        f"re-attributed={r['re_attributed']} flagged={r['flagged']} -> "
        f"{'PASS ✓ install works' if ok else 'CHECK — unexpected counts'}\n")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="citation-guard",
        description="Local validated citation-faithfulness guard (AttrScore 3-tier: verify/re-attribute/flag).")
    ap.add_argument("--input", help="input JSON file (defaults to stdin)")
    ap.add_argument("--no-reattribute", action="store_true",
                    help="skip re-attribution (verify then flag only); use out-of-domain, where "
                         "re-attribution ranking is unreliable")
    ap.add_argument("--remove", action="store_true",
                    help="drop unsupported cited sentences instead of flagging them "
                         "(opt-in; default is flag-mode, no silent deletion)")
    ap.add_argument("--out", help="write full JSON result here (defaults to stdout)")
    ap.add_argument("--selftest", action="store_true",
                    help="run the bundled example to confirm the install works (and warm the model), then exit")
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest()

    raw = Path(a.input).read_text() if a.input else sys.stdin.read()
    data = json.loads(raw)
    verified, report = guard(data.get("answer", ""), data.get("ctxs", []),
                             reattribute=not a.no_reattribute, remove=a.remove)
    result = {"verified_answer": verified, "report": report}

    r = report
    sys.stderr.write(
        f"[citation-guard] cited={r['n_cited']} verified={r['verified']} re-attributed={r['re_attributed']} "
        f"flagged={r['flagged']}  -> manual checks reduced {100 * (r['manual_check_reduction'] or 0):.0f}%\n")
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    if a.out:
        Path(a.out).write_text(payload)
        print(f"written: {a.out}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
