"""citation-guard core — a local, validated citation-faithfulness trust layer.

Given an LLM answer that cites a provided context set by [N], verify each cited sentence against its
cited passage with a deterministic, gold-validated attribution model (AttrScore); if unsupported,
re-attribute the claim to a better provided passage; else flag. Unlike LLM self-verification (which is
unreliable — quality rubrics barely separate answers with 5x more unsupported citations), the decision
is made by an external, validated verifier. Runs locally (3B model, single GPU or CPU).

Scope: checks *attribution locality* (is the claim supported by the cited provided passage), NOT
conclusion correctness (e.g. in-vitro vs clinical) and NOT whether a reference exists in the world.
The verifier is moderate (gold kappa ~0.46) and prompt-sensitive; use flag-mode (default) and treat
the output as a triage, not a guarantee.
"""
import math
import re

ATTR_MODEL = "osunlp/attrscore-flan-t5-xl"
# Deployed prompt: a deliberately shortened, higher-recall paraphrase of AttrScore's default
# (SciFact kappa=0.46, recall 0.90 vs the default's kappa=0.33, recall 0.66). The default over-flags
# inference-requiring supported claims; this one is selected on the supported-class recall axis (the
# guard's safety requirement) and validated on gold. Prompt choice materially changes results.
PROMPT = ("As an Attribution Validator, verify whether the given reference can support the claim. "
          "Answer with Attributable, Extrapolatory, or Contradictory.\nClaim: {c}\nReference: {r}")
LABELS = ("Attributable", "Extrapolatory", "Contradictory")
CLAIM_CHARS, REF_CHARS, MAX_LEN = 600, 1500, 1024   # one passage/token budget, used everywhere

SENT = re.compile(r"(?<=[.!?])\s+")
CITE = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")   # one citation group: [1] or [1, 2]
NUM = re.compile(r"\d+")

_tok = _model = _torch = None
_lab_ids = None


def _load():
    global _tok, _model, _torch, _lab_ids
    if _model is None:
        import sys
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        sys.stderr.write(f"[citation-guard] loading attribution model {ATTR_MODEL} "
                         f"(first run downloads it once, ~3 GB; cached afterwards)...\n")
        _torch = torch
        _tok = AutoTokenizer.from_pretrained(ATTR_MODEL)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _model = AutoModelForSeq2SeqLM.from_pretrained(ATTR_MODEL, dtype=torch.bfloat16).to(dev).eval()
        _lab_ids = [_tok(lab, return_tensors="pt").input_ids.to(dev) for lab in LABELS]
    return _tok, _model


def _encode(claim: str, ref: str):
    tok, model = _load()
    return tok([PROMPT.format(c=str(claim)[:CLAIM_CHARS], r=str(ref)[:REF_CHARS])],
               return_tensors="pt", truncation=True, max_length=MAX_LEN).to(model.device)


def p_attributable(claim: str, ref: str) -> float:
    """Continuous score s = P(Attributable | claim, ref): softmax over the teacher-forced total
    log-probabilities of the three label strings {Attributable, Extrapolatory, Contradictory}.
    This is the score the conformal layer calibrates and that re-attribution ranks by."""
    if not str(ref).strip():
        return 0.0
    tok, model = _load()
    enc = _encode(claim, ref)
    lps = []
    with _torch.no_grad():
        for lab in _lab_ids:
            out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, labels=lab)
            lps.append(-out.loss.item() * lab.shape[1])   # total log-prob of the label sequence
    m = max(lps)
    ex = [math.exp(lp - m) for lp in lps]
    return ex[0] / sum(ex)


def supported(claim: str, ref: str) -> bool:
    """Binary verify step: True iff AttrScore's greedy verdict for (claim, ref) is Attributable."""
    if not str(ref).strip():
        return False
    tok, model = _load()
    enc = _encode(claim, ref)
    with _torch.no_grad():
        out = tok.decode(model.generate(**enc, max_new_tokens=8)[0], skip_special_tokens=True)
    return out.strip().lower().startswith("attribut")


def guard(answer: str, ctxs, reattribute: bool = True, remove: bool = False):
    """Run the three-step guard on each cited sentence: verify -> re-attribute -> flag.

    For each cited sentence: (1) *verify* the claim against its cited passage(s); if supported, keep.
    (2) If unsupported and ``reattribute`` is set, *re-attribute* by ranking the other provided
    passages by P(Attributable) and re-pointing the citation to the top one, but only when that
    passage actually supports the claim (keep the claim, fix only the pointer). (3) Otherwise *flag*
    the citation as ``[N UNVERIFIED]`` (default; ``remove=True`` drops the sentence instead, opt-in,
    no silent deletion in the default flag-mode).

    ctxs: list of dicts with a 'text' field (the provided passages, 1-indexed by [N]).
    Returns (verified_answer, report).
    """
    n = len(ctxs)
    texts = [str(c.get("text", "")) for c in ctxs]   # truncation is applied uniformly in _encode
    out_sents, audit = [], []
    n_cited = n_verified = n_reattr = n_flagged = 0
    for s in SENT.split(answer or ""):
        cs = [int(x) for g in CITE.findall(s) for x in NUM.findall(g)]
        cs = [c for c in cs if 1 <= c <= n]
        if not cs:
            out_sents.append(s)
            continue
        n_cited += 1
        claim = CITE.sub("", s).strip()
        if supported(claim, " ".join(texts[c - 1] for c in cs)):
            n_verified += 1
            out_sents.append(s)
            audit.append({"claim": claim[:160], "cited": cs, "status": "verified"})
            continue
        hit = None
        if reattribute:
            cands = [j for j in range(1, n + 1) if j not in cs]
            if cands:
                best = max(cands, key=lambda j: p_attributable(claim, texts[j - 1]))
                if supported(claim, texts[best - 1]):   # re-verify the top-ranked passage
                    hit = best
        if hit is not None:
            n_reattr += 1
            s2 = CITE.sub("\x00", s, count=1)        # mark the first citation group
            s2 = CITE.sub("", s2)                    # drop any remaining (now-stale) groups
            s2 = s2.replace("\x00", f"[{hit}]")      # insert the re-attributed citation
            out_sents.append(s2)
            audit.append({"claim": claim[:160], "cited": cs, "status": "re-attributed", "to": hit})
        else:
            n_flagged += 1
            if remove:
                audit.append({"claim": claim[:160], "cited": cs, "status": "removed"})
            else:
                out_sents.append(CITE.sub(lambda m: m.group(0)[:-1] + " UNVERIFIED]", s))
                audit.append({"claim": claim[:160], "cited": cs, "status": "flagged"})
    report = {
        "n_cited": n_cited, "verified": n_verified, "re_attributed": n_reattr, "flagged": n_flagged,
        "manual_check_targets": n_flagged,
        # fraction of cited sentences a reviewer need not re-check by hand; re-attributed sentences are
        # verifier-self-scored (their direction is audited in the paper, not their exact count).
        "manual_check_reduction": round(1 - n_flagged / n_cited, 3) if n_cited else None,
        "audit": audit,
    }
    return " ".join(out_sents), report
