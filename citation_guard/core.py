"""citation-guard core — a local, validated citation-faithfulness guard for cited scientific synthesis.

Given an LLM answer that cites a provided context set by [N], verify each cited sentence against its
cited passage with a deterministic, gold-validated attribution model (AttrScore, 3B); if unsupported,
re-attribute the claim to a better provided passage; else flag. The support decision is made by an
external, gold-validated verifier, not by asking the generator to self-check. Runs locally (single GPU
or CPU).

Re-attribution is a swappable, commodity slot. The default ranker is a deterministic lexical BM25 (no
extra model): on gold it recovers the supporting passage about as well as the best open generator and far
better than the verifier score, while staying free and reproducible. Pass ``reattribute="verifier"`` to
rank by the attribution score instead. A lexical ranker only *proposes* a candidate; the verifier then
*re-checks* support before the pointer is moved, so the ranker need not tell support from contradiction.

Scope: checks *attribution locality* (is the claim supported by the cited provided passage), NOT
conclusion correctness (e.g. in-vitro vs clinical) and NOT whether a reference exists in the world. The
verifier is an imperfect instrument validated on gold (supported-class recall ~0.90 on SciFact, ~0.94 on a
held-out split); a split-conformal layer (see ``reproduce/``) turns its score into a distribution-free
catch-rate guarantee. Use flag-mode (default) and treat the output as a triage, not a guarantee.
"""
import math
import re

ATTR_MODEL = "osunlp/attrscore-flan-t5-xl"
# Deployed prompt: a deliberately shortened, higher-recall paraphrase of AttrScore's default. It is
# selected on supported-class recall (the guard's safety axis: do not drop genuine citations) -- 0.90 on
# SciFact vs the canonical prompt's 0.66, and re-validated at 0.94 on a held-out gold split. The canonical
# prompt over-flags inference-requiring supported claims. Prompt choice materially changes results.
PROMPT = ("As an Attribution Validator, verify whether the given reference can support the claim. "
          "Answer with Attributable, Extrapolatory, or Contradictory.\nClaim: {c}\nReference: {r}")
LABELS = ("Attributable", "Extrapolatory", "Contradictory")
CLAIM_CHARS, REF_CHARS, MAX_LEN = 600, 1500, 1024   # one passage/token budget, used everywhere

SENT = re.compile(r"(?<=[.!?])\s+")
CITE = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")   # one citation group: [1] or [1, 2]
NUM = re.compile(r"\d+")
WORD = re.compile(r"[a-z0-9]+")                # BM25 tokenizer (lowercased alphanumerics)

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


def _bm25_best(query: str, cand_idx, texts, k1: float = 1.5, b: float = 0.75):
    """Rank candidate passages (1-indexed in ``cand_idx``) for ``query`` by BM25 and return the best
    index, or ``None`` if nothing overlaps. Deterministic, lexical, no model: the default re-attribution
    ranker. It only proposes a candidate; ``guard()`` re-verifies it with the attribution model before
    moving the pointer, so a lexical ranker is safe even though it cannot tell support from contradiction."""
    from collections import Counter
    toks = {j: WORD.findall(str(texts[j - 1]).lower()) for j in cand_idx}
    q = WORD.findall(str(query).lower())
    if not q or not any(toks.values()):
        return None
    dl = {j: len(t) for j, t in toks.items()}
    avgdl = (sum(dl.values()) / len(dl)) or 1.0
    nc = len(toks)
    df = {t: sum(1 for tt in toks.values() if t in tt) for t in set(q)}
    best, best_s = None, 0.0
    for j, tt in toks.items():
        tf = Counter(tt)
        s = 0.0
        for t in q:
            f = tf.get(t, 0)
            if f:
                idf = math.log((nc - df[t] + 0.5) / (df[t] + 0.5) + 1)
                s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl[j] / avgdl))
        if s > best_s:
            best_s, best = s, j
    return best


def guard(answer: str, ctxs, reattribute=True, remove: bool = False):
    """Run the three-step guard on each cited sentence: verify -> re-attribute -> flag.

    For each cited sentence: (1) *verify* the claim against its cited passage(s); if supported, keep.
    (2) If unsupported and ``reattribute`` is truthy, *re-attribute* by ranking the other provided
    passages and re-pointing the citation to the top one, but only when the verifier confirms that
    passage actually supports the claim (keep the claim, fix only the pointer). (3) Otherwise *flag*
    the citation as ``[N UNVERIFIED]`` (default; ``remove=True`` drops the sentence instead, opt-in,
    no silent deletion in the default flag-mode).

    reattribute: ``True``/``"bm25"`` (default) ranks candidates with a deterministic lexical BM25;
        ``"verifier"`` ranks by the attribution score P(Attributable); ``False`` disables re-attribution.
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
                if reattribute == "verifier":
                    best = max(cands, key=lambda j: p_attributable(claim, texts[j - 1]))
                else:                                    # default: deterministic BM25 (swappable slot)
                    best = _bm25_best(claim, cands, texts)
                if best is not None and supported(claim, texts[best - 1]):   # verifier re-checks before move
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
