"""Smoke tests for citation-guard: exercise the verify -> re-attribute -> flag dispatch, citation
parsing, and report counts WITHOUT loading the 3 GB attribution model (the model-backed functions
are monkeypatched). Run: pytest -q."""
import re

import citation_guard.core as core
from citation_guard import guard, p_attributable, supported  # noqa: F401  (import surface check)


def _kw(text):
    return set(re.findall(r"[a-z]+", text.lower()))


def _fake_supported(claim, ref):
    # toy stand-in for AttrScore: "supported" iff claim and ref share an alphabetic token
    return bool(_kw(claim) & _kw(ref))


def _fake_p(claim, ref):
    return 1.0 if _fake_supported(claim, ref) else 0.0


def _patch(monkeypatch):
    monkeypatch.setattr(core, "supported", _fake_supported)
    monkeypatch.setattr(core, "p_attributable", _fake_p)


def test_three_tier_dispatch(monkeypatch):
    """One sentence each that should verify, re-attribute, and flag."""
    _patch(monkeypatch)
    ctxs = [{"text": "alpha"}, {"text": "qqq"}, {"text": "beta"}]
    answer = "Alpha claim [1]. Beta claim [2]. Zeta claim [2]."
    verified, r = guard(answer, ctxs)
    assert r["n_cited"] == 3
    assert r["verified"] == 1        # Alpha [1] -> cited passage 1 supports
    assert r["re_attributed"] == 1   # Beta [2] -> re-pointed to passage 3 (BM25 lexical match on "beta")
    assert r["flagged"] == 1         # Zeta [2] -> no provided passage overlaps, none supports
    assert "[3]" in verified         # re-attribution moved the pointer to passage 3
    assert "UNVERIFIED" in verified  # flag-mode leaves a visible marker, no deletion


def test_bm25_reattribution_is_default(monkeypatch):
    """Default re-attribution ranks candidates by BM25 (lexical), then the verifier re-checks support."""
    # The claim shares words with passage 3; passage 1 does not. The verifier confirms any overlap.
    monkeypatch.setattr(core, "supported", _fake_supported)
    ctxs = [{"text": "unrelated tokens"}, {"text": "cited but wrong"}, {"text": "mitochondria membrane potential"}]
    verified, r = guard("Mitochondria membrane potential drives ATP synthesis [2].", ctxs)
    assert r["re_attributed"] == 1
    assert "[3]" in verified and "[1]" not in verified


def test_verifier_reattribution_ranks_by_score(monkeypatch):
    """With reattribute='verifier', ranking uses P(Attributable), not lexical overlap."""
    # No lexical overlap with any candidate, so BM25 would find nothing; the verifier score must drive it.
    monkeypatch.setattr(core, "supported", lambda c, r: "high" in r)
    monkeypatch.setattr(core, "p_attributable", lambda c, r: 0.9 if "high" in r else 0.1)
    ctxs = [{"text": "low score"}, {"text": "cited but unsupported"}, {"text": "high score"}]
    verified, r = guard("Some claim [2].", ctxs, reattribute="verifier")
    assert r["re_attributed"] == 1
    assert "[3]" in verified and "[1]" not in verified


def test_bm25_does_not_use_verifier_score(monkeypatch):
    """On the default path, a high verifier score on a lexically-unrelated passage is NOT followed."""
    # BM25 finds no overlap -> nothing proposed -> flagged, even though p_attributable would score passage 3 high.
    monkeypatch.setattr(core, "supported", lambda c, r: "high" in r)
    monkeypatch.setattr(core, "p_attributable", lambda c, r: 0.9 if "high" in r else 0.1)
    ctxs = [{"text": "zzz"}, {"text": "cited"}, {"text": "high score"}]
    verified, r = guard("Some claim [2].", ctxs)            # default = bm25
    assert r["flagged"] == 1 and r["re_attributed"] == 0    # no lexical overlap -> BM25 proposes nothing


def test_flag_is_default_no_silent_deletion(monkeypatch):
    """Default mode flags (keeps the sentence with a marker); --remove drops it."""
    monkeypatch.setattr(core, "supported", lambda c, r: False)
    monkeypatch.setattr(core, "p_attributable", lambda c, r: 0.0)
    ctxs = [{"text": "unrelated"}]
    flagged_ans, rf = guard("Unsupported claim [1].", ctxs)            # default = flag
    assert rf["flagged"] == 1 and "UNVERIFIED" in flagged_ans
    removed_ans, rr = guard("Unsupported claim [1].", ctxs, remove=True)
    assert rr["flagged"] == 1 and "UNVERIFIED" not in removed_ans      # sentence dropped


def test_citation_regex_and_uncited_passthrough(monkeypatch):
    _patch(monkeypatch)
    ctxs = [{"text": "alpha"}, {"text": "beta"}]
    verified, r = guard("Grouped cite [1, 2]. A plain sentence with no citation.", ctxs)
    assert r["n_cited"] == 1                       # only the cited sentence counts
    assert "A plain sentence with no citation." in verified
