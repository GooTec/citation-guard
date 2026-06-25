# Changelog

All notable changes to `citation-guard` will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [SemVer](https://semver.org/).

## [0.2.0] — 2026-06-25

Aligned with the accompanying paper (submitted to ACM Transactions on Intelligent Systems and Technology,
Special Issue on LLM-Driven Agentic AI). The verifier instrument is unchanged; re-attribution and the
surrounding framing are updated to match the evaluated method.

### Changed
- **Re-attribution default is now a deterministic lexical BM25** (no extra model), the swappable-slot
  default in the paper: on QASA gold (n=1375) BM25 recovers the supporting passage at recall@1 0.69, close
  to the best open generator and well above the verifier score (0.58), while staying free and reproducible.
  The verifier still *re-checks* the BM25-proposed passage before any pointer is moved.
- `guard(..., reattribute=...)` accepts `True`/`"bm25"` (default), `"verifier"` (rank by attribution
  score), or `False`. New CLI flag `--reattribute-by {bm25,verifier}`.
- Verifier selection reframed: at a *matched catch rate* the candidate verifiers separate comparably, so
  AttrScore-3B is adopted on **cost** (local) at a high-recall operating point (0.90 on SciFact, 0.94 held
  out), rather than on Cohen's kappa (de-emphasized due to the base-rate paradox).
- Documentation: "trust layer" -> "guard"; "Patterns" -> the ACM TIST submission; QASA evaluated on the
  full test set (n=1375).

### Added
- `_bm25_best()` BM25 ranker and tests for the BM25 default, the `"verifier"` option, and that the default
  path does not follow the verifier score on lexically unrelated passages.

## [0.1.0] — 2026-05-30 (unreleased)

Initial release accompanying the first paper submission (working draft).

### Added
- `citation_guard.core.guard()` — 3-step verify → re-attribute → flag pipeline.
- `citation_guard.cli` — `citation-guard` console script (`pip install citation-guard`).
- `reproduce/` — 1-H100, ~2-hour reproducibility kit (BioASQ n=200, 1 model × 4 pipelines).
  - `reproduce/data/` — BioASQ validation slice, QASA calibration scores (~3 MB).
  - `reproduce/scripts/` — numbered, idempotent (resume-by-id) reproduction scripts.
  - `reproduce/expected_results/` — JSON of expected citF1 / unsupported% / CI per cell.
- `openclaude-plugin/` — *optional* openclaude (`/sci-cite-guard`) skill + opt-in trust-gate hook. **Not a runtime dependency of the pip package.**

### Dependencies
- Runtime: `transformers>=4.40`, `torch>=2.0`. No openclaude dependency.
- Reproduce-only: `vllm==0.21.0`, `paper-qa`, `openai` (declared in `reproduce/requirements.txt`).

### Verifier
- Default verifier: `osunlp/attrscore-flan-t5-xl` (3B, bf16; auto-downloaded from HuggingFace).
- Prompt frozen to the gold-validated configuration (selected on supported-class recall; see the paper).

### Notes
- Code license: MIT. Verifier model weights are subject to the upstream license at `osunlp/attrscore-flan-t5-xl`.
- Reproducibility expectations: citation-F1 means within ±0.01, unsupported-rate point estimates within ±0.5pp on a single H100 80GB (vLLM 0.21 + transformers 4.x; verifier results are deterministic at temperature 0).
