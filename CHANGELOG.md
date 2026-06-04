# Changelog

All notable changes to `citation-guard` will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [SemVer](https://semver.org/).

## [0.1.0] — 2026-05-30 (unreleased)

Initial release accompanying the *Patterns* submission (working draft v0.8).

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
- Prompt frozen to the gold-validated configuration (SciFact κ=0.46, QASA κ=0.52; see paper §C2, Table 3).

### Notes
- Code license: MIT. Verifier model weights are subject to the upstream license at `osunlp/attrscore-flan-t5-xl`.
- Reproducibility expectations: citation-F1 means within ±0.01, unsupported-rate point estimates within ±0.5pp on a single H100 80GB (vLLM 0.21 + transformers 4.x; verifier results are deterministic at temperature 0).
