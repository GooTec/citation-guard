---
name: "Scientific Citation Guard (deterministic, local)"
description: "After a cited scientific synthesis, verify every cited sentence against the provided passages with a gold-validated local attribution model (AttrScore), re-attribute mis-cited claims to a better provided passage, and flag the genuinely unsupported — a deterministic alternative to LLM self-verification, which does not reliably catch citation drift."
allowed-tools: ["Bash", "Read", "Write"]
effort: "low"
---

# /sci-cite-guard

A **deterministic, local** citation trust layer for cited scientific synthesis. Use it as the
post-synthesis check instead of (or after) `/sci-cite-verify`.

## Why not just ask the model to self-verify?

`/sci-cite-verify` implements OpenScholar's *LLM self-verification*. We measured that this does not
work: even when explicitly asked to score citation support and shown the passages, LLM judges barely
separate answers with 5× more unsupported citations (per-item corr with the true unsupported rate
≈ −0.1 to −0.26). So this skill delegates the decision to a **deterministic attribution model
(AttrScore, validated on SciFact/QASA gold, κ≈0.5)** rather than the LLM's own judgment.

## When to invoke

After producing or refining a scientific synthesis answer that cites a fixed, provided passage set by
`[N]`. You need the answer text and the list of passages (`ctxs`).

## Protocol

### Step 1 — assemble input
Write the answer and passages to a JSON file:
```json
{"question": "<the question, optional>",
 "answer": "<synthesis text with [N] citations>",
 "ctxs": [{"text": "<passage 1 text>"}, {"text": "<passage 2 text>"}, ...]}
```
Use the **Write** tool to save it to `/tmp/cite_guard_in.json`.

### Step 2 — run the guard (deterministic, local, single-GPU)
Run the guard. The first line **auto-installs the engine** if it isn't already present, so there is no
separate setup step for the user:
```bash
command -v citation-guard >/dev/null 2>&1 || pipx install citation-guard 2>/dev/null || pip install --user citation-guard
citation-guard --input /tmp/cite_guard_in.json --out /tmp/cite_guard_out.json
```
(First use downloads the 3B attribution model from Hugging Face, ~3 GB, once; GPU recommended, CPU works
but is slow. If you cloned this repo instead of installing from PyPI, use `pip install -e citation-guard/`.)
- **default** (full 3-tier verify -> re-attribute -> flag): keep verified citations; re-point mis-cited
  claims to a better provided passage; flag the rest as `[N UNVERIFIED]` (no silent deletion).
- `--no-reattribute`: skip re-attribution, only flag unsupported (most conservative; use out-of-domain).
- `--remove`: drop unsupported cited sentences (use only when brevity matters more than recall).
The guard prints a one-line summary to stderr (`cited / verified / re-attributed / flagged → manual
checks reduced N%`) and writes the full result to the `--out` file.

### Step 3 — return to the user
Read `/tmp/cite_guard_out.json` and present:
- the **verified answer** (`verified_answer`), and
- the **audit** (`report.audit`): which cited sentences were verified, re-attributed (and to which
  passage), or flagged.

Tell the user they only need to manually check the **flagged** sentences — the verified ones are
supported by their cited passage per the validated verifier, and the re-attributed ones now point at a
passage that supports the claim. State the residual caveat: the verifier's gold recall is ≈0.90, so a
small fraction of unsupported citations among the "verified" set may be missed; flag-mode is the
default for this reason (no silent deletion).

## Output format

```
## Citation guard (deterministic, AttrScore)
cited: N | verified: X | re-attributed: Y | flagged: Z  → manual checks reduced ~P%

## Verified answer
<verified_answer with re-attributed citations fixed and unsupported ones marked [N⚠UNVERIFIED]>

## To check manually (flagged only)
- "<claim excerpt>"  (cited [k]; no provided passage supports it)
```

## Notes
- Runs locally on a single GPU with a 3B model; no API/frontier dependency.
- The verifier prompt is the gold-validated configuration (SciFact κ=0.46); the canonical AttrScore
  prompt over-flags (κ=0.33). Do not change the prompt without re-validating on gold.
- Scope: this checks *attribution locality* (is the claim supported by the cited provided passage),
  not *conclusion correctness* (e.g., in-vitro vs clinical, contradictory evidence) and not whether a
  reference exists in the world.
