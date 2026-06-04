# citation-guard — openclaude integration

Use `citation-guard` inside openclaude as a **deterministic citation trust layer** (a validated
replacement for the LLM-self-verify `sci-cite-verify` skill).

## Simplest: one-command plugin install
```text
/plugin marketplace add  https://github.com/GooTec/citation-guard   # or a local clone dir
/plugin install          citation-guard@bionexus-citation-guard
```
Then call **`/sci-cite-guard`** after any cited synthesis. The skill **auto-installs the engine** on
first use (`pipx`/`pip install citation-guard`), so steps 1–2 below are handled for you. The sections
below are the manual equivalent / details.

## 1. Install the engine (manual; skill does this automatically)
```bash
pip install citation-guard          # or: pip install -e /path/to/citation-guard
```
(First run downloads the 3B attribution model `osunlp/attrscore-flan-t5-xl` from Hugging Face, ~3 GB once.)

## 2. Add the skill
Copy the skill into your openclaude skills dir:
```bash
cp -r openclaude-plugin/skills/sci-cite-guard  <your-project>/.claude/skills/
```
Now the agent can invoke **`/sci-cite-guard`** after any cited synthesis: it writes `{question, answer,
ctxs}` to JSON, runs `citation-guard`, and returns the verified answer + which citations were verified /
re-attributed / flagged — you only re-check the **flagged** ones.

## 3. (Optional, opt-in) run it automatically on every synthesis
By default the guard is invoked on demand via the skill. If you want it to run **automatically** as a
standing trust gate, add a hook to `.claude/settings.json`. ⚠️ This loads a 3B model on each trigger —
enable only if you want continuous checking and have a GPU.

```jsonc
// .claude/settings.json  (opt-in — adjust the matcher/command to your workflow)
{
  "hooks": {
    "Stop": [
      {
        "matcher": "*",
        "hooks": [
          { "type": "command",
            "command": "citation-guard --input \"$CLAUDE_LAST_SYNTHESIS_JSON\"",
            "statusMessage": "citation-guard: flag unsupported citations" }
        ]
      }
    ]
  }
}
```
(You must arrange for the agent's last cited answer + its ctxs to be written to a JSON the hook reads —
the `/sci-cite-guard` skill already does this on demand, which is the recommended, lower-overhead path.)

## Notes
- **Flag-mode is the safe default** (no silent deletion); the full pipeline verify -> re-attribute ->
  flag runs by default. Use `--no-reattribute` out-of-domain (where re-attribution ranking is
  unreliable), and `--remove` only when brevity matters more than recall.
- Scope: checks attribution *locality* (claim ⊆ cited passage), not conclusion correctness or whether a
  reference exists in the world. Verifier is moderate (gold κ≈0.5) — treat output as triage.
- The older `sci-cite-verify` skill (LLM self-verification) is **deprecated** — our data shows LLM
  self-verification does not reliably catch citation drift; use `sci-cite-guard`.

## Install as a plugin (verified against current openclaude schema)
This directory **is** a ready openclaude plugin. Verified against this build's
`src/utils/plugins/schemas.ts` (`PluginManifestSchema`) + `src/schemas/hooks.ts` on 2026-05-27:

- Manifest lives at **`.claude-plugin/plugin.json`** (already present here).
- Component dirs are auto-discovered: **`skills/`** (each `<skill>/SKILL.md`) and, if present,
  **`hooks/hooks.json`**, `commands/`, `agents/`. The manifest's optional `skills` / `hooks` fields
  *add to* these convention dirs.
- ⚠️ Field names are **`skills`** and **`hooks`** — *not* `skillsPath` / `hooksConfig` (those were an
  earlier/incorrect guess). `name` must be kebab-case (no spaces); `version` is semver; `author` is
  `{ name, email?, url? }`.

```jsonc
// .claude-plugin/plugin.json  (this is what ships here; skills/ is auto-discovered, no `skills` field needed)
{
  "name": "citation-guard",
  "version": "0.1.0",
  "description": "Local, validated citation-faithfulness guard (verify / re-attribute / flag).",
  "author": { "name": "BioNexus" },
  "license": "MIT",
  "keywords": ["hallucination", "citation", "attribution", "faithfulness", "RAG"]
}
```

To use locally: add this dir as a plugin via `/plugin` (or a local marketplace pointing at it).
The opt-in `Stop` hook above (§3) is intentionally **not** bundled in `hooks/hooks.json` (that would
auto-run on enable) — keep it in user `settings.json` so the 3B model loads only when you opt in.
