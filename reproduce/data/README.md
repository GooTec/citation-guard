# Reproduction data — sources and licences

These files support `reproduce/scripts/`. They are derived artifacts or structural manifests, not
raw third-party datasets. We do **not** redistribute BioASQ passages; we rebuild the slice from the
public source on demand.

| file | what it is | source | licence / terms |
|------|------------|--------|-----------------|
| `bioasq_slice_manifest.json` | Structural manifest of the n=200 validation slice: row indices, PubMed IDs, and question hashes only — **no passage text**. | derived from BioASQ Task B (via `rag-datasets/rag-mini-bioasq`) | MIT (this repo). Holds only identifiers, not third-party text. |
| `make_bioasq_slice.py` | Rebuilds `bioasq_validation_n200.json` from the public dataset using the manifest, then checksum-verifies it (`md5 == expected`). | — | MIT (this repo). |
| `qasa_calibration_scores.json` | Per-item AttrScore P(Attributable) scores for the QASA split-conformal calibration (n=500 supported / 500 distractor). **Derived scores only — no third-party text.** | QASA (Lee et al.), via ScholarQABench `single_paper_tasks/qasa_test.jsonl` | Scores are our output (MIT). QASA itself: see the QASA / ScholarQABench upstream licences. |

## Rebuilding the BioASQ slice

The slice `bioasq_validation_n200.json` is **not** shipped (BioASQ passages are published PubMed
abstracts under the BioASQ challenge terms, and carry author-contact metadata). Rebuild it from the
public `rag-datasets/rag-mini-bioasq` HuggingFace dataset:

```bash
python data/make_bioasq_slice.py                       # downloads from HuggingFace (needs network)
python data/make_bioasq_slice.py --from-parquet DIR    # offline: DIR/{qa,corpus}.parquet
```

The script reconstructs the exact slice (`output = answer.strip()`,
`json.dumps(..., ensure_ascii=False, indent=2)`) and aborts unless the md5 matches the manifest's
`expected_md5` (`d47b5cfd70b83e6f49822a6d005f0048`). The per-question hash check fails loudly if the
upstream row order ever changes, so the reproduction can never silently drift. The reproduce scripts
(`02_generate.py`, `03_guard.py`, `reproduce_minimal.sh`) build it automatically on first use.

ScholarQABench / SciFact raw data are **not** redistributed here either; obtain them from their
upstream repositories (e.g. `AkariAsai/OpenScholar`, `allenai/scifact`) under their own licences.
