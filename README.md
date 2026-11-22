# Visual reasoning

Public control: **057_quality — 0.99978**, reported by the team. The current study strengthens or simplifies its existing count branch. It does not add all alternatives to the final ensemble.

## Run the current study

Update the existing runtime private dataset with `artefacts/kaggle/study.zip`; retain the original base input dataset. Kaggle unpacks ZIP uploads automatically. Open the single `notebooks/base.ipynb`, choose T4 ×2, Internet Off, and Restart & Run All.

Defaults:

```python
SELECTED = ["060_count"]
SEEDS = [853]
REPLAY = ["057_quality", "062_mixed", "063_consensus"]
```

The quick replay first writes `057_quality.csv`, `062_mixed.csv` and `063_consensus.csv` under `/kaggle/working`, without image encoding or fitting. Set `SELECTED=[]` for only that replay. 061 was slightly worse on public; 058/059 are complete without a useful improvement. Repeat only 060, then inspect its matched 054/853 control. Existing 057 is reproduced from registered frozen components. Live 061 inference also skips both 053 checkpoint networks; a live 057 rerun needs the previous full deployment bundle, while its frozen replay remains available in this study bundle.

| Config | Single change from 054/057 |
|---|---|
| 058_count | Supervise the inserted class +1 inside its known donor box and zero outside |
| 059_count | Joint class-density correction on ConvNeXt |
| 060_count | Remove paired insertion and its second image forward |
| 061_slim | Completed: slightly worse public; retain 057 |
| 062_mixed | Replace only 054/421 with 060/421, keeping 054/853 |
| 063_consensus | Require primary confirmation for the existing auxiliary swap |

Training uses train only, fixed 18 epochs and effective batch 96, with common decoded RGB and foreground caches. Each completed count run saves `candidate.csv` inside `artefacts/results/<config>/<seed>/<timestamp>`. It replaces the count slot of 057 at weight .25, preserving the other branches. Test is used only for inference/export. Partial smoke runs do not export competition submissions.

Read `leader_replacement` / `leader_metrics` for comparison against full 057; the ordinary `metrics` field retains the historical L2 comparison. Reports also include the matched single-seed 054 control, absolute count errors, pair errors/ties, training telemetry and source provenance. Download the complete Output to retain the canonical trained weights; result ZIPs intentionally avoid duplicating them.

## Layout

- `src`: reusable runtime, training, export and packaging code.
- `configs`: monotonically increasing `NNN_word.yaml` configurations.
- `notebooks/base.ipynb`: the only maintained notebook.
- `tests`: finished-code regression checks.
- `inputs`: explicit immutable model/teacher assets.
- `artefacts/{results,weights,kaggle,archive}`: output only; runtime never reads them.
- `.work`: disposable caches. `docs/.research`: ignored analyses and temporary studies.
- `KaggleTemplate.md`: verified evidence, sources, experiment status and protocols.
- `docs/archive/legacy.zip`: historical evidence, never a runtime import.

`python -m src.package` explicitly packages chosen full train-only runs and registered assets. Runtime uses checksummed read-only input datasets, never checkpoint fallback into outputs. Frozen component layouts are checked against CSV hashes, row IDs and scene order.

The measured 5:43 for 056/057 is frozen inference including val/test, with cached SigLIP predictions. It is not fresh training. The complete final dependency graph still needs an end-to-end measurement against the 50-minute limit.
