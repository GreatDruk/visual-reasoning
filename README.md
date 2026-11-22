# Visual reasoning

Current Kaggle entry point: `notebooks/base.ipynb` with `artefacts/kaggle/057.zip`, competition input, T4 ×2, Internet Off. It caches historical R50/thaw/sigft/teacher and retrains C044, U045 and counters053/054 before applying the fixed057 composition. The already-submitted `moodle/` folder is unchanged. Cold066 and full frozen inference are not the current notebook default.

Public control: **057_quality — 0.99978**, reported by the team. The current study strengthens or simplifies its existing count branch. It does not add all alternatives to the final ensemble.

## Run the current study

Study 064/065 is complete and is not recommended for another run or submission: neither repairs an error of 057; native refit introduces one error and doubled resolution introduces three. The instructions below reproduce the completed study. Retain 057; detailed count and pair attribution is recorded in KaggleTemplate.md.

Update the existing runtime private dataset with `artefacts/kaggle/detail.zip`; retain the original base input dataset. Attach only one runtime overlay. Kaggle unpacks ZIP uploads automatically. Open the single `notebooks/base.ipynb`, choose T4 ×2, Internet Off, and Restart & Run All.

Defaults:

```python
SELECTED = ["064_detail", "065_detail"]
SEEDS = None
REPLAY = []
```

The current study compares native and doubled input resolution using the same train-only 060/421 checkpoint. CNN/FPN weights stay frozen; spatial features are extracted once onto each GPU, then only the existing readout and mass head train. 064 is the native-resolution refit control; 065 changes only the scale. These runs evaluate validation and save weights, without exporting test CSVs. Download the complete Output. Public leader 057 remains unchanged; 061–063 are not recommended for more submissions.

| Config | Single change from 054/057 |
|---|---|
| 058_count | Supervise the inserted class +1 inside its known donor box and zero outside |
| 059_count | Joint class-density correction on ConvNeXt |
| 060_count | Remove paired insertion and its second image forward |
| 061_slim | Completed: slightly worse public; retain 057 |
| 062_mixed | Replace only 054/421 with 060/421, keeping 054/853 |
| 063_consensus | Require primary confirmation for the existing auxiliary swap |
| 064_detail | Frozen-feature readout refit at native resolution, initialized from 060/421 |
| 065_detail | Same as 064 with doubled image and feature-map resolution |

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
