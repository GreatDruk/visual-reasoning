# Repository conventions

The active work is the train-only visual foundation, ResNet plus SigLIP. Read `KaggleTemplate.md` and the selected YAML before modifying experiments. Course rules live in `docs/rules.md`; the working constraint is stricter: test may be used for inference and pairing, never for fitting preprocessing, representations, features or models.

- Keep one notebook, `notebooks/base.ipynb`. Put reusable logic in `src`, finished-code checks in `tests`. Do not add experiment-specific scripts or notebook code generators to the maintained tree.
- Configurations are `configs/NNN_word.yaml`: a globally increasing three-digit ID and exactly one lowercase word. Historical IDs 001–018 are archived. Allocate the next unused ID; never recycle an ID. Explain hypotheses inside YAML, not in filenames. A seed repeat is a run of the same configuration.
- `inputs` is immutable. `.work` is scratch. `artefacts` has exactly `results`, `weights`, `kaggle`, `archive`; runtime must never read from it, including its own checkpoints. No implicit cache or checkpoint fallback into outputs.
- Put temporary analyses, pictures, validation annotations, migration helpers and local dependency work in ignored `docs/.research` or `.work`. Do not import them from maintained code or package them for Kaggle.
- Record verified findings, planned experiment status, sources and results in `KaggleTemplate.md`. Distinguish historical val, fresh training, smoke checks and confirmed Kaggle results. Report ties separately.
- Change one branch at a time against the fixed `r50_img + sigft` reference. Keep the other branch, blend weight, parser and executor fixed. Revisit later ensemble layers only after a substantial foundation improvement.
- Run relevant tests. Never describe the CPU smoke checks as a full T4 training result or as validation of the 50-minute limit.

The former implementation is preserved in `docs/archive/legacy.zip` with checksums. It is historical evidence, not an active runtime dependency.
