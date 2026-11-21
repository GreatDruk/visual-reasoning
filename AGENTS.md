# Repository conventions

The active work is the train-only visual foundation and the next object-recovery/union layer. Preserve three candidate foundations: ConvNeXt 044 plus frozen SigLIP, ResNet 023 plus frozen SigLIP, and ConvNeXt 044 alone. Read `KaggleTemplate.md` and the selected YAML before modifying experiments. Course rules live in `docs/rules.md`; the working constraint is stricter: test may be used for inference and pairing, never for fitting preprocessing, representations, features or models.

- Keep one notebook, `notebooks/base.ipynb`. Put reusable logic in `src`, finished-code checks in `tests`. Do not add experiment-specific scripts or notebook code generators to the maintained tree.
- Configurations are `configs/NNN_word.yaml`: a globally increasing three-digit ID and exactly one lowercase word. Historical IDs 001–018 are archived. Allocate the next unused ID; never recycle an ID. Explain hypotheses inside YAML, not in filenames. A seed repeat is a run of the same configuration.
- `inputs` is immutable. `.work` is scratch. `artefacts` has exactly `results`, `weights`, `kaggle`, `archive`; runtime must never read from it, including its own checkpoints. No implicit cache or checkpoint fallback into outputs.
- Put temporary analyses, pictures, validation annotations, migration helpers and local dependency work in ignored `docs/.research` or `.work`. Do not import them from maintained code or package them for Kaggle.
- Record verified findings, planned experiment status, sources and results in `KaggleTemplate.md`. Distinguish historical val, fresh training, smoke checks and confirmed Kaggle results. Report ties separately.
- Change one factor at a time. Keep the historical `r50_img + sigft` reference and the three documented candidate foundations fixed when testing later layers; report gains over each foundation separately. Keep parser/executor fixed. The user authorized moving to the recovery/union layer after the 041–044 analysis; only one independent-seed confirmation of 044 is currently prioritized at the foundation level. Do not silently add all candidate foundations to the deployed ensemble.
- Run relevant tests. Never describe the CPU smoke checks as a full T4 training result or as validation of the 50-minute limit.

The former implementation is preserved in `docs/archive/legacy.zip` with checksums. It is historical evidence, not an active runtime dependency.
