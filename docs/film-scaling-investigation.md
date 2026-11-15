# Контролируемая диагностика масштабирования FiLM

Результаты и решения по стадиям сохраняются в `artefacts/film_scaling_investigation.md`; подробные локальные данные — в `artefacts/film_scaling/`. Это train-only исследование, не оценка generalization и не изменение final pipeline.

## Входные свидетельства

- `artefacts/film_collapse/medium_w9/`: успешное запоминание1152 строк, resolved Hydra config, vocabulary, model state и история.
- `artefacts/film_collapse/verified_receipt.json`: предыдущие проверки freezing, objective, размера и абсолютного warmup.
- `scripts/film_scaling_audit.py`: аудит train и детерминированные вложенные manifests целых пар.
- `scripts/film_scaling_run.py`: загрузка существующего recipe, вызов существующего `Trainer`, train-only измерения и snapshots. Отдельного training loop нет.

## Контроль optimizer steps

Опциональные параметры текущего `Trainer`:

- `train.max_steps`: ограничение числа успешных optimizer updates, включая остановку внутри эпохи; `null` сохраняет прежнее поведение.
- `train.scheduler.total_steps`: абсолютный horizon scheduler, независимый от размера набора и ограничения запуска; `null` использует прежний `steps_per_epoch * epochs`.
- `train.scheduler.warmup_steps`: абсолютный warmup; `null` использует `warmup_epochs`.
- `train.snapshot_steps`: шаги callback для внешних диагностических snapshots. Измерения не являются objective.

Нельзя приравнивать одинаковое число optimizer steps к одинаковому числу эпох. При изменении размера набора одновременно меняются разнообразие и повторные предъявления; без дополнительного контроля это не причинная абляция размера.

## Продолжение со старого checkpoint

Старый `model.pt` не содержит AdamW moments или scheduler. Поэтому обе warm-start ветки A/B одинаково пересоздают optimizer и scheduler, начиная с одних model/BN buffers, с одинаковыми LR/dropout/weight decay и stochastic streams. Контроль A продолжает на исходных1152 строках, B — на полном train. Это restart, не точное продолжение исходной траектории. Если качество падает только у B, одинаковый сброс optimizer сам по себе не объясняет различие.

Каждый новый checkpoint сохраняет model, optimizer, scheduler, step/epoch и RNG. Рядом сохраняются config, vocabulary, ids и logits по полному train. Исходные1152 строки и остальные train examples измеряются отдельно; оставшиеся строки не называются holdout, поскольку могут разделять изображения и captions с исходным subset.

Для image sensitivity используется фиксированная перестановка между разными изображениями; text sensitivity — обмен подписями внутри пары. Обе величины — mean absolute logit delta, не метрика правильного reasoning. Signed margin — logit(true)−logit(false); tie даёт0.5 в pair accuracy.

Все команды запускаются из корня repo с существующей `.venv` и локальными ImageNet weights. Новые output directories должны отсутствовать; старые результаты не перезаписываются. Val/test не читаются. MPS jobs выполняются последовательно; MPS не гарантирует побитовую детерминированность.
