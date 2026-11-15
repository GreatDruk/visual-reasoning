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

Старый `model.pt` не содержит AdamW moments или scheduler. Поэтому обе warm-start ветки A/B одинаково пересоздают optimizer и scheduler, начиная с одних model/BN buffers, с одинаковыми LR/dropout/weight decay и stochastic streams. Контроль A продолжает на исходных1152 строках, B — на полном train. Это restart, не точное продолжение исходной траектории. Падение B оценивается относительно A на одинаковых steps: сам контроль тоже может временно ухудшиться после возвращения peak LR. Взаимодействие reset с новым распределением не исключается.

Каждый новый checkpoint сохраняет model, optimizer, scheduler, step/epoch и RNG. Рядом сохраняются config, vocabulary, ids и logits по полному train. Исходные1152 строки и остальные train examples измеряются отдельно; оставшиеся строки не называются holdout, поскольку могут разделять изображения и captions с исходным subset.

Для image sensitivity используется фиксированная перестановка между разными изображениями; text sensitivity — обмен подписями внутри пары. Обе величины — mean absolute logit delta, не метрика правильного reasoning. Signed margin — logit(true)−logit(false); tie даёт0.5 в pair accuracy.

В receipt дополнительно сохраняется image delta / logit std: raw sensitivity может упасть из-за уменьшения уверенности, а не полной потери зависимости от изображения.

Все команды запускаются из корня repo с существующей `.venv` и локальными ImageNet weights. Новые output directories должны отсутствовать; старые результаты не перезаписываются. Val/test не читаются. MPS jobs выполняются последовательно; MPS не гарантирует побитовую детерминированность.

## Воспроизведение и проверка

Нужен исходный каталог `artefacts/film_collapse/medium_w9/` с config, vocabulary, diagnostic manifest, ids и model.pt. Его initial state проверяется точным SHA-256; old checkpoint не перезаписывается.

```bash
bash scripts/film_scaling_reproduce.sh artefacts/film_scaling_reproduction_02
```

Сценарий последовательно выполняет audit, общие read-only baseline measurements,4608-row scaling, warm-start A/B/C, выбранную read-only диагностику и численную проверку.9216-row manifest создаётся, но fit на нём не запускается: на4608 уже обнаружена деградация, а следующий размер при тех же steps не разделит diversity и repetitions.

`scripts/film_scaling_receipt.py` проверяет actual optimizer counters, checkpoint/model identity, sampler RNG, replay сохранённых batches, LR trajectories, BN modes и пересчитывает BCE/AUC/pair accuracy из aligned logits. Full-order hash реконструирован из sampler/seed, не является online записью каждого batch. Для B/C отдельно измеряется качество seen/unseen строк дополнительной фазы.

`scripts/film_scaling_gradients.py` не вызывает optimizer: gradients измеряются на сохранённых states с выключенным dropout, отдельно при eval-BN и train-BN; BN buffers возвращаются перед каждым category batch. Crossed weight/BN substitutions оцениваются на всех1152 исходных строках. Для точных norm/dot длинные vectors один раз переводятся в CPU float64 после неизменного float32 forward/backward. Порядок `--checkpoint` важен: первый state — базовый. Настройки сохраняются в `procedure.json`; matrices и counterfactual metrics — в `gradient_interference.json`.

Все предсказания и градиенты относятся к train; перенос на val/test не измеряется. Протокол воспроизводим, побитовая идентичность MPS не гарантируется. Снимки сохраняют optimizer/scheduler/RNG для проверки состояния; автоматический resume из середины эпохи не добавляется. Полное воспроизведение выполняется из зафиксированного seed и recipe.

