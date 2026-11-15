# Длительное обучение ResNet-50 + GRU + FiLM на 4608 строках

**Ответ:** более длительное обучение с ненулевым constant LR позволило почти полностью запомнить те же4608 строки. Раннее слабое качество не было доказанным capacity limit этой модели. Это результат fitted train, не generalization.

## Вопрос и контроль

Проверяется, достаточно ли одной непрерывной более длинной оптимизации для обучения на прежних 4608 fitted train rows. Это диагностика fitted train, не generalization и не новый выбор финальной модели. Используются только `data/train.csv` и train JPEG; val/test не читаются.

Исходные свидетельства: [scaling investigation](film_scaling_investigation.md), [проверенный receipt](film_scaling/verified_receipt.json). Старый `scaling_4608` завершил1080 updates с cosine LR, практически достигшим нуля. Этот checkpoint не продолжался.

Новый run: `film_long_training/`. Exact IDs взяты из `film_scaling/subsets/nested_4608.json`; множество и порядок совпали с `scaling_4608/fitted_ids.npy`. Initial model SHA-256 совпал с `f4a604d547485e4561690c306cd33bac90f0459b57b1ecad4adcb8721ce60c55`; до первого update проверены также vocabulary, optimizer groups и начальные Python/NumPy/torch/MPS/sampler RNG старого run.

Recipe: ResNet-50 ImageNet V2 layer3, последние два residual-блока обучаются; GRU256 bidirectional; четыре FiLM-блока; MLP512→1024→ReLU→Dropout→1. Batch32, целые пары, seed42; CNN/task LR3e-5/3e-4; warmup9 абсолютных optimizer steps, затем constant LR; AdamW weight decay1e-4; dropout0.1; clipping1.0. CNN BN eval, FiLM BN train. Без augmentation. Выполнена одна непрерывная trajectory: optimizer и scheduler созданы один раз, без restart/restore между контрольными точками.

Все quality-метрики и category/level разрезы считаются по полным 4608 fitted rows в eval. Sensitivity и MLP statistics — на неизменном 288-row fitted probe, совпадающем с прежним scaling run; дополнительный read-only gradient probe использует первые32 строки этого probe в eval без dropout. Его gradients не смешиваются с actual training gradients/updates из `diagnostics.jsonl`. Step0 имеет диагностические gradients, но не optimizer update.

Checkpoints0/1080/2160/4320 сохраняют model, AdamW, scheduler, Python/NumPy/torch/device RNG, sampler RNG, epoch-start sampler state и batch cursor, processed rows, scaler state и LR следующего update. Cursor позволяет воспроизвести недооконченную эпоху, не перезапуская sampler permutation. При заметном падении 36-row monitoring probe (AUC −0.20 от максимума) или preclip gradient norm>100 сохраняется дополнительный диагностический state; noisy probe не заменяет quality на всех4608 строках.

Внутренний бюджет Trainer —29 минут, из него30 секунд резервируется на финальную оценку/snapshot; внешний wall-clock limit —30 минут. Bounded run при исчерпании времени возвращает последний state с `stop_reason=time_budget`, а не выдаёт незавершённые30 эпох за завершённый эксперимент. Legacy fixed-epoch refit без `max_steps` сохраняет прежний отказ при нехватке времени.

## Воспроизведение

Нужны существующая `.venv`, train CSV/JPEG, локальные torchvision ImageNet weights и сохранённые исходные артефакты. Output должен отсутствовать; имена старых каталогов не переиспользуются.

```bash
.venv/bin/python scripts/film_scaling_run.py run artefacts/film_long_training_reproduction_02 \
  --reference artefacts/film_collapse/medium_w9 \
  --subset artefacts/film_scaling/subsets/nested_4608.json \
  --baseline-dir artefacts/film_scaling/scaling_4608 \
  --start initial --steps 4320 --snapshots 0,1080,2160,4320 \
  --schedule constant --horizon 4320 --warmup 9 --task-lr 3e-4 \
  --device mps --fitted-only --snapshot-reserve-sec 30 --anomaly-snapshots
```

Фактический запуск был обёрнут в `caffeinate -i`, `/usr/bin/time -p` и внешний timeout1800с. [Лог](logs/film_long_training_01.log): wall903.53с, fitting со snapshots14.9889мин; оба лимита соблюдены. `stop_reason=max_steps`, не `time_budget`; выполнены30 полных эпох /138240 предъявленных строк, по30 предъявлений каждой fitted строки. Проверка досрочной остановки здесь синтетическая CPU-регрессия, а не реально сработавший time-budget stop.

Повторная read-only проверка сохранённых outputs, без обучения или inference:

```bash
.venv/bin/python scripts/film_long_training_receipt.py \
  --run artefacts/film_long_training \
  --baseline artefacts/film_scaling/scaling_4608 \
  --output artefacts/film_long_training/reverified_receipt_02.json
```

Сравнение с cosine1080 меняет LR trajectory уже до1080, поэтому разность результатов не является чистым эффектом дополнительных предъявлений. Внутри нового run после warmup данные и LR не меняются; при этом AdamW moments, weights и BN statistics естественно эволюционируют.

## Проверка инструментов и исходного состояния

Перед запуском выполнено 12 CPU tests: `.venv/bin/python -m unittest discover -s tests -p 'test_diagnostics.py'`, OK за4.872с; [лог](film_long_training_tests.log). Проверены trace invariance, выравнивание predictions, clipping/BN modes, absolute scheduler controls и новые регрессии: constant warmup9 без decay, один optimizer на всех snapshots, time-budget stop как точный prefix непрерывного fit, sampler replay из epoch-start RNG и cursor, сохранение live optimizer/scheduler counters, прежний отказ fixed-epoch refit при нехватке времени. Синтетические fixtures тестов не используют реальные val/test.

Отдельный CPU smoke новых gradient/sensitivity helpers подтвердил неизменность weights, BN buffers, RNG и `.grad`: [лог](film_long_training_smoke.log). Первая искусственная fixture с повторяющимися image indices не допускала применяемую круговую image derangement и дала `StopIteration`; [диагностика](film_long_training_smoke_error.log) сохранена. Исправлена только throwaway fixture, production recipe не менялась; training при этой ошибке не запускался.

Реальный step0 сохранён до первого update: optimizer state пуст, scheduler counter0, cursor0. Exact initial model/RNG/groups проверены до сохранения checkpoint и повторно сверены read-only verifier. Tiny отличие initial AUC от старого run (1.9e-7) не свидетельствует о разных весах: состав inference batches различается, а MPS не гарантирует побитовую детерминированность; максимальное отличие initial logits меньше2e-7. AMP на MPS не использовался.

В Git входят source и этот отчёт; большие checkpoints/predictions остаются локальными и gitignored. Исключение `.gitignore` относится только к этому отчёту, не к данным или весам.

## Измеренные контрольные точки

Метрики на всех4608 fitted строках; один update предъявляет32 строки, один полный проход —144 updates.

| Режим LR | Шаг | Эквивалент эпох | BCE | ROC-AUC | Accuracy | Pair accuracy | Mean signed margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Constant | 0 | 0 | 0.695021 | 0.496865 | 0.500000 | 0.488281 | −0.000126 |
| Constant | 1080 | 7.5 | 0.675341 | 0.606298 | 0.573568 | 0.598090 | 0.115541 |
| Constant | 2160 | 15 | 0.451142 | 0.866856 | 0.766276 | 0.817708 | 3.023375 |
| Constant | 4320 | 30 | 0.071427 | 0.997108 | 0.974392 | 0.993056 | 19.362337 |
| Старый cosine | 1080 | 7.5 | 0.681679 | 0.587837 | 0.562717 | 0.587240 | 0.069376 |

На всех контрольных точках проверены live AdamW/scheduler counters, sampler state/cursor, aligned logits и метрики: [итоговый receipt](film_long_training/verified_receipt.json). Дополнительные точечные проверки сохранены в [step1080](film_long_training/step1080_inspection.json) и [step2160](film_long_training/step2160_inspection.json).

На1080 разность AUC к старому cosine составляет+0.018461 при одинаковом числе updates и предъявлений. Это сравнение разных LR trajectories, а не эффект дополнительных повторов; один недетерминированный seed не устанавливает устойчивую величину эффекта schedule. Внутри нового run после warmup LR постоянен: дальнейший рост качества не вызван новым LR restart.

После1080 восстанавливаются signed margins и зависимость от изображения. Качество выросло, хотя доля нулей MLP не уменьшилась: сама эта доля не является достаточным объяснением раннего слабого обучения.

## Чувствительность, logits и MLP

Logit statistics —по всем4608 fitted строкам; остальные поля —по одному и тому же288-row fitted probe. Image/text delta —mean absolute logit change при image derangement / swap captions внутри пар. Нормализация image delta использует logit std **того же probe**, не full-fitted std.

| Шаг | Среднее logits | Std logits | Image delta | Text delta | Image delta / probe std | MLP pre mean | MLP pre std | Нули после ReLU, % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | −0.118601 | 0.026154 | 0.023490 | 0.006742 | 1.027813 | 0.010423 | 0.353264 | 48.2968 |
| 1080 | 0.051706 | 0.295542 | 0.022293 | 0.315703 | 0.076847 | −0.778437 | 1.210774 | 87.0782 |
| 2160 | −0.004403 | 3.057508 | 1.436730 | 2.852726 | 0.514015 | −1.283730 | 2.229581 | 87.9523 |
| 4320 | 0.541061 | 11.498626 | 8.600203 | 17.916239 | 0.807233 | −1.951669 | 3.508684 | 88.8882 |

Рост raw image delta сопровождается восстановлением нормализованной чувствительности после1080, а не только увеличением масштаба logits. Это показывает использование изображения на fitted probe, но не доказывает правильные общие правила reasoning. Полные preactivation/activation quantiles, zero/negative fractions и all-zero units сохранены на каждом checkpoint в `measurement_stepNNNN.json`; [сводка](film_long_training/analysis_summary.json).

## Градиенты, обновления, LR и BN

Ни optimizer updates, ни LR не исчезают к поздним checkpoints. Нормы ниже —actual sampled training gradients **до** clipping; относительные updates —реальные L2 изменения параметров за соответствующий update / L2 weights до него. Batches на разных steps различаются, поэтому это не matched-batch causal comparison; gradient RMS и все модули сохранены в trace.

| Шаг | GRU gradient norm | Readout gradient norm | CNN layer3.5 gradient norm | GRU relative update | Readout relative update | CNN layer3.5 relative update |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1080 | 0.093773 | 0.618821 | 0.019931 | 9.361e-4 | 1.489e-3 | 6.400e-5 |
| 2160 | 0.315435 | 0.976139 | 1.438250 | 5.442e-4 | 6.991e-4 | 1.682e-4 |
| 4320 | 0.546238 | 0.355291 | 0.836414 | 5.702e-4 | 1.041e-3 | 9.511e-5 |

На step0 actual update равен0 по определению; отдельные eval-probe gradient norms GRU/fusion/CNN равны0.064346/1.569281/0.140189. Read-only `autograd.grad` не меняет `.grad` training loop. Этот фиксированный32-row gradient probe сохранён на всех четырёх checkpoints отдельно от actual training trace.

LR следующего update на step0: CNN3.333333e-6/task3.333333e-5; на1080/2160/4320 используемый и следующий LR одинаковы: CNN3e-5/task3e-4. Это абсолютный warmup9, не9 эпох. Все19 captured optimizer steps имеют конечные gradients; максимальный captured total preclip norm17.3374. Ни nonfinite stop, ни порог extra-snapshot (>100), ни большой AUC drop monitoring probe не сработали; сохранены ровно четыре запланированных snapshots. Утверждение о максимуме относится к captured steps, не ко всем4320 norms.

CNN BN buffers побитово неизменны; pretrained counters не обнулялись и остались673830. Все шесть FiLM BN counters точно равны числу updates на каждом snapshot. FiLM running mean/variance эволюционируют вместе с weights, но не препятствуют позднему fitted learning:

| Шаг | FiLM running mean L2 | FiLM running variance L2 | L2 изменения FiLM running mean/variance с предыдущего snapshot |
| --- | ---: | ---: | ---: |
| 0 | 0 | 33.9411 | — |
| 1080 | 98.4325 | 667.4338 | 651.1784 |
| 2160 | 102.9873 | 796.0983 | 484.5448 |
| 4320 | 113.1812 | 520.6683 | 443.8033 |

Источник: [saved-state BN/update analysis](film_long_training/bn_update_analysis.json). Эти нормы описательны, а не независимые причинные вклады BN и weights; нового BN counterfactual fit не проводилось.

## Категории и уровни

На4320 нет категории около chance; наиболее слабым остаётся `spatial_counting`. Полные AUC/BCE **на каждом** checkpoint лежат в `measurement_stepNNNN.json` и пересчитаны verifier из aligned logits.

| Категория | Fitted AUC, step4320 | Fitted BCE, step4320 |
| --- | ---: | ---: |
| attribute | 0.996750 | 0.100092 |
| binding | 0.998840 | 0.063618 |
| comparison | 0.997574 | 0.064658 |
| counting | 0.998825 | 0.046392 |
| negation | 0.999374 | 0.028767 |
| position | 0.998993 | 0.049178 |
| spatial | 0.998459 | 0.056854 |
| spatial_counting | 0.979507 | 0.187423 |
| spatial_negation | 0.998993 | 0.045861 |

| Уровень | Fitted AUC, step4320 | Fitted BCE, step4320 |
| --- | ---: | ---: |
| 1 | 0.996750 | 0.100092 |
| 2 | 0.998825 | 0.046392 |
| 3 | 0.999027 | 0.046192 |
| 4 | 0.998802 | 0.053016 |
| 5 | 0.994364 | 0.099314 |

## Интерпретация гипотез

| Гипотеза | Вывод в проверенном режиме | Основание и граница |
| --- | --- | --- |
| H1: недостаточно optimizer steps | Сильно поддержана при constant LR | Между1080→2160→4320 fitted AUC, pair accuracy и margins растут, BCE снижается; active LR и реальные updates сохраняются. Эффект дополнительных steps внутри этой trajectory не смешан с LR restart. |
| H2: долгое обучение не решает проблему | Не поддержана для этих4608 строк | Низкого fitted plateau к30 эпохам нет; достигнута почти полная memorization во всех категориях. Это не утверждение о полном train или других seeds. |
| H3: нестабильность оптимизации | Не поддержана как устойчивый поздний collapse | Full-fitted checkpoints улучшаются, image sensitivity восстанавливается, gradients конечны. Малый36-row monitor колеблется по AUC/BCE; отсутствие transient degradation между редкими full checkpoints не доказано. |

**Основной ответ:** current ResNet-50 + GRU + FiLM + MLP способен почти запомнить4608 строк без архитектурных изменений. Старый cosine1080 результат не устанавливал архитектурный потолок. Однако нельзя утверждать, что старая неудача объясняется **только** числом steps: одновременно заменена затухающая LR trajectory. Новый run показывает достаточность сочетания дополнительных предъявлений с продолжающейся оптимизацией при constant LR.

Ни высокий fitted AUC, ни сильная image sensitivity не доказывают generalization, cross-modal reasoning или качество на полном train. AUC не равен ровно1.0; остаточные ошибки и более слабый spatial_counting сохранены, а не скрыты.

## Ограничения и одно следующее действие

- Один seed, MPS с недетерминированными backward kernels; это не CUDA/Kaggle runtime proof и не оценка статистической устойчивости.
- Subset прежний, категориально-сбалансированный, а не пропорциональный полному train. Не выполнены full-train fit и оценка на остальных train rows.
- Vocabulary намеренно сохранён от исходной серии:56 entries; `nine` отображается в UNK. Val/test не участвовали ни в vocabulary, ни в fit, ни в диагностике.
- Quality рассчитана в eval/no_grad без augmentation, а online loss —при training dropout/FiLM BN; эти losses не считаются одинаковыми. Sensitivity использует только train probe.
- Четыре full-fitted measurements не определяют качество на каждом промежуточном update. [Monitoring curve](film_long_training/analysis_summary.json) даёт лишь малый36-row probe, не замену full-fitted метрик.

**Следующий отдельный эксперимент:** один read-only generalization gate для зафиксированного `checkpoint_step4320.pt` на официальном val, без нового fitting, изменения vocabulary, выбора checkpoint по val или tuning. Измерить AUC/accuracy/pair accuracy и category/level разрезы; image permutation на фиксированном val probe —диагностика зависимости, не training objective. Этот эксперимент здесь **не запускался**: val/test не читались.

Продление ради ровноAUC1.0 мало меняет ответ о достижимости memorization, поэтому bounded continuation не запускается. Full train, architecture sweep и дополнительные long fits не выполняются.

## Артефакты и итоговая проверка

- `film_long_training/{config.yaml,invocation.json,vocab.json,train_ids.npy,fitted_ids.npy}`: воспроизводимый recipe, exact IDs и provenance13 source files.
- `checkpoint_step{0000,1080,2160,4320}.pt`: model + полный AdamW/scheduler + Python/NumPy/torch/MPS RNG + sampler state и epoch cursor; inactive CUDA/scaler fields явно `None`. `model.pt` побитово соответствует final checkpoint по state hash.
- `logits_stepNNNN.npy`:4608 aligned fitted logits; `measurement_stepNNNN.json`: все quality/sensitivity/activation measurements. `diagnostics.jsonl`: actual sampled batches, pre/postclip gradients, real updates, LR и BN modes.
- [Verified receipt](film_long_training/verified_receipt.json): status `verified_max_steps`; все checkpoints достигнуты, пропусков нет; пересчитаны все fitted/category/level metrics и probe summaries; проверены optimizer counters, LR и sampler cursor. Первые1080 batches дают прежний replay hash `3fb5479ac39ad13497778a36f31cf7781c88f10d42d3c7df2d90b44d53d25cde`; matched7 старых и19 новых captured batches. Полный order hash реконструирован sampler replay, а не выдаётся за online запись каждого batch.
- [Protected inputs manifest](film_long_training_inputs.json): все51 fingerprints совпали после run, включая прежние scaling/medium checkpoints и outputs, final notebook/preset, archived configs и final submission. Прежние результаты не перезаписаны; destructive Git operations и push не выполнялись.

Длинный fit, read-only verifier,12 focused CPU tests и smoke завершены. Следующая проверка generalization только предложена; текущий эксперимент завершён.
