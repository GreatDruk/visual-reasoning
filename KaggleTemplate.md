### Текущий маршрут

- Обычный запуск: `uv run python -m src.train name_exp=debug save_config=false`. Train-only, ResNet-50 ImageNet V2 layer3 + GRU hidden 256 + cosine, BCE, без аугментаций и twin normalization. Словарь не видит val; test не читается до явно включённого `train.predict_test=true`.
- Сравнение fusion: defaults → `experiment=pooled_mlp` → `experiment=film`; отдельный `experiment=lab_baseline` сохраняет ResNet-18. Затем отдельные абляции pair loss, перефразов, hflip и ResNet-101. При приросте AUC <0.002 остаётся предыдущий более простой вариант.
- Holdout — официальный val, не случайные строки. `val_para` — диагностика общих с аугментацией правил перефразирования, **не независимый OOD-набор**. Category/level/pair_id/id не являются входами classifier.
- `results.json` содержит архитектуру, seed, best epoch, устройство/версии, время подготовки, fitting/inference, peak CUDA memory и метрики. Рядом сохраняются resolved config, vocabulary, state_dict и выровненные ids/logits. `cache_min` включает загрузку кэша при попадании и не означает cold extraction time.
- Новый кэш имеет префикс `eval-v2_` и различает None/V1/V2; старые файлы не удаляются. Smoke-limit сокращает набор изображений до feature extraction. Замороженный backbone остаётся в eval.
- Исторические результаты до исправления validation vocabulary leakage — **historical/pre-fix**, не сравнение той же процедуры. Старые JSON и `configs/archive` не редактируются. ID 001–005 заняты, включая shell failures; перед основным запуском нужен следующий после максимального занятого ID во всех run/log/archive/submission путях.
- Серия A–H и повторы двух лучших вариантов завершены на MPS; измерения и выбор — в Solution ниже. [Final notebook](notebooks/02_final.ipynb) заново обучает выбранную модель. Допуск на Kaggle пока не подтверждён.
- Финальный refit: train+val, словарь train+val, фиксированное число эпох, новый fit без contest checkpoint, затем test inference в порядке sample submission (`id,label`). Исчерпание лимита до завершения фиксированных эпох — ошибка.
- Диагностика FiLM на MPS: [16 train-пар, 200 optimizer steps](artefacts/hydra/debug_film_overfit/2026-11-14_15-42-05/results.json), training loss 0.703211 → 0.000529. Критерий запоминания выполнен; это не benchmark качества и не Kaggle timing. MPS предупреждает о недетерминированных kernels даже при фиксированном seed.
- Kaggle cold offline proof и лимит ≤50 минут пока **не подтверждены**. В текущей среде есть MPS, но нет CUDA, Kaggle CLI и credentials. Из скриншота установлен slug: `assessment-of-logical-statements-in-an-image-text-problem`; Kaggle показывает `Invitation required`. Нужны доступ к Kaggle Default GPU и разрешённый локальный mount ImageNet-весов. Сабмит без отдельного поручения не отправляется.
- Для явной оценки fitted строк: `train.eval.train=true` сохраняет `train_fit` в eval/no_grad без аугментаций; `train.eval.holdout=false` не читает val, если нет `train_on_val`. В history сохраняются также category AUC; `stop_reason` различает `epochs`, `early_stopping`, `time_budget`. Early stopping требует метрику реально выбранного eval split.
- `data.train_pairs_per_category=N` детерминированно выбирает N целых пар каждой категории до словаря и загрузки изображений; несовместим с `data.limit` и `train_on_val`. Это диагностика, не holdout.
- `train.optimizer.image_encoder_lr` задаёт отдельный LR trainable CNN; `null` оставляет общий LR. Группы AdamW разделяются по identity параметров, scheduler сохраняет отношение LR. Partial unfreeze отключает feature cache и сохраняет CNN BN в eval.
- Контролируемый кандидат: `experiment=film_finetune` — два последних residual-блока **layer3**, FiLM + MLP, batch32; это пакет изменений, не причинная абляция размораживания. Перед полным train→val обязателен `experiment=film_memorize`: 16 пар × 9 категорий = 288 строк train-only, gate по восстановленному checkpoint (AUC≥0.98, accuracy/pair accuracy≥0.95, каждый category AUC≥0.95). Ни test, ни текущий final этим циклом не меняются.

### To-do
- [ ] EDA только train/val: число подписей по split/category/level, подписи на картинку, структура pair_id, словарь и пересечение шаблонов. Test — только inference и явно разрешённый поиск близнецов.
- [x] Валидационный фреймворк: общий AUC, AUC по 9 категориям, pair accuracy (tie=0.5), фиксированные сиды
- [ ] Аудит: text-only и image-only бейзлайны
- [ ] Бейзлайн: замороженный CLIP/SigLIP + MLP над [i, t, i*t, |i-t|]
- [x] Сравнение ResNet-50 + GRU с cosine / pooled MLP / FiLM; выбран cosine
- [x] Абляция BCE + 0.5·pairwise по pair_id
- [x] Label-preserving аугментации: hflip с заменой left↔right и перефразы; логические негативы не генерируются
- [ ] MAC как второй кандидат
- [ ] Relation Network голова по ячейкам feature map для уровней 4–5
- [ ] Поиск близнецов в test + нормализация внутри пары (softmax по паре)
- [ ] Ансамбль / маршрутизация по category
- [ ] Финальный notebook создан и проверен локально; остаётся Restart & Run All офлайн на Kaggle Default GPU, ≤50 мин
- [ ] Moodle-пакет: команда, вклад, ID сабмита, ноутбук, final_submission.csv, таблица экспериментов (архитектура, val Accuracy + AUC, runtime, seed), версии библиотек

### Notes
- Задача: по картинке (240×160, CLEVR-подобная сцена) и утверждению предсказать P(утверждение истинно). Метрика ROC-AUC.
- Данные: train 5000 картинок, val 1000, test 6000. 9 категорий, 5 уровней, баланс 50/50 в каждой категории.
- pair_id (только train/val): две подписи к одной картинке, отличаются одной деталью, ровно одна истинна.
- В test есть формулировки, которых нет в train/val: нужна аугментация/каноникализация текста.
- category и level есть и в test: можно маршрутизировать модели по категориям.
- Разрешено: классификаторы torchvision с ImageNet-весами (любые слои), text-only энкодеры (RNN, BERT и т.п.), CLIP/SigLIP/OpenCLIP, любые свои слои; train+val для финала; ансамбли; аугментации; неразмеченные test-подписи для поиска близнецов.
- Запрещено: детекция/сегментация (даже из torchvision), совместные VLM (BLIP, ViLT, Qwen-VL, OWL-ViT, Grounding DINO...), LLM, внешние данные (CLEVR, NLVR, ShapeWorld) и модели, обученные на похожих данных; ручная разметка test.
- Аугментации: vflip, поворот, color jitter нельзя; crop/zoom рискованно; лёгкие яркость/контраст/blur/шум можно.
- Лимиты: 5 сабмитов/день, 1 финальный сабмит, команда до 3 человек, мердж до 18.11, финал 22.11 14:00 МСК.
- Идея из FiLM: замороженный ResNet-101 conv4 + текстовый генератор γ/β, 97.7% на CLEVR. При 240×160 сетка conv4 = 15×10, возможно стоит апсемплить вход или брать conv3.
- CLOSURE: FiLM/MAC/NMN плохо обобщаются на новые комбинации, поэтому test-only формулировки закрывать через текст.
- Hard negatives без hard positives ломают устойчивость к парафразам (Kamath et al., 2024).

### Naming
- Ноутбуки: `notebooks/0X_<name>.ipynb` (двузначный номер по порядку: `01_eda`, `02_lab_baseline`, ...).
- Эксперименты: сквозной трёхзначный номер `00X_<name>` (`001_lab_baseline`, `002_clip_mlp`, `003_film_bce`, ...). Номер не переиспользуется, даже если эксперимент провалился.
- Конфиги: каждый запуск, который является основной проверкой модели (попадает в таблицу Solution), сохраняется полным скомпонованным конфигом в `configs/archive/00X_<name>.yaml`, с тем же номером и именем, что и эксперимент. Архив только дописывается, старые файлы не редактируются.
- Пробные и отладочные запуски в архив не идут.
- В таблице Solution колонка ID = `00X`, по ней находится конфиг в архиве.

### Questions
- Можно ли использовать test-картинки для self-supervised / fit representations? Раздел 4 разрешает test-подписи для поиска близнецов, но ниже запрещено fit на test.
- Разрешён ли детерминированный грамматический парсер подписей в структуру/программу?
- Разрешён ли DINOv2 (не в torchvision, не CLIP-семейство)?
- Как подключать предобученные веса офлайн при лимите 20 MB: Kaggle Dataset/Model не считается?
- Имя колонки в сабмите: `label` (data.md) или `target` (rules)?
- Таблица экспериментов просит validation Accuracy, метрика AUC: указывать обе?
- Нормализация скоров внутри найденных пар близнецов в test точно не считается «exploit an unintended property»?

### Solution

Обучение только на train, официальный val не участвовал в fitting и словаре. Seed 42; все времена ниже — MPS, не Kaggle. Полные метрики, версии, aligned predictions и история находятся в `artefacts/hydra/<ID>/...`; [receipt сравнения](artefacts/experiment_selection.json) содержит точные пути. Архивы ниже append-only.

| Этап / config | Модель / изменение | val AUC | val accuracy | pair accuracy | fit, мин (MPS) |
| --- | --- | ---: | ---: | ---: | ---: |
| [A / 006](configs/archive/006_lab_baseline.yaml) | ResNet-18 + GRU + cosine | 0.598546 | 0.559167 | 0.575000 | 2.3227 |
| [B / 007](configs/archive/007_r50_cosine.yaml) | ResNet-50 + GRU + cosine | 0.602662 | 0.559167 | 0.574333 | 2.8215 |
| [C / 008](configs/archive/008_pooled_mlp.yaml) | ResNet-50 + GRU + pooled MLP | 0.517263 | 0.517833 | 0.515667 | 2.5858 |
| [D / 009](configs/archive/009_film_bce.yaml) | ResNet-50 + GRU + FiLM | 0.553643 | 0.532667 | 0.538667 | 13.4636 |
| [E / 010](configs/archive/010_pair_loss.yaml) | B + pair loss | 0.600500 | 0.555333 | 0.572667 | 2.4239 |
| [F / 011](configs/archive/011_text_aug.yaml) | B + перефразы p=0.3 | 0.602710 | 0.559167 | 0.566667 | 1.2954 |
| [G / 012](configs/archive/012_hflip.yaml) | B + перефразы p=0.3 + hflip p=0.5 | 0.601989 | 0.555333 | 0.565333 | 1.5624 |
| [H / 013](configs/archive/013_resnet101.yaml) | B с ResNet-101 | 0.598363 | 0.555000 | 0.561667 | 1.3235 |

По правилу прироста ≥0.002 после B более сложные варианты не заменили incumbent. Для проверки seed-устойчивости повторены B и F; исторический 002 в сравнение не включён.

| Вариант | Config IDs (seeds 42 / 43 / 44) | mean val AUC | sample STD | mean fit, мин (MPS) |
| --- | --- | ---: | ---: | ---: |
| B: cosine, BCE, без аугментаций | [007](configs/archive/007_r50_cosine.yaml) / [014](configs/archive/014_repeat_b_s43.yaml) / [015](configs/archive/015_repeat_b_s44.yaml) | 0.605125 | 0.002294 | 1.8355 |
| F: B + перефразы | [011](configs/archive/011_text_aug.yaml) / [016](configs/archive/016_repeat_f_s43.yaml) / [017](configs/archive/017_repeat_f_s44.yaml) | 0.572403 | 0.055700 | 0.9806 |

Это повторы одного holdout, **не cross-validation**. У F seed 44 дал AUC 0.508120 и best epoch 1; результат сохранён, не исключён из среднего. У победителя best epochs [20, 19, 19], поэтому final refit — 19 эпох. FiLM запоминает малую выборку, но не выиграл holdout; spatial/binding категории остаются близкими к случайному уровню. `val_para` остаётся диагностикой общих правил переписывания.

Дополнительная [диагностика сохранённых checkpoints](artefacts/quality_diagnosis.json), без нового обучения: cosine слабо решает пространственные категории уже на train, а FiLM имеет большой разрыв train/val. При случайной перестановке изображений val AUC обоих вариантов падает к случайному уровню — изображение используется, но это не доказывает правильный reasoning. Ошибка преждевременного spatial pooling в FiLM не обнаружена: в архиве 009 `input_pool=null`, кэш содержит карты `[N,1024,10,15]`. Полученный `final` — артефакт воспроизводимого baseline, **не удовлетворительное решение по качеству**; выбор лучшего из слабых вариантов не заменяет диагностику обучения.

### Контролируемое дообучение FiLM

ResNet-50 V2 выбран по уже измеренному frozen-сравнению: 007 val AUC 0.602662 против 013 ResNet-101 0.598363. Это не доказывает превосходство ResNet-50 при fine-tuning; дообученный ResNet-101 здесь не сравнивается.

`experiment=film_memorize` прошёл gate по **восстановленному checkpoint**, не training loss: [результат](artefacts/hydra/debug_recovery_memorize/2026-11-14_17-31-52/results.json), train_fit AUC/accuracy/pair accuracy = 1.000, все 9 category AUC = 1.000. Выбраны 144 пары / 288 строк / 143 изображения; словарь только из этих строк, val/test не читались. Завершены 120 эпох / 1080 optimizer steps; выбран epoch 30 по accuracy, `stop_reason=epochs`, fit 5.6784 мин на MPS. Разморожены два последних residual-блока layer3, LR CNN/task = 3e-5/3e-4; CNN BN остаётся в eval. Это проверка запоминания, **не оценка generalization**.

Полный [018 / config](configs/archive/018_film_finetune.yaml), [результаты](artefacts/hydra/018_film_finetune/2026-11-14_17-37-54/results.json), [лог](artefacts/logs/018_film_finetune.log), [receipt цикла](artefacts/recovery_cycle.json): 29 986 train-строк / 5 000 изображений, неизменные 6 000 val-строк / 1 000 изображений. `train_fit`, `val`, `val_para` оценены после восстановления лучшего checkpoint; ids совпадают с CSV, scores и метрики конечны. Config/vocab/model/history/ids/logits сохранены; словарь только train. `val_para` AUC ≈ 0.5006.

| Вариант | train AUC | val AUC | val accuracy | val pair accuracy | reasoning macro AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| 007: frozen cosine | 0.698578 | 0.602662 | 0.559167 | 0.574333 | 0.499910 |
| 009: frozen FiLM/cosine | 0.814666 | 0.553643 | 0.532667 | 0.538667 | 0.487911 |
| 018: partial fine-tuning FiLM/MLP | 0.501066 | 0.499794 | 0.497833 | 0.499333 | 0.505065 |

| Вариант | binding | position | spatial | spatial_counting | spatial_negation |
| --- | ---: | ---: | ---: | ---: | ---: |
| 007 | 0.518974 | 0.500395 | 0.488110 | 0.497422 | 0.494647 |
| 009 | 0.515996 | 0.519134 | 0.482780 | 0.470469 | 0.451179 |
| 018 | 0.498210 | 0.499703 | 0.503878 | 0.514883 | 0.508651 |

| Вариант | best epoch | завершено эпох | stop_reason | fit, мин | device |
| --- | ---: | ---: | --- | ---: | --- |
| 007 | 20 | 20 | не записан | 2.8215 | MPS |
| 009 | 17 | 20 | не записан | 13.4636 | MPS |
| 018 | 3 | 7 | time_budget | 40.4097 | MPS |

**Гипотеза не прошла improvement gate:** Δval AUC к 007 = −0.102868 вместо требуемых ≥0.02; macro пяти reasoning-категорий 0.505065 вместо ≥0.60. Обучение на полном train остаётся ограниченным (train AUC <0.95); spatial reasoning не решён. Это не случай высокого train AUC при плохом переносе и не доказательство потолка архитектуры: из-за лимита 45 минут выполнены только 7 из 30 эпох. Полное время с post-fit evaluation — 43.3574 мин; бюджет автоматически не продлевался.

Старые train AUC взяты из `quality_diagnosis.json`, остальные старые поля — исходных results 007/009; старые артефакты не менялись. Все три сравниваемых запуска — seed42, но различаются head, fine-tuning, batch size и число завершённых эпох: причинный эффект одного параметра и статистическая значимость не установлены. MPS предупреждает о недетерминированных backward kernels; это не CUDA/Kaggle runtime proof. Test/submission/final preset и notebook не менялись, refit train+val и Kaggle submission не выполнялись.

#### Диагностика коллапса 018 без нового fit

[Измерения на сохранённом checkpoint](artefacts/training_mechanics_diagnosis.json): на одном train-batch из 32 строк проверены CSV→image/caption/label, пары и соответствие decoded JPEG исходному tensor — расхождения не найдены. CNN-параметры действительно изменились; градиенты GRU, FiLM, MLP и двух CNN-блоков конечны и ненулевые. Диагностический optimizer step выполнен только на копиях моделей; сохранённый checkpoint не менялся. Это не подтверждает семантику всех меток: визуально проверены только два простых train-примера attribute/counting.

При перестановке изображений mean absolute logit delta на этом batch упал с 0.02455 у initial модели до 0.000505 у checkpoint 018. CNN feature maps реагируют на изображения (mean absolute delta 0.3968), но сигнал подавляется внутри FiLM/MLP. После MLP ReLU 92.7% элементов нулевые, 900 из 1024 units полностью нулевые на этом batch. На ещё трёх batches: 894–904 полностью нулевых units, image delta 0.000716–0.001301; у memorization checkpoint на тех же входах image delta 3.724–4.674. Это сравнение чувствительности, **не сравнение качества**, и не доказательство, что units всегда неактивны.

На проверенном batch переключение только FiLM BN в train при выключенном dropout оставило BCE около 0.693; простой train/eval BN mismatch не объяснил наблюдение. CPU/MPS forward различался максимум на 6.7e-8 в logits; backward между backend не сравнивался.

Вывод: наблюдается коллапс визуального пути, а не только слабое обобщение. **Первичная причина пока не установлена:** нет доказанного дефекта indexing/optimizer; влияние dropout, LR, weight decay и динамики FiLM/MLP не разделено. Код и checkpoint не исправляются наугад; нового длинного fit не было. Успех запоминания с выключенной регуляризацией не являлся проверкой работоспособности full-data режима.

#### GELU только в MLP readout: отдельная гипотеза

[GELU](https://arxiv.org/abs/1606.08415) использует `x·Φ(x)`, а не жёсткое обнуление отрицательной полуоси; GELU применяется, например, в [ConvNeXt torchvision](https://docs.pytorch.org/vision/main/_modules/torchvision/models/convnext.html). [SiLU/Swish](https://docs.pytorch.org/docs/main/generated/torch.nn.SiLU.html) — другая гладкая альтернатива. Это не универсально лучшие активации: у GELU/SiLU градиенты тоже могут быть очень малы на больших отрицательных входах. [SwiGLU/GEGLU](https://arxiv.org/html/2002.05202) меняют структуру FFN (добавляют отдельную gated projection), а не только активацию; здесь они не добавляются.

Новый [preset `experiment=film_gelu`](configs/experiment/film_gelu.yaml) отличается от `film_finetune` ровно одним полем: `model.fusion.readout_activation=gelu`. По умолчанию остаётся `relu`; ReLU в CNN и conv-блоках FiLM, архитектура/ширина, LR, dropout и данные не меняются. Это позволяет проверить отдельную гипотезу вместо пакета изменений. Старые checkpoint/config/archive/final сохраняют прежнее поведение.

Выполнены реальный forward/backward на 8 train-строках с локальными ImageNet weights на MPS и существующая CPU partial-update регрессия. Градиенты CNN/GRU/FiLM/readout конечны и ненулевые; отрицательные preactivations MLP передают градиент через GELU. Прежний ReLU checkpoint воспроизводит сохранённые logits с максимальным отклонением 8.6e-8 на проверенных строках. **Quality fit GELU не запускался, улучшение AUC не установлено.** Простая подмена активации в уже обученном checkpoint не считается обучением или доказанным исправлением коллапса.




#### Поэтапная диагностика обучения

Поиск сформулирован без предположения о виновности ReLU: «FiLM CNN+GRU запоминает малую выборку, но остаётся около chance на полном наборе пар; CNN features меняются между изображениями, а logits становятся image-insensitive и visual gradients уменьшаются. Как различить gradient imbalance, activation saturation, BatchNorm mismatch и modality collapse?» Отдельные запросы: `FiLM training gamma beta ReLU batch normalization`, `multimodal gradient imbalance modality dominance diagnostics`, `What Makes Training Multi-Modal Classification Networks Hard`.

Первичные источники и границы переноса:
- [Оригинальный FiLM](https://arxiv.org/html/1709.07871): ReLU/BN работали в исходной модели; подавление каналов, отрицательные beta и большой диапазон gamma являются частью обученного conditioning, а не сами по себе доказательством дефекта. Задача и объём CLEVR отличаются от текущих; чужие данные/weights не используются.
- [Wang et al., CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_What_Makes_Training_Multi-Modal_Classification_Networks_Hard_CVPR_2020_paper.html): модальности могут обучаться и переобучаться с разной скоростью. Это основание наблюдать отдельные пути, не доказательство нашей причины.
- [Peng et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Peng_Balanced_Multimodal_Learning_via_On-the-Fly_Gradient_Modulation_CVPR_2022_paper.pdf): анализ under-optimization и gradient imbalance. OGM/GE, дополнительный шум и новые heads здесь не добавляются.

`train.diagnostics.enabled=true` включает `diagnostics.jsonl` и `diagnostic_probe.json` в output текущего запуска. Трассируются CNN, sentence encoder, stem, gamma/beta, FiLM-блоки, projection, pooling summary и MLP до/после активации; записываются actual batch ids/images/captions/labels/logits, LR, градиенты до/после clipping (norm и RMS), реальные изменения весов и BN modes/running statistics. Градиент на отрицательных входах MLP измеряется tensor hook **до in-place ReLU**, а не `retain_grad` на изменяемом tensor.

Фиксированный probe выбирается только из fitted train: 2 пары каждой категории, всего 36 строк для полного train. В `eval/no_grad` сохраняются raw logits и метрики, logits после image derangement и swap подписей внутри пар, чувствительность к изображениям по стадиям. Manifest содержит ids/labels/captions/images, перестановки и SHA-256 начального state_dict. Снимки: до обучения, шаги 1/10/50, далее каждые 100 шагов, конец эпохи и восстановленный checkpoint. Probe — не holdout и не дополнительная training objective.

[`experiment=film_trace`](configs/experiment/film_trace.yaml) делает один проход по полному train; val/test не читаются, early stopping выключен. Первый warmup epoch совпадает с режимом 018. Для matched A/B меняется только `model.fusion.readout_activation=relu|gelu`; это debug, без числового experiment ID и архивирования. Логирование по умолчанию выключено.

Параллельная [интервенция на checkpoint 018](artefacts/gelu_checkpoint_intervention.json), без optimizer steps: GELU почти убрал точные нули активаций, но image sensitivity осталась порядка 0.001. Это не обученный GELU результат и не доказанный фикс. В первоначальном замере preactivation gradient обнаружена методическая ошибка `retain_grad` + in-place ReLU; поле исправлено после проверки tensor hook, который для отрицательных ReLU-входов корректно даёт нулевой градиент.

Проверка инструмента: 15 CPU-регрессий прошли. На реальном training loop с dropout включение/выключение trace даёт побитово одинаковые итоговые parameters, BN buffers и torch RNG state; probe logits согласованы с manifest/labels и пересчитанными метриками. Отдельная регрессия проверяет корректное различение нулевого pre-ReLU gradient и ненулевого GELU gradient на отрицательных входах.

Расширение того же trace: `train.diagnostics.channel_stats=true` сохраняет по каждому FiLM-каналу каждого блока и каждому MLP unit mean/std, p05/p50/p95, долю отрицательных значений и долю `abs(x)<0.05` (наблюдательный порог, не gate). Для MLP дополнительно сохраняются нормы градиента по каналам, отдельно на отрицательных preactivation, до in-place ReLU. На фиксированном probe доступны те же распределения; новые CPU-проверки подтверждают ось каналов, значения квантилей и сохранение training state/RNG.

[`experiment=film_recipe_memorize`](configs/experiment/film_recipe_memorize.yaml) повторяет 288-строчный тест с model/dropout/weight decay/LR/BN mode/cosine/warmup от `film_finetune`/018. Меняются только train subset, отсутствие holdout и train-only checkpoint selection. Это 30 эпох / 270 optimizer steps; старый облегчённый memorization уже достиг всех метрик 1.000 на шаге 270, а не только после 1080 шагов. Отдельные абляции: dropout-off (оба dropout GRU/MLP = 0, weight decay прежний) и weight-decay-off (только weight decay = 0, dropout прежний). Изменения не складываются.

018 остаётся ограниченным compute результатом: 7 из 30 эпох, выбран checkpoint эпохи 3. Ни этот запуск, ни одноэпоховые traces не являются полноценной проверкой потенциала fine-tuning.

##### Измерения по пяти проверкам

Полный [receipt](artefacts/staged_diagnosis.json) объединяет пути runs, конфиги, snapshots, реальные predictions и ограничения. Traces сохранены в `artefacts/hydra/debug_trace_{relu,gelu,relu_nodrop}_01/.../diagnostics.jsonl`; все три запуска — один проход по 29 986 train-строкам / 938 optimizer steps на MPS. Фиксированный probe содержит 36 fitted строк, не является holdout и не используется как objective.

**1. Ранние optimizer steps.** У ReLU на шагах 0/1/10/50/100 резкого обвала нет. На шаге 100 фактический task LR = 3.198e-5, относительное изменение readout-весов за этот шаг = 2.312e-4; данные не указывают на огромный одиночный update. Ухудшение становится выраженным в интервале 300–400 и продолжается дальше. Разрешение поздних snapshots — 100 шагов: точный одиночный «сломавший шаг» этим логом не установлен.

| Шаг ReLU | Image sensitivity: mean abs logit delta | Нулевые элементы MLP ReLU, % |
| ---: | ---: | ---: |
| 0 | 0.020812 | 47.11 |
| 1 | 0.024225 | 47.29 |
| 10 | 0.035993 | 50.43 |
| 50 | 0.071654 | 48.80 |
| 100 | 0.095835 | 48.73 |
| 300 | 0.103385 | 60.41 |
| 400 | 0.048723 | 70.15 |
| 938 | 0.011724 | 89.76 |

**2. FiLM отдельно от MLP.** Разброс и зависимость от картинки — разные измерения. Ниже `std` активаций и mean absolute delta при image derangement для одной и той же стадии во времени; масштабы разных слоёв нельзя напрямую сравнивать. Вариативность признаков сама по себе не доказывает полезную для reasoning информацию.

| Стадия ReLU | std, шаг 100 | std, шаг 938 | Image delta, шаг 100 | Image delta, шаг 938 |
| --- | ---: | ---: | ---: | ---: |
| ResNet до FiLM | 0.933211 | 0.945998 | 0.394425 | 0.403483 |
| После четвёртого FiLM-блока | 1.421996 | 2.957774 | 1.248844 | 0.743821 |
| После pooling, вход readout | 0.632175 | 0.876149 | 0.560509 | 0.139442 |
| Перед MLP ReLU | 1.602080 | 1.540635 | 0.336283 | 0.099756 |
| После MLP ReLU | 0.931060 | 0.789315 | 0.171317 | 0.011080 |

[Перекрёстная подстановка initial/trained trunk и readout](artefacts/readout_counterfactual_relu_01.json), без fitting, локализует значительную часть неактивности в обученных MLP-весах, а не в исчезновении различий на выходе CNN. Matched GELU с теми же начальным state SHA-256, данными и LR сохранил градиенты отрицательных входов, но к шагу 938 image delta всё равно уменьшился до 0.012138. Отключение только dropout readout при сохранении GRU dropout дало 0.022132; это влияние на диагностический симптом, **не доказанное улучшение generalization**.

**3. Gamma/beta и каналы.** [Полные распределения](artefacts/channel_diagnosis.json) сохраняют каждый канал, а не только общий mean. На ReLU step 938 доля `abs(gamma)<0.05` = 0.434%; negative beta по блокам находятся в диапазоне 47.85–48.46%, beta p05/p50/p95 = −0.6609 / 0.01675 / 0.80844. При этом negative preactivation MLP = 89.76%. Массового обнуления gamma на этом probe не обнаружено; отрицательные beta сами по себе не являются дефектом FiLM. В checkpoint 018 доля почти нулевых gamma = 2.686%, negative preactivation MLP = 92.85%.

**4–5. Memorization с полным recipe и отдельные абляции.** Во всех трёх случаях — те же 144 пары / 288 строк / 143 изображения, vocabulary из fitted строк, ReLU, те же initial parameter values (для dropout-off нормализован только индекс Linear после удаления parameter-free Dropout), LR/BN/cosine/warmup, 30 эпох / 270 steps. Данные/порядок sampled batches/LR совпадают; dropout и weight decay не отключались вместе. Метрики ниже — после восстановления checkpoint на всех 288 fitted строках.

| Условие | train AUC | accuracy | pair accuracy | min category AUC | Первый gate, epoch |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Recipe 018 без ослабления](artefacts/hydra/debug_recipe_memorize_01/2026-11-14_19-36-59/results.json) | 1.000 | 1.000 | 1.000 | 1.000 | 21 |
| [Только dropout GRU/MLP = 0](artefacts/hydra/debug_recipe_memorize_nodrop_01/2026-11-14_19-41-27/results.json) | 1.000 | 1.000 | 1.000 | 1.000 | 17 |
| [Только weight decay = 0](artefacts/hydra/debug_recipe_memorize_nowd_01/2026-11-14_19-43-37/results.json) | 1.000 | 1.000 | 1.000 | 1.000 | 20 |

Выбранные epochs — 26/20/25 соответственно; все runs завершили 30 эпох с `stop_reason=epochs`. Gate прежний: AUC≥0.98, accuracy/pair accuracy≥0.95, каждая category AUC≥0.95. Один seed и недетерминированный MPS не позволяют объявить разницу в сроке прохождения gate статистически устойчивой.

**Диагноз уточнён:** full-training recipe умеет запоминать 288 строк и без отключения регуляризации. ReLU не установлен как единственная причина; GELU не устранил наблюдаемую динамику автоматически. Dropout readout влияет на неактивность и [шум margin](artefacts/readout_dropout_noise.json), но не доказан как объяснение слабого full-data качества. [CPU/MPS backward](artefacts/backend_gradient_comparison.json) на проверенном batch не показал крупного расхождения: максимум относительного L2-отличия групповых градиентов около 0.71%, при близких направлениях. Это не полная гарантия backend correctness.

Первопричина слабого full-data обучения **остаётся неустановленной**; данные показывают позднее подавление визуального пути, а не моментальное повреждение первым optimizer step. Состав/сложность набора, взаимодействие оптимизации и число повторных предъявлений ещё не разделены. 018 с 7/30 эпохами и checkpoint эпохи 3 не закрывает этот вопрос. Новые проверки не читали val/test, не меняли final/submission и не являются результатом holdout или Kaggle.

### Воспроизведение финала

[notebooks/02_final.ipynb](notebooks/02_final.ipynb) использует один `python -m src.train experiment=final`, пустой временный feature cache и только явно приложенные ресурсы. Ожидаются supplied CSV/JPEG в `data/` и официальный `resources/torch/hub/checkpoints/resnet50-11ad3fa6.pth`; на Kaggle notebook также умеет подключить единственный подходящий ресурс из `../input`. SHA-256 весов проверяется. Зависимости должны быть установлены офлайн под фактический runtime; скрытых скачиваний и contest checkpoint нет.

Все code-ячейки последовательно выполнены в свежем Python-процессе на **Apple MPS**, в изолированной копии исходников с явно подготовленными локальными ресурсами. [Receipt](artefacts/final_notebook_mps_run.json): 19 завершённых эпох train+val, полный cold запуск 2.4074 мин, исходники 125 024 байт. Создан [final_submission.csv](final_submission.csv): 35 992 уникальных sample id в нужном порядке, `id,label`, конечные вероятности в [0,1]. Это локальная проверка исполнения ячеек, **не GUI Restart & Run All на Kaggle и не доказательство сетевой изоляции**.

Версии этого запуска: Python 3.14.3, torch 2.15.0, torchvision 0.29.2, Hydra 1.4.0rc3, NumPy 2.5.4, Polars 2.1.0, scikit-learn 1.9.1. CUDA AMP/bf16/fp16 и peak VRAM здесь не измерены. Kaggle admission заблокирован отсутствием доступа к Default GPU/credentials и неподтверждённым способом подключения ImageNet-весов. Slug известен: [страница соревнования](https://www.kaggle.com/competitions/assessment-of-logical-statements-in-an-image-text-problem). Следующее действие для runtime proof — предоставить этот runtime и разрешённый mount, выполнить notebook с интернетом off и сохранить новый receipt. Сабмит не отправлялся; проблема качества модели этим не решается.

### Annotations

| №   | Annotation |
| --- | ---------- |
| 1   | [FiLM](https://arxiv.org/pdf/1709.07871): условная модуляция feature maps текстом, замороженный ResNet-101 |
| 2   | [MAC](https://arxiv.org/pdf/1803.03067): итеративное reasoning, data efficient (85.5% на 10% CLEVR) |
| 3   | [Relation Networks](https://en.wikipedia.org/wiki/Relation_network): попарные отношения ячеек feature map + координаты |
| 4   | [CoordConv](https://arxiv.org/pdf/1807.03247): явные координатные каналы для позиционных задач |
| 5   | [TbD-net](https://arxiv.org/pdf/1803.05268): NMN с масками внимания, нужны программы |
| 6   | [CLOSURE](https://ar5iv.arxiv.org/html/1912.05783): провалы систематического обобщения у FiLM/MAC/NMN |
| 7   | [NegCLIP / ARO](https://arxiv.org/pdf/2210.01936): CLIP как bag-of-words, лечение hard negatives |
| 8   | [Kamath et al., ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02149.pdf): hard negatives без hard positives вредят |
| 9   | [Slot Attention](https://arxiv.org/pdf/2006.15055): object-centric представления без масок |
| 10  | [NLVR](https://lil.nlp.cornell.edu/nlvr/index.html), [ShapeWorld](https://arxiv.org/pdf/1704.04517): похожие true/false бенчмарки (только методика, данные нельзя) |

### Final Ensemble

| Model | CV  | Public LB | Private LB |
| ----- | --- | --------- | ---------- |
|       |     |           |            |

### Processed ideas
#### Good
- pass
#### Neutral
- pass
#### Bad
- pass
#### Some experimental ideas I didn't implemented
- Slot Attention (unsupervised на train+val картинках) как object-centric энкодер
- NMN / Vector-NMN с программами из грамматического парсера (если парсер разрешат)
- Каноникализация подписей парсером вместо/вместе с аугментацией перестановкой клауз
- CLIP/SigLIP patch-токены вместо пулингового эмбеддинга + cross-attention
- Category-specific модели: отдельная голова для counting/comparison
