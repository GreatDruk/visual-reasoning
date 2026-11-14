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
