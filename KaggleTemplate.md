### Текущее состояние

- Лучшая модель — трёхэтапная на замороженном SigLIP2: голова выбора из пары → ветка подсчёта → геометрическая ветка spatial_counting. **val AUC 0.9774 / 0.9776** (seeds 42 / 43), обучение только на train.
- Подписи всех сплитов канонизируются в формы train (`data.canonicalize: true`), оценки нормализуются внутри пары (`train.postprocess.twin_normalize: true`). Пара = две строки с одной картинкой и категорией; в train, val и test такая группа всегда ровно из двух строк, ровно одна истинна.
- Признаки SigLIP2 считаются один раз и кэшируются (`paths.features_dir`); обучаются только голова и текстовый GRU.
- Финальный notebook ([02_final.ipynb](notebooks/02_final.ipynb)) **ещё обучает старого кандидата B**; перевод на трёхэтапную модель, офлайн-веса SigLIP2 и проверка лимита 50 мин на Kaggle GPU не сделаны.

Воспроизведение лучшей модели (train → val; каждый этап берёт чекпойнт предыдущего через `model.init_from`):

```bash
uv run python -m src.train experiment=pair_best_stage1 name_exp=debug_pair_best_stage1
uv run python -m src.train experiment=pair_best_stage2 model.init_from=debug_pair_best_stage1 name_exp=debug_pair_best_stage2
uv run python -m src.train experiment=pair_geometric_stage2 model.init_from=debug_pair_best_stage2 name_exp=debug_pair_geometric
```

### To-do
- [x] Анализ сдвига train/val/test (`scripts/shift_analysis.py`): картинки не различаются (adversarial AUC ≈ 0.50), 25.4% строк test — шаблоны подписей, которых нет в train
- [x] Канонизация подписей + twin normalization
- [x] Голова выбора из пары (`src/models/pair_choice.py`)
- [x] Ветка подсчёта (`src/models/count_distribution.py`)
- [x] Геометрическая ветка spatial_counting (`src/models/geometric_count.py`)
- [ ] Присвоить ID и заархивировать конфиги прогонов из таблицы ниже (пока это debug-прогоны без `configs/archive`)
- [ ] Поднять counting (0.856), spatial (0.930), spatial_negation (0.896)
- [ ] Финальный notebook на трёхэтапной модели: train+val, SigLIP2 офлайн, Restart & Run All на Kaggle Default GPU ≤ 50 мин
- [ ] Moodle-пакет: команда, вклад, ID сабмита, ноутбук, final_submission.csv, таблица экспериментов (архитектура, val Accuracy + AUC, runtime, seed), версии библиотек

### Notes
- Задача: по картинке (240×160, CLEVR-подобная сцена) и утверждению предсказать P(утверждение истинно). Метрика ROC-AUC.
- Данные: train 5000 картинок / 29 986 подписей, val 1000 / 6000, test 6000 / 35 992. 9 категорий, 5 уровней, баланс 50/50 в каждой категории. На картинку 6 подписей = 3 категории × пара.
- Словарь 55 слов, OOV в test нет. Сдвиг только в формулировках: правила `src/text_aug.py` покрывают 100% шаблонов test; канонизация проверена на всех 71 978 подписях (смысл не меняется).
- Разрешено: классификаторы torchvision с ImageNet-весами, text-only энкодеры, CLIP/SigLIP/OpenCLIP, любые свои слои; train+val для финала; неразмеченный test (анализ, поиск близнецов, unsupervised-шаги).
- Запрещено: детекция/сегментация (даже из torchvision), совместные VLM, LLM, внешние данные (CLEVR, NLVR, ShapeWorld), скрытые метки test.
- Решения команды: без pseudo-labeling на test. Ансамбли допустимы на финальном этапе; пока идёт работа над моделью, копии одной модели по сидам массово не гоняем.
- Аугментации: hflip только с заменой left↔right в подписи; vflip, поворот, color jitter нельзя.
- Лимиты: 5 сабмитов/день, 1 финальный сабмит, команда до 3 человек, мердж до 18.11, финал 22.11 14:00 МСК.

### Naming
- Ноутбуки: `notebooks/0X_<name>.ipynb`.
- Эксперименты: сквозной трёхзначный номер `00X_<name>`, номер не переиспользуется. Занято до 018 включительно.
- Конфиги основных прогонов (таблица Solution) — полным скомпонованным конфигом в `configs/archive/00X_<name>.yaml`; архив только дописывается. Debug-прогоны не архивируются.

### Questions
- Разрешён ли детерминированный разбор подписей (канонизация, извлечение числа, фильтра, отношения, опорного объекта)? **Уже используется** — подтвердить у преподавателя.
- Разрешён ли DINOv2 (не torchvision, не CLIP-семейство)?
- Как подключать предобученные веса офлайн при лимите 20 MB: Kaggle Dataset/Model не считается?
- Таблица экспериментов просит validation Accuracy, метрика AUC: указывать обе?
- ~~Использование test~~ — неразмеченный test разрешён. ~~Twin normalization~~ — разрешена, включена. ~~Колонка сабмита~~ — `label`, Kaggle принимает.

### Solution

#### Модели выбора из пары (текущая линия)

Все прогоны: обучение только на train, оценка на официальном val, seed 42, если не указано иное. Debug-прогоны в `artefacts/hydra/<name>/`. Accuracy = pair accuracy (после twin normalization). Время fit: MPS, если не указано CPU.

| Шаг | Прогон | Изменение | val AUC | val acc | fit, мин |
| --- | --- | --- | ---: | ---: | ---: |
| 0 | debug_canon_off | кандидат B (ResNet-50 + GRU + cosine), как было | 0.5994 | 0.5565 | 1.2 |
| 1 | debug_canon_on | + канонизация (twin-normalized AUC 0.6338; val_para 0.5974 → 0.6338) | 0.6111 | 0.5630 | 1.3 |
| 2 | debug_pair_siglip / _repro | голова выбора из пары, SigLIP2, сетка 8×12 | 0.9433 (CPU) / 0.9297 | 0.8633 / 0.8380 | 3.9 / 2.4 |
| 3 | debug_pair_A_fullgrid | полная сетка 16×24 | 0.9543 | 0.8667 | 2.5 |
| 4 | debug_pair_B_wide | + голова шире (dim 256, 8 queries) | 0.9566 | 0.8793 | 3.7 |
| 5 | debug_pair_D_block8 | + признаки 8-го блока SigLIP2 вместо последнего | 0.9615 | 0.8847 | 3.6 |
| 6 | debug_pair_F_flip | + hflip с заменой left↔right (= `pair_best_stage1`) | 0.9675 | 0.8943 | 12.8 |
| 7 | debug_count_on_F | + ветка подсчёта поверх F (= `pair_best_stage2`) | 0.9745 | 0.9060 | 1.8 |
| 8 | debug_geometry_v1_s42 / s43 | + геометрическая ветка spatial_counting (`pair_geometric_stage2`) | **0.9774 / 0.9776** | 0.9147 / 0.9153 | 13.2 / 13.5 (CPU) |

AUC по слабым категориям (остальные — attribute, binding, negation, position — 0.99–1.00 начиная с шага 5):

| Шаг | comparison | counting | spatial | spatial_counting | spatial_negation |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.767 | 0.636 | 0.495 | 0.520 | 0.521 |
| 5 | 0.946 | 0.748 | 0.929 | 0.548 | 0.891 |
| 6 | 0.972 | 0.781 | 0.930 | 0.547 | 0.896 |
| 7 | 0.972 | 0.856 | 0.930 | 0.600 | 0.896 |
| 8 (s42 / s43) | 0.972 | 0.856 | 0.930 | 0.832 / 0.844 | 0.896 |

spatial_counting на val — всего 160 пар. 95% bootstrap CI по парам: шаг 7 [0.509, 0.683], шаг 8 s42 [0.765, 0.886], s43 [0.779, 0.898] — интервалы не пересекаются.

Проверки честности: пары на val/test строятся только по картинке, категории и тексту, метки и `pair_id` на вход не идут; порядок строк внутри пары в CSV случаен (первая истинна в 49–50%); AUC пересчитан отдельно по сохранённым логитам против `val.csv`; при перестановке картинок между подписями AUC падает до 0.49; обучение повторено с нуля.

#### Публичный лидерборд

| Сабмит | Модель | Public LB |
| --- | --- | ---: |
| pair_count_val09745.csv | шаг 7, обучение на train | 0.96851 |
| pair_count_trainval.csv | шаг 7, train+val, фиксированные эпохи 20 / 23 | 0.96779 |
| pair_count_ens2.csv | среднее логитов двух строк выше (ансамбль — к финалу, не сейчас) | 0.97340 |

#### Что не сработало
- Длинное обучение (до 60 эпох) — 0.9489: раньше начинается переобучение.
- 6-й блок SigLIP2 — 0.9516, хуже 8-го.
- Геометрическая ветка без супервизии локализации — spatial_counting 0.670 вместо 0.83.
- Классический подсчёт объектов без обучения (`scripts/classical_count_probe.py`): точное число объектов на 30.4% отложенных картинок, ±1 — 77%. Neuro-symbolic через ручную сегментацию отклонён.
- Train+val вместо train для шага 7 на LB не дал прироста (0.96779 против 0.96851, в пределах разброса между запусками).

#### Старая линия (ResNet + GRU, 006–018)

Обучение только на train, seed 42, MPS. Конфиги в `configs/archive`.

| ID | Модель | val AUC | val acc | pair acc |
| --- | --- | ---: | ---: | ---: |
| [006](configs/archive/006_lab_baseline.yaml) | ResNet-18 + GRU + cosine | 0.598546 | 0.559167 | 0.575000 |
| [007](configs/archive/007_r50_cosine.yaml) | ResNet-50 + GRU + cosine (кандидат B) | 0.602662 | 0.559167 | 0.574333 |
| [008](configs/archive/008_pooled_mlp.yaml) | ResNet-50 + GRU + pooled MLP | 0.517263 | 0.517833 | 0.515667 |
| [009](configs/archive/009_film_bce.yaml) | ResNet-50 + GRU + FiLM | 0.553643 | 0.532667 | 0.538667 |
| [010](configs/archive/010_pair_loss.yaml) | B + pair loss | 0.600500 | 0.555333 | 0.572667 |
| [011](configs/archive/011_text_aug.yaml) | B + перефразы p=0.3 | 0.602710 | 0.559167 | 0.566667 |
| [012](configs/archive/012_hflip.yaml) | B + перефразы + hflip | 0.601989 | 0.555333 | 0.565333 |
| [013](configs/archive/013_resnet101.yaml) | B с ResNet-101 | 0.598363 | 0.555000 | 0.561667 |
| [018](configs/archive/018_film_finetune.yaml) | FiLM + частичное дообучение layer3 | 0.499794 | 0.497833 | 0.499333 |

Повторы B по сидам 42/43/44 (007/014/015): mean val AUC 0.605125, STD 0.002294. Вся линия упиралась в около 0.60: признаки ResNet плохо разделяли объекты, а модель оценивала каждую подпись отдельно. Диагностика коллапса FiLM (018) — в [artefacts/film_long_training_investigation.md](artefacts/film_long_training_investigation.md).

### Annotations

| №   | Annotation |
| --- | ---------- |
| 1   | [FiLM](https://arxiv.org/pdf/1709.07871): условная модуляция feature maps текстом |
| 2   | [MAC](https://arxiv.org/pdf/1803.03067): итеративное reasoning, data efficient |
| 3   | [Relation Networks](https://en.wikipedia.org/wiki/Relation_network): попарные отношения ячеек feature map + координаты |
| 4   | [CoordConv](https://arxiv.org/pdf/1807.03247): явные координатные каналы для позиционных задач |
| 5   | [CLOSURE](https://ar5iv.arxiv.org/html/1912.05783): провалы систематического обобщения у FiLM/MAC/NMN |
| 6   | [NegCLIP / ARO](https://arxiv.org/pdf/2210.01936): CLIP как bag-of-words, лечение hard negatives |
| 7   | [Kamath et al., ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02149.pdf): hard negatives без hard positives вредят |
| 8   | [Slot Attention](https://arxiv.org/pdf/2006.15055): object-centric представления без масок |

### Processed ideas
#### Good
- Сравнение двух подписей пары вместо оценки каждой отдельно: 0.63 → 0.93.
- Канонизация подписей: снимает сдвиг формулировок test (val_para = val по построению).
- Полная сетка признаков, промежуточный слой ViT (блок 8), hflip с заменой left↔right.
- Явная числовая ветка для counting и геометрическая (опорный объект + полуплоскость) для spatial_counting.
#### Neutral
- Train+val для финального обучения: на LB в пределах шума.
#### Bad
- FiLM / pooled MLP / частичное дообучение ResNet на линии с одиночной оценкой подписи.
- Классическая сегментация и подсчёт без обучения.
#### Not tried yet
- Slot Attention как object-centric энкодер.
- Геометрический модуль для spatial и spatial_negation (сейчас только spatial_counting).
- Дообучение SigLIP2 вместо замороженных признаков.
