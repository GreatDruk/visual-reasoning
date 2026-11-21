### Текущее состояние

- **Сейчас лучший — пайплайн тиммейта `count_swap` + TTA ветки r50_img: public LB 0.99972** (лидеры 1.00000). Подробности и очередь идей — в разделе «Линия count_swap и TTA» ниже. Всё, что ниже в этом пункте, относится к нашей прежней линии.
- Лучшая модель — трёхэтапная на замороженном SigLIP2: голова выбора из пары → ветка подсчёта → геометрическая ветка (существование, spatial-маршруты, comparison через подсчёт). val AUC 0.9844, public LB 0.98019 (`pair_geometric_comparison`, seed 42, обучение только на train). Плюс объектные слоты (Slot Attention поверх SigLIP2): **val AUC 0.9924, public LB 0.98999**. Слабее всего counting 0.943 и spatial_counting 0.941.
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
- [x] Объектные слоты (Slot Attention): spatial 0.991, spatial_negation 0.995, comparison 0.991
- [ ] Поднять counting (0.943) и spatial_counting (0.941): слияние объектов в плотных сценах — 16 слотов, дообучение слотов, супервизия числа объектов
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
| 8 | debug_geometry_v1_s42 / s43 | + геометрическая ветка spatial_counting (`pair_geometric_stage2`) | 0.9774 / 0.9776 | 0.9147 / 0.9153 | 13.2 / 13.5 (CPU) |
| 10 | debug_slots_k11 (s42) + sub_geometric_full | Slot Attention (K=11, предобучение без меток на train+val+test, `src/slots_pretrain.py`) + голова-программа подсчёта по слотам (`pair_slots_all`); отступ слотов складывается с отступом шага 9 | **0.9924** (s43: 0.9922) | — | 32 (слоты) + 14 |
| 9 | debug_shift_existence_s42 / s43; sub_geometric_full (MPS) | + супервизия существования (attribute/binding/negation), геометрия на инференсе для spatial / spatial_negation, comparison через подсчёт (`pair_geometric_comparison`) | **0.9829 / 0.9838; 0.9844** | 0.9253 / 0.9290; 0.9310 | 13.3 / 13.8 (CPU); 14.8 |

AUC по слабым категориям (остальные — attribute, binding, negation, position — 0.99–1.00 начиная с шага 5):

| Шаг | comparison | counting | spatial | spatial_counting | spatial_negation |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.767 | 0.636 | 0.495 | 0.520 | 0.521 |
| 5 | 0.946 | 0.748 | 0.929 | 0.548 | 0.891 |
| 6 | 0.972 | 0.781 | 0.930 | 0.547 | 0.896 |
| 7 | 0.972 | 0.856 | 0.930 | 0.600 | 0.896 |
| 8 (s42 / s43) | 0.972 | 0.856 | 0.930 | 0.832 / 0.844 | 0.896 |
| 9 (MPS, сабмит) | 0.987 | 0.856 | 0.968 | 0.913 | 0.978 |
| 10 (шаг 9 + слоты) | 0.991 | 0.943 | 0.991 | 0.941 | 0.995 |

spatial_counting на val — всего 160 пар. 95% bootstrap CI по парам: шаг 7 [0.509, 0.683], шаг 8 s42 [0.765, 0.886], s43 [0.779, 0.898] — интервалы не пересекаются.

Проверки честности: пары на val/test строятся только по картинке, категории и тексту, метки и `pair_id` на вход не идут; порядок строк внутри пары в CSV случаен (первая истинна в 49–50%); AUC пересчитан отдельно по сохранённым логитам против `val.csv`; при перестановке картинок между подписями AUC падает до 0.49; обучение повторено с нуля.

#### Публичный лидерборд

| Сабмит | Модель | Public LB |
| --- | --- | ---: |
| pair_count_val09745.csv | шаг 7, обучение на train | 0.96851 |
| pair_count_trainval.csv | шаг 7, train+val, фиксированные эпохи 20 / 23 | 0.96779 |
| pair_count_ens2.csv | среднее логитов двух строк выше (ансамбль — к финалу, не сейчас) | 0.97340 |
| pair_geometric_val09843.csv | шаг 9 (`pair_geometric_comparison`, seed 42, train only), val 0.9844 | 0.98019 |
| slots_geometric_val09924.csv | шаг 10: шаг 9 + отступ слотов (`scripts/slot_submission.py`), val 0.9924 | **0.98999** |

#### Линия count_swap и TTA (2026-11-20/21)

Основа — пайплайн тиммейта (`swap_upstream_refit.py` → `count_swap_refit.py` → `count_evidence.compose`, ноутбук `18_count_swap_refit`). Val почти насыщен (у лидера 2 ошибочные пары, обе spatial_counting), поэтому решения принимаются по val-AUC отдельных веток и по LB.

TTA для `r50_img` (`kaggle_suite_detectors.py`, `detector_eval`): исходный вид + отражение (боксы зеркалятся обратно, left↔right в разборе сцены не нужен, т.к. сцена декодируется после возврата координат) + сдвиги на полшага сетки детектора (stride 4 → 2 px, replicate-pad). Скоры `joint` усредняются.

| Вариант ветки r50_img | val AUC ветки |
| --- | ---: |
| один вид | 0.998744 |
| + отражение | 0.999220 |
| 4 вида: p, f, s22, s22f (коммит fb5df7c) | 0.999529 |
| 8 видов: p, f, s11, s11f, s22, s22f, s33, s33f | 0.999576 |

| Сабмит | Что | Public LB |
| --- | --- | ---: |
| count_swap | исходный пайплайн тиммейта | 0.99970 |
| demote_slots_t8 | понижение уверенности по несогласию со слотами | 0.99963 |
| count_swap_r50tta_w01 | count_swap logit + 0.1·(TTA r50 − один вид), отражение | 0.99971 |
| count_swap_r50tta4_w01 | то же, 4 вида | **0.99972** |
| count_swap_r50tta8_w01 | то же, 8 видов | 0.99972 |

Сабмиты с поправкой w=0.1 — прокси. В полном пайплайне r50 входит в основу ансамбля с полным весом: нужно пересобрать `clevr_count_tv.zip` (`package_count_swap_refit.py`) и перезапустить `18_count_swap_refit`.

Вес поправки на val (лидер на val + w·Δ, Δ = 4 вида − один): w=0.05 — 2 ошибочные пары, 0.1 — 3, 0.3 — 4, 0.56 — 7, 1.0 — 20. Покатегорийные веса по формуле пайплайна (0.56 на лёгких, 0.375 на счётных) — 5. Вывод: поднимать вес аддитивной поправки нельзя; полный эффект TTA проверяется только перезапуском пайплайна (нужны union и counter).

Ветка `occ` на val (собрана локально: r50 + `sigft` + `rescue`, без union): исходный r50 — 0.999732 / 16 ошибочных пар; полный TTA r50 (4 вида) — 0.999670 / 18, 8 видов — 0.999679 / 19; масштабирование r50 не спасает. Доля TTA: +0.1·(4v−p) — 0.999735 / 15, +0.25 — 0.999733 / 14, +0.5 — 0.999717 / 14. Поэтому в `detector_eval` теперь `TTA_WEIGHT = .25`: полный TTA в пайплайне, скорее всего, ухудшил бы LB.

Не сработало: сдвиги окна кропа для `sigft` (полпатча ViT): исходный 0.99868, сдвиг 4 px 0.99859, 8 px 0.99844, среднее p+4+8 0.99860; TTA (отражение) для `sigft` — 0.99868 → 0.99866, откачено; ошибки парсера на новых шаблонах test — не подтвердились; понижение по слотам — LB хуже; подмешивание сырого r50 к лидеру — хуже на val; покатегорийная калибровка — не проверяема на насыщенном val.

Очередь идей:
1. Те же виды TTA для r50 внутри union-ветки (вход для всех категорий). Нужны `initial/union_{421,853}.pt` из `clevr_count_tv.zip` (их нет ни в переданных архивах, ни в Kaggle-датасете).
2. Отражение для dense counter (`initial/counter_{421,853}.pt`), бьёт по counting/spatial_counting. Счётчик инвариантен к отражению.
3. TTA для rescue-ветки и локальных фрагментных моделей (риск: пороговые гейты).
4. Перевести `TTA_VIEWS` на 8 видов (+0.00005 на val ветки) после проверки в полном пайплайне.

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
