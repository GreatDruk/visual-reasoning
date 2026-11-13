### To-do
- [ ] EDA: число подписей по split/category/level, подписи на картинку, структура pair_id, словарь, кластеры шаблонов, пересечение шаблонов train/val/test
- [ ] Валидационный фреймворк: общий AUC, AUC по 9 категориям, pair accuracy (score true twin > false twin), фиксированные сиды
- [ ] Аудит: text-only и image-only бейзлайны
- [ ] Бейзлайн: замороженный CLIP/SigLIP + MLP над [i, t, i*t, |i-t|]
- [ ] Основная модель: FiLM на замороженном ResNet-101 (torchvision, ImageNet) + CoordConv + GRU/BERT
- [ ] Лосс BCE + λ·pairwise по pair_id
- [ ] Аугментации: hflip с заменой left↔right, перестановка клауз (как позитивы), логические негативы n→n±1, left↔right
- [ ] MAC как второй кандидат
- [ ] Relation Network голова по ячейкам feature map для уровней 4–5
- [ ] Поиск близнецов в test + нормализация внутри пары (softmax по паре)
- [ ] Ансамбль / маршрутизация по category
- [ ] Финальный ноутбук: Restart & Run All, офлайн, ≤50 мин на Kaggle GPU, сиды, ≤20 MB кода
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

| ID  | Approach | CV  | CV STD | LB  | Date |
| --- | -------- | --- | ------ | --- | ---- |
|     | Text-only (аудит) |     |        |     |      |
|     | Image-only (аудит) |     |        |     |      |
|     | Модель из лабы (dual encoder, cos) |     |        |     |      |
|     | CLIP/SigLIP frozen + MLP |     |        |     |      |
|     | FiLM (ResNet-101 frozen + CoordConv + GRU) |     |        |     |      |
|     | FiLM + BCE + pairwise loss |     |        |     |      |
|     | FiLM + аугментации текста и hflip |     |        |     |      |
|     | MAC |     |        |     |      |
|     | FiLM + RN-голова |     |        |     |      |
|     | + нормализация внутри пар близнецов |     |        |     |      |

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
