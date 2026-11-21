# Visual reasoning

Исследуем три фиксированные основы и следующий слой восстановления объектов, сохраняя parser и executor. Обоснования, матрица и результаты — в [KaggleTemplate.md](KaggleTemplate.md).

```text
configs/                  019_base.yaml … 055_count.yaml
src/                      восприятие, программы, обучение, оценка и упаковка
tests/                    проверки готового кода
notebooks/base.ipynb       единый Kaggle-ноутбук
inputs/                   неизменяемые входы; не отслеживаются Git
.work/                    временные кэши; не отслеживаются Git
artefacts/results/        отчёты и прогнозы
artefacts/weights/        обученные параметры
artefacts/kaggle/         ZIP для private dataset
artefacts/archive/        предыдущие артефакты
docs/.research/            временные исследования; не отслеживаются Git
docs/archive/legacy.zip   проверенный снимок прежнего исходного кода
```

Установить окружение: `uv sync --extra dev`. Проверки: `uv run pytest`. Зависимости Kaggle должны быть доступны офлайн; ноутбук ничего не устанавливает из сети. Версии сохраняются в каждом отчёте.

Локальный контроль:

```powershell
uv run python -m src.run --config configs/019_base.yaml --data data
```

Выбранный опыт использует тот же entry point. Для проверки кода на нескольких сценах добавьте `--limit 4 --steps 1`; такие результаты явно помечаются `smoke` и не являются метрикой модели.

Пакет для Kaggle:

```powershell
uv run python -m src.package --inputs inputs --output artefacts/kaggle/base.zip
```

Загрузить `base.zip` в private dataset, подключить данные соревнования, открыть `notebooks/base.ipynb`, выбрать T4 ×2, отключить интернет. Для первого исследования основы использовать `SELECTED = ["019_base", "020_resnet", "029_siglip"]`. Текущий notebook выбирает counters 051–055 и требует актуальный runtime с foundation checkpoint, donors и L2 anchors. `SEEDS=None` использует seed каждого конфига; `SEEDS=[853]` повторяет выбранные варианты.

Если Kaggle распаковал ZIP при загрузке, ноутбук находит корень проекта по `package.json` и каталогам `src`, `configs`, `inputs` и читает их непосредственно из датасета. Копировать веса в рабочий каталог не нужно. Сохранённый файл `base.zip` также поддерживается; контрольные суммы проверяются в обоих случаях.

Для обновления кода поверх уже загруженного набора весов можно собрать маленький пакет:

```powershell
uv run python -m src.package --code-only --output artefacts/kaggle/runtime.zip
```

Подключить `runtime.zip` отдельным private dataset вместе с прежним полным датасетом, обновить `notebooks/base.ipynb` и выполнить **Restart & Run All**. После автоматической распаковки ноутбук находит `runtime.json`, проверяет контрольные суммы и копирует только `src` и `configs`; веса продолжает читать из прежнего датасета. Одновременно подключать несколько runtime-пакетов нельзя. Альтернатива — обновить полный `base.zip`, тогда runtime-датасет не нужен.

Очередь выполняет независимые конфигурации по одной на GPU. Логи идут сразу с префиксом конфига и GPU; прогресс основных стадий — примерно раз в 30 секунд и на границах этапов. В конце сохраняются полные логи в `artefacts/results/logs` и сводка `suite.json`. Ошибка конфигурации показывается сразу, остальные независимые задания продолжаются; итоговая ошибка перечисляет неудачные опыты. Размер физического батча выбирается измерением скорости и VRAM; у ResNet проверяются все активные этапы curriculum и тот же loss, что при обучении, включая count-loss. Общий batch ResNet остаётся 128 через накопление градиентов. Loss и его аккумуляторы считаются в FP32, даже если выходы модели FP16. Двухкарточный DDP пока не реализован и не измерен.

При обновлении кода нужно обновить существующий runtime-датасет и выполнить Restart & Run All с новым notebook. Для union необходим текущий пакет с папкой `foundations`, содержащей фиксированные train-only 044/853 и 023/812. Этот пакет больше прежнего code-only overlay; исходный base dataset остаётся подключённым.

Для воспроизводимой сборки такого исследовательского пакета:

```powershell
uv run python -m src.package --code-only --output artefacts/kaggle/runtime.zip --foundation convnext=artefacts/results/044_convnext/853/20261121_175510_913616 --foundation resnet=artefacts/results/023_resnet/812/20261121_152340_135948
```

Упаковщик читает явно выбранные результаты офлайн и помещает checkpoint, контрольные val scores, provenance и SHA256 в immutable input bundle. Обучение и inference читают их только из входного dataset; `artefacts` остаётся исключительно выходом. Notebook проверяет эти файлы потоково и не копирует веса в working/output. Это ускорение исследования следующего уровня, не замена полного обучения всех моделей при финальном benchmark.

Сначала suite подготавливает frozen object tables четырёх фаз для трёх источников, распределяя источники между GPU. Затем запускает шесть маленьких голов на двух GPU. Варианты 046–049 используют один кэш; ROI 049 извлекаются из имеющейся карты CNN. Кэш только в scratch; ordinary features FP32, ROI FP16. Отчёты подготовки — `results/preparation`, общее время — `results/timing.json`. В каждом опыте сохраняются `views.npz`, `blends.npz`, `objects.npz`, checkpoint головы и сравнение со всеми тремя основами. `source_replay` показывает расхождение текущего plain source scoring с сохранённой основой. Bootstrap оставлен для локального анализа после скачивания, чтобы не задерживать GPU-очередь.

Все новые fit-операции используют только train. Вторая ветка смеси фиксируется на контрольных прогнозах. Исторический val — исследовательский ориентир, многократно использованный для разработки. Ноутбук пока не является финальным train+val обучением и генератором соревнательного сабмита.

Новые fit-запуски сохраняют `objects.npz` с именами val-сцен, признаками объектной таблицы, позициями и valid-маской. Для SigLIP признаки включают PCA-колонки. Это диагностический выход для разбора пропусков и дублей; обучение его не читает. Текущая матрица, измеренные результаты и следующая группа запусков находятся в `KaggleTemplate.md`.

Пакет требует подготовленного `inputs/manifest.json`. Входы не восстанавливаются молча из `artefacts` и не скачиваются автоматически. Исторические teacher targets и контрольные checkpoint — явно объявленные team-зависимости.

Текущий этап — dense counts 051–055. Обновлённый `runtime.zip` содержит те же основы, небольшой train-donor asset и три фиксированных L2 validation anchor. Source CNN каждого counter обучается 18 эпох; L2-ветки для этой исследовательской оценки не переобучаются. Доноры проверяются против полного train.csv, не включают val; foreground cache и JPEG decode общие. Наблюдаемые foreground-пиксели защищены при вставке, однако исходные маски и классы являются псевдометками.

Notebook по умолчанию запускает 051–055 на seed 421. Очередь сначала готовит общие данные, затем распределяет счётчики между GPU. Размер батча профилируется по полному loss/backward (включая paired images и joint density); effective batch остаётся 96, BN statistics заморожены. В `counts.npz` сохраняются histogram, mass map и query counts; в отчёте — абсолютная count MAE, доля ошибок менее половины объекта, acceptance rate вставок и сравнение трёх L2 anchors. `branch_metrics` относится только к counting/comparison; полный AUC находится в `metrics`/`blends`. Пространственные категории этот counter не изменяет.

Дополнительные входы упаковщика задаются явными `--asset ROLE=PATH` и `--anchor ROLE=NPZ`. Исследовательские anchors — сохранённые прогнозы ранее зафиксированных моделей, не обучающие метки и не test inference pipeline. В итоговом fresh-training запуске весь L2 и counter должны воспроизводиться из обучения. `result.zip` каждого counter содержит отчёты, а веса отдельно лежат в `artefacts/weights` — они не дублируются внутри вложенного ZIP.
# Today's two submission candidates

Update the existing runtime dataset with `artefacts/kaggle/runtime.zip`, retain the original base input dataset, and run `notebooks/base.ipynb` on T4 ×2 with Internet Off. Defaults 056_compact/057_quality share frozen train-only checkpoint inference and write two CSV files directly under `/kaggle/working` before validation diagnostics. Historical SigLIP val/test predictions are explicit checksummed input assets; val matches the fixed SigLIP reference exactly. This is an inference comparison, not proof of final fresh training within 50 minutes. Late evidence uses a train-derived placement prior and a guarded paired-count swap; full late-stage investigation remains pending.

