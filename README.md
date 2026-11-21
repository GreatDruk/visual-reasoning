# Visual reasoning

Исследуем основу `r50_img + sigft`, сохраняя parser и executor. Обоснования, матрица и результаты — в [KaggleTemplate.md](KaggleTemplate.md).

```text
configs/                  019_base.yaml … 039_siglip.yaml
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

Загрузить `base.zip` в private dataset, подключить данные соревнования, открыть `notebooks/base.ipynb`, выбрать T4 ×2, отключить интернет. Сначала оставить `SELECTED = ["019_base", "020_resnet", "029_siglip"]`. Затем менять только список конфигов. `SEEDS = [421, 853]` повторяет выбранные варианты на двух seed.

Все новые fit-операции используют только train. Вторая ветка смеси фиксируется на контрольных прогнозах. Исторический val — исследовательский ориентир, многократно использованный для разработки. Ноутбук пока не является финальным train+val обучением и генератором соревнательного сабмита.

Пакет требует подготовленного `inputs/manifest.json`. Входы не восстанавливаются молча из `artefacts` и не скачиваются автоматически. Исторические teacher targets и контрольные checkpoint — явно объявленные team-зависимости.
