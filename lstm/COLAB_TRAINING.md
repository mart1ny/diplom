# Обучение LSTM в Google Colab

## Где брать весь pipeline

Готовый one-click pipeline лежит здесь:

- [colab_oneclick_pipeline.py](/Users/danilvlasuk/Desktop/diplom/lstm/colab_oneclick_pipeline.py)

Датасет, который можно просто загрузить в Colab, лежит здесь:

- [synthetic_queues_dataset.csv](/Users/danilvlasuk/Desktop/diplom/lstm/data/synthetic_queues_dataset.csv)

## Как запускать

1. Загрузи `synthetic_queues_dataset.csv` в Colab.
2. Выполни установку зависимостей:

```python
!git clone https://github.com/mart1ny/diplom.git
%cd diplom
!pip install -r requirements.txt
```

3. Запусти pipeline одной командой:

```python
!python lstm/colab_oneclick_pipeline.py
```

Скрипт сам:
- найдёт `csv`,
- выберет `cuda`, если GPU доступна,
- запустит обучение с захардкоженными параметрами,
- сохранит артефакты модели.

## Куда положить датасет в Colab

Скрипт ищет файл по одному из путей:

- `/content/synthetic_queues_dataset.csv`
- `/content/synthetic_queues.csv`
- `lstm/data/synthetic_queues_dataset.csv`
- `data/lstm/synthetic_queues.csv`

Самый простой вариант: загрузи файл в корень Colab как
`/content/synthetic_queues_dataset.csv`.

## Что получится на выходе

После выполнения появятся файлы:

- `models/demand_forecaster/demand_lstm.pt`
- `models/demand_forecaster/demand_lstm_scaler.json`
- `models/demand_forecaster/training_summary.json`

## Как скачать результат

```python
from google.colab import files

files.download("models/demand_forecaster/demand_lstm.pt")
files.download("models/demand_forecaster/demand_lstm_scaler.json")
files.download("models/demand_forecaster/training_summary.json")
```
