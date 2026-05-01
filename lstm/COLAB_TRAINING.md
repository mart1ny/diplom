# Обучение LSTM в Google Colab

## 1. Подготовка среды

```python
!git clone https://github.com/mart1ny/diplom.git
%cd diplom
!pip install -r requirements.txt
```

Если хочешь только training-контур без всего backend-стека:

```python
!pip install torch numpy
```

## 2. Генерация синтетического датасета

```python
!python lstm/generate_synthetic_dataset.py \
  --output data/lstm/synthetic_queues.jsonl \
  --summary data/lstm/synthetic_queues.summary.json \
  --lights 4 \
  --days 60 \
  --step-minutes 5 \
  --seed 42
```

Что получится:
- `data/lstm/synthetic_queues.jsonl`
- `data/lstm/synthetic_queues.summary.json`

## 3. Обучение модели

На GPU в Colab:

```python
!python lstm/train_demand_forecaster.py \
  --data data/lstm/synthetic_queues.jsonl \
  --output-dir models/demand_forecaster \
  --seq-len 12 \
  --horizon 3 \
  --epochs 25 \
  --batch-size 128 \
  --hidden-size 96 \
  --layers 2 \
  --dropout 0.1 \
  --device cuda
```

Артефакты:
- `models/demand_forecaster/demand_lstm.pt`
- `models/demand_forecaster/demand_lstm_scaler.json`
- `models/demand_forecaster/training_summary.json`

## 4. Скачать модель из Colab

```python
from google.colab import files

files.download("models/demand_forecaster/demand_lstm.pt")
files.download("models/demand_forecaster/demand_lstm_scaler.json")
files.download("models/demand_forecaster/training_summary.json")
```

## 5. Что крутить в первую очередь

- Если модель недообучается: увеличь `--epochs` до `40-60`
- Если модель переобучается: уменьшай `--hidden-size` и увеличивай `--dropout`
- Если хочешь более дальний прогноз: увеличь `--horizon`
- Если хочешь больше контекста: увеличь `--seq-len`

## 6. Рекомендуемый baseline

Для первого рабочего результата:

```python
!python lstm/train_demand_forecaster.py \
  --data data/lstm/synthetic_queues.jsonl \
  --output-dir models/demand_forecaster \
  --seq-len 12 \
  --horizon 3 \
  --epochs 20 \
  --batch-size 128 \
  --hidden-size 64 \
  --layers 2 \
  --dropout 0.1 \
  --device cuda
```
