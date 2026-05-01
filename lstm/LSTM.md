# LSTM-прогнозирование нагрузки

В папке `lstm/` теперь есть рабочий минимальный контур для обучения модели прогноза очередей:

- `generate_synthetic_dataset.py` — генерация синтетического датасета
- `demand_forecaster.py` — подготовка признаков, scaler, dataset, модель и predictor wrapper
- `train_demand_forecaster.py` — обучение и сохранение артефактов
- `COLAB_TRAINING.md` — готовые команды для Google Colab

## Формат данных

Каждая строка `jsonl`:

```json
{
  "light_id": "intersection_A",
  "timestamp": "2026-01-05T07:30:00Z",
  "approach": "north",
  "queue_len": 12.4,
  "risk_score": 0.31,
  "weekday": 0,
  "hour": 7,
  "minute": 30,
  "is_weekend": false,
  "is_holiday": false,
  "weather": "rain"
}
```

## Быстрый локальный запуск

Генерация датасета:

```bash
python lstm/generate_synthetic_dataset.py \
  --output data/lstm/synthetic_queues.jsonl \
  --days 45 \
  --lights 3 \
  --step-minutes 5
```

Обучение:

```bash
python lstm/train_demand_forecaster.py \
  --data data/lstm/synthetic_queues.jsonl \
  --output-dir models/demand_forecaster \
  --seq-len 12 \
  --horizon 3 \
  --epochs 20
```

Артефакты модели:

- `models/demand_forecaster/demand_lstm.pt`
- `models/demand_forecaster/demand_lstm_scaler.json`
- `models/demand_forecaster/training_summary.json`

## Следующий шаг интеграции

После обучения модель нужно будет подключить в основной пайплайн:

1. копить историю `queue_len/risk_score` по подходам;
2. вызывать прогноз перед LP-оптимизацией;
3. использовать `effective_demand = alpha * current + (1 - alpha) * forecast`.
