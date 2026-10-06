# Spaceship Titanic — результаты

## Baseline RF v1

- Пайплайн: признаки каюты/расходов → медианная импутация → категоризация → RandomForest.
- Параметры: `numTrees=200`, `maxDepth=8`, `maxBins=64`, `seed=42`.
- Holdout: 20%; метрика — accuracy.
- Модель: `models/rf_v1`; submission: `output/submission.csv`.
- Accuracy на реальных данных: **не измерена**.
- Kaggle public score: **не отправлялся**.

## Журнал экспериментов

После запуска добавьте дату, точную команду, параметры, seed, локальную accuracy, пути модели и submission. Публичный score записывайте после получения оценки Kaggle. Результаты синтетических тестов не являются оценкой качества на соревновании.
