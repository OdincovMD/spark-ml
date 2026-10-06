# Digit Recognizer — результаты

## Baseline multinomial LR v1

- Пайплайн: 784 пикселя → нормализация `/255` → multinomial LogisticRegression.
- Параметры: `maxIter=100`, `regParam=0.01`, split seed `42`.
- Holdout: 20%; метрика — accuracy.
- Модель: `models/multinomial_lr_v1`; submission: `output/submission.csv`.
- Accuracy на реальных данных: **не измерена**.
- Kaggle public score: **не отправлялся**.

## Журнал экспериментов

Для каждого запуска сохраняйте дату, команду, seed, параметры оптимизации, локальную accuracy и пути артефактов. После отправки добавьте Kaggle score. Не используйте synthetic-test accuracy как результат соревнования.
