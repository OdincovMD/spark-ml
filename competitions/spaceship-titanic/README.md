# Spaceship Titanic

**Бинарная классификация пассажиров на Scala и Spark ML.**

[Соревнование Kaggle](https://www.kaggle.com/competitions/spaceship-titanic) · [Описание данных](https://www.kaggle.com/competitions/spaceship-titanic/data) · [Результаты](RESULTS.md) · [Документация проекта](../../docs/README.md)

Цель — предсказать `Transported`: был ли пассажир перенесён в другое измерение. Метрика соревнования — accuracy. Submission имеет формат `PassengerId,Transported` со значениями `True` или `False`.

## Данные

```bash
kaggle competitions download -c spaceship-titanic -p competitions/spaceship-titanic/data/raw
unzip competitions/spaceship-titanic/data/raw/spaceship-titanic.zip -d competitions/spaceship-titanic/data/raw
```

Используйте свою учётную запись Kaggle и проверьте правила соревнования. Нужны `data/raw/train.csv` и `data/raw/test.csv`. Ожидается исходный порядок колонок Kaggle:

```text
PassengerId,HomePlanet,CryoSleep,Cabin,Destination,Age,VIP,RoomService,FoodCourt,ShoppingMall,Spa,VRDeck,Name
```

В обучающем CSV дополнительно требуется последняя колонка `Transported`. Возраст и расходы читаются как числа, остальные поля — как строки. Идентификаторы должны быть уникальны и иметь формат `gggg_pp`, например `0001_01`; ведущие нули сохраняются. Пропуски признаков допустимы.

## Пайплайн

1. `SQLTransformer` выделяет палубу, номер и сторону каюты, номер участника группы, сумму расходов и признак отсутствия расходов. `Name` и сам идентификатор не используются классификатором.
2. `Imputer` заполняет числовые пропуски медианами обучающих данных. В суммарных расходах пропущенное значение трактуется как 0.
3. `StringIndexer` кодирует категории и поддерживает неизвестные значения при инференсе.
4. `VectorAssembler` формирует вектор; `RandomForestClassifier` делает предсказание.

Все стадии сохраняются в одной модели. Holdout: 20%, seed `42`. По умолчанию: 200 деревьев, глубина 8, `maxBins=64`. Baseline не учитывает размер всей группы и не объединяет train/test для вычисления признаков.

## Обучение и инференс

Команды выполняются из корня репозитория:

```bash
# Проверить accuracy на holdout и сохранить модель, не читая test.csv.
./bin/sbt-local "spaceshipTitanic/run train"

# Только загрузить модель и предсказать test.csv.
./bin/sbt-local "spaceshipTitanic/run predict"

# Полный цикл за один запуск.
./bin/sbt-local "spaceshipTitanic/run run --model models/rf_v2 --output output/submission_v2.csv"

# Настроить модель и split.
./bin/sbt-local "spaceshipTitanic/run train --num-trees 300 --max-depth 10 --seed 1337 --model models/rf_seed1337"

# Новая партия пассажиров: train.csv не требуется.
./bin/sbt-local "spaceshipTitanic/run predict --input data/raw/new_passengers.csv --model models/rf_v1 --output output/new_predictions.csv"
```

Пути относительны к `competitions/spaceship-titanic`. Модель по умолчанию: `models/rf_v1`; результат: `output/submission.csv`, отсортированный по `PassengerId`. Для замены существующих артефактов нужен `--overwrite`. Полный CLI: `./bin/sbt-local "spaceshipTitanic/run --help"`.

## Проверка и отправка

```bash
./bin/sbt-local "spaceshipTitanic/test"
kaggle competitions submit -c spaceship-titanic -f competitions/spaceship-titanic/output/submission.csv -m "Scala Spark RF v1"
```

Перед отправкой проверьте число предсказаний, уникальность ID и значения `True`/`False`. Зафиксируйте локальную accuracy и параметры в [RESULTS.md](RESULTS.md); добавляйте публичный score только после получения результата Kaggle.
