# Spaceship Titanic — результаты

## Запуск 2026-10-31: расширенные признаки

Добавлен переключатель `--feature-set basic|extended`. `basic` сохраняет исходные 16 признаков вектора. `extended` добавляет восемь: номер группы, число пропущенных расходов, число сервисов с положительными расходами и пять долей сервисов в общей сумме. При нулевой сумме доли равны нулю; все вычисления используют только строку пассажира. Подготовка признаков сохраняется в модели и не требует train.csv при инференсе.

Сравнение выполнено на прежних реальных CSV (8 693 обучающих и 4 277 тестовых строк, SHA-256 ниже). Общие параметры: GBT, 100 итераций, глубина 5, шаг 0.05, `maxBins=64`, holdout 20%. Для каждого seed обе конфигурации используют одинаковый split; seed также передаётся классификатору. Медианы и категории обучаются только на training split. После оценки каждая модель заново обучена на всех размеченных строках и сохранена отдельно.

| Seed | `basic` accuracy | `extended` accuracy | Изменение |
| --- | --- | --- | --- |
| 42 | 0.82279 | 0.81623 | −0.00656 |
| 1337 | 0.79025 | 0.79906 | +0.00881 |
| Среднее по двум seed | 0.80652 | 0.80765 | ≈ +0.00113 |

Исходный результат seed 42 воспроизведён. Среднее улучшилось примерно на 0.11 процентного пункта, но эффект неодинаков на двух разбиениях: на seed 42 расширенные признаки ухудшили результат. Это экспериментальный вариант, а не подтверждённое устойчивое улучшение. По умолчанию остаётся `basic`; прежний GBT v1 остаётся вариантом с подтверждённой оценкой Kaggle. Настройки уже оцениваются по этим holdout, поэтому результаты не являются независимой оценкой после подбора. Новые модели в Kaggle не отправлялись; их public accuracy не измерена.

Окружение: Eclipse Temurin JDK 11.0.32.1, sbt 1.11.3, Scala 2.12.18, Spark 3.5.9; heap приложения 2 GiB, `local[4]`, 8 shuffle partitions. Команды из корня:

```bash
export JAVA_TOOL_OPTIONS='-XX:ActiveProcessorCount=4 -Dspark.master=local[4] -Dspark.sql.shuffle.partitions=8'
export SPARK_LOCAL_IP=127.0.0.1

for seed in 42 1337; do
  for features in basic extended; do
    ./bin/sbt-local "spaceshipTitanic/run train --classifier gbt --max-depth 5 --max-iter 100 --step-size 0.05 --feature-set $features --seed $seed --model models/gbt_${features}_seed${seed}_20261031"
  done
done

./bin/sbt-local "spaceshipTitanic/run predict --train data/raw/not-needed.csv --model models/gbt_extended_seed42_20261031 --output output/submission_gbt_extended_seed42_20261031.csv"
./bin/sbt-local "spaceshipTitanic/test" "core/test"
```

Артефакты с указанными именами уже существуют; для повторения задайте новые пути или явный `--overwrite`. Как и в предыдущем запуске, в PATH добавлен временный launcher `/tmp/spark-ml-tools/bin/sbt`, использующий локальный `sbt-launch.jar` 1.11.3, и задан `JAVA_HOME=/home/hardbox/.local/share/jdk-11-temurin`. Зависимости взяты из кешей `bin/sbt-local`.

Пройдены 7 тестов: Spaceship Titanic — 5, core — 2. Проверены сохранение/загрузка расширенного GBT, инференс без train.csv, исходный RF, неизвестные категории, ведущие нули ID, пропуски и нулевые расходы, отклонение неизвестного набора признаков и защита CSV от перезаписи. Журнал: `output/feature_tests_20261031.log`. Журнал четырёх обучений и отдельного инференса: `output/feature_comparison_20261031.log`.

Отдельный `predict` занял 42 секунды и успешно отработал с несуществующим `--train`. В `output/submission_gbt_extended_seed42_20261031.csv` проверены заголовок `PassengerId,Transported`, 4 277 уникальных ID, полное совпадение ID и порядка с test.csv, ведущие нули и значения `True`/`False` (2 154 / 2 123). SHA-256: `1709b544a9bbc985aa464f8277fffcbf47d4306c64694cbcc809afba8afd943d`. В metadata всех четырёх моделей подтверждены параметры GBT и размеры вектора: 16 для `basic`, 24 для `extended`. Сводка: `output/feature_verification_20261031.json`.

Перед правками прочитаны `docs/README.md`, корневой README и руководство модуля. `.gitnexus` отсутствует, `npx` недоступен; зависимости проверены непосредственно по исходникам. Данные, модели, CSV и журналы в Git не добавляются.

## Запуск 2026-10-26, Europe/Moscow

Выполнено сравнение RandomForest и Gradient Boosted Trees на реальных данных [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic). Оба алгоритма используют один сохранённый Spark ML pipeline с подготовкой признаков, медианной импутацией и кодированием категорий. При оценке эти преобразования обучаются только на training split; после оценки модель заново обучается на всех размеченных строках.

Лучший локальный вариант — **GBT v1: holdout accuracy 0.82279**. Реальный сабмит **57670441** обработан Kaggle со статусом **`COMPLETE`**, **public accuracy 0.79588**.

## Данные и происхождение

Официальный архив скачан через авторизованный Kaggle CLI после принятия правил соревнования. До этого для первого запуска использована [публичная копия train/test](https://github.com/You-sha/Spaceship-Titanic). Оба файла побайтово совпали с официальным архивом; все последующие запуски используют CSV из него. Архив и распакованные исходники сохранены локально в `data/raw/kaggle_verified/`; рабочие CSV — в `data/raw/`.

Обучение: **8 693 строки**, классы `True`: 4 378, `False`: 4 315. Тест: **4 277 строк**. Проверены уникальность PassengerId, отсутствие пересечения train/test ID и полное совпадение порядка тестовых ID с sample submission. Тестовые labels не использовались.

| Файл | Размер, байт | SHA-256 |
| --- | --- | --- |
| `train.csv` | 805 421 | `17336d553f49ebdf6ecb266d2b5d3746e5dd308445f7c7864141c4f28d2a88d0` |
| `test.csv` | 372 487 | `3c02b0703423d27d51a657a111b15a37b8d03b3c1de325e3a1e6b26f6572fc88` |
| `sample_submission.csv` | 59 902 | `c77c0a5ed1f3d2728d6add6079266a543d12755aebaf00f230ff1966b364ebed` |

Манифест скачивания: `data/raw/provenance.json`. CSV, модели, credentials и журналы не входят в Git.

## Сравнение моделей

Общие настройки: случайный holdout 20%, seed 42, `maxBins=64`, одинаковые признаки каюты, расходов и номера участника группы. Медианы и StringIndexer обучаются отдельно внутри каждого pipeline; validation не используется при их `fit`.

| Вариант | Алгоритм | Деревья / итерации | Глубина | Шаг GBT | Holdout accuracy |
| --- | --- | --- | --- | --- | --- |
| `rf_v1` | RandomForest | 200 | 8 | — | 0.81623 |
| `rf_v2` | RandomForest | 300 | 10 | — | 0.81862 |
| **`gbt_v1`** | **GBT** | **100** | **5** | **0.05** | **0.82279** |

Лучший вариант — **`gbt_v1`**, улучшение относительно baseline: **0.00656**, или 0.656 процентного пункта accuracy. Он выбран по локальному holdout до отправки на Kaggle. По умолчанию CLI сохраняет прежний RF baseline; для лучшего варианта передайте параметры GBT явно.

Настройки выбираются по этому holdout, поэтому лучшая метрика после подбора не является независимой оценкой качества. Участники одной группы могут оказаться в обеих частях случайного split. Train/test не объединяются для подготовки признаков; групповые labels и внешние ответы не используются.

## Воспроизведение

Окружение: Scala 2.12.18, Spark 3.5.9, sbt 1.11.3, Eclipse Temurin JDK 11.0.32.1. Приложение: heap 2 GiB, `local[4]`, 8 shuffle partitions. Команды выполняются из корня; относительные пути внутри приложения отсчитываются от `competitions/spaceship-titanic/`.

```bash
export JAVA_TOOL_OPTIONS='-XX:ActiveProcessorCount=4 -Dspark.master=local[4] -Dspark.sql.shuffle.partitions=8'
export SPARK_LOCAL_IP=127.0.0.1

./bin/sbt-local "spaceshipTitanic/run train --classifier rf --num-trees 200 --max-depth 8 --seed 42 --model models/rf_v1"
./bin/sbt-local "spaceshipTitanic/run train --classifier rf --num-trees 300 --max-depth 10 --seed 42 --model models/rf_v2"
./bin/sbt-local "spaceshipTitanic/run train --classifier gbt --max-depth 5 --max-iter 100 --step-size 0.05 --seed 42 --model models/gbt_v1"
```

Для повторения существующего запуска используйте новые пути модели/submission либо явный `--overwrite`. На этой машине JDK установлен в `/home/hardbox/.local/share/jdk-11-temurin`, launcher sbt — `/tmp/spark-ml-tools/bin/sbt`; перед запуском использованы `JAVA_HOME=/home/hardbox/.local/share/jdk-11-temurin` и `PATH=/tmp/spark-ml-tools/bin:$JAVA_HOME/bin:$PATH`. Launcher использует официальный `sbt-launch.jar` 1.11.3 и настройки кешей из `bin/sbt-local`.

Отдельный инференс лучшей модели:

```bash
./bin/sbt-local "spaceshipTitanic/run predict --train data/raw/not-needed.csv --model models/gbt_v1 --output output/submission_gbt_v1.csv"
```

`data/raw/not-needed.csv` не существует: успешный запуск подтверждает, что `predict` читает только test.csv и сохранённую модель, не обучает преобразования и не обращается к training CSV.

Время `train` (оценка на holdout, обучение на всех данных и сохранение): RF v1 — 104 с, RF v2 — 130 с, GBT v1 — 139 с. Настройки сохранённых классификаторов проверены непосредственно в metadata; сводка — `output/experiments.json`.

Журналы обучения: `output/run_rf_v1.log`, `output/run_rf_v2.log`, `output/run_gbt_v1.log`. Журнал отдельного инференса: `output/predict_gbt_v1.log`.

## Проверка submission

`output/submission_gbt_v1.csv` создан отдельным `predict` из сохранённого `models/gbt_v1/`. Инференс занял 27 с.

- Заголовок строго `PassengerId,Transported`, 4 277 строк без заголовка.
- Все тестовые PassengerId встречаются ровно один раз; ведущие нули сохранены, порядок полностью совпадает с test.csv и sample_submission.csv.
- Предсказания: `True` — 2 249, `False` — 2 028; других значений нет.
- SHA-256: `ea4edfb632417005f2054b5b73f9fcd860a957802fcfc3e01f84e5aa91b019ca`.
- Сводка проверок: `output/verification.json`.

## Отправка в Kaggle

2026-10-26 отправлен именно `output/submission_gbt_v1.csv` с SHA-256, указанным выше. Kaggle подтвердил **submission ID 57670441**, статус **`COMPLETE`**, **public accuracy 0.79588**. В API указано время `2026-10-26T15:09:36.790000` UTC (18:09:36 Europe/Moscow). Описание: `Scala Spark GBT v1; 100 trees depth5 step0.05; seed42; holdout accuracy0.82279`.

Публичная оценка ниже локального holdout на 0.02691. Случайное разбиение и подбор трёх конфигураций по одному holdout не гарантируют совпадения с leaderboard; оценка Kaggle не подменяется локальной метрикой. Отправлен один вариант, выбранный по holdout до получения публичной оценки.

```bash
kaggle competitions submit spaceship-titanic \
  -f competitions/spaceship-titanic/output/submission_gbt_v1.csv \
  -m "Scala Spark GBT v1; 100 trees depth5 step0.05; seed42; holdout accuracy0.82279" \
  --wait 45 --poll-interval 5
kaggle competitions submissions spaceship-titanic --format json
```

Использован Kaggle CLI 2.2.5, в текущем окружении доступный как `PYTHONPATH=/tmp/spark-ml-tools/python python3 -m kaggle`. Журнал отправки: `output/kaggle_submit.log`; подтверждение статуса и оценки API: `output/kaggle_score.json`. Правила приняты пользователем до официального скачивания и отправки; credentials остаются вне репозитория.

## Проверки кода

```bash
./bin/sbt-local compile "spaceshipTitanic/test" "core/test"
```

Общая компиляция репозитория прошла. Успешны **6 тестов**: Spaceship Titanic — 4, core — 2.

Проверки Spaceship Titanic охватывают обучение/сохранение RF, загрузку и предсказания без train.csv, сохранение настроек и препроцессинга GBT, пропуски, неизвестные категории, формат идентификаторов, ошибочные данные и параметры. Тесты core проверяют общий CLI и защиту CSV от перезаписи. Финальный журнал: `output/final_check.log`.

Перед изменениями прочитаны `docs/README.md`, корневой README и руководство модуля. Индекс `.gitnexus` отсутствует; CLI недоступен (`npx: command not found`), зависимости проверены непосредственно по Scala-коду и `build.sbt`.
