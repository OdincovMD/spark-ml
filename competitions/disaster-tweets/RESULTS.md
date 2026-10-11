# Результаты NLP with Disaster Tweets

## Запуск 2026-10-11, Europe/Moscow

Обучение, сохранение полного `PipelineModel` и отдельный инференс выполнены на реальных данных соревнования. Лучший локальный вариант — **`tfidf_lr_v3`: F1 0.77005, accuracy 0.79063**. Сабмит **57084278** успешно обработан Kaggle: **публичный F1 0.77750**, статус `COMPLETE`.

## Данные и происхождение

Kaggle CLI успешно авторизовался с локальными credentials вне репозитория. До принятия правил прямое скачивание отклонялось, поэтому для обучения использована [публичная копия исходных CSV](https://github.com/Krishnarohith10/nlp-getting-started). После принятия правил официальные CSV скачаны через Kaggle API: SHA-256 всех трёх файлов полностью совпадает с использованной копией. Переобучение для смены источника не требуется.

Данные содержат 7 613 обучающих строк (3 271 положительная, 4 342 отрицательных) и 3 263 тестовых строки. Размеры: train 987 712 байт, test 420 783 байта, sample submission 22 746 байт. Официальный архив и результаты сравнения находятся в локальном `data/raw/kaggle_verified/`.

Проверены уникальность идентификаторов, отсутствие пересечения train/test ID и совпадение тестовых ID с шаблоном submission. Тестовые labels не использовались. Заголовки — исходные Kaggle, включая `keyword` и `location`; train содержит 435 текстов с переводами строк.

SHA-256 скачанных файлов:

| Файл | SHA-256 |
| --- | --- |
| `train.csv` | `61111c6dc31eaffa34d1e1fa62e2395325c9bc3b38bba1941a5f1ed9b3fa60df` |
| `test.csv` | `33375dc852696331c5ad30d078ea4d20e357b74052be6225fecc2a2b0f468bd2` |
| `sample_submission.csv` | `ec431de21357caad931108491d696954cd77c53149b7ce13380ca4d00d4da2db` |

Локальный `data/raw/provenance.json` содержит URL каждого файла, checksum и число строк. CSV, модели и журналы запуска исключены из Git.

## Сравнение моделей

Все варианты: seed 42, случайный holdout 20%, HashingTF 32 768 признаков, слова и биграммы, IDF с `minDocFreq=2`, L2-нормализация строк, LogisticRegression до 80 итераций. IDF и классификатор обучались только на training split для оценки. Затем каждый пайплайн обучен заново на всех 7 613 размеченных строках.

| Вариант | Стандартизация колонок LR | regParam | Порог | Accuracy | F1 (`target=1`) |
| --- | --- | --- | --- | --- | --- |
| `tfidf_lr_v1` | Включена | 0.03 | 0.5 | 0.73140 | 0.67172 |
| `tfidf_lr_v2` | Выключена | 0.001 | 0.5 | 0.74518 | 0.59519 |
| **`tfidf_lr_v3`** | **Выключена** | **0.0001** | **0.4** | **0.79063** | **0.77005** |

Выбор сделан по бинарному F1, основной метрике соревнования. По сравнению с первым вариантом F1 улучшился на 0.09833. Финальный код отключает поколоночную стандартизацию и использует параметры v3 по умолчанию. v1 измерен до этого изменения; его историческая модель сохранена, но нынешняя команда обучения не воспроизводит включённую стандартизацию v1.

Метрики относятся к локальному случайному holdout. Параметры выбирались по этому holdout; похожие и повторяющиеся тексты могут попасть в обе части. Оценка не является независимой оценкой после подбора и не заменяет leaderboard.

## Воспроизведение лучшего запуска

Окружение: Scala 2.12.18, Spark 3.5.9, sbt launcher 1.11.3, Eclipse Temurin JDK 11.0.32.1. Heap приложения 2 GiB, Spark `local[4]`, 8 shuffle partitions. На этой машине JDK 11 установлен в `~/.local/share/jdk-11-temurin`; временный launcher sbt — `/tmp/spark-ml-tools/bin/sbt` (использует официальный `sbt-launch` 1.11.3 и параметры `bin/sbt-local`).

```bash
# Настройки ресурсов для воспроизведения измеренного запуска.
export JAVA_TOOL_OPTIONS='-XX:ActiveProcessorCount=4 -Dspark.master=local[4] -Dspark.sql.shuffle.partitions=8'
export SPARK_LOCAL_IP=127.0.0.1

./bin/sbt-local "disasterTweets/run train --seed 42 --validation-fraction 0.2 --num-features 32768 --max-iter 80 --reg-param 0.0001 --threshold 0.4 --model models/tfidf_lr_v3"
./bin/sbt-local "disasterTweets/run predict --train data/raw/not-needed.csv --model models/tfidf_lr_v3 --output output/submission_tfidf_lr_v3.csv"
```

Команды выполняются из корня с доступными JDK 11 и sbt в PATH. В текущем окружении перед ними использовались `JAVA_HOME=/home/hardbox/.local/share/jdk-11-temurin` и `PATH=/tmp/spark-ml-tools/bin:$JAVA_HOME/bin:$PATH`. Артефакты уже существуют: для повторного обучения используйте новые пути или явный `--overwrite`.

Обучение v3 заняло 44 секунды, отдельный инференс — 22 секунды. В `predict` передан заведомо отсутствующий `--train`, что подтверждает отсутствие чтения обучающего CSV. Полный пайплайн: `models/tfidf_lr_v3/`. Submission: `output/submission_tfidf_lr_v3.csv`.

Проверка готового CSV:

- Заголовок строго `id,target`; 3 263 строки без учёта заголовка.
- Каждый test ID присутствует ровно один раз, порядок — по числовому ID.
- Предсказания: 1 450 значений `1`, 1 813 значений `0`.
- SHA-256: `d76bb7c4c05d82740061a9bf3d378e00ef4e2604f3ebf21e512dcceb2595e18b`.
- В metadata сохранённого классификатора подтверждены `regParam=0.0001`, `threshold=0.4`, `standardization=false`, `maxIter=80`.

Предыдущие модели и CSV сохранены с суффиксами v1/v2. Журналы: `output/run_v1.log`, `output/run_v2.log`, `output/run_v3.log`; сводка проверок — `output/verification.json`.

## Отправка в Kaggle

Выполнена реальная попытка отправки `output/submission_tfidf_lr_v3.csv` через Kaggle API `competition_submit` в `nlp-getting-started`, описание: `Scala Spark TF-IDF words/bigrams LR v3; seed 42; holdout F1 0.77005`.

Первая попытка до принятия правил завершилась HTTP 403 `RulesAcceptanceRequired`; её ответ сохранён в `output/kaggle_submission_status_initial.json`.

После принятия правил отправка повторена 2026-10-11. Kaggle выдал **submission ID 57084278**, затем подтвердил статус **`COMPLETE`**, отсутствие ошибки обработки и **публичный F1 0.77750**. Отправлен тот же CSV с SHA-256, указанным выше. Ответы API сохранены в `output/kaggle_submission_status.json` и `output/kaggle_score.json`.

Команды для отправки нового результата и проверки статуса:

```bash
kaggle competitions submit -c nlp-getting-started \
  -f competitions/disaster-tweets/output/submission_tfidf_lr_v3.csv \
  -m "Scala Spark TF-IDF words/bigrams LR v3; seed 42; holdout F1 0.77005"
kaggle competitions submissions -c nlp-getting-started
```

В текущем окружении CLI доступен как `PYTHONPATH=/tmp/spark-ml-tools/python python3 -m kaggle`.

## Проверки кода

Прошли 10 тестов: `core/test` (2), `spaceshipTitanic/test` (2), `digitRecognizer/test` (2), `disasterTweets/test` (4). После изменения нормализации и после выбора финальных настроек по умолчанию повторно прошли 4 теста Disaster Tweets, включая удаление URL/mentions после загрузки модели. Выполнена общая компиляция репозитория; финальный журнал — `output/final_check.log`.

Индекс `.gitnexus` и его CLI отсутствовали; зависимости проверены непосредственно по Scala-коду и `build.sbt` после чтения `docs/README.md`.
