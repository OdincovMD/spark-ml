# NLP with Disaster Tweets

**Классификация текстов на Scala и Spark ML:** предсказать, описывает ли твит настоящую катастрофу (`target=1`).

[Соревнование Kaggle](https://www.kaggle.com/competitions/nlp-getting-started) · [Данные](https://www.kaggle.com/competitions/nlp-getting-started/data) · [Результаты](RESULTS.md) · [Документация проекта](../../docs/README.md)

Метрика соревнования — бинарный F1 положительного класса: `2*TP/(2*TP+FP+FN)`. Локальный запуск выводит этот F1 и accuracy. Это не weighted F1 Spark и не оценка leaderboard.

## Данные

Присоединитесь к соревнованию и примите его правила в своём аккаунте Kaggle, затем скачайте данные:

```bash
kaggle competitions download -c nlp-getting-started -p competitions/disaster-tweets/data/raw
unzip competitions/disaster-tweets/data/raw/nlp-getting-started.zip -d competitions/disaster-tweets/data/raw
```

Ожидаются исходные CSV с заголовком `id,keyword,location,text,target` для обучения и `id,keyword,location,text` для инференса. `id` — уникальное неотрицательное целое в диапазоне Long, `target` — строго `0` или `1`. Пустые тексты и metadata допустимы. Парсер поддерживает запятые, переводы строк и удвоенные кавычки внутри quoted-полей. Отсутствующие колонки, дубликаты идентификаторов и некорректные labels отклоняются.

## Пайплайн

1. Сохранённый `SQLTransformer` удаляет URL и mentions, добавляет `keyword`, заменяя `%20` пробелом. `location` и `id` не используются как признаки.
2. `RegexTokenizer` приводит текст к нижнему регистру и выделяет слова длиной от двух символов. `NGram` добавляет биграммы.
3. Векторизатор кодирует слова и биграммы в разреженный вектор: исходный `HashingTF` либо обучаемый `CountVectorizer` с `--vectorizer count`. `IDF` обучается только на training split при валидации. `Normalizer` выполняет L2-нормализацию.
4. `LogisticRegression` выполняет бинарную классификацию без поколоночной стандартизации: TF-IDF уже нормализован по строкам. Порог решения сохраняется вместе с моделью.

Holdout случайный, по умолчанию 20% с seed 42. Одинаковые или похожие твиты могут попасть в обе части; эта оценка не заменяет leaderboard. После проверки пайплайн заново обучается на всех размеченных строках и сохраняется целиком.

`CountVectorizer` строит словарь слов и биграмм, встречающихся как минимум в двух обучающих твитах, и сохраняет его в `PipelineModel`. Каждый термин имеет собственную координату без хеш-коллизий; `--num-features` ограничивает размер словаря сверху. При проверке на holdout словарь строится только по training split. Неизвестные слова при инференсе игнорируются; если известных терминов нет, классификатор получает нулевой вектор. Словарь и IDF при `predict` не переобучаются. Прежний `HashingTF` остаётся вариантом по умолчанию; существующие модели загружаются как прежде.

Сравнение на реальных данных от 2026-11-01: CountVectorizer улучшил F1 на всех трёх seed (42, 1337, 2026), средний F1 вырос с **0.75460 до 0.75897** при одинаковых параметрах классификатора. Публичная оценка нового варианта не измерена. Метрики по seed и команды — в [RESULTS.md](RESULTS.md).

## Обучение и инференс

Из корня репозитория, с JDK 11 и sbt 1.11.3:

```bash
./bin/sbt-local "disasterTweets/run --help"
./bin/sbt-local "disasterTweets/run train --seed 42 --model models/tfidf_lr_v3"
./bin/sbt-local "disasterTweets/run predict --model models/tfidf_lr_v3 --output output/submission_tfidf_lr_v3.csv"

# Объединённый запуск с новыми путями артефактов.
./bin/sbt-local "disasterTweets/run run --model models/tfidf_lr_v4 --output output/submission_tfidf_lr_v4.csv"

# Обучаемый словарь вместо хеширования; отдельные артефакты.
./bin/sbt-local "disasterTweets/run train --vectorizer count --model models/count_tfidf_lr_v1"
./bin/sbt-local "disasterTweets/run predict --model models/count_tfidf_lr_v1 --output output/submission_count_tfidf_lr_v1.csv"
```

`predict` загружает только `--model` и `--input`; `train.csv` ему не нужен. Он не переобучает IDF или классификатор. Относительные пути при `sbt run` отсчитываются от `competitions/disaster-tweets/`. По умолчанию используется `models/tfidf_lr_v3` и `output/submission.csv`. Задавайте новые пути или явно добавляйте `--overwrite` для замены существующих артефактов.

| Параметр | По умолчанию | Назначение |
| --- | --- | --- |
| `--vectorizer` | `hashing` | `hashing` — HashingTF; `count` — обучаемый словарь CountVectorizer |
| `--num-features` | `32768` | Размер HashingTF или максимальный размер словаря CountVectorizer, положительное целое |
| `--max-iter` | `80` | Максимум итераций LogisticRegression |
| `--reg-param` | `0.0001` | Неотрицательная конечная L2-регуляризация |
| `--threshold` | `0.4` | Порог положительного класса, строго между 0 и 1 |
| `--seed` | `42` | Seed holdout |
| `--validation-fraction` | `0.2` | Доля holdout |

Submission — один CSV `id,target` с целочисленными предсказаниями `0`/`1`, отсортированный по числовому `id`.

Векторизатор сохраняется вместе с моделью. При `predict` повторять `--vectorizer` и `--num-features` не нужно: используются сохранённые стадии и словарь.

## Отправка в Kaggle

```bash
kaggle competitions submit -c nlp-getting-started \
  -f competitions/disaster-tweets/output/submission_tfidf_lr_v3.csv \
  -m "Scala Spark TF-IDF word/bigram LogisticRegression v3"
kaggle competitions submissions -c nlp-getting-started
```

Отправка требует рабочей авторизации и принятых правил соревнования. Обучение и `predict` только создают локальные артефакты; CLI Kaggle отправляет файл отдельно. Записывайте публичный F1 в `RESULTS.md` только после обработки сабмита Kaggle.

## Проверка

```bash
./bin/sbt-local "disasterTweets/test" "core/test"
```

Тесты проверяют обучение и сохранение обоих векторизаторов, загрузку и инференс после удаления обучающего CSV, исключение редких и неизвестных терминов из словаря, пустые тексты, multiline CSV с кавычками, соответствие идентификаторов, защиту от перезаписи и бинарный F1. Реальный запуск и происхождение данных фиксируются в `RESULTS.md`. Credentials, CSV и модели исключены из Git.
