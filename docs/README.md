# Документация Spark ML Kaggle Lab

## Назначение и границы проекта

Репозиторий содержит локальные Spark ML эксперименты и пакетные пайплайны для пяти соревнований Kaggle. Он предназначен для разработки признаков, сравнения моделей и воспроизводимого получения submission CSV. Предсказания выполняются CLI-командами; HTTP-сервиса и автоматической отправки результатов нет.

Начните с [главной страницы проекта](../README.md). Схема данных и команды конкретного соревнования находятся в его `README.md`, измеренные результаты — в `RESULTS.md`.

## Архитектура

| sbt-проект | Каталог | Ответственность |
| --- | --- | --- |
| `core` | `modules/core` | CLI новых модулей, holdout, обучение, загрузка `PipelineModel`, запись CSV |
| `titanic` | `competitions/titanic` | Эксперименты классификации, CV, ансамбли и постобработка |
| `housePrices` | `competitions/house-prices` | Регрессия `log1p(SalePrice)`, holdout и ансамбли |
| `spaceshipTitanic` | `competitions/spaceship-titanic` | Типизированный CSV, признаки каюты/расходов, RandomForest/GBT |
| `digitRecognizer` | `competitions/digit-recognizer` | Порядок строк MNIST, нормализация пикселей, multinomial LogisticRegression |
| `disasterTweets` | `competitions/disaster-tweets` | Multiline CSV твитов, TF-IDF слов/биграмм, LogisticRegression, бинарный F1 |

Новые модули зависят от `core` и реализуют `ClassificationCompetition`: чтение/проверку данных, построение пайплайна и формирование submission. `BatchClassification` управляет жизненным циклом. Titanic и House Prices сохраняют существующие интерфейсы; XGBoost подключён только к этим двум модулям.

## Окружение

Используйте JDK 11 и sbt 1.11.3. Версии Scala/Spark закреплены в `build.sbt`. Запускайте команды из корня через `./bin/sbt-local`; обёртка требует доступный `sbt` и создаёт локальные `.sbt-home`, `.sbt-cache`, `.ivy2`, `.coursier`.

Новые приложения и тесты запускаются в отдельной JVM с лимитом heap 2 GiB. Spark работает локально с `local[*]`. Для больших экспериментов изменяйте настройки ресурсов осознанно; multinomial LogisticRegression по 784 пикселям существенно тяжелее табличной задачи Spaceship Titanic.

## CLI новых модулей

```bash
./bin/sbt-local "spaceshipTitanic/run train"
./bin/sbt-local "spaceshipTitanic/run predict"
./bin/sbt-local "spaceshipTitanic/run run"
./bin/sbt-local "digitRecognizer/run --help"
./bin/sbt-local "disasterTweets/run train"
./bin/sbt-local "disasterTweets/run predict"
```

Вызов без режима эквивалентен `run`.

| Параметр | По умолчанию | Использование |
| --- | --- | --- |
| `--train` | `data/raw/train.csv` | Размеченные данные для `train`/`run` |
| `--input` | `data/raw/test.csv` | Данные без целевой колонки для `predict`/`run` |
| `--model` | См. руководство модуля | Полный сохранённый Spark-пайплайн |
| `--output` | `output/submission.csv` | Один CSV-файл предсказаний |
| `--seed` | `42` | Seed случайного holdout и поддерживающей его модели |
| `--validation-fraction` | `0.2` | Доля holdout, строго между 0 и 1 |
| `--overwrite` | Выключен | Разрешить замену существующих артефактов |

Дополнительные параметры: Spaceship Titanic — `--classifier rf|gbt`, `--num-trees` (RF), `--max-depth`, `--max-iter` и `--step-size` (GBT); Digit Recognizer — `--max-iter`, `--reg-param`; Disaster Tweets — `--num-features`, `--max-iter`, `--reg-param`, `--threshold`.

Все относительные пути при `sbt run` отсчитываются от каталога соревнования. Например, `--model models/rf_v1` для `spaceshipTitanic` указывает на `competitions/spaceship-titanic/models/rf_v1`.

## Контракт обучения и инференса

`train` проверяет данные и делит их случайно с заданным seed. Пайплайн обучается на training split, затем оценивается accuracy на holdout. После проверки новый экземпляр модели обучается на всех размеченных строках и сохраняется вместе с преобразованиями признаков. Пустые split и training split с одним классом отклоняются.

`predict` читает только входной CSV и сохранённую модель. В нём отсутствуют `fit`, доступ к `train.csv` и пересчёт параметров импутации/категоризации. Используйте артефакт своего модуля, совместимые версии Spark и тот же входной формат. Модель можно переносить целиком, включая все подкаталоги `metadata` и `stages`.

Соревнование может добавить метрики через `ClassificationCompetition.validationMetrics`; общий lifecycle выводит их вместе с accuracy на том же holdout. Disaster Tweets считает F1 положительного класса `target=1`, сохраняет IDF обучающей части и порог LogisticRegression. Submission имеет заголовок `id,target` и сортируется по числовому идентификатору.

Spaceship Titanic сохраняет строковые `PassengerId` с ведущими нулями и выводит `Transported` как `True`/`False`. Digit Recognizer назначает `ImageId` по физическому порядку строк исходного единственного CSV, начиная с 1, до любых Spark-преобразований. Это предотвращает сопоставление предсказаний с неверными изображениями.

Submission записывается как один файл с заголовком. Нельзя заменить каталог путём `--output`. При разрешённой перезаписи CSV временно сохраняется резервная копия; при ошибке перемещения восстанавливается прежний файл. Экспорт через одну партицию рассчитан на размеры этих соревнований.

## Проверка

```bash
# Собрать весь репозиторий.
./bin/sbt-local compile

# Проверить новые модули на синтетических данных, без Kaggle credentials.
./bin/sbt-local "core/test" "spaceshipTitanic/test" "digitRecognizer/test" "disasterTweets/test"

# Проверить конкретный модуль.
./bin/sbt-local "spaceshipTitanic/testOnly *SpaceshipTitanicPipelineSpec"
```

ScalaTest 3.2.19 используется в `core` и трёх новых модулях. Тесты создают временные CSV и модели; проверяют полный цикл обучения/сохранения/инференса, порядок и типы идентификаторов, неизвестные категории, пропуски, некорректные labels/pixels и защиту результатов от перезаписи. У старых модулей автоматические тесты пока отсутствуют. Порог покрытия не задан.

При изменениях модели дополнительно запускайте проверку на реальных данных. Сравнивайте одинаковые split/seed, проверяйте заголовок и число строк submission, диапазон предсказаний. Синтетические тесты подтверждают работоспособность кода, но не качество модели на leaderboard.

## Данные и результаты

Скачивайте данные через свою учётную запись Kaggle после проверки условий соревнования. Храните credentials вне репозитория. `data/`, `models/`, `output/`, архивы и CSV исключены из Git; не добавляйте их принудительно ради отчёта.

В `RESULTS.md` записывайте дату, команду, seed, параметры, локальную метрику, путь модели и submission. Публичную оценку указывайте только после действительной отправки. До реального запуска метрики помечаются «не измерено».

Измеренный запуск Spaceship Titanic от 2026-10-26: GBT с 100 итерациями, глубиной 5 и шагом 0.05 — локальная accuracy 0.82279; подтверждённая Kaggle public accuracy 0.79588 (submission 57670441, `COMPLETE`). Сравнение с двумя RandomForest, происхождение данных и команды — в [RESULTS.md модуля](../competitions/spaceship-titanic/RESULTS.md).

## Разработка и работа агента

Перед правками прочитайте этот документ, руководство затрагиваемого модуля и проверьте `.gitnexus` (при наличии CLI — `npx gitnexus status`). Если индекс отсутствует или устарел, явно зафиксируйте ограничение и проверяйте зависимости непосредственно в коде. Не запускайте генерацию индекса без учёта того, что она может обновить агентские файлы.

Следуйте Scala-стилю существующего кода: два пробела, `UpperCamelCase` для типов, `lowerCamelCase` для методов. Новый модуль добавляйте в `build.sbt` и главную таблицу соревнований, вместе с README, журналом результатов и проверками модели. Подробнее — [AGENTS.md](../AGENTS.md).
