# Digit Recognizer

**Распознавание рукописных цифр MNIST средствами Spark ML.**

[Соревнование Kaggle](https://www.kaggle.com/competitions/digit-recognizer) · [Описание данных](https://www.kaggle.com/competitions/digit-recognizer/data) · [Результаты](RESULTS.md) · [Документация проекта](../../docs/README.md)

Задача — классифицировать изображение размером 28 × 28 в один из десяти классов `0…9`. Метрика — accuracy. Здесь используется линейный baseline по пикселям; это отправная точка для экспериментов, без заявленного качества нейросетей.

## Данные

```bash
kaggle competitions download -c digit-recognizer -p competitions/digit-recognizer/data/raw
unzip competitions/digit-recognizer/data/raw/digit-recognizer.zip -d competitions/digit-recognizer/data/raw
```

Формат исходных файлов Kaggle:

| Файл | Заголовок | Значения |
| --- | --- | --- |
| `train.csv` | `label,pixel0,…,pixel783` | Цифра `0…9` и 784 целочисленных пикселя |
| `test.csv` | `pixel0,…,pixel783` | 784 целочисленных пикселя без label |

Диапазон пикселей: `0…255`. Пропущенные значения, пустые строки, неверный порядок заголовка и неправильное число полей отклоняются. Поддерживается исходный числовой CSV Kaggle без кавычек и многострочных значений. Вход должен быть **одним CSV-файлом**, а не каталогом или glob. `ImageId` присваивается по исходному порядку строк, начиная с 1; одинаковые изображения не удаляются.

## Пайплайн

`VectorAssembler` собирает пиксели в фиксированном порядке. `ElementwiseProduct` делит значения на 255. `LogisticRegression` с `family=multinomial` предсказывает один из десяти классов. Нормализация и классификатор сохраняются в полном `PipelineModel`.

По умолчанию: `maxIter=100`, `regParam=0.01`, holdout 20%, seed `42`. Модель сначала оценивается на holdout, затем обучается заново на всех размеченных строках. Seed задаёт split; классификатор обучается детерминированным оптимизатором. Пиксельная модель тяжелее табличных baseline; время обучения зависит от числа строк и итераций.

## Обучение и инференс

Из корня репозитория:

```bash
# Сохранить модель после локальной оценки.
./bin/sbt-local "digitRecognizer/run train"

# Применить существующую модель, не читая train.csv.
./bin/sbt-local "digitRecognizer/run predict"

# Обучить и создать submission за один запуск.
./bin/sbt-local "digitRecognizer/run run --model models/lr_v2 --output output/submission_v2.csv"

# Изменить параметры оптимизации.
./bin/sbt-local "digitRecognizer/run train --max-iter 150 --reg-param 0.005 --model models/lr_150"

# Предсказать новую партию изображений того же CSV-формата.
./bin/sbt-local "digitRecognizer/run predict --input data/raw/new_digits.csv --output output/new_predictions.csv"
```

Пути отсчитываются от `competitions/digit-recognizer`. Модель по умолчанию: `models/multinomial_lr_v1`. Результат: один файл `output/submission.csv` с заголовком `ImageId,Label`, отсортированный по `ImageId`. Перезапись включается явно через `--overwrite`.

Для самостоятельной партии `ImageId` снова начинается с 1. Для submission Kaggle всегда используйте полный оригинальный `test.csv`, сохраняя его порядок.

## Проверка и отправка

```bash
./bin/sbt-local "digitRecognizer/test"
kaggle competitions submit -c digit-recognizer -f competitions/digit-recognizer/output/submission.csv -m "Scala Spark multinomial LR v1"
```

Проверьте число строк, непрерывные ID `1…N` и labels в диапазоне `0…9`. Реальные метрики и параметры фиксируйте в [RESULTS.md](RESULTS.md). Синтетические тесты проверяют сохранение модели и соответствие предсказания исходной строке, но не заменяют оценку на данных Kaggle.
