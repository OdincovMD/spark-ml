# Spark ML Kaggle Lab

Scala/Spark workspace for Kaggle competitions. Each competition lives in its own folder under `competitions/` and can be wired as a separate sbt subproject.

## Competitions

Started:

- [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic): binary classification, feature engineering, model comparison, and post-processing.
- [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques): tabular regression with `SalePrice` target and `Id,SalePrice` submission format.

Both are Kaggle Getting Started competitions with small enough data to iterate quickly while still exercising Spark CSV IO, feature engineering, ML pipelines, evaluation, and Kaggle submission formatting.


## Layout

```text
.
├── build.sbt
├── bin/sbt-local
├── competitions/
│   ├── titanic/
│   │   ├── README.md
│   │   ├── data/
│   │   │   ├── raw/
│   │   │   ├── interim/
│   │   │   └── processed/
│   │   ├── models/
│   │   ├── output/
│   │   └── src/main/
│   │       ├── resources/log4j2.properties
│   │       └── scala/kaggle/sparkml/titanic/TitanicPipeline.scala
│   └── house-prices/
│       ├── README.md
│       ├── RESULTS.md
│       ├── data/
│       ├── models/
│       ├── output/
│       └── src/main/scala/kaggle/sparkml/houseprices/HousePricesPipeline.scala
└── project/build.properties
```

## Prerequisites

- Java 8+ on `PATH`.
- sbt and Metals.
- Kaggle CLI, if you want to download data and submit from the terminal.

The local sbt wrapper keeps launcher and dependency caches inside this repository, which is useful in restricted environments:

```bash
./bin/sbt-local "titanic/compile"
./bin/sbt-local "housePrices/compile"
```

## Kaggle Setup

Install and configure the Kaggle CLI outside this repo:

```bash
pip install kaggle
mkdir -p ~/.kaggle
# Put kaggle.json from your Kaggle account settings into ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Accept the competition rules on Kaggle, then download the data.

Titanic:

```bash
kaggle competitions download -c titanic -p competitions/titanic/data/raw
unzip competitions/titanic/data/raw/titanic.zip -d competitions/titanic/data/raw
```

House Prices:

```bash
kaggle competitions download -c house-prices-advanced-regression-techniques -p competitions/house-prices/data/raw
unzip competitions/house-prices/data/raw/house-prices-advanced-regression-techniques.zip -d competitions/house-prices/data/raw
```

## Train And Create Submission

Titanic:

```bash
./bin/sbt-local "titanic/run"
```

This reads:

- `competitions/titanic/data/raw/train.csv`
- `competitions/titanic/data/raw/test.csv`

And writes:

- `competitions/titanic/output/submission.csv`
- `competitions/titanic/models/random_forest_pipeline`

Submit:

```bash
kaggle competitions submit -c titanic \
  -f competitions/titanic/output/submission.csv \
  -m "Scala Spark RandomForest baseline"
```

House Prices:

```bash
./bin/sbt-local "housePrices/run"
```

This reads:

- `competitions/house-prices/data/raw/train.csv`
- `competitions/house-prices/data/raw/test.csv`

And writes:

- `competitions/house-prices/output/submission.csv`
- `competitions/house-prices/models/rf_log_price_baseline`

Submit:

```bash
kaggle competitions submit -c house-prices-advanced-regression-techniques \
  -f competitions/house-prices/output/submission.csv \
  -m "Scala Spark RF log-price baseline"
```
