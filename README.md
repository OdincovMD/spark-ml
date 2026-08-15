# Spark ML Kaggle Lab

Scala/Spark workspace for Kaggle competitions. Each competition lives in its own folder under `competitions/` and can be wired as a separate sbt subproject.

## First Competition

Start with [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic).

Why this one:

- It is a Kaggle Getting Started competition with a rolling leaderboard.
- The data is small enough to iterate quickly while still exercising Spark CSV IO, feature engineering, ML pipelines, evaluation, and Kaggle submission formatting.
- The target is binary classification, so the first baseline can stay focused and understandable.

## Layout

```text
.
├── build.sbt
├── bin/sbt-local
├── competitions/
│   └── titanic/
│       ├── README.md
│       ├── data/
│       │   ├── raw/
│       │   ├── interim/
│       │   └── processed/
│       ├── models/
│       ├── output/
│       └── src/main/
│           ├── resources/log4j2.properties
│           └── scala/kaggle/sparkml/titanic/TitanicPipeline.scala
└── project/build.properties
```

## Prerequisites

- Java 8+ on `PATH`.
- sbt and Metals.
- Kaggle CLI, if you want to download data and submit from the terminal.

The local sbt wrapper keeps launcher and dependency caches inside this repository, which is useful in restricted environments:

```bash
./bin/sbt-local "titanic/compile"
```

## Kaggle Setup

Install and configure the Kaggle CLI outside this repo:

```bash
pip install kaggle
mkdir -p ~/.kaggle
# Put kaggle.json from your Kaggle account settings into ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Accept the Titanic competition rules on Kaggle, then download the data:

```bash
kaggle competitions download -c titanic -p competitions/titanic/data/raw
unzip competitions/titanic/data/raw/titanic.zip -d competitions/titanic/data/raw
```

## Train And Create Submission

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
