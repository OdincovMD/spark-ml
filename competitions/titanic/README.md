# Titanic

Kaggle: <https://www.kaggle.com/competitions/titanic>

Goal: predict whether passengers survived the Titanic shipwreck and create a Kaggle-compatible CSV with `PassengerId` and `Survived`.

## Data

Place Kaggle files here:

```text
competitions/titanic/data/raw/train.csv
competitions/titanic/data/raw/test.csv
```

Download with Kaggle CLI after accepting the rules:

```bash
kaggle competitions download -c titanic -p competitions/titanic/data/raw
unzip competitions/titanic/data/raw/titanic.zip -d competitions/titanic/data/raw
```

## Run

```bash
./bin/sbt-local "titanic/run"
```

The sbt task runs from `competitions/titanic`, so custom paths are relative to that directory unless you pass absolute paths.

Custom paths:

```bash
./bin/sbt-local 'titanic/run --train data/raw/train.csv --test data/raw/test.csv --output output/submission.csv --model models/random_forest_pipeline'
```

## Baseline

The first baseline uses Spark ML:

- simple SQL feature engineering for title, family size, alone flag, and fare per person;
- imputation for `Age` and `Fare`;
- string indexing for categorical features;
- vector assembly;
- random forest classifier;
- holdout accuracy printed to stdout;
- single-file `submission.csv` for Kaggle.
