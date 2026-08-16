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

Default experiment: `rf_tuned_v2`.

The sbt task runs from `competitions/titanic`, so custom paths are relative to that directory unless you pass absolute paths.

Custom paths:

```bash
./bin/sbt-local 'titanic/run --experiment rf_tuned_v2 --train data/raw/train.csv --test data/raw/test.csv --output output/submission.csv --model models'
```

List experiments:

```bash
./bin/sbt-local "titanic/run --list-experiments"
```

Run one experiment:

```bash
./bin/sbt-local "titanic/run --experiment baseline_001 --output output/submission_baseline_check.csv"
./bin/sbt-local "titanic/run --experiment rf_tuned_v2"
./bin/sbt-local "titanic/run --experiment gbt_v1"
./bin/sbt-local "titanic/run --experiment lr_v1"
./bin/sbt-local "titanic/run --experiment xgb_v1"
./bin/sbt-local "titanic/run --experiment ensemble_v1"
```

Fast sanity check for a single candidate instead of the full grid:

```bash
./bin/sbt-local "titanic/run --experiment rf_tuned_v2 --fast --cv-folds 2 --output output/submission_rf_tuned_v2_check.csv"
```

Run the full comparison:

```bash
./bin/sbt-local "titanic/run --experiment all"
```

`all` writes:

- `output/experiments.csv`
- `output/submission_<experiment>.csv`
- `output/submission.csv` copied from the best local CV result

## Experiments

`baseline_001` keeps the first Spark ML RandomForest pipeline:

- simple SQL feature engineering for title, family size, alone flag, and fare per person;
- imputation for `Age` and `Fare`;
- string indexing for categorical features;
- vector assembly;
- random forest classifier;

New experiments use label-free train+test feature engineering:

- normalized titles, surname group size, ticket group size;
- deck, cabin presence, cabin count, ticket prefix;
- family size buckets, fare per person, log fare;
- grouped age/fare imputation;
- 5-fold CV by default with `--cv-folds`.

Available model families:

- `rf_tuned_v2`: Spark ML RandomForest grid.
- `gbt_v1`: Spark ML Gradient-Boosted Trees grid.
- `lr_v1`: one-hot categorical features, scaled numeric features, LogisticRegression grid.
- `xgb_v1`: XGBoost4J-Spark grid.
- `ensemble_v1`: soft-vote average of RF, GBT, and XGBoost probabilities.
