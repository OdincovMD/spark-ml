# House Prices

Kaggle: <https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques>

Goal: predict the final sale price for each home and create a Kaggle-compatible CSV with `Id` and `SalePrice`.

This competition complements Titanic with a tabular regression task. The current default trains an ElasticNet linear model on `log1p(SalePrice)` with one-hot categorical features and writes predictions back in dollars.

## Data

Place Kaggle files here:

```text
competitions/house-prices/data/raw/train.csv
competitions/house-prices/data/raw/test.csv
competitions/house-prices/data/raw/sample_submission.csv
competitions/house-prices/data/raw/data_description.txt
```

Download with Kaggle CLI after accepting the rules:

```bash
kaggle competitions download -c house-prices-advanced-regression-techniques -p competitions/house-prices/data/raw
unzip competitions/house-prices/data/raw/house-prices-advanced-regression-techniques.zip -d competitions/house-prices/data/raw
```

## Run

```bash
./bin/sbt-local "housePrices/run"
```

The sbt task runs from `competitions/house-prices`, so custom paths are relative to that directory unless you pass absolute paths.

Current ensemble candidate:

```bash
./bin/sbt-local "housePrices/run --algo ensemble --max-depth 3 --xgb-rounds 500 --xgb-eta 0.03 --ensemble-xgb-weight 0.4 --output output/submission_lr_xgb_v4.csv --model models/lr_xgb_ensemble_v4"
```

Feature-engineered ElasticNet candidate:

```bash
./bin/sbt-local "housePrices/run --algo lr --advanced-features --reg-param 0.01 --elastic-net 0.8 --output output/submission_lr_fe_v2.csv --model models/lr_elastic_net_fe_v2"
```

All training rows are retained by default. The experimental `--remove-outliers` option is preserved for reproducibility, but it worsened the public score.

Tune without writing a model/submission:

```bash
./bin/sbt-local "housePrices/run --algo lr --validation-only --reg-param 0.01 --elastic-net 0.8"
```

Evaluate several deterministic holdouts in one run:

```bash
./bin/sbt-local "housePrices/run --algo lr --validation-only --validation-seeds 42,1337,2026"
```

RandomForest baseline:

```bash
./bin/sbt-local "housePrices/run --algo rf --output output/submission_rf.csv --model models/rf_log_price_baseline"
```

Gradient-Boosted Trees baseline:

```bash
./bin/sbt-local "housePrices/run --algo gbt --output output/submission_gbt.csv --model models/gbt_log_price_baseline"
```

Dense-vector XGBoost:

```bash
./bin/sbt-local "housePrices/run --algo xgb --max-depth 3 --xgb-rounds 500 --xgb-eta 0.03 --output output/submission_xgb.csv --model models/xgb_log_price"
```

Submit:

```bash
kaggle competitions submit -c house-prices-advanced-regression-techniques -f competitions/house-prices/output/submission_lr_xgb_v4.csv -m "Scala Spark ElasticNet 60 XGBoost 40 log blend"
```

## Baseline

Current pipeline:

- infer numeric and categorical columns from `train.csv`;
- add house age, remodel age, total square footage, bathroom, porch, quality-area, and garage-age features;
- add selected `log1p` variants for skewed numeric features;
- optionally enable the broader, lower-scoring v2 feature set with `--advanced-features`;
- retain all training rows by default;
- read Kaggle `NA` markers as nulls so numeric columns with missing values stay numeric;
- median-impute numeric features;
- fill/index categorical features with `handleInvalid = keep`;
- use one-hot categorical features and scaled numeric features for the default `lr` model;
- support dense-vector XGBoost and a log-space ElasticNet/XGBoost ensemble;
- set tree `maxBins = 512` by default for House Prices categorical cardinalities;
- train on `log1p(SalePrice)`;
- evaluate validation log-RMSE;
- write `Id,SalePrice` submission.
