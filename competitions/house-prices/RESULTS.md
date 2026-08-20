# House Prices Results

## Baseline 001 - Scala Spark RandomForest Log Price

- Date: 2026-08-20
- Kaggle public score: `0.14311`
- Submission: `output/submission_rf_check.csv`
- Message: `Scala Spark RF log-price baseline`
- Local validation log-RMSE: `0.12197`

Pipeline summary:

- Dynamic numeric/categorical feature detection.
- Simple derived features: `HouseAgeAtSale`, `RemodAgeAtSale`, `TotalSF`, `TotalBathrooms`.
- Kaggle `NA` markers read as nulls before schema inference.
- Median imputation for numeric features.
- Categorical indexing for string/categorical features.
- Model target: `log1p(SalePrice)`.
- Model: Spark ML `RandomForestRegressor`, `numTrees = 500`, `maxDepth = 10`, `maxBins = 512`.

## Linear 001 - Scala Spark ElasticNet Log Price

- Date: 2026-08-20
- Kaggle public score: TBD
- Submission: `output/submission_lr_reg01_en08.csv`
- Message: `Scala Spark LR ElasticNet log-price v1`
- Local validation log-RMSE: `0.11452`

Pipeline changes versus RF baseline:

- `MSSubClass`, `MoSold`, and `YrSold` treated as categorical.
- Categorical features use `StringIndexer` plus `OneHotEncoder`.
- Numeric features use median imputation and `StandardScaler`.
- Added log features for skewed area variables.
- Added `TotalPorchSF`, `OverallQualXTotalSF`, and `GarageAgeAtSale`.
- Model: Spark ML `LinearRegression`, `regParam = 0.01`, `elasticNetParam = 0.8`, `maxIter = 250`.

Other local probes:

- `output/submission_lr_v1.csv`: local validation log-RMSE `0.12081`, `regParam = 0.003`, `elasticNetParam = 0.35`.
- `output/submission_lr_reg001_en015.csv`: local validation log-RMSE `0.12679`, `regParam = 0.001`, `elasticNetParam = 0.15`.
- `output/submission_lr_reg005_en05.csv`: local validation log-RMSE `0.11811`, `regParam = 0.005`, `elasticNetParam = 0.5`.
- validation-only probe: local validation log-RMSE `0.11635`, `regParam = 0.007`, `elasticNetParam = 0.65`.
- validation-only probe: local validation log-RMSE `0.11699`, `regParam = 0.015`, `elasticNetParam = 0.9`.

New Kaggle scores should be added here after submission with:

- experiment name;
- local validation log-RMSE from stdout;
- Kaggle public score;
- submission file path.
