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
- Kaggle public score: `0.13185`
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

## Linear 002 - Feature Engineering and Training Outlier Filter

- Date: 2026-08-21
- Kaggle public score: `0.13414`
- Submission: `output/submission_lr_fe_v2.csv`
- Suggested message: `Scala Spark ElasticNet log-price FE v2`
- Local validation log-RMSE: `0.11281` (seed `42`)
- Previous pipeline on the identical split: `0.11452`

Pipeline changes versus Linear 001:

- Expanded `log1p` variants for skewed area, basement, porch, room-count, and rare-value features.
- Added `OverallQual` square/cube terms and quality-area interactions.
- Added total finished home square footage and pool/second-floor/garage/basement/fireplace flags.
- Removed the two rows with `GrLivArea >= 4000` and `SalePrice < 300000` only from model fitting; validation observations remain untouched.

Conclusion: the single seed-42 holdout selected an overfit feature set. The public score regressed from `0.13185` to `0.13414`, so advanced features are no longer the default.

## Linear 003 - Baseline Features with Two-Outlier Filter

- Date: 2026-08-21
- Kaggle public score: pending submission
- Submission: `output/submission_lr_v3.csv`
- Suggested message: `Scala Spark ElasticNet v1 plus two-outlier filter`

Three-seed validation comparison:

- Baseline v1 without filtering: `0.11452`, `0.11708`, `0.13001`; mean `0.12053`, stddev `0.00678`.
- Baseline v1 with the two-outlier training filter: `0.11436`, `0.11661`, `0.11329`; mean `0.11475`, stddev `0.00138`.

New Kaggle scores should be added here after submission with:

- experiment name;
- local validation log-RMSE from stdout;
- Kaggle public score;
- submission file path.
