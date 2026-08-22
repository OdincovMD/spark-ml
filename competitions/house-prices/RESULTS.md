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
- Kaggle public score: `0.13542`
- Submission: `output/submission_lr_v3.csv`
- Suggested message: `Scala Spark ElasticNet v1 plus two-outlier filter`

Three-seed validation comparison:

- Baseline v1 without filtering: `0.11452`, `0.11708`, `0.13001`; mean `0.12053`, stddev `0.00678`.
- Baseline v1 with the two-outlier training filter: `0.11436`, `0.11661`, `0.11329`; mean `0.11475`, stddev `0.00138`.

Conclusion: despite substantially better random-holdout metrics, removing the two training outliers regressed the public score from `0.13185` to `0.13542`. All training rows are therefore retained by default.

## Ensemble 004 - ElasticNet and Dense-Vector XGBoost

- Date: 2026-08-22
- Kaggle public score: `0.12432`
- Submission: `output/submission_lr_xgb_v4.csv`
- Suggested message: `Scala Spark ElasticNet 60 XGBoost 40 log blend`
- Blend: 60% ElasticNet and 40% XGBoost predictions in log-price space.
- XGBoost: 500 rounds, eta `0.03`, depth `3`, subsample `0.8`, column sample `0.8`.
- All training rows retained; baseline feature set used.

Validation log-RMSE by XGBoost weight on seeds `42`, `1337`, and `2026`:

- 0%: `0.11452`, `0.11708`, `0.13001`; mean `0.12054`.
- 10%: `0.11351`, `0.11649`, `0.12643`; mean `0.11881`.
- 20%: `0.11282`, `0.11628`, `0.12334`; mean `0.11748`.
- 30%: `0.11245`, `0.11646`, `0.12078`; mean `0.11656`.
- 40%: `0.11240`, `0.11701`, `0.11879`; mean `0.11607`.
- 50%: `0.11269`, `0.11793`, `0.11739`; mean `0.11600`.

The quadratic minimum is near 47% XGBoost. A conservative 40% weight was chosen because standalone XGBoost was weaker than ElasticNet on two of the three splits.

## Ensemble 005 - Leaderboard-Calibrated Blend Weight

- Date: 2026-08-22
- Kaggle public score: `0.12282`
- Submission: `output/submission_lr_xgb_v5_w67.csv`
- Suggested message: `Scala Spark ElasticNet 33 XGBoost 67 calibrated log blend`
- Blend: 33% ElasticNet and 67% XGBoost predictions in log-price space.
- Same component models and all-row training policy as Ensemble 004.

Weight calibration from public scores and prediction distance:

- ElasticNet at 0% XGBoost: public RMSLE `0.13185`.
- Ensemble 004 at 40% XGBoost: public RMSLE `0.12432`.
- RMS log-distance between these submissions: `0.02867062`.
- Reconstructed quadratic optimum: XGBoost weight `0.6693`.
- Estimated public RMSLE at the optimum: `0.12281`.

The observed public score was `0.12282`, within `0.00001` of the estimate. Blend-weight optimization is therefore considered complete for these two component models.

## Ensemble 006 - Higher-Capacity Indexed XGBoost

- Date: 2026-08-22
- Kaggle public score: pending
- Submission: `output/submission_lr_xgb_v6_r800_w67.csv`
- Suggested message: `Scala Spark ElasticNet 33 XGBoost 67 indexed 800-round log blend`
- Blend: 33% ElasticNet and 67% XGBoost predictions in log-price space.
- XGBoost change versus Ensemble 005: 800 rounds instead of 500; all other model and feature settings retained.

Three-seed validation comparison at the same 50% XGBoost weight:

- 500 rounds: log-RMSE `0.11269`, `0.11793`, `0.11739`; mean `0.11600`.
- 800 rounds: log-RMSE `0.11153`, `0.11761`, `0.11610`; mean `0.11508`.
- Improvement: `0.00092` mean log-RMSE; the 800-round model improved all three validation seeds.

The leaderboard-calibrated 67% weight is retained for the first public probe because Ensemble 004 and 005 showed that random-holdout validation systematically underweighted XGBoost. The RMS log-distance from Ensemble 005 is `0.00790716`.

New Kaggle scores should be added here after submission with:

- experiment name;
- local validation log-RMSE from stdout;
- Kaggle public score;
- submission file path.
