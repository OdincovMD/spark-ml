# Titanic Results

## Baseline 001 - Scala Spark RandomForest

- Date: 2026-08-15
- Kaggle public score: `0.77272`
- Submission: `output/submission.csv`
- Message: `Scala Spark RandomForest baseline`
- Local validation accuracy: `0.8207`
- Local validation AUC: `0.8999`

Pipeline summary:

- Median imputation for `Age` and `Fare`.
- Categorical indexing for `Sex`, `Embarked`, `Pclass`, and extracted `Title`.
- Derived features: `FamilySize`, `IsAlone`, `FarePerPerson`.
- Model: Spark ML `RandomForestClassifier`, `numTrees = 200`, `maxDepth = 6`, `seed = 42`.

## Next Experiments

The runner now supports `baseline_001`, `rf_tuned_v2`, `gbt_v1`, `lr_v1`, `xgb_v1`, `ensemble_v1`, and `all`.

New Kaggle scores should be added here after submission with:

- experiment name;
- local CV accuracy/AUC from stdout or `output/experiments.csv`;
- Kaggle public score;
- submission file path.
