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

## Postprocess 001 - Fast CV Probe

- Date: 2026-08-18
- Base model: first `rf_tuned_v2` candidate, `--fast --cv-folds 2`
- Best local postprocess candidate: `postprocess_family_ticket_v1`
- Local CV accuracy: `0.8227`
- Local CV AUC: `0.8683`
- Candidate submission: `output/submission_postprocess_family_ticket_v1.csv`

Fast CV candidates:

- `postprocess_threshold_047`: accuracy `0.8182`, AUC `0.8714`
- `postprocess_threshold_050`: accuracy `0.8160`, AUC `0.8714`
- `postprocess_threshold_053`: accuracy `0.8193`, AUC `0.8714`
- `postprocess_family_v1`: accuracy `0.8216`, AUC `0.8687`
- `postprocess_ticket_v1`: accuracy `0.8216`, AUC `0.8722`
- `postprocess_family_ticket_v1`: accuracy `0.8227`, AUC `0.8683`
- `postprocess_conservative_v1`: accuracy `0.8182`, AUC `0.8638`

Kaggle public scores:

- `submission.csv` / `FE+RF`: `0.77990` (`326 / 418`)
- `submission_postprocess_threshold_053.csv`: `0.78708` (`329 / 418`)
- `submission_postprocess_ticket_v1.csv`: `0.78708` (`329 / 418`)
- `submission_postprocess_family_v1.csv`: `0.79186` (`331 / 418`)
- `submission_postprocess_family_ticket_v1.csv`: `0.79186` (`331 / 418`)

Current best: `submission_postprocess_family_v1.csv` and `submission_postprocess_family_ticket_v1.csv`, both `0.79186`.
