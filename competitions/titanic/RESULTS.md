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

