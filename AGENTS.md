# Repository Guidelines

## Project Structure & Module Organization

This Scala 2.12.18/Spark 3.5.9 workspace contains two sbt subprojects: `titanic` and `housePrices`. Each lives under `competitions/<competition>/`, with pipeline code in `src/main/scala/kaggle/sparkml/<package>/` and logging configuration in `src/main/resources/`. Keep competition-specific changes within that module. Use its `README.md` for experiment commands and `RESULTS.md` for recorded metrics. Store input CSVs in `data/raw/`, intermediate datasets in `data/interim/` or `data/processed/`, saved models in `models/`, and submissions in `output/`.

## Build, Test, and Development Commands

Install Java and sbt; `project/build.properties` pins sbt 1.11.3. Run commands from the repository root using the wrapper, which keeps dependency caches locally:

```bash
./bin/sbt-local "titanic/compile" "housePrices/compile"
./bin/sbt-local "titanic/run --list-experiments"
./bin/sbt-local "titanic/run"
./bin/sbt-local "housePrices/run"
```

These compile both modules, list Titanic experiments, or train and generate submissions. Training requires Kaggle `train.csv` and `test.csv` in each module's `data/raw/`. Runtime paths are relative to the competition directory. Use distinct `--output` and `--model` paths when preserving earlier runs.

## Coding Style & Naming Conventions

Follow existing Scala style: two-space indentation, multiline parameter alignment, `UpperCamelCase` types/objects, and `lowerCamelCase` methods/variables. Keep packages under `kaggle.sparkml`. Use descriptive experiment identifiers such as `rf_tuned_v2` and submission filenames such as `submission_lr_xgb_v6.csv`. No formatter or linter is configured.

## Testing Guidelines

No automated test framework, test sources, or coverage threshold is configured. Compile changed modules and validate pipeline changes with reproducible runs:

```bash
./bin/sbt-local "titanic/run --fast --cv-folds 2 --output output/submission_check.csv --model models/check"
./bin/sbt-local "housePrices/run --validation-only --validation-seeds 42,1337,2026"
```

Compare Titanic accuracy/AUC or House Prices log-RMSE using identical settings. Check submission headers (`PassengerId,Survived` or `Id,SalePrice`) and row counts against test inputs. Record settings and metrics in the module's `RESULTS.md`.

## Commit & Pull Request Guidelines

History uses short free-form subjects; no enforced convention is evident. Write descriptive subjects naming the competition and change. PRs should explain the change, include validation commands and before/after metrics, link relevant issues, and update affected documentation.

## Configuration & Agent Instructions

Keep Kaggle credentials outside the repository; respect ignored datasets, models, outputs, and caches. Before edits or proposed changes, read `docs/README.md` and check `.gitnexus`. If unavailable, report that limitation and consult the root and competition READMEs.
