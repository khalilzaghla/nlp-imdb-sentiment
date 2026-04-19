# nlp-imdb-sentiment

This repository contains a sentiment analysis baseline project for the IMDB Large Movie Review Dataset.

## Project structure

- `data/` - dataset storage, including `raw/aclImdb`.
- `src/` - reusable code for data loading, preprocessing, and model evaluation.
- `notebooks/` - exploratory analysis and notebook experimentation.
- `reports/` - saved baseline results and summary files.
- `models/` - trained model artifacts.
- `outputs/` - run-level artifacts and metrics.
- `tests/` - unit tests and validation code.

## How to run

1. Ensure the dataset is available locally at `data/raw/aclImdb`.
2. From the repository root, run:

```bash
python src/evaluation/bow_tfidf_baseline.py
```

This script will:
- load the IMDB train/test splits
- preprocess the text with a shared pipeline
- train and evaluate multiple BoW and TF-IDF logistic regression baselines
- print confusion matrices and classification reports
- save run-level JSON metrics and a CSV summary to `reports/`

## Baseline implementation

The code now provides reusable API functions in `src/data/imdb_loader.py`, `src/preprocessing/text_preprocessing.py`, and `src/evaluation/baselines.py`.

### Key functions

- `load_imdb_dataset(raw_dir)` — load train/test texts and labels.
- `preprocess_texts(texts, max_length)` — shared text cleaning and optional truncation.
- `run_baseline_experiments(raw_data_dir, results_dir, max_length)` — run multiple model configurations and save results.

## Baseline results

After running the script, review the generated files in `reports/`:

- `baseline_results.csv` — summary table of all experiments and metrics.
- `best_baseline_results_<timestamp>.json` — best-performing configuration details.
- individual JSON run artifacts for each configuration.

## Notes

- The baseline models use scikit-learn pipelines for reproducibility.
- The preprocessing pipeline includes lowercasing, HTML break removal, whitespace normalization, and optional truncation.
- The experiment suite compares multiple hyperparameter configurations for both BoW and TF-IDF baselines.
