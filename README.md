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

### SVM Baseline

For SVM-based baselines using the same TF-IDF features:

```bash
python src/evaluation/svm_baseline_main.py
```

This script will:
- load and preprocess the IMDB data (same as LR baseline)
- train SVM models with LinearSVC on TF-IDF features
- evaluate performance and compare to Logistic Regression
- save results to `reports/svm_baseline_results.csv`

See `notebooks/svm_baseline_demo.ipynb` for an interactive demonstration.

## Baseline implementation

The code now provides reusable API functions in `src/data/imdb_loader.py`, `src/preprocessing/text_preprocessing.py`, and `src/evaluation/baselines.py`.

### Key functions

- `load_imdb_dataset(raw_dir)` — load train/test texts and labels.
- `preprocess_texts(texts, max_length)` — shared text cleaning and optional truncation.
- `run_baseline_experiments(raw_data_dir, results_dir, max_length)` — run multiple Logistic Regression model configurations and save results.
- `run_svm_experiments(raw_data_dir, results_dir, max_length)` — run multiple SVM model configurations and save results.

### SVM Baseline Details

The SVM baseline in `src/evaluation/svm_baseline.py` provides:
- `build_svm_pipeline()` — create TF-IDF + LinearSVC pipelines
- `evaluate_svm_pipeline()` — compute metrics and print reports
- `compare_svm_to_logistic_regression()` — direct performance comparison

SVM models use LinearSVC for efficiency with sparse TF-IDF features.

## Baseline results

After running the scripts, review the generated files in `reports/`:

- `baseline_results.csv` — summary table of all Logistic Regression experiments and metrics.
- `svm_baseline_results.csv` — summary table of all SVM experiments and metrics.
- `best_baseline_results_<timestamp>.json` — best-performing Logistic Regression configuration details.
- `svm_vs_lr_comparison_<timestamp>.json` — comparison between SVM and LR performance.
- individual JSON run artifacts for each configuration.

## Notes

- The baseline models use scikit-learn pipelines for reproducibility.
- The preprocessing pipeline includes lowercasing, HTML break removal, whitespace normalization, and optional truncation.
- The experiment suite compares multiple hyperparameter configurations for both BoW and TF-IDF baselines.
# IMDB Sentiment Analysis – NLP Text Classification

This project builds and compares NLP models for binary sentiment classification (positive vs negative) using the IMDB movie reviews dataset. The goal is to go from simple bag‑of‑words baselines to more advanced representation and modeling techniques.

## Project Overview

- **Task:** Sentiment analysis on movie reviews (binary: positive vs negative)
- **Dataset:** IMDB movie review dataset (train/test splits with labeled reviews)
- **Baselines:** Bag‑of‑Words and TF‑IDF features with traditional ML classifiers
- **Planned extensions:**
  - Better text preprocessing and tokenization
  - Word embeddings or contextual models
  - Deeper architectures (e.g., RNN/CNN/transformer‑based)

## Repository Structure

This repository follows a standard NLP project skeleton to keep experimentation organized:

- `data/` – Raw and processed datasets (not tracked by Git, see below)
- `src/` – Source code for data loading, preprocessing, modeling, and training
- `notebooks/` – Exploratory analysis and experimentation
- `models/` – Saved model checkpoints and vectorizers
- `outputs/` – Evaluation reports, metrics, and plots
- `tests/` – Unit tests for core components (preprocessing, data pipeline, etc.)

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/khalilzaghla/nlp-imdb-sentiment.git
cd nlp-imdb-sentiment
```

### 2. Create and activate an environment

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

*(If you don’t have a `requirements.txt` yet, this is a good place to add it.)*

### 4. Download the dataset

Briefly explain where the IMDB data comes from and how to place it, for example:

1. Download the IMDB movie review dataset from [link or instructions].
2. Place the files under `data/raw/` with the following layout:
   - `data/raw/train/`
   - `data/raw/test/`

## Usage

Examples (adapt to your actual scripts):

```bash
# Train a BoW/TF-IDF baseline model
python src/train_baseline.py --config configs/bow_tfidf_logreg.yaml

# Evaluate a trained model
python src/evaluate.py --model-path models/bow_tfidf_logreg.joblib
```

Key CLI options:

- `--config`: Path to a YAML configuration file with model and training settings
- `--model-path`: Path to a saved model to load for evaluation

## Experiments and Results

Once you have runs, summarize them here in a small table:

| Model                    | Features   | Accuracy | F1-score |
|--------------------------|-----------|----------|----------|
| Logistic Regression      | BoW       | 0.86     | 0.87    |
| Linear SVM               | TF-IDF    | 0.9     | 0.89    |

Add a short note about what you learned from these results and what you plan to try next.

## Development

### Branching strategy

Development happens in feature branches (for example, `feature/bow-tfidf-baseline`) and is merged into `main` via pull requests. This keeps experiments isolated and the `main` branch stable.

### Testing

If you add tests:

```bash
pytest
```

Describe what is currently covered by tests (preprocessing, tokenization, etc.) and what you plan to add.

## Roadmap

- Add configuration management for experiments
- Implement and compare deep learning models
- Add experiment tracking and visualization of metrics
- Improve documentation and examples

## License

Specify the license here (e.g., MIT) once you decide on one.
