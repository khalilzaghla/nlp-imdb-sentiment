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
