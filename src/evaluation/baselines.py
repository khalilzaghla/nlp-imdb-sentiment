import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from src.data.imdb_loader import load_imdb_dataset
from src.preprocessing.text_preprocessing import preprocess_texts


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"


def build_text_pipeline(
    vectorizer_type: str,
    vectorizer_params: Dict[str, Any],
    classifier_params: Dict[str, Any],
) -> Pipeline:
    """Build a scikit-learn pipeline for text classification."""
    if vectorizer_type == "bow":
        vectorizer = CountVectorizer(**vectorizer_params)
    elif vectorizer_type == "tfidf":
        vectorizer = TfidfVectorizer(**vectorizer_params)
    else:
        raise ValueError(f"Unknown vectorizer_type={vectorizer_type}")

    classifier = LogisticRegression(**classifier_params)
    return Pipeline([("vectorizer", vectorizer), ("classifier", classifier)])


def fit_pipeline(
    pipeline: Pipeline,
    X_train: Sequence[str],
    y_train: Sequence[int],
) -> Pipeline:
    """Fit the text classification pipeline on training data."""
    return pipeline.fit(X_train, y_train)


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    """Compute standard classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def evaluate_pipeline(
    pipeline: Pipeline,
    X_test: Sequence[str],
    y_test: Sequence[int],
    label_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Evaluate a fitted pipeline and print a classification report."""
    y_pred = pipeline.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)

    print("\n=== Evaluation summary ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 score: {metrics['f1']:.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=label_names, digits=4))

    return {
        "metrics": metrics,
        "classification_report": classification_report(
            y_test, y_pred, target_names=label_names, digits=4, output_dict=True
        ),
        "y_pred": y_pred,
    }


def flatten_report_row(metrics: Dict[str, Any], baseline_name: str) -> Dict[str, Any]:
    """Convert a metrics dictionary into a flat row for CSV tracking."""
    vectorizer = metrics["vectorizer_params"]
    classifier = metrics["classifier_params"]
    return {
        "baseline": baseline_name,
        "config_name": metrics["config_name"],
        "accuracy": f"{metrics['metrics']['accuracy']:.4f}",
        "precision": f"{metrics['metrics']['precision']:.4f}",
        "recall": f"{metrics['metrics']['recall']:.4f}",
        "f1": f"{metrics['metrics']['f1']:.4f}",
        "ngram_range": vectorizer.get("ngram_range"),
        "max_features": vectorizer.get("max_features"),
        "min_df": vectorizer.get("min_df"),
        "max_df": vectorizer.get("max_df"),
        "classifier_C": classifier.get("C"),
        "solver": classifier.get("solver"),
    }


def save_metrics(
    metrics: Dict[str, Any],
    target_dir: Path,
    baseline_name: str,
) -> None:
    """Save run-level metrics into JSON and CSV summary files."""
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = target_dir / f"{baseline_name}_{metrics['config_name']}_{timestamp}.json"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    summary_path = target_dir / "baseline_results.csv"
    summary_row = flatten_report_row(metrics, baseline_name)
    write_header = not summary_path.exists()

    with summary_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "baseline",
                "config_name",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "ngram_range",
                "max_features",
                "min_df",
                "max_df",
                "classifier_C",
                "solver",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(summary_row)

    print(f"Saved run metrics to {json_path}")
    print(f"Appended summary row to {summary_path}")


def run_baseline_experiments(
    raw_data_dir: Path,
    results_dir: Optional[Path] = None,
    max_length: Optional[int] = None,
) -> Dict[str, Any]:
    """Train and evaluate a set of BoW and TF-IDF baseline configurations."""
    if results_dir is None:
        results_dir = REPORTS_DIR

    train_texts, train_labels, test_texts, test_labels = load_imdb_dataset(raw_data_dir)
    train_texts = preprocess_texts(train_texts, max_length=max_length)
    test_texts = preprocess_texts(test_texts, max_length=max_length)

    configs = [
        {
            "baseline": "bow",
            "config_name": "bow_unigrams",
            "vectorizer_params": {
                "lowercase": True,
                "ngram_range": (1, 1),
                "min_df": 5,
                "max_df": 0.95,
                "max_features": 20000,
            },
            "classifier_params": {
                "solver": "liblinear",
                "penalty": "l2",
                "C": 1.0,
                "max_iter": 2000,
                "random_state": 42,
            },
        },
        {
            "baseline": "bow",
            "config_name": "bow_unigrams_bigrams",
            "vectorizer_params": {
                "lowercase": True,
                "ngram_range": (1, 2),
                "min_df": 3,
                "max_df": 0.95,
                "max_features": 30000,
            },
            "classifier_params": {
                "solver": "liblinear",
                "penalty": "l2",
                "C": 1.0,
                "max_iter": 2000,
                "random_state": 42,
            },
        },
        {
            "baseline": "bow",
            "config_name": "bow_high_capacity",
            "vectorizer_params": {
                "lowercase": True,
                "ngram_range": (1, 2),
                "min_df": 2,
                "max_df": 0.98,
                "max_features": 50000,
            },
            "classifier_params": {
                "solver": "liblinear",
                "penalty": "l2",
                "C": 2.0,
                "max_iter": 2000,
                "random_state": 42,
            },
        },
        {
            "baseline": "tfidf",
            "config_name": "tfidf_unigrams",
            "vectorizer_params": {
                "lowercase": True,
                "ngram_range": (1, 1),
                "min_df": 5,
                "max_df": 0.95,
                "max_features": 20000,
                "smooth_idf": True,
                "use_idf": True,
            },
            "classifier_params": {
                "solver": "liblinear",
                "penalty": "l2",
                "C": 1.0,
                "max_iter": 2000,
                "random_state": 42,
            },
        },
        {
            "baseline": "tfidf",
            "config_name": "tfidf_unigrams_bigrams",
            "vectorizer_params": {
                "lowercase": True,
                "ngram_range": (1, 2),
                "min_df": 3,
                "max_df": 0.95,
                "max_features": 30000,
                "smooth_idf": True,
                "use_idf": True,
            },
            "classifier_params": {
                "solver": "liblinear",
                "penalty": "l2",
                "C": 1.5,
                "max_iter": 2000,
                "random_state": 42,
            },
        },
        {
            "baseline": "tfidf",
            "config_name": "tfidf_more_ngrams",
            "vectorizer_params": {
                "lowercase": True,
                "ngram_range": (1, 3),
                "min_df": 2,
                "max_df": 0.98,
                "max_features": 40000,
                "smooth_idf": True,
                "use_idf": True,
            },
            "classifier_params": {
                "solver": "liblinear",
                "penalty": "l2",
                "C": 1.0,
                "max_iter": 2000,
                "random_state": 42,
            },
        },
    ]

    best_by_baseline: Dict[str, Dict[str, Any]] = {}

    for config in configs:
        baseline_name = config["baseline"]
        print(f"\nRunning {baseline_name} config: {config['config_name']}")

        pipeline = build_text_pipeline(
            vectorizer_type=baseline_name,
            vectorizer_params=config["vectorizer_params"],
            classifier_params=config["classifier_params"],
        )
        fitted_pipeline = fit_pipeline(pipeline, train_texts, train_labels)
        evaluation = evaluate_pipeline(
            fitted_pipeline,
            test_texts,
            test_labels,
            label_names=["negative", "positive"],
        )

        run_metrics = {
            "baseline": baseline_name,
            "config_name": config["config_name"],
            "vectorizer_params": config["vectorizer_params"],
            "classifier_params": config["classifier_params"],
            "metrics": evaluation["metrics"],
            "classification_report": evaluation["classification_report"],
            "timestamp": datetime.now().isoformat(),
            "n_train": len(train_texts),
            "n_test": len(test_texts),
        }

        save_metrics(run_metrics, results_dir, baseline_name)

        current_best = best_by_baseline.get(baseline_name)
        if current_best is None or run_metrics["metrics"]["f1"] > current_best["metrics"]["f1"]:
            best_by_baseline[baseline_name] = run_metrics

    print("\n=== Best configurations ===")
    for baseline_name, best in best_by_baseline.items():
        print(f"{baseline_name}: {best['config_name']} - F1={best['metrics']['f1']:.4f}")

    return best_by_baseline
