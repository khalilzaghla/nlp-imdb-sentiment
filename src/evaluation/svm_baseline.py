import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.data.imdb_loader import load_imdb_dataset
from src.preprocessing.text_preprocessing import preprocess_texts


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"


def build_svm_pipeline(
    vectorizer_type: str,
    vectorizer_params: Dict[str, Any],
    classifier_params: Dict[str, Any],
) -> Pipeline:
    """Build a scikit-learn pipeline for text classification using SVM.

    Args:
        vectorizer_type: Either "bow" for CountVectorizer or "tfidf" for TfidfVectorizer.
        vectorizer_params: Parameters for the vectorizer.
        classifier_params: Parameters for the LinearSVC classifier.

    Returns:
        A Pipeline with vectorizer and SVM classifier.
    """
    if vectorizer_type == "bow":
        vectorizer = CountVectorizer(**vectorizer_params)
    elif vectorizer_type == "tfidf":
        vectorizer = TfidfVectorizer(**vectorizer_params)
    else:
        raise ValueError(f"Unknown vectorizer_type={vectorizer_type}")

    classifier = LinearSVC(**classifier_params)
    return Pipeline([("vectorizer", vectorizer), ("classifier", classifier)])


def fit_svm_pipeline(
    pipeline: Pipeline,
    X_train: Sequence[str],
    y_train: Sequence[int],
) -> Pipeline:
    """Fit the SVM text classification pipeline on training data.

    Args:
        pipeline: The SVM pipeline to fit.
        X_train: Training text documents.
        y_train: Training labels.

    Returns:
        The fitted pipeline.
    """
    return pipeline.fit(X_train, y_train)


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    """Compute standard classification metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary with accuracy, precision, recall, and f1 scores.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def evaluate_svm_pipeline(
    pipeline: Pipeline,
    X_test: Sequence[str],
    y_test: Sequence[int],
    label_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Evaluate a fitted SVM pipeline and print results.

    Args:
        pipeline: The fitted SVM pipeline.
        X_test: Test text documents.
        y_test: Test labels.
        label_names: Optional names for the labels.

    Returns:
        Dictionary containing metrics and evaluation results.
    """
    y_pred = pipeline.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)

    print("\n=== SVM Evaluation Summary ===")
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


def flatten_svm_report_row(metrics: Dict[str, Any], baseline_name: str) -> Dict[str, Any]:
    """Convert SVM metrics dictionary into a flat row for CSV tracking.

    Args:
        metrics: The metrics dictionary from evaluation.
        baseline_name: Name of the baseline (e.g., "svm_tfidf").

    Returns:
        Flattened dictionary suitable for CSV writing.
    """
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
        "max_iter": classifier.get("max_iter"),
    }


def save_svm_metrics(
    metrics: Dict[str, Any],
    target_dir: Path,
    baseline_name: str,
) -> None:
    """Save SVM run-level metrics into JSON and CSV summary files.

    Args:
        metrics: The metrics dictionary to save.
        target_dir: Directory to save the files in.
        baseline_name: Name of the baseline for file naming.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = target_dir / f"{baseline_name}_{metrics['config_name']}_{timestamp}.json"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    summary_path = target_dir / "svm_baseline_results.csv"
    summary_row = flatten_svm_report_row(metrics, baseline_name)
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
                "max_iter",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(summary_row)

    print(f"Saved SVM run metrics to {json_path}")
    print(f"Appended summary row to {summary_path}")


def run_svm_experiments(
    raw_data_dir: Path,
    results_dir: Optional[Path] = None,
    max_length: Optional[int] = None,
) -> Dict[str, Any]:
    """Train and evaluate SVM baseline configurations on TF-IDF features.

    Args:
        raw_data_dir: Path to the raw IMDB data directory.
        results_dir: Directory to save results. Defaults to REPORTS_DIR.
        max_length: Optional maximum token length per document.

    Returns:
        Dictionary with best results for each baseline type.
    """
    if results_dir is None:
        results_dir = REPORTS_DIR

    train_texts, train_labels, test_texts, test_labels = load_imdb_dataset(raw_data_dir)
    train_texts = preprocess_texts(train_texts, max_length=max_length)
    test_texts = preprocess_texts(test_texts, max_length=max_length)

    # SVM configurations - focusing on TF-IDF as it's typically better for SVM
    configs = [
        {
            "baseline": "svm_tfidf",
            "config_name": "svm_tfidf_unigrams",
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
                "C": 1.0,
                "max_iter": 2000,
                "random_state": 42,
                "dual": False,  # More efficient for n_samples > n_features
            },
        },
        {
            "baseline": "svm_tfidf",
            "config_name": "svm_tfidf_unigrams_bigrams",
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
                "C": 1.0,
                "max_iter": 2000,
                "random_state": 42,
                "dual": False,
            },
        },
        {
            "baseline": "svm_tfidf",
            "config_name": "svm_tfidf_high_capacity",
            "vectorizer_params": {
                "lowercase": True,
                "ngram_range": (1, 2),
                "min_df": 2,
                "max_df": 0.98,
                "max_features": 50000,
                "smooth_idf": True,
                "use_idf": True,
            },
            "classifier_params": {
                "C": 2.0,
                "max_iter": 3000,
                "random_state": 42,
                "dual": False,
            },
        },
    ]

    best_by_baseline: Dict[str, Dict[str, Any]] = {}

    for config in configs:
        baseline_name = config["baseline"]
        print(f"\nRunning {baseline_name} config: {config['config_name']}")

        pipeline = build_svm_pipeline(
            vectorizer_type="tfidf",  # All SVM configs use TF-IDF
            vectorizer_params=config["vectorizer_params"],
            classifier_params=config["classifier_params"],
        )
        fitted_pipeline = fit_svm_pipeline(pipeline, train_texts, train_labels)
        evaluation = evaluate_svm_pipeline(
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

        save_svm_metrics(run_metrics, results_dir, baseline_name)

        current_best = best_by_baseline.get(baseline_name)
        if current_best is None or run_metrics["metrics"]["f1"] > current_best["metrics"]["f1"]:
            best_by_baseline[baseline_name] = run_metrics

    print("\n=== Best SVM Configurations ===")
    for baseline_name, best in best_by_baseline.items():
        print(f"{baseline_name}: {best['config_name']} - F1={best['metrics']['f1']:.4f}")

    return best_by_baseline


def load_logistic_regression_results(results_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load the best Logistic Regression results for comparison.

    Args:
        results_dir: Directory containing the results. Defaults to REPORTS_DIR.

    Returns:
        Dictionary with the best LR results, or empty dict if not found.
    """
    if results_dir is None:
        results_dir = REPORTS_DIR

    # Look for the most recent best baseline results file
    lr_files = list(results_dir.glob("best_baseline_results_*.json"))
    if not lr_files:
        print("Warning: No Logistic Regression baseline results found for comparison")
        return {}

    # Get the most recent file
    latest_file = max(lr_files, key=lambda f: f.stat().st_mtime)

    with latest_file.open("r", encoding="utf-8") as handle:
        lr_results = json.load(handle)

    return lr_results


def compare_svm_to_logistic_regression(
    svm_results: Dict[str, Any],
    lr_results: Dict[str, Any],
) -> None:
    """Print a comparison table between SVM and Logistic Regression results.

    Args:
        svm_results: Results from SVM experiments.
        lr_results: Results from Logistic Regression experiments.
    """
    if not lr_results:
        print("No Logistic Regression results available for comparison")
        return

    print("\n" + "="*60)
    print("COMPARISON: SVM vs Logistic Regression on TF-IDF Features")
    print("="*60)

    # Print header
    print("<25")
    print("-" * 65)

    # Compare TF-IDF results
    svm_tfidf = svm_results.get("svm_tfidf")
    lr_tfidf = lr_results.get("tfidf")

    if svm_tfidf and lr_tfidf:
        svm_metrics = svm_tfidf["metrics"]
        lr_metrics = lr_tfidf["metrics"]

        print("<25"
              "<8.4f"
              "<8.4f"
              "<8.4f"
              "<8.4f")

        print("<25"
              "<8.4f"
              "<8.4f"
              "<8.4f"
              "<8.4f")

        # Calculate differences
        print("<25"
              "<+8.4f"
              "<+8.4f"
              "<+8.4f"
              "<+8.4f")

    print("\nSVM Configuration:", svm_tfidf["config_name"] if svm_tfidf else "N/A")
    print("LR Configuration:", lr_tfidf["config_name"] if lr_tfidf else "N/A")