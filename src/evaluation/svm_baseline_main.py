import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.svm_baseline import (
    REPORTS_DIR,
    compare_svm_to_logistic_regression,
    load_logistic_regression_results,
    run_svm_experiments,
)

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "aclImdb"


def main() -> None:
    """Run SVM baseline experiments and compare to Logistic Regression."""
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Loading IMDB data from: {DATA_DIR}")
    print("\nRunning SVM baseline experiments...")

    # Run SVM experiments
    svm_results = run_svm_experiments(
        raw_data_dir=DATA_DIR,
        results_dir=REPORTS_DIR,
        max_length=250,  # Same as LR baseline for fair comparison
    )

    # Load Logistic Regression results for comparison
    print("\nLoading Logistic Regression results for comparison...")
    lr_results = load_logistic_regression_results(REPORTS_DIR)

    # Compare results
    compare_svm_to_logistic_regression(svm_results, lr_results)

    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = REPORTS_DIR / f"svm_vs_lr_comparison_{timestamp}.json"

    comparison_data = {
        "svm_results": svm_results,
        "lr_results": lr_results,
        "timestamp": datetime.now().isoformat(),
    }

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(comparison_data, handle, indent=2)

    print(f"\nSaved comparison summary to {summary_path}")


if __name__ == "__main__":
    main()