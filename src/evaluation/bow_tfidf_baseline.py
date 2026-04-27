import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.baselines import REPORTS_DIR, run_baseline_experiments

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "aclImdb"


def main() -> None:
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Loading IMDB data from: {DATA_DIR}")

    best_results = run_baseline_experiments(
        raw_data_dir=DATA_DIR,
        results_dir=REPORTS_DIR,
        max_length=250,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = REPORTS_DIR / f"best_baseline_results_{timestamp}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(best_results, handle, indent=2)

    print(f"\nSaved best result summary to {summary_path }")


if __name__ == "__main__":
    main()
