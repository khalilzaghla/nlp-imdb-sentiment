from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parent

    directories = [
        project_root / "data",
        project_root / "data" / "raw",
        project_root / "data" / "processed",
        project_root / "models",
        project_root / "notebooks",
        project_root / "outputs",
        project_root / "reports",
        project_root / "src",
        project_root / "src" / "data",
        project_root / "src" / "evaluation",
        project_root / "src" / "models",
        project_root / "src" / "preprocessing",
        project_root / "src" / "utils",
        project_root / "tests",
    ]

    for directory in directories:
        if directory.exists():
            print(f"Exists:  {directory}")
        else:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"Created: {directory}")


if __name__ == "__main__":
    main()
