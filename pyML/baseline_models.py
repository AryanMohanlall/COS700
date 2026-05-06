DEPENDENCY_ERROR = """This script requires third-party packages that are not installed.

Install them with:
  python -m pip install pandas numpy scikit-learn
"""


try:
    from feature_benchmark.baseline_cli import parse_args
    from feature_benchmark.baseline_runner import run_baseline_models
except ImportError as exc:
    raise SystemExit(f"{DEPENDENCY_ERROR}\nMissing import: {exc}")


if __name__ == "__main__":
    run_baseline_models(parse_args())
