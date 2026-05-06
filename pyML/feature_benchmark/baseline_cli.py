import argparse


MODEL_CHOICES = [
    "random_forest",
    "xgboost",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train full-feature baseline classifiers for a tabular classification dataset."
    )
    parser.add_argument("--csv", required=True, help="Path to the dataset CSV file.")
    parser.add_argument(
        "--output-dir",
        default="outputs/baseline_models",
        help="Directory where result files will be written.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODEL_CHOICES,
        choices=MODEL_CHOICES,
        help="Baseline models to train and compare.",
    )
    parser.add_argument(
        "--random-search-iterations",
        type=int,
        default=12,
        help="Number of random hyperparameter samples to evaluate per model.",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.15,
        help="Fraction of the full dataset reserved for validation.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of the full dataset reserved for final testing.",
    )
    parser.add_argument(
        "--latency-repeats",
        type=int,
        default=25,
        help="Number of repeated prediction passes used to estimate detection latency.",
    )
    parser.add_argument(
        "--top-features",
        type=int,
        default=20,
        help="How many top-ranked features to include in the report preview per model.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible sampling, splitting, and tuning.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional stratified sample size for faster experimentation.",
    )
    parser.add_argument(
        "--target-column",
        default=None,
        help="Target/label column. If omitted, the loader will try to infer it.",
    )
    parser.add_argument(
        "--positive-label",
        default=None,
        help="Optional label value to treat as the positive class.",
    )
    return parser.parse_args()
