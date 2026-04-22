import argparse


SELECTOR_CHOICES = [
    "variance_threshold",
    "chi2",
    "info_gain",
    "forward_selection",
    "backward_elimination",
    "l1",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark feature-selection algorithms on a tabular classification dataset."
    )
    parser.add_argument("--csv", required=True, help="Path to the dataset CSV file.")
    parser.add_argument(
        "--output-dir",
        default="outputs/feature_selection",
        help="Directory where result files will be written.",
    )
    parser.add_argument(
        "--selectors",
        nargs="+",
        default=SELECTOR_CHOICES,
        choices=SELECTOR_CHOICES,
        help="Feature-selection algorithms to benchmark.",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[10, 20, 30, 50],
        help="Feature counts to test for selectors that keep the top-k features.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of stratified cross-validation folds.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible sampling and training.",
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
