import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark feature-selection methods for ransomware detection."
    )
    parser.add_argument("--csv", required=True, help="Path to the input CSV file.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for result files.")
    parser.add_argument(
        "--selectors",
        nargs="+",
        default=["baseline", "mutual_info", "f_classif", "l1", "random_forest"],
        choices=["baseline", "mutual_info", "f_classif", "l1", "random_forest", "rfe"],
        help="Feature-selection methods to benchmark.",
    )
    parser.add_argument(
        "--classifiers",
        nargs="+",
        default=["logistic", "random_forest"],
        choices=["logistic", "random_forest", "svm", "knn", "xgboost"],
        help="Evaluation classifiers used to score selected features.",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[10, 20, 30, 50],
        help="Feature subset sizes to test for selectors that use k.",
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of stratified CV folds.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional stratified sample size for quicker experiments.",
    )
    parser.add_argument(
        "--benign-label",
        default="Benign",
        help="Label value treated as benign when the dataset has multiple classes.",
    )
    parser.add_argument(
        "--target-column",
        default=None,
        help="Optional manual override for the label column.",
    )
    return parser.parse_args()
