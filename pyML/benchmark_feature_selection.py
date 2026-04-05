import argparse
import csv
import json
import warnings
from pathlib import Path
from time import perf_counter


DEPENDENCY_ERROR = """This script requires third-party packages that are not installed.

Install them with:
  python -m pip install pandas numpy scikit-learn
"""


try:
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, f_classif, mutual_info_classif
    from sklearn.impute import SimpleImputer
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
except ImportError as exc:
    raise SystemExit(f"{DEPENDENCY_ERROR}\nMissing import: {exc}")

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


warnings.filterwarnings(
    "ignore",
    message=r"'penalty' was deprecated in version 1\.8.*",
    category=FutureWarning,
    module=r"sklearn\.linear_model\._logistic",
)
warnings.filterwarnings(
    "ignore",
    message=r"Inconsistent values: penalty=l1 with l1_ratio=0\.0.*",
    category=UserWarning,
    module=r"sklearn\.linear_model\._logistic",
)


TARGET_CANDIDATES = ["label", "class", "target", "legitimate"]
BENIGN_LABELS = {"benign", "normal", "legitimate", "clean", "goodware", "safe", "0", "false", "no"}
LIKELY_ID_COLUMNS = {
    "",
    "unnamed: 0",
    "name",
    "md5",
    "sha1",
    "sha256",
    "flow id",
    "source ip",
    "destination ip",
    "timestamp",
}


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


def sniff_delimiter(csv_path: Path) -> str:
    sample = csv_path.read_text(encoding="utf-8-sig", errors="ignore")[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;|\t")
        return dialect.delimiter
    except Exception:
        if "|" in sample and sample.count("|") > sample.count(","):
            return "|"
        return ","


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def find_target_column(df: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Requested target column '{requested}' was not found.")
        return requested

    normalized = {str(col).strip().lower(): col for col in df.columns}
    for candidate in TARGET_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]

    raise ValueError(
        f"Could not infer the target column. Available columns include: {list(df.columns[:15])}"
    )


def binarize_target(series: pd.Series, target_column: str, benign_label: str) -> pd.Series:
    cleaned = series.astype(str).str.strip()
    unique_values = sorted(cleaned.dropna().unique().tolist())
    lowered = {value.lower() for value in unique_values}

    if target_column.strip().lower() == "legitimate":
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.isna().any():
            raise ValueError("The 'legitimate' target column could not be safely converted to numeric values.")
        return (numeric == 0).astype(int)

    if len(unique_values) == 2:
        benign_candidates = lowered & BENIGN_LABELS
        if benign_candidates:
            return (~cleaned.str.lower().isin(benign_candidates)).astype(int)

        numeric = pd.to_numeric(series, errors="coerce")
        if not numeric.isna().any():
            return (numeric != 0).astype(int)

        ordered = {unique_values[0]: 0, unique_values[1]: 1}
        return cleaned.map(ordered).astype(int)

    benign_value = benign_label.strip().lower()
    if benign_value not in lowered:
        raise ValueError(
            f"Dataset has multiple classes {unique_values[:10]}, but benign label '{benign_label}' was not found."
        )

    return (cleaned.str.lower() != benign_value).astype(int)


def infer_drop_columns(df: pd.DataFrame, target_column: str) -> list[str]:
    drop_columns = []
    for column in df.columns:
        lowered = column.strip().lower()
        if lowered == target_column.strip().lower():
            continue
        if lowered in LIKELY_ID_COLUMNS:
            drop_columns.append(column)
    return drop_columns


def load_dataset(csv_path: Path, target_column: str | None, benign_label: str) -> tuple[pd.DataFrame, pd.Series, dict]:
    delimiter = sniff_delimiter(csv_path)
    df = pd.read_csv(csv_path, sep=delimiter, low_memory=False)
    df = normalize_columns(df)

    target = find_target_column(df, target_column)
    y = binarize_target(df[target], target, benign_label)

    drop_columns = infer_drop_columns(df, target)
    X = df.drop(columns=[target, *drop_columns], errors="ignore")

    X_numeric = X.apply(pd.to_numeric, errors="coerce")
    all_nan_columns = X_numeric.columns[X_numeric.isna().all()].tolist()
    X_numeric = X_numeric.drop(columns=all_nan_columns, errors="ignore")

    nunique = X_numeric.nunique(dropna=False)
    constant_columns = nunique[nunique <= 1].index.tolist()
    X_numeric = X_numeric.drop(columns=constant_columns, errors="ignore")

    metadata = {
        "csv_path": str(csv_path),
        "delimiter": delimiter,
        "target_column": target,
        "dropped_identifier_columns": drop_columns,
        "dropped_all_nan_columns": all_nan_columns,
        "dropped_constant_columns": constant_columns,
        "rows": int(len(df)),
        "usable_numeric_features": int(X_numeric.shape[1]),
        "class_balance": {
            "benign_0": int((y == 0).sum()),
            "ransomware_1": int((y == 1).sum()),
        },
    }

    if X_numeric.shape[1] == 0:
        raise ValueError("No usable numeric features remained after preprocessing.")

    return X_numeric, y, metadata


def maybe_sample(X: pd.DataFrame, y: pd.Series, sample_size: int | None, random_state: int):
    if sample_size is None or sample_size >= len(X):
        return X, y, None

    sample_size = max(sample_size, 2)
    sampled_indices = (
        y.groupby(y, group_keys=False)
        .apply(
            lambda class_values: class_values.sample(
                n=max(1, round(sample_size * len(class_values) / len(y))),
                random_state=random_state,
            )
        )
        .index
    )

    sampled_indices = sampled_indices[:sample_size]
    sampled_X = X.loc[sampled_indices].reset_index(drop=True)
    sampled_y = y.loc[sampled_indices].reset_index(drop=True)

    return sampled_X, sampled_y, int(len(sampled_X))


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                slice(0, None),
            )
        ],
        remainder="drop",
    )


def make_classifier(name: str, random_state: int):
    if name == "logistic":
        return LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            max_iter=2000,
            random_state=random_state,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        )
    if name == "svm":
        return SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=random_state,
        )
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=5, weights="distance")
    if name == "xgboost":
        if XGBClassifier is None:
            raise SystemExit(
                "xgboost is not installed. Install it with:\n"
                "  python -m pip install xgboost"
            )
        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
        )
    raise ValueError(f"Unsupported classifier: {name}")


def make_selector(name: str, k: int, random_state: int):
    if name == "baseline":
        return "passthrough"
    if name == "mutual_info":
        return SelectKBest(score_func=mutual_info_classif, k=k)
    if name == "f_classif":
        return SelectKBest(score_func=f_classif, k=k)
    if name == "l1":
        estimator = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            class_weight="balanced",
            max_iter=2000,
            random_state=random_state,
        )
        return SelectFromModel(estimator=estimator, threshold=-np.inf, max_features=k)
    if name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        )
        return SelectFromModel(estimator=estimator, threshold=-np.inf, max_features=k)
    if name == "rfe":
        estimator = LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            max_iter=2000,
            random_state=random_state,
        )
        return RFE(estimator=estimator, n_features_to_select=k, step=0.1)
    raise ValueError(f"Unsupported selector: {name}")


def get_selector_support(selector, feature_names: list[str]) -> list[str]:
    if selector == "passthrough":
        return feature_names

    support = selector.get_support()
    return [feature for feature, keep in zip(feature_names, support) if keep]


def benchmark_configuration(
    X: pd.DataFrame,
    y: pd.Series,
    selector_name: str,
    classifier_name: str,
    k: int | None,
    folds: int,
    random_state: int,
) -> tuple[dict, list[dict]]:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    feature_names = X.columns.tolist()
    feature_rows = []

    fold_metrics = {
        "f1": [],
        "precision": [],
        "recall": [],
        "roc_auc": [],
        "pr_auc": [],
        "fit_seconds": [],
        "selected_feature_count": [],
    }

    for fold_index, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        selector = make_selector(selector_name, k=k, random_state=random_state)
        classifier = make_classifier(classifier_name, random_state=random_state)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("selector", selector),
                ("classifier", classifier),
            ]
        )

        start = perf_counter()
        pipeline.fit(X_train, y_train)
        fold_metrics["fit_seconds"].append(perf_counter() - start)
        y_pred = pipeline.predict(X_test)
        y_score = pipeline.predict_proba(X_test)[:, 1]

        fitted_selector = pipeline.named_steps["selector"]
        selected_features = get_selector_support(fitted_selector, feature_names)

        for feature_name in selected_features:
            feature_rows.append(
                {
                    "selector": selector_name,
                    "classifier": classifier_name,
                    "k": len(selected_features),
                    "fold": fold_index,
                    "feature": feature_name,
                }
            )

        fold_metrics["f1"].append(f1_score(y_test, y_pred, zero_division=0))
        fold_metrics["precision"].append(precision_score(y_test, y_pred, zero_division=0))
        fold_metrics["recall"].append(recall_score(y_test, y_pred, zero_division=0))
        fold_metrics["roc_auc"].append(roc_auc_score(y_test, y_score))
        fold_metrics["pr_auc"].append(average_precision_score(y_test, y_score))
        fold_metrics["selected_feature_count"].append(len(selected_features))

    result = {
        "selector": selector_name,
        "classifier": classifier_name,
        "k": "all" if selector_name == "baseline" else k,
        "mean_f1": float(np.mean(fold_metrics["f1"])),
        "std_f1": float(np.std(fold_metrics["f1"])),
        "mean_precision": float(np.mean(fold_metrics["precision"])),
        "mean_recall": float(np.mean(fold_metrics["recall"])),
        "mean_roc_auc": float(np.mean(fold_metrics["roc_auc"])),
        "mean_pr_auc": float(np.mean(fold_metrics["pr_auc"])),
        "mean_fit_seconds": float(np.mean(fold_metrics["fit_seconds"])),
        "mean_selected_feature_count": float(np.mean(fold_metrics["selected_feature_count"])),
    }

    return result, feature_rows


def format_results_table(df: pd.DataFrame, limit: int = 10) -> str:
    display_df = df.head(limit).copy()
    if display_df.empty:
        return "No benchmark rows to display."

    display_df = display_df.rename(
        columns={
            "selector": "selector",
            "classifier": "model",
            "k": "k",
            "mean_f1": "f1",
            "std_f1": "f1_std",
            "mean_precision": "precision",
            "mean_recall": "recall",
            "mean_roc_auc": "roc_auc",
            "mean_pr_auc": "pr_auc",
            "mean_fit_seconds": "fit_s",
            "mean_selected_feature_count": "features",
        }
    )

    for col in ["f1", "f1_std", "precision", "recall", "roc_auc", "pr_auc", "fit_s", "features"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(
                lambda value: f"{value:.4f}" if isinstance(value, (int, float, np.floating)) else value
            )

    return display_df[
        ["selector", "model", "k", "f1", "f1_std", "precision", "recall", "pr_auc", "fit_s", "features"]
    ].to_string(index=False)


def summarize_best_by_group(results_df: pd.DataFrame, group_column: str) -> str:
    if results_df.empty:
        return "No results available."

    best_rows = (
        results_df.sort_values(
            by=["mean_f1", "mean_selected_feature_count", "mean_fit_seconds"],
            ascending=[False, True, True],
        )
        .groupby(group_column, as_index=False)
        .first()
        .sort_values(by=["mean_f1", "mean_selected_feature_count"], ascending=[False, True])
    )

    lines = []
    for _, row in best_rows.iterrows():
        lines.append(
            f"- {row[group_column]}: selector={row['selector']}, model={row['classifier']}, "
            f"k={row['k']}, f1={row['mean_f1']:.4f}, features={row['mean_selected_feature_count']:.1f}, "
            f"fit_s={row['mean_fit_seconds']:.4f}"
        )
    return "\n".join(lines)


def summarize_top_features(frequency_df: pd.DataFrame, results_df: pd.DataFrame, top_n: int = 10) -> str:
    if frequency_df.empty or results_df.empty:
        return "No selected-feature frequency data available."

    best_row = results_df.iloc[0]
    best_features = frequency_df[
        (frequency_df["selector"] == best_row["selector"])
        & (frequency_df["classifier"] == best_row["classifier"])
        & (frequency_df["k"].astype(str) == str(best_row["k"]))
    ].sort_values(by=["fold_selection_count", "feature"], ascending=[False, True])

    if best_features.empty:
        return "No selected features recorded for the best configuration."

    lines = [
        f"Best configuration: selector={best_row['selector']}, model={best_row['classifier']}, "
        f"k={best_row['k']}, f1={best_row['mean_f1']:.4f}, "
        f"features={best_row['mean_selected_feature_count']:.1f}"
    ]
    for _, row in best_features.head(top_n).iterrows():
        lines.append(f"- {row['feature']}: selected in {int(row['fold_selection_count'])} fold(s)")
    return "\n".join(lines)


def run_benchmark(args):
    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_lines = []
    file_stem = csv_path.stem.lower().replace(" ", "_")

    def emit(message: str):
        print(message)
        log_lines.append(message)

    X, y, metadata = load_dataset(csv_path, args.target_column, args.benign_label)
    X, y, sampled_rows = maybe_sample(X, y, args.sample_size, args.random_state)

    if sampled_rows is not None:
        metadata["sampled_rows"] = sampled_rows

    benchmark_rows = []
    feature_rows = []

    max_k = X.shape[1]
    k_values = sorted({k for k in args.k_values if k <= max_k})
    if not k_values:
        k_values = [max_k]

    for selector_name in args.selectors:
        selector_k_values = [None] if selector_name == "baseline" else k_values
        for k in selector_k_values:
            for classifier_name in args.classifiers:
                result, selected_features = benchmark_configuration(
                    X=X,
                    y=y,
                    selector_name=selector_name,
                    classifier_name=classifier_name,
                    k=k,
                    folds=args.folds,
                    random_state=args.random_state,
                )
                benchmark_rows.append(result)
                feature_rows.extend(selected_features)
                emit(
                    f"Finished selector={selector_name} classifier={classifier_name} "
                    f"k={result['k']} mean_f1={result['mean_f1']:.4f} "
                    f"features={result['mean_selected_feature_count']:.1f}"
                )

    results_df = pd.DataFrame(benchmark_rows).sort_values(
        by=["mean_f1", "mean_selected_feature_count", "mean_fit_seconds"],
        ascending=[False, True, True],
    )
    feature_df = pd.DataFrame(feature_rows)

    if not feature_df.empty:
        frequency_df = (
            feature_df.groupby(["selector", "classifier", "k", "feature"])
            .size()
            .reset_index(name="fold_selection_count")
            .sort_values(
                by=["selector", "classifier", "k", "fold_selection_count", "feature"],
                ascending=[True, True, True, False, True],
            )
        )
    else:
        frequency_df = pd.DataFrame(
            columns=["selector", "classifier", "k", "feature", "fold_selection_count"]
        )

    results_path = output_dir / f"{file_stem}_benchmark_results.csv"
    features_path = output_dir / f"{file_stem}_selected_feature_frequencies.csv"
    metadata_path = output_dir / f"{file_stem}_run_metadata.json"
    summary_path = output_dir / f"{file_stem}_benchmark_summary.txt"

    results_df.to_csv(results_path, index=False)
    frequency_df.to_csv(features_path, index=False)

    metadata.update(
        {
            "selectors": args.selectors,
            "classifiers": args.classifiers,
            "k_values": k_values,
            "folds": args.folds,
            "random_state": args.random_state,
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    emit("")
    emit("Run overview:")
    emit(
        f"- dataset={csv_path.name}, rows={len(X)}, features={X.shape[1]}, "
        f"benign={int((y == 0).sum())}, ransomware={int((y == 1).sum())}"
    )
    emit(
        f"- selectors={', '.join(args.selectors)}, models={', '.join(args.classifiers)}, "
        f"folds={args.folds}, k_values={', '.join(str(k) for k in k_values)}"
    )
    emit("")
    emit("Leaderboard:")
    emit(format_results_table(results_df, limit=10))
    emit("")
    emit("Best per selector:")
    emit(summarize_best_by_group(results_df, "selector"))
    emit("")
    emit("Best per model:")
    emit(summarize_best_by_group(results_df, "classifier"))
    emit("")
    emit("Top recurring features for the best configuration:")
    emit(summarize_top_features(frequency_df, results_df, top_n=10))
    emit("")
    emit(f"Saved results to {results_path}")
    emit(f"Saved feature frequencies to {features_path}")
    emit(f"Saved metadata to {metadata_path}")
    emit(f"Saved summary to {summary_path}")

    summary_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run_benchmark(parse_args())
