import csv
from pathlib import Path

import pandas as pd


TARGET_CANDIDATES = ["label", "class", "target", "legitimate"]
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
    normalized = df.copy()
    normalized.columns = [str(column).strip() for column in normalized.columns]
    return normalized


def find_target_column(df: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Target column '{requested}' was not found in the dataset.")
        return requested

    lowered = {str(column).strip().lower(): column for column in df.columns}
    for candidate in TARGET_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]

    preview = ", ".join(map(str, df.columns[:15]))
    raise ValueError(f"Could not infer the target column. First columns: {preview}")


def binarize_target(series: pd.Series, target_column: str, positive_label: str | None) -> pd.Series:
    cleaned = series.astype(str).str.strip()
    unique_values = cleaned.dropna().unique().tolist()

    if target_column.strip().lower() == "legitimate":
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.isna().any():
            raise ValueError("The 'legitimate' target column could not be converted to numeric values.")
        return (numeric == 0).astype(int)

    if positive_label is not None:
        return (cleaned.str.lower() == positive_label.strip().lower()).astype(int)

    if len(unique_values) != 2:
        raise ValueError(
            "The dataset target has more than two classes. Pass --positive-label to define the positive class."
        )

    numeric = pd.to_numeric(series, errors="coerce")
    if not numeric.isna().any():
        return (numeric != 0).astype(int)

    ordered = sorted(map(str, unique_values))
    mapping = {ordered[0]: 0, ordered[1]: 1}
    return cleaned.map(mapping).astype(int)


def infer_drop_columns(df: pd.DataFrame, target_column: str) -> list[str]:
    columns_to_drop = []
    for column in df.columns:
        lowered = column.strip().lower()
        if lowered == target_column.strip().lower():
            continue
        if lowered in LIKELY_ID_COLUMNS:
            columns_to_drop.append(column)
    return columns_to_drop


def load_dataset(
    csv_path: Path,
    target_column: str | None,
    positive_label: str | None,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    delimiter = sniff_delimiter(csv_path)
    raw_df = pd.read_csv(csv_path, sep=delimiter, low_memory=False)
    df = normalize_columns(raw_df)

    target = find_target_column(df, target_column)
    y = binarize_target(df[target], target, positive_label)

    dropped_identifier_columns = infer_drop_columns(df, target)
    X = df.drop(columns=[target, *dropped_identifier_columns], errors="ignore")

    X_numeric = X.apply(pd.to_numeric, errors="coerce")
    all_nan_columns = X_numeric.columns[X_numeric.isna().all()].tolist()
    X_numeric = X_numeric.drop(columns=all_nan_columns, errors="ignore")

    constant_columns = X_numeric.columns[X_numeric.nunique(dropna=False) <= 1].tolist()
    X_numeric = X_numeric.drop(columns=constant_columns, errors="ignore")

    if X_numeric.shape[1] == 0:
        raise ValueError("No usable numeric features remained after preprocessing.")

    metadata = {
        "csv_path": str(csv_path),
        "delimiter": delimiter,
        "target_column": target,
        "rows": int(len(df)),
        "usable_numeric_features": int(X_numeric.shape[1]),
        "class_balance": {
            "class_0": int((y == 0).sum()),
            "class_1": int((y == 1).sum()),
        },
        "dropped_identifier_columns": dropped_identifier_columns,
        "dropped_all_nan_columns": all_nan_columns,
        "dropped_constant_columns": constant_columns,
    }
    return X_numeric, y, metadata


def maybe_sample(
    X: pd.DataFrame,
    y: pd.Series,
    sample_size: int | None,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series, int | None]:
    if sample_size is None or sample_size >= len(X):
        return X, y, None

    class_zero = max(1, round(sample_size * (y == 0).mean()))
    class_one = max(1, sample_size - class_zero)

    sampled_zero = y[y == 0].sample(n=min(class_zero, int((y == 0).sum())), random_state=random_state)
    sampled_one = y[y == 1].sample(n=min(class_one, int((y == 1).sum())), random_state=random_state)
    sampled_indices = sampled_zero.index.union(sampled_one.index)

    sampled_X = X.loc[sampled_indices].reset_index(drop=True)
    sampled_y = y.loc[sampled_indices].reset_index(drop=True)
    return sampled_X, sampled_y, int(len(sampled_X))
