import csv
from pathlib import Path

import pandas as pd


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
