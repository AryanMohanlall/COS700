from itertools import combinations
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, chi2, mutual_info_classif
from sklearn.model_selection import StratifiedKFold


warnings.filterwarnings(
    "ignore",
    message=r"Features .* are constant\.",
    category=UserWarning,
    module=r"sklearn\.feature_selection\._univariate_selection",
)
warnings.filterwarnings(
    "ignore",
    message=r"invalid value encountered in divide",
    category=RuntimeWarning,
    module=r"sklearn\.feature_selection\._univariate_selection",
)


def _prepare_numeric_frame(X: pd.DataFrame) -> pd.DataFrame:
    return X.copy().fillna(X.median(numeric_only=True))


def _prepare_non_negative_frame(X: pd.DataFrame) -> pd.DataFrame:
    prepared = _prepare_numeric_frame(X)
    min_values = prepared.min(axis=0)
    shift = min_values.where(min_values < 0, other=0).abs()
    return prepared + shift


def _variance_scores(X: pd.DataFrame) -> np.ndarray:
    selector = VarianceThreshold()
    selector.fit(X)
    return selector.variances_


def compute_feature_scores(
    X: pd.DataFrame,
    y: pd.Series,
    selector_name: str,
    random_state: int,
) -> np.ndarray:
    if selector_name == "variance_threshold":
        return _variance_scores(_prepare_numeric_frame(X))
    if selector_name == "info_gain":
        prepared = _prepare_numeric_frame(X)
        return mutual_info_classif(prepared, y, random_state=random_state)
    if selector_name == "chi2":
        prepared = _prepare_non_negative_frame(X)
        return chi2(prepared, y)[0]
    raise ValueError(f"Unsupported selector: {selector_name}")


def select_top_k(feature_names: list[str], scores: np.ndarray, k: int) -> list[str]:
    ranked = (
        pd.DataFrame({"feature": feature_names, "score": scores})
        .fillna({"score": 0.0})
        .sort_values(by=["score", "feature"], ascending=[False, True])
    )
    return ranked.head(k)["feature"].tolist()


def average_jaccard(selected_feature_sets: list[set[str]]) -> float:
    if len(selected_feature_sets) < 2:
        return 1.0

    similarities = []
    for left, right in combinations(selected_feature_sets, 2):
        union = left | right
        if not union:
            similarities.append(1.0)
            continue
        similarities.append(len(left & right) / len(union))
    return float(np.mean(similarities)) if similarities else 1.0


def benchmark_selector(
    X: pd.DataFrame,
    y: pd.Series,
    selector_name: str,
    k: int,
    folds: int,
    random_state: int,
) -> tuple[dict, list[dict]]:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    feature_names = X.columns.tolist()
    feature_rows = []
    selected_sets = []
    fold_times = []
    score_summaries = []

    for fold_index, (train_idx, _test_idx) in enumerate(cv.split(X, y), start=1):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]

        start = perf_counter()
        scores = compute_feature_scores(X_train, y_train, selector_name, random_state=random_state)
        fold_times.append(perf_counter() - start)

        selected_features = select_top_k(feature_names, scores, k)
        selected_sets.append(set(selected_features))
        score_summaries.append(float(np.nanmean(scores)))

        for feature_name in selected_features:
            feature_rows.append(
                {
                    "selector": selector_name,
                    "k": k,
                    "fold": fold_index,
                    "feature": feature_name,
                }
            )

    result = {
        "selector": selector_name,
        "k": k,
        "mean_fit_seconds": float(np.mean(fold_times)),
        "mean_score": float(np.mean(score_summaries)),
        "stability_jaccard": average_jaccard(selected_sets),
        "mean_selected_feature_count": float(k),
    }
    return result, feature_rows
