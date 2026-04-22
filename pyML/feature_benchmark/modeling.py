from itertools import combinations
import os
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector, VarianceThreshold, chi2, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

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


def _prepare_numeric_frame(X: pd.DataFrame) -> pd.DataFrame:
    return X.copy().fillna(X.median(numeric_only=True))


def _prepare_non_negative_frame(X: pd.DataFrame) -> pd.DataFrame:
    prepared = _prepare_numeric_frame(X)
    min_values = prepared.min(axis=0)
    shift = min_values.where(min_values < 0, other=0).abs()
    return prepared + shift


def _build_selection_estimator(random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _build_evaluation_estimator(random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=random_state,
                ),
            ),
        ]
    )


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


def wrapper_candidate_columns(
    X: pd.DataFrame,
    y: pd.Series,
    k: int,
    random_state: int,
    candidate_limit: int = 20,
) -> list[str]:
    limit = min(len(X.columns), max(k + 1, candidate_limit))
    info_gain_scores = compute_feature_scores(X, y, "info_gain", random_state=random_state)
    return select_top_k(X.columns.tolist(), info_gain_scores, limit)


def select_via_estimator(
    X: pd.DataFrame,
    y: pd.Series,
    selector_name: str,
    k: int,
    random_state: int,
) -> list[str]:
    if k >= len(X.columns):
        return X.columns.tolist()

    estimator = _build_selection_estimator(random_state=random_state)
    candidate_columns = wrapper_candidate_columns(X, y, k, random_state=random_state)
    if k >= len(candidate_columns):
        return candidate_columns[:k]
    X_candidates = X[candidate_columns]

    if selector_name == "forward_selection":
        selector = SequentialFeatureSelector(
            estimator=estimator,
            n_features_to_select=k,
            direction="forward",
            scoring="f1",
            cv=2,
            n_jobs=1,
        )
        selector.fit(X_candidates, y)
        support = selector.get_support()
        return [feature for feature, keep in zip(X_candidates.columns.tolist(), support) if keep]

    if selector_name == "backward_elimination":
        selector = SequentialFeatureSelector(
            estimator=estimator,
            n_features_to_select=k,
            direction="backward",
            scoring="f1",
            cv=2,
            n_jobs=1,
        )
        selector.fit(X_candidates, y)
        support = selector.get_support()
        return [feature for feature, keep in zip(X_candidates.columns.tolist(), support) if keep]

    if selector_name == "l1":
        l1_estimator = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            class_weight="balanced",
            max_iter=2000,
            random_state=random_state,
        )
        pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "selector",
                    SelectFromModel(
                        estimator=l1_estimator,
                        threshold=-np.inf,
                        max_features=k,
                    ),
                ),
            ]
        )
        pipeline.fit(X, y)
        support = pipeline.named_steps["selector"].get_support()
        return [feature for feature, keep in zip(X.columns.tolist(), support) if keep]

    raise ValueError(f"Unsupported model-based selector: {selector_name}")


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
    fold_f1_scores = []

    for fold_index, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        start = perf_counter()
        if selector_name in {"info_gain", "variance_threshold", "chi2"}:
            scores = compute_feature_scores(X_train, y_train, selector_name, random_state=random_state)
            selected_features = select_top_k(feature_names, scores, k)
            selection_model = "n/a"
        else:
            selected_features = select_via_estimator(
                X_train,
                y_train,
                selector_name,
                k,
                random_state=random_state,
            )
            selection_model = "logistic_regression"
        fold_times.append(perf_counter() - start)

        selected_sets.append(set(selected_features))

        evaluator = _build_evaluation_estimator(random_state=random_state)
        evaluator.fit(X_train[selected_features], y_train)
        y_pred = evaluator.predict(X_test[selected_features])
        fold_f1_scores.append(f1_score(y_test, y_pred, zero_division=0))

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
        "selection_algorithm": selector_name,
        "model": selection_model,
        "time": float(np.mean(fold_times)),
        "amount_of_features_chosen": float(k),
        "fitness_score": float(np.mean(fold_f1_scores)),
        "stability_jaccard": average_jaccard(selected_sets),
        "k": k,
    }
    return result, feature_rows
