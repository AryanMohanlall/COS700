from time import perf_counter
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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
