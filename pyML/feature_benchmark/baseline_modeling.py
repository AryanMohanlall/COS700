import json
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import ParameterSampler, train_test_split
from sklearn.pipeline import Pipeline

from .modeling import XGBClassifier


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    validation_size: float,
    test_size: float,
    random_state: int,
):
    if not 0 < validation_size < 1:
        raise ValueError("--validation-size must be between 0 and 1.")
    if not 0 < test_size < 1:
        raise ValueError("--test-size must be between 0 and 1.")
    if validation_size + test_size >= 1:
        raise ValueError("The sum of --validation-size and --test-size must be less than 1.")

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    validation_share_of_train_val = validation_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=validation_share_of_train_val,
        stratify=y_train_val,
        random_state=random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def _random_forest_space():
    return {
        "n_estimators": [200, 300, 500, 700],
        "max_depth": [None, 8, 12, 16, 24],
        "min_samples_split": [2, 4, 8, 12],
        "min_samples_leaf": [1, 2, 4, 6],
        "max_features": ["sqrt", "log2", 0.5, 0.75],
    }


def _xgboost_space():
    return {
        "n_estimators": [200, 300, 500, 700],
        "max_depth": [3, 4, 6, 8],
        "learning_rate": [0.03, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "reg_alpha": [0.0, 0.1, 0.5],
        "reg_lambda": [1.0, 1.5, 2.0],
    }


def _parameter_space(model_name: str):
    if model_name == "random_forest":
        return _random_forest_space()
    if model_name == "xgboost":
        if XGBClassifier is None:
            raise SystemExit(
                "xgboost is not installed. Install it with:\n"
                "  pyML\\.venv\\Scripts\\python.exe -m pip install xgboost"
            )
        return _xgboost_space()
    raise ValueError(f"Unsupported model: {model_name}")


def _make_estimator(model_name: str, params: dict, random_state: int):
    if model_name == "random_forest":
        model = RandomForestClassifier(
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
            **params,
        )
    elif model_name == "xgboost":
        if XGBClassifier is None:
            raise SystemExit(
                "xgboost is not installed. Install it with:\n"
                "  pyML\\.venv\\Scripts\\python.exe -m pip install xgboost"
            )
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            importance_type="gain",
            n_jobs=1,
            random_state=random_state,
            **params,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


def _classification_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def _measure_prediction_latency(estimator: Pipeline, X_test: pd.DataFrame, repeats: int) -> dict:
    repeats = max(1, int(repeats))
    estimator.predict(X_test)

    durations = []
    for _ in range(repeats):
        start = perf_counter()
        estimator.predict(X_test)
        durations.append(perf_counter() - start)

    average_seconds = float(np.mean(durations))
    row_count = max(1, len(X_test))
    return {
        "prediction_latency_seconds": average_seconds,
        "prediction_latency_ms": average_seconds * 1000.0,
        "prediction_latency_per_row_ms": (average_seconds / row_count) * 1000.0,
    }


def _extract_feature_importance(estimator: Pipeline, feature_names: list[str], model_name: str) -> pd.DataFrame:
    fitted_model = estimator.named_steps["model"]
    if not hasattr(fitted_model, "feature_importances_"):
        raise ValueError(f"The fitted model '{model_name}' does not expose feature_importances_.")

    importances = np.asarray(fitted_model.feature_importances_, dtype=float)
    importance_df = pd.DataFrame(
        {
            "feature_selection_algorithm": "full_feature_set",
            "feature": feature_names,
            "importance_score": importances,
            "model": model_name,
        }
    ).sort_values(by=["importance_score", "feature"], ascending=[False, True]).reset_index(drop=True)
    importance_df["rank"] = np.arange(1, len(importance_df) + 1)
    importance_df["importance_source"] = "feature_importances_"
    return importance_df[
        [
            "model",
            "feature_selection_algorithm",
            "rank",
            "feature",
            "importance_score",
            "importance_source",
        ]
    ]


def tune_and_evaluate_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_search_iterations: int,
    latency_repeats: int,
    random_state: int,
):
    candidate_params = list(
        ParameterSampler(
            _parameter_space(model_name),
            n_iter=max(1, random_search_iterations),
            random_state=random_state,
        )
    )

    best = None
    search_rows = []

    for iteration, params in enumerate(candidate_params, start=1):
        estimator = _make_estimator(model_name, params, random_state=random_state)
        start = perf_counter()
        estimator.fit(X_train, y_train)
        fit_seconds = perf_counter() - start
        val_predictions = estimator.predict(X_val)
        metrics = _classification_metrics(y_val, val_predictions)

        row = {
            "model": model_name,
            "feature_selection_algorithm": "full_feature_set",
            "search_iteration": iteration,
            "fit_seconds": float(fit_seconds),
            "validation_f1_score": metrics["f1_score"],
            "validation_accuracy": metrics["accuracy"],
            "validation_precision": metrics["precision"],
            "validation_recall": metrics["recall"],
            "params_json": json.dumps(params, sort_keys=True),
        }
        search_rows.append(row)

        candidate = {
            "estimator": estimator,
            "params": params,
            "fit_seconds": fit_seconds,
            "validation_metrics": metrics,
        }
        if best is None:
            best = candidate
            continue

        if metrics["f1_score"] > best["validation_metrics"]["f1_score"]:
            best = candidate
            continue
        if (
            metrics["f1_score"] == best["validation_metrics"]["f1_score"]
            and fit_seconds < best["fit_seconds"]
        ):
            best = candidate

    best_estimator = best["estimator"]
    combined_X = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    combined_y = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

    final_estimator = _make_estimator(model_name, best["params"], random_state=random_state)
    train_start = perf_counter()
    final_estimator.fit(combined_X, combined_y)
    final_fit_seconds = perf_counter() - train_start

    test_predictions = final_estimator.predict(X_test)
    test_metrics = _classification_metrics(y_test, test_predictions)
    latency = _measure_prediction_latency(final_estimator, X_test, repeats=latency_repeats)
    importance_df = _extract_feature_importance(final_estimator, combined_X.columns.tolist(), model_name)

    summary = {
        "model": model_name,
        "feature_selection_algorithm": "full_feature_set",
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1_score": test_metrics["f1_score"],
        "validation_f1_score": best["validation_metrics"]["f1_score"],
        "validation_accuracy": best["validation_metrics"]["accuracy"],
        "validation_precision": best["validation_metrics"]["precision"],
        "validation_recall": best["validation_metrics"]["recall"],
        "fit_seconds": float(final_fit_seconds),
        "prediction_latency_ms": latency["prediction_latency_ms"],
        "prediction_latency_per_row_ms": latency["prediction_latency_per_row_ms"],
        "train_rows": int(len(combined_X)),
        "test_rows": int(len(X_test)),
        "feature_count": int(combined_X.shape[1]),
        "random_search_iterations": int(len(candidate_params)),
        "best_params_json": json.dumps(best["params"], sort_keys=True),
    }
    return summary, pd.DataFrame(search_rows), importance_df
