from pathlib import Path

import pandas as pd

from .baseline_modeling import split_dataset, tune_and_evaluate_model
from .baseline_reporting import (
    write_baseline_report,
    write_feature_importance_outputs,
    write_metadata_output,
    write_metrics_outputs,
    write_search_outputs,
)
from .data import load_dataset, maybe_sample


def run_baseline_models(args):
    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, metadata = load_dataset(csv_path, args.target_column, args.positive_label)
    X, y, sampled_rows = maybe_sample(X, y, args.sample_size, args.random_state)
    if sampled_rows is not None:
        metadata["sampled_rows"] = sampled_rows

    metadata["evaluated_rows"] = int(len(X))
    metadata["evaluated_class_balance"] = {
        "class_0": int((y == 0).sum()),
        "class_1": int((y == 1).sum()),
    }
    metadata["models"] = args.models
    metadata["validation_size"] = float(args.validation_size)
    metadata["test_size"] = float(args.test_size)
    metadata["random_search_iterations"] = int(args.random_search_iterations)
    metadata["latency_repeats"] = int(args.latency_repeats)

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X,
        y,
        validation_size=args.validation_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    metadata["split_rows"] = {
        "train": int(len(X_train)),
        "validation": int(len(X_val)),
        "test": int(len(X_test)),
    }

    metric_rows = []
    search_frames = []
    feature_importance_tables = {}

    for model_name in args.models:
        summary, search_df, importance_df = tune_and_evaluate_model(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            random_search_iterations=args.random_search_iterations,
            latency_repeats=args.latency_repeats,
            random_state=args.random_state,
        )
        metric_rows.append(summary)
        search_frames.append(search_df)
        feature_importance_tables[model_name] = importance_df
        print(
            f"Finished model={model_name} "
            f"f1={summary['f1_score']:.4f} "
            f"latency_ms={summary['prediction_latency_ms']:.3f}"
        )

    metrics_df = pd.DataFrame(metric_rows).sort_values(
        by=["f1_score", "recall", "precision", "accuracy"],
        ascending=[False, False, False, False],
    )
    ordered_metric_columns = [
        "model",
        "feature_selection_algorithm",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "validation_accuracy",
        "validation_precision",
        "validation_recall",
        "validation_f1_score",
        "fit_seconds",
        "prediction_latency_ms",
        "prediction_latency_per_row_ms",
        "train_rows",
        "test_rows",
        "feature_count",
        "random_search_iterations",
        "best_params_json",
    ]
    metrics_df = metrics_df[ordered_metric_columns]
    search_df = pd.concat(search_frames, ignore_index=True) if search_frames else pd.DataFrame()

    file_stem = csv_path.stem.lower().replace(" ", "_")
    metrics_path, metrics_xlsx_path = write_metrics_outputs(output_dir, file_stem, metrics_df)
    search_path = write_search_outputs(output_dir, file_stem, search_df)
    importance_paths = write_feature_importance_outputs(output_dir, file_stem, feature_importance_tables)
    metadata_path = write_metadata_output(output_dir, file_stem, metadata)
    report_path = output_dir / f"{file_stem}_baseline_report.md"
    write_baseline_report(report_path, metrics_df, feature_importance_tables, metadata, args.top_features)

    print("")
    print("Run overview")
    print(
        f"- dataset={csv_path.name}, rows={len(X)}, features={X.shape[1]}, "
        f"class_0={int((y == 0).sum())}, class_1={int((y == 1).sum())}"
    )
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved Excel table to {metrics_xlsx_path}")
    print(f"Saved random-search trials to {search_path}")
    for model_name, importance_path in importance_paths.items():
        print(f"Saved {model_name} feature importance to {importance_path}")
    print(f"Saved metadata to {metadata_path}")
    print(f"Saved report to {report_path}")
