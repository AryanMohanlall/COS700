import json
from pathlib import Path

import pandas as pd

from .data import load_dataset, maybe_sample
from .modeling import benchmark_configuration
from .reporting import (
    build_recommendation,
    chart_bar_svg,
    format_results_table,
    get_features_for_result,
    label_configuration,
    summarize_best_by_group,
    summarize_top_features,
    write_markdown_report,
)


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

    metadata["evaluated_rows"] = int(len(X))
    metadata["evaluated_class_balance"] = {
        "benign_0": int((y == 0).sum()),
        "ransomware_1": int((y == 1).sum()),
    }

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

    report_path = output_dir / f"{file_stem}_benchmark_report.md"
    leaderboard_chart_path = output_dir / f"{file_stem}_leaderboard_f1.svg"
    feature_chart_path = output_dir / f"{file_stem}_top_features.svg"

    top_chart_rows = results_df.head(8)
    chart_paths = {}
    if chart_bar_svg(
        labels=[label_configuration(row) for _, row in top_chart_rows.iterrows()],
        values=top_chart_rows["mean_f1"].tolist(),
        title="Top configurations by F1 score",
        output_path=leaderboard_chart_path,
    ):
        chart_paths["leaderboard"] = leaderboard_chart_path

    best_features_for_chart = get_features_for_result(frequency_df, results_df.iloc[0]).head(10)
    if chart_bar_svg(
        labels=best_features_for_chart["feature"].tolist(),
        values=best_features_for_chart["fold_selection_count"].astype(float).tolist(),
        title="Most recurring features in the best configuration",
        output_path=feature_chart_path,
        value_formatter=lambda value: f"{int(value)} fold(s)",
    ):
        chart_paths["features"] = feature_chart_path

    write_markdown_report(
        report_path=report_path,
        results_df=results_df,
        frequency_df=frequency_df,
        metadata=metadata,
        chart_paths=chart_paths,
    )

    emit("")
    emit("Run overview")
    emit(
        f"- dataset={csv_path.name}, rows={len(X)}, features={X.shape[1]}, "
        f"benign={int((y == 0).sum())}, ransomware={int((y == 1).sum())}"
    )
    emit(
        f"- selectors={', '.join(args.selectors)}, models={', '.join(args.classifiers)}, "
        f"folds={args.folds}, k_values={', '.join(str(k) for k in k_values)}"
    )
    emit("")
    emit("Quick recommendation")
    emit(build_recommendation(results_df).replace("**", ""))
    emit("")
    emit("Leaderboard")
    emit(format_results_table(results_df, limit=10))
    emit("")
    emit("Best per selector")
    emit(summarize_best_by_group(results_df, "selector"))
    emit("")
    emit("Best per model")
    emit(summarize_best_by_group(results_df, "classifier"))
    emit("")
    emit("Top recurring features for the best configuration")
    emit(summarize_top_features(frequency_df, results_df, top_n=10))
    emit("")
    emit("Readable report")
    emit(f"- Start here: {report_path}")
    if "leaderboard" in chart_paths:
        emit(f"- F1 chart: {leaderboard_chart_path}")
    if "features" in chart_paths:
        emit(f"- Stable-feature chart: {feature_chart_path}")
    emit("")
    emit(f"Saved results to {results_path}")
    emit(f"Saved feature frequencies to {features_path}")
    emit(f"Saved metadata to {metadata_path}")
    emit(f"Saved summary to {summary_path}")

    summary_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
