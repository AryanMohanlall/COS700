import json
from pathlib import Path

import pandas as pd

from .data import load_dataset, maybe_sample
from .modeling import benchmark_selector
from .reporting import build_recommendation, chart_bar_svg, label_configuration, summarize_top_features, write_markdown_report


def run_benchmark(args):
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
    metadata["selectors"] = args.selectors
    metadata["folds"] = args.folds

    max_k = X.shape[1]
    k_values = sorted({k for k in args.k_values if 0 < k <= max_k})
    if not k_values:
        k_values = [max_k]
    metadata["k_values"] = k_values

    result_rows = []
    selected_feature_rows = []

    for selector_name in args.selectors:
        for k in k_values:
            result, selected_features = benchmark_selector(
                X=X,
                y=y,
                selector_name=selector_name,
                k=k,
                folds=args.folds,
                random_state=args.random_state,
            )
            result_rows.append(result)
            selected_feature_rows.extend(selected_features)
            print(
                f"Finished selector={selector_name} "
                f"k={result['k']} stability={result['stability_jaccard']:.4f}"
            )

    results_df = pd.DataFrame(result_rows).sort_values(
        by=["fitness_score", "amount_of_features_chosen", "time"],
        ascending=[False, True, True],
    )
    ordered_columns = [
        "time",
        "amount_of_features_chosen",
        "model",
        "selection_algorithm",
        "fitness_score",
        "stability_jaccard",
        "k",
    ]
    results_df = results_df[ordered_columns]

    feature_df = pd.DataFrame(selected_feature_rows)
    if not feature_df.empty:
        feature_df = (
            feature_df.groupby(["selector", "k", "feature"])
            .size()
            .reset_index(name="fold_selection_count")
            .sort_values(
                by=["selector", "k", "fold_selection_count", "feature"],
                ascending=[True, True, False, True],
            )
        )
    else:
        feature_df = pd.DataFrame(columns=["selector", "k", "feature", "fold_selection_count"])

    file_stem = csv_path.stem.lower().replace(" ", "_")
    results_path = output_dir / f"{file_stem}_benchmark_results.csv"
    features_path = output_dir / f"{file_stem}_selected_feature_frequencies.csv"
    metadata_path = output_dir / f"{file_stem}_run_metadata.json"
    report_path = output_dir / f"{file_stem}_benchmark_report.md"
    svg_path = output_dir / f"{file_stem}_fitness_score.svg"

    results_df.to_csv(results_path, index=False)
    feature_df.to_csv(features_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_markdown_report(report_path, results_df, feature_df, metadata)
    chart_bar_svg(
        labels=[label_configuration(row) for _, row in results_df.head(10).iterrows()],
        values=results_df.head(10)["fitness_score"].tolist(),
        title="Top configurations by fitness score (F1)",
        output_path=svg_path,
    )

    print("")
    print("Run overview")
    print(
        f"- dataset={csv_path.name}, rows={len(X)}, features={X.shape[1]}, "
        f"class_0={int((y == 0).sum())}, class_1={int((y == 1).sum())}"
    )
    print(build_recommendation(results_df))
    print(summarize_top_features(feature_df, results_df))
    print(f"Saved results to {results_path}")
    print(f"Saved feature frequencies to {features_path}")
    print(f"Saved metadata to {metadata_path}")
    print(f"Saved report to {report_path}")
    print(f"Saved chart to {svg_path}")
