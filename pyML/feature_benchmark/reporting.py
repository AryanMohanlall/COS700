from pathlib import Path

import pandas as pd


def label_configuration(row: pd.Series) -> str:
    return f"{row['selector']} (k={row['k']})"


def build_recommendation(results_df: pd.DataFrame) -> str:
    if results_df.empty:
        return "No recommendation available because the benchmark produced no rows."

    best = results_df.iloc[0]
    compact_pool = results_df[
        results_df["stability_jaccard"] >= best["stability_jaccard"] - 0.02
    ].sort_values(
        by=["mean_selected_feature_count", "stability_jaccard", "mean_fit_seconds"],
        ascending=[True, False, True],
    )
    compact = compact_pool.iloc[0]

    if label_configuration(best) == label_configuration(compact):
        return (
            f"Current front-runner: {label_configuration(best)} with "
            f"stability={best['stability_jaccard']:.4f}, mean_score={best['mean_score']:.4f}, "
            f"and about {best['mean_selected_feature_count']:.1f} selected features."
        )

    return (
        f"Highest stability: {label_configuration(best)} at {best['stability_jaccard']:.4f}. "
        f"If you want a leaner subset with nearly the same stability, use "
        f"{label_configuration(compact)} at {compact['stability_jaccard']:.4f} with "
        f"about {compact['mean_selected_feature_count']:.1f} features."
    )


def summarize_top_features(feature_df: pd.DataFrame, results_df: pd.DataFrame, top_n: int = 10) -> str:
    if feature_df.empty or results_df.empty:
        return "No selected-feature frequency data available."

    best = results_df.iloc[0]
    filtered = feature_df[
        (feature_df["selector"] == best["selector"])
        & (feature_df["k"].astype(str) == str(best["k"]))
    ].sort_values(by=["fold_selection_count", "feature"], ascending=[False, True])

    if filtered.empty:
        return "No recurring selected features were recorded for the best configuration."

    lines = [f"Best configuration: {label_configuration(best)}"]
    for _, row in filtered.head(top_n).iterrows():
        lines.append(f"- {row['feature']}: selected in {int(row['fold_selection_count'])} fold(s)")
    return "\n".join(lines)


def write_markdown_report(
    report_path: Path,
    results_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    metadata: dict,
):
    lines = [
        f"# Feature Selection Benchmark Report: {Path(metadata['csv_path']).name}",
        "",
        "## Summary",
        "",
        build_recommendation(results_df),
        "",
        "## Dataset",
        "",
        f"- Rows evaluated: **{metadata['evaluated_rows']:,}**",
        f"- Usable numeric features: **{metadata['usable_numeric_features']:,}**",
        f"- Class 0 count: **{metadata['evaluated_class_balance']['class_0']:,}**",
        f"- Class 1 count: **{metadata['evaluated_class_balance']['class_1']:,}**",
        f"- Folds: **{metadata['folds']}**",
        f"- Selectors: **{', '.join(metadata['selectors'])}**",
        f"- k values: **{', '.join(str(k) for k in metadata['k_values'])}**",
        "",
        "## Top Results",
        "",
    ]

    if results_df.empty:
        lines.append("_No benchmark rows were generated._")
    else:
        top_rows = results_df.head(10)
        lines.extend(
            [
                "| Configuration | Stability | Mean score | Features | Fit seconds |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for _, row in top_rows.iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        label_configuration(row),
                        f"{row['stability_jaccard']:.4f}",
                        f"{row['mean_score']:.4f}",
                        f"{row['mean_selected_feature_count']:.1f}",
                        f"{row['mean_fit_seconds']:.2f}",
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Stable Features",
            "",
            summarize_top_features(feature_df, results_df),
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
