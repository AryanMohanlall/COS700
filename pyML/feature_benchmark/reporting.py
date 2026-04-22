from pathlib import Path
import html

import pandas as pd


def _format_score(value) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.4f}"


def label_configuration(row: pd.Series) -> str:
    return f"{row['selection_algorithm']} (k={row['k']})"


def build_recommendation(results_df: pd.DataFrame) -> str:
    if results_df.empty:
        return "No recommendation available because the benchmark produced no rows."

    best = results_df.iloc[0]
    compact_pool = results_df[
        results_df["fitness_score"] >= best["fitness_score"] - 0.02
    ].sort_values(
        by=["amount_of_features_chosen", "fitness_score", "time"],
        ascending=[True, False, True],
    )
    compact = compact_pool.iloc[0]

    if label_configuration(best) == label_configuration(compact):
        return (
            f"Current front-runner: {label_configuration(best)} with "
            f"fitness_score={_format_score(best['fitness_score'])}, stability={best['stability_jaccard']:.4f}, "
            f"and about {best['amount_of_features_chosen']:.1f} selected features."
        )

    return (
        f"Highest fitness_score: {label_configuration(best)} at {_format_score(best['fitness_score'])}. "
        f"If you want a leaner subset with nearly the same fitness_score, use "
        f"{label_configuration(compact)} at {_format_score(compact['fitness_score'])} with "
        f"about {compact['amount_of_features_chosen']:.1f} features."
    )


def summarize_top_features(feature_df: pd.DataFrame, results_df: pd.DataFrame, top_n: int = 10) -> str:
    if feature_df.empty or results_df.empty:
        return "No selected-feature frequency data available."

    best = results_df.iloc[0]
    filtered = feature_df[
        (feature_df["selector"] == best["selection_algorithm"])
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
                "| Configuration | Stability | Fitness score | Features | Time |",
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
                        _format_score(row["fitness_score"]),
                        f"{row['amount_of_features_chosen']:.1f}",
                        f"{row['time']:.2f}",
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


def chart_bar_svg(labels: list[str], values: list[float], title: str, output_path: Path) -> bool:
    if not labels or not values:
        return False

    width = 920
    row_height = 34
    top = 64
    left = 300
    right = 120
    bottom = 34
    height = top + bottom + row_height * len(labels)
    max_value = max(values) if max(values) > 0 else 1
    bar_width = width - left - right
    palette = ["#1f6feb", "#0f9d58", "#d97706", "#c2410c", "#7c3aed", "#b91c1c"]

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="34" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111111">{html.escape(title)}</text>',
    ]

    for index, (label, value) in enumerate(zip(labels, values)):
        y = top + index * row_height
        fill = palette[index % len(palette)]
        scaled = max(2, (value / max_value) * bar_width)
        svg.extend(
            [
                f'<text x="24" y="{y + 21}" font-family="Arial, sans-serif" font-size="14" fill="#222222">{html.escape(label[:44])}</text>',
                f'<rect x="{left}" y="{y + 5}" width="{scaled:.2f}" height="21" rx="4" fill="{fill}"/>',
                f'<text x="{left + scaled + 10}" y="{y + 21}" font-family="Arial, sans-serif" font-size="14" fill="#111111">{value:.4f}</text>',
            ]
        )

    svg.append("</svg>")
    output_path.write_text("\n".join(svg) + "\n", encoding="utf-8")
    return True
