import html
from pathlib import Path

import numpy as np
import pandas as pd


def format_results_table(df: pd.DataFrame, limit: int = 10) -> str:
    display_df = df.head(limit).copy()
    if display_df.empty:
        return "No benchmark rows to display."

    display_df = display_df.rename(
        columns={
            "selector": "selector",
            "classifier": "model",
            "k": "k",
            "mean_f1": "f1",
            "std_f1": "f1_std",
            "mean_precision": "precision",
            "mean_recall": "recall",
            "mean_roc_auc": "roc_auc",
            "mean_pr_auc": "pr_auc",
            "mean_fit_seconds": "fit_s",
            "mean_selected_feature_count": "features",
        }
    )

    for col in ["f1", "f1_std", "precision", "recall", "roc_auc", "pr_auc", "fit_s", "features"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(
                lambda value: f"{value:.4f}" if isinstance(value, (int, float, np.floating)) else value
            )

    return display_df[
        ["selector", "model", "k", "f1", "f1_std", "precision", "recall", "pr_auc", "fit_s", "features"]
    ].to_string(index=False)


def label_configuration(row: pd.Series) -> str:
    return f"{row['selector']} + {row['classifier']} (k={row['k']})"


def fmt_float(value, digits: int = 4) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def fmt_percent(value) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def markdown_cell(value) -> str:
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def markdown_table(df: pd.DataFrame, columns: list[tuple[str, str]], limit: int | None = None) -> str:
    if df.empty:
        return "_No rows available._"

    display_df = df.head(limit).copy() if limit else df.copy()
    headers = [markdown_cell(heading) for _, heading in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for _, row in display_df.iterrows():
        values = []
        for column, _ in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                value = fmt_float(value)
            values.append(markdown_cell(value))
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def chart_bar_svg(
    labels: list[str],
    values: list[float],
    title: str,
    output_path: Path,
    value_formatter=fmt_float,
) -> bool:
    if not labels or not values:
        return False

    width = 920
    row_height = 34
    top = 64
    left = 290
    right = 120
    bottom = 34
    height = top + bottom + row_height * len(labels)
    max_value = max(values) if max(values) > 0 else 1
    bar_width = width - left - right

    palette = ["#2166ac", "#b2182b", "#1b7837", "#762a83", "#9970ab", "#5aae61"]
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
                f'<text x="24" y="{y + 21}" font-family="Arial, sans-serif" font-size="14" fill="#222222">{html.escape(label[:42])}</text>',
                f'<rect x="{left}" y="{y + 5}" width="{scaled:.2f}" height="21" rx="4" fill="{fill}"/>',
                f'<text x="{left + scaled + 10}" y="{y + 21}" font-family="Arial, sans-serif" font-size="14" fill="#111111">{html.escape(value_formatter(value))}</text>',
            ]
        )

    svg.append("</svg>")
    output_path.write_text("\n".join(svg) + "\n", encoding="utf-8")
    return True


def get_features_for_result(frequency_df: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    if frequency_df.empty:
        return frequency_df

    matching_k = str(row["k"])
    if matching_k == "all":
        matching_k = str(int(round(row["mean_selected_feature_count"])))

    return frequency_df[
        (frequency_df["selector"] == row["selector"])
        & (frequency_df["classifier"] == row["classifier"])
        & (frequency_df["k"].astype(str) == matching_k)
    ].sort_values(by=["fold_selection_count", "feature"], ascending=[False, True])


def summarize_best_by_group(results_df: pd.DataFrame, group_column: str) -> str:
    if results_df.empty:
        return "No results available."

    best_rows = (
        results_df.sort_values(
            by=["mean_f1", "mean_selected_feature_count", "mean_fit_seconds"],
            ascending=[False, True, True],
        )
        .groupby(group_column, as_index=False)
        .first()
        .sort_values(by=["mean_f1", "mean_selected_feature_count"], ascending=[False, True])
    )

    lines = []
    for _, row in best_rows.iterrows():
        lines.append(
            f"- {row[group_column]}: selector={row['selector']}, model={row['classifier']}, "
            f"k={row['k']}, f1={row['mean_f1']:.4f}, features={row['mean_selected_feature_count']:.1f}, "
            f"fit_s={row['mean_fit_seconds']:.4f}"
        )
    return "\n".join(lines)


def summarize_top_features(frequency_df: pd.DataFrame, results_df: pd.DataFrame, top_n: int = 10) -> str:
    if frequency_df.empty or results_df.empty:
        return "No selected-feature frequency data available."

    best_row = results_df.iloc[0]
    best_features = get_features_for_result(frequency_df, best_row)

    if best_features.empty:
        return "No selected features recorded for the best configuration."

    lines = [
        f"Best configuration: selector={best_row['selector']}, model={best_row['classifier']}, "
        f"k={best_row['k']}, f1={best_row['mean_f1']:.4f}, "
        f"features={best_row['mean_selected_feature_count']:.1f}"
    ]
    for _, row in best_features.head(top_n).iterrows():
        lines.append(f"- {row['feature']}: selected in {int(row['fold_selection_count'])} fold(s)")
    return "\n".join(lines)


def build_recommendation(results_df: pd.DataFrame) -> str:
    if results_df.empty:
        return "No recommendation could be generated because the benchmark produced no rows."

    best = results_df.iloc[0]
    compact_pool = results_df[
        results_df["mean_f1"] >= best["mean_f1"] - 0.002
    ].sort_values(by=["mean_selected_feature_count", "mean_f1", "mean_fit_seconds"], ascending=[True, False, True])
    compact = compact_pool.iloc[0] if not compact_pool.empty else best

    if label_configuration(best) == label_configuration(compact):
        return (
            f"Use **{label_configuration(best)}** as the current front-runner. "
            f"It reached F1={fmt_float(best['mean_f1'])}, recall={fmt_percent(best['mean_recall'])}, "
            f"and used about {fmt_float(best['mean_selected_feature_count'], 1)} features."
        )

    return (
        f"Highest F1: **{label_configuration(best)}** with F1={fmt_float(best['mean_f1'])}. "
        f"For a smaller feature set within 0.002 F1 of the best score, consider "
        f"**{label_configuration(compact)}** with F1={fmt_float(compact['mean_f1'])} and "
        f"about {fmt_float(compact['mean_selected_feature_count'], 1)} features."
    )


def write_markdown_report(
    report_path: Path,
    results_df: pd.DataFrame,
    frequency_df: pd.DataFrame,
    metadata: dict,
    chart_paths: dict[str, Path],
):
    best = results_df.iloc[0] if not results_df.empty else None
    baseline_pool = results_df[results_df["selector"] == "baseline"] if not results_df.empty else pd.DataFrame()
    baseline = baseline_pool.iloc[0] if not baseline_pool.empty else None
    best_features = get_features_for_result(frequency_df, best).head(12) if best is not None else pd.DataFrame()

    top_results = results_df.head(10).copy()
    if not top_results.empty:
        top_results["configuration"] = top_results.apply(label_configuration, axis=1)
        top_results["F1"] = top_results["mean_f1"].map(fmt_float)
        top_results["Recall"] = top_results["mean_recall"].map(fmt_percent)
        top_results["Precision"] = top_results["mean_precision"].map(fmt_percent)
        top_results["PR-AUC"] = top_results["mean_pr_auc"].map(fmt_float)
        top_results["Fit seconds"] = top_results["mean_fit_seconds"].map(lambda value: fmt_float(value, 2))
        top_results["Feature count"] = top_results["mean_selected_feature_count"].map(lambda value: fmt_float(value, 1))

    if not best_features.empty:
        best_features = best_features.copy()
        folds = max(1, int(metadata["folds"]))
        best_features["Stability"] = best_features["fold_selection_count"].map(
            lambda count: f"{int(count)}/{folds} folds"
        )

    lines = [
        f"# Feature Selection Benchmark Report: {Path(metadata['csv_path']).name}",
        "",
        "## Plain-English Summary",
        "",
        build_recommendation(results_df),
        "",
        "This benchmark compares feature-selection methods for ransomware detection. "
        "Higher F1 means a better balance between catching ransomware and avoiding false alarms. "
        "Recall is especially important when missing ransomware is costly. PR-AUC is useful when the classes are imbalanced.",
        "",
        "## Dataset And Run",
        "",
        f"- Rows evaluated: **{metadata.get('evaluated_rows', metadata['rows']):,}**",
        f"- Usable numeric features: **{metadata['usable_numeric_features']:,}**",
        f"- Benign samples: **{metadata.get('evaluated_class_balance', metadata['class_balance'])['benign_0']:,}**",
        f"- Ransomware samples: **{metadata.get('evaluated_class_balance', metadata['class_balance'])['ransomware_1']:,}**",
        f"- Cross-validation folds: **{metadata['folds']}**",
        f"- Selectors: **{', '.join(metadata['selectors'])}**",
        f"- Classifiers: **{', '.join(metadata['classifiers'])}**",
        f"- k values tested: **{', '.join(str(k) for k in metadata['k_values'])}**",
        "",
        "## Leaderboard",
        "",
    ]

    if "leaderboard" in chart_paths:
        lines.extend([f"![Top configurations]({chart_paths['leaderboard'].name})", ""])

    lines.extend(
        [
            markdown_table(
                top_results,
                [
                    ("configuration", "Configuration"),
                    ("F1", "F1"),
                    ("Recall", "Recall"),
                    ("Precision", "Precision"),
                    ("PR-AUC", "PR-AUC"),
                    ("Feature count", "Features"),
                    ("Fit seconds", "Fit seconds"),
                ],
            ),
            "",
        ]
    )

    if best is not None and baseline is not None:
        feature_reduction = 1 - (best["mean_selected_feature_count"] / baseline["mean_selected_feature_count"])
        f1_delta = best["mean_f1"] - baseline["mean_f1"]
        lines.extend(
            [
                "## Best Vs. No Feature Selection",
                "",
                f"- Best configuration: **{label_configuration(best)}**",
                f"- Best baseline: **{label_configuration(baseline)}**",
                f"- F1 change vs baseline: **{f1_delta:+.4f}**",
                f"- Feature reduction vs baseline: **{feature_reduction:.1%}**",
                "",
            ]
        )

    lines.extend(["## Most Stable Features In The Best Configuration", ""])

    if "features" in chart_paths:
        lines.extend([f"![Recurring selected features]({chart_paths['features'].name})", ""])

    lines.extend(
        [
            markdown_table(
                best_features,
                [
                    ("feature", "Feature"),
                    ("Stability", "Selected in"),
                ],
            ),
            "",
            "## How To Read This",
            "",
            "- Prefer a selector that keeps F1 and recall high while using fewer features than the baseline.",
            "- Features that appear in every fold are more stable and easier to defend in a research write-up.",
            "- If two rows have nearly identical F1, the row with fewer features is usually easier to explain.",
            "",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
