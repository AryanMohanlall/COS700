import json
from pathlib import Path

import pandas as pd

from .reporting import write_results_xlsx


def write_baseline_report(
    report_path: Path,
    metrics_df: pd.DataFrame,
    feature_importance_tables: dict[str, pd.DataFrame],
    metadata: dict,
    top_features: int,
):
    lines = [
        f"# Baseline Model Report: {Path(metadata['csv_path']).name}",
        "",
        "## Dataset",
        "",
        f"- Rows evaluated: **{metadata['evaluated_rows']:,}**",
        f"- Usable numeric features: **{metadata['usable_numeric_features']:,}**",
        f"- Class 0 count: **{metadata['evaluated_class_balance']['class_0']:,}**",
        f"- Class 1 count: **{metadata['evaluated_class_balance']['class_1']:,}**",
        f"- Validation size: **{metadata['validation_size']:.2f}**",
        f"- Test size: **{metadata['test_size']:.2f}**",
        f"- Random-search iterations per model: **{metadata['random_search_iterations']}**",
        "",
        "## Model Metrics",
        "",
    ]

    if metrics_df.empty:
        lines.append("_No model rows were generated._")
    else:
        lines.extend(
            [
                "| Model | Feature selection | Accuracy | Precision | Recall | F1 score | Fit seconds | Latency ms | Per-row ms |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for _, row in metrics_df.iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["model"]),
                        str(row["feature_selection_algorithm"]),
                        f"{row['accuracy']:.4f}",
                        f"{row['precision']:.4f}",
                        f"{row['recall']:.4f}",
                        f"{row['f1_score']:.4f}",
                        f"{row['fit_seconds']:.2f}",
                        f"{row['prediction_latency_ms']:.3f}",
                        f"{row['prediction_latency_per_row_ms']:.6f}",
                    ]
                )
                + " |"
            )

    lines.extend(["", "## Best Hyperparameters", ""])
    for _, row in metrics_df.iterrows():
        lines.append(f"### {row['model']}")
        lines.append("")
        lines.append(f"`{row['best_params_json']}`")
        lines.append("")

    lines.extend(["## Top Features", ""])
    for model_name, importance_df in feature_importance_tables.items():
        lines.append(f"### {model_name}")
        lines.append("")
        if importance_df.empty:
            lines.append("_No feature importance data available._")
            lines.append("")
            continue
        lines.extend(
            [
                "| Feature selection | Rank | Feature | Importance score |",
                "| --- | --- | --- | --- |",
            ]
        )
        for _, row in importance_df.head(top_features).iterrows():
            lines.append(
                f"| {row['feature_selection_algorithm']} | {int(row['rank'])} | {row['feature']} | {float(row['importance_score']):.6f} |"
            )
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def write_metrics_outputs(output_dir: Path, file_stem: str, metrics_df: pd.DataFrame):
    csv_path = output_dir / f"{file_stem}_baseline_metrics.csv"
    xlsx_path = output_dir / f"{file_stem}_baseline_metrics.xlsx"
    metrics_df.to_csv(csv_path, index=False)
    write_results_xlsx(xlsx_path, metrics_df)
    return csv_path, xlsx_path


def write_search_outputs(output_dir: Path, file_stem: str, search_df: pd.DataFrame):
    csv_path = output_dir / f"{file_stem}_random_search_trials.csv"
    search_df.to_csv(csv_path, index=False)
    return csv_path


def write_feature_importance_outputs(
    output_dir: Path,
    file_stem: str,
    feature_importance_tables: dict[str, pd.DataFrame],
):
    output_paths = {}
    for model_name, importance_df in feature_importance_tables.items():
        csv_path = output_dir / f"{file_stem}_{model_name}_feature_importance.csv"
        importance_df.to_csv(csv_path, index=False)
        output_paths[model_name] = csv_path
    return output_paths


def write_metadata_output(output_dir: Path, file_stem: str, metadata: dict):
    metadata_path = output_dir / f"{file_stem}_baseline_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path
