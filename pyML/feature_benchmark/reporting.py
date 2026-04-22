from pathlib import Path
import html
from io import BytesIO
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd


def _format_score(value) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.4f}"


def label_configuration(row: pd.Series) -> str:
    return f"{row['selection_algorithm']} + {row['model']} (k={row['k']})"


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
        & (feature_df["model"] == best["model"])
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
        f"- Models: **{', '.join(metadata['models'])}**",
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


def _excel_column_name(index: int) -> str:
    name = ""
    current = index
    while current > 0:
        current, remainder = divmod(current - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _excel_cell(value, row_index: int, col_index: int) -> str:
    cell_ref = f"{_excel_column_name(col_index)}{row_index}"
    style = "1" if row_index == 1 else "0"

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return f'<c r="{cell_ref}" s="{style}"/>'

    if isinstance(value, bool):
        numeric = 1 if value else 0
        return f'<c r="{cell_ref}" s="{style}"><v>{numeric}</v></c>'

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{cell_ref}" s="{style}"><v>{value}</v></c>'

    text = escape(str(value))
    return f'<c r="{cell_ref}" s="{style}" t="inlineStr"><is><t>{text}</t></is></c>'


def write_results_xlsx(output_path: Path, results_df: pd.DataFrame) -> None:
    sheet_name = "Results"
    rows = [results_df.columns.tolist(), *results_df.astype(object).where(pd.notnull(results_df), None).values.tolist()]
    max_col = len(results_df.columns)
    max_row = len(rows)
    last_cell = f"{_excel_column_name(max_col)}{max_row}"

    width_xml = []
    for idx, column in enumerate(results_df.columns, start=1):
        series = results_df[column].astype(str).tolist()
        max_length = max([len(str(column)), *[len(value) for value in series]], default=len(str(column)))
        width = min(max(max_length + 2, 12), 28)
        width_xml.append(
            f'<col min="{idx}" max="{idx}" width="{width}" customWidth="1"/>'
        )

    row_xml = []
    for row_index, row in enumerate(rows, start=1):
        cells = "".join(_excel_cell(value, row_index, col_index) for col_index, value in enumerate(row, start=1))
        row_xml.append(f'<row r="{row_index}">{cells}</row>')

    sheet_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetViews>
    <sheetView workbookViewId="0">
      <pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>
    </sheetView>
  </sheetViews>
  <dimension ref="A1:{last_cell}"/>
  <sheetFormatPr defaultRowHeight="15"/>
  <cols>
    {''.join(width_xml)}
  </cols>
  <sheetData>
    {''.join(row_xml)}
  </sheetData>
  <autoFilter ref="A1:{last_cell}"/>
</worksheet>
'''

    workbook_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="{escape(sheet_name)}" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>
'''

    styles_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="2">
    <font>
      <sz val="11"/>
      <name val="Calibri"/>
    </font>
    <font>
      <b/>
      <sz val="11"/>
      <name val="Calibri"/>
    </font>
  </fonts>
  <fills count="2">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
  </fills>
  <borders count="1">
    <border><left/><right/><top/><bottom/><diagonal/></border>
  </borders>
  <cellStyleXfs count="1">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0"/>
  </cellStyleXfs>
  <cellXfs count="2">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/>
  </cellXfs>
  <cellStyles count="1">
    <cellStyle name="Normal" xfId="0" builtinId="0"/>
  </cellStyles>
</styleSheet>
'''

    content_types_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
'''

    root_rels_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
'''

    workbook_rels_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>
'''

    core_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
 xmlns:dc="http://purl.org/dc/elements/1.1/"
 xmlns:dcterms="http://purl.org/dc/terms/"
 xmlns:dcmitype="http://purl.org/dc/dcmitype/"
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Feature Selection Results</dc:title>
  <dc:creator>Codex</dc:creator>
</cp:coreProperties>
'''

    app_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
 xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Microsoft Excel</Application>
  <TitlesOfParts>
    <vt:vector size="1" baseType="lpstr">
      <vt:lpstr>{escape(sheet_name)}</vt:lpstr>
    </vt:vector>
  </TitlesOfParts>
</Properties>
'''

    buffer = BytesIO()
    with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED) as workbook:
        workbook.writestr("[Content_Types].xml", content_types_xml)
        workbook.writestr("_rels/.rels", root_rels_xml)
        workbook.writestr("docProps/core.xml", core_xml)
        workbook.writestr("docProps/app.xml", app_xml)
        workbook.writestr("xl/workbook.xml", workbook_xml)
        workbook.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        workbook.writestr("xl/styles.xml", styles_xml)
        workbook.writestr("xl/worksheets/sheet1.xml", sheet_xml)

    output_path.write_bytes(buffer.getvalue())
