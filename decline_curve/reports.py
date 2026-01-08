"""Report generation module for decline curve analysis.

This module provides single-page summary reports for wells and fields,
including HTML and PDF output formats.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .artifacts import FitArtifact
from .fit_diagnostics import DiagnosticsResult
from .fitting import FitResult
from .logging_config import get_logger

logger = get_logger(__name__)

# Try to import reportlab for PDF generation
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Image,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.debug(
        "reportlab not available. Install with: pip install reportlab. "
        "PDF generation will not be available."
    )


def generate_well_report(
    fit_result: FitResult,
    diagnostics: Optional[DiagnosticsResult] = None,
    artifact: Optional[FitArtifact] = None,
    output_path: Optional[str] = None,
    format: str = "html",
) -> str:
    """Generate single-page well report.

    Args:
        fit_result: FitResult object
        diagnostics: Optional DiagnosticsResult
        artifact: Optional FitArtifact
        output_path: Output file path (auto-generated if None)
        format: Output format ('html' or 'pdf')

    Returns:
        Path to generated report file

    Example:
        >>> report_path = generate_well_report(fit_result, diagnostics)
        >>> print(f"Report saved to {report_path}")
    """
    if output_path is None:
        well_id = artifact.metadata.run_id if artifact else "well_001"
        output_path = f"{well_id}_report.html"

    logger.info(f"Generating well report: {output_path} (format: {format})")

    format_handlers = {
        "pdf": lambda: _handle_pdf_report(
            fit_result, diagnostics, artifact, output_path
        ),
        "html": lambda: _handle_html_report(
            fit_result, diagnostics, artifact, output_path
        ),
    }

    handler = format_handlers.get(format, format_handlers["html"])
    return handler()


def _handle_pdf_report(
    fit_result: FitResult,
    diagnostics: Optional[DiagnosticsResult],
    artifact: Optional[FitArtifact],
    output_path: str,
) -> str:
    """Handle PDF report generation."""
    if not REPORTLAB_AVAILABLE:
        logger.warning(
            "reportlab not available. Install with: pip install reportlab. "
            "Saving HTML instead."
        )
        return _handle_html_report(fit_result, diagnostics, artifact, output_path)

    pdf_path = (
        output_path.replace(".html", ".pdf")
        if output_path.endswith(".html")
        else output_path
    )
    if not pdf_path.endswith(".pdf"):
        pdf_path += ".pdf"
    _generate_pdf_report(fit_result, diagnostics, artifact, pdf_path)
    return pdf_path


def _handle_html_report(
    fit_result: FitResult,
    diagnostics: Optional[DiagnosticsResult],
    artifact: Optional[FitArtifact],
    output_path: str,
) -> str:
    """Handle HTML report generation."""
    html_content = _generate_html_report(fit_result, diagnostics, artifact)
    with open(output_path, "w") as f:
        f.write(html_content)
    return output_path


def _generate_pdf_report(
    fit_result: FitResult,
    diagnostics: Optional[DiagnosticsResult],
    artifact: Optional[FitArtifact],
    output_path: str,
    plot_path: Optional[str] = None,
) -> None:
    """Generate PDF report using reportlab."""
    if not REPORTLAB_AVAILABLE:
        raise ImportError(
            "reportlab not available. Install with: pip install reportlab"
        )

    doc = SimpleDocTemplate(output_path, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()

    title = Paragraph("Decline Curve Analysis Report", styles["Title"])
    story.append(title)
    story.append(Spacer(1, 0.2 * inch))

    model_name = (
        fit_result.model.name if hasattr(fit_result.model, "name") else "Unknown"
    )
    story.append(Paragraph("<b>Model Information</b>", styles["Heading2"]))
    story.append(Paragraph(f"Model: {model_name}", styles["Normal"]))
    story.append(
        Paragraph(
            f"Fit Status: {'Success' if fit_result.success else 'Failed'}",
            styles["Normal"],
        )
    )

    if diagnostics:
        grade = diagnostics.grade if diagnostics else "N/A"
        quality_score = diagnostics.quality_score if diagnostics else None
        if grade != "N/A":
            story.append(Paragraph(f"Grade: {grade}", styles["Normal"]))
        if quality_score is not None:
            story.append(
                Paragraph(f"Quality Score: {quality_score:.2f}", styles["Normal"])
            )

    story.append(Spacer(1, 0.2 * inch))

    params = fit_result.params
    param_data = [["Parameter", "Value"]]
    param_data.extend([[param, f"{value:.4f}"] for param, value in params.items()])

    param_table = Table(param_data)
    param_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 12),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )
    story.append(param_table)
    story.append(Spacer(1, 0.2 * inch))

    if diagnostics:
        story.append(Paragraph("<b>Diagnostics</b>", styles["Heading2"]))
        diag_data = [
            ["Metric", "Value"],
            ["RMSE", f"{diagnostics.metrics.get('rmse', 0):.4f}"],
            ["MAE", f"{diagnostics.metrics.get('mae', 0):.4f}"],
            ["R²", f"{diagnostics.metrics.get('r_squared', 0):.4f}"],
        ]

        diag_table = Table(diag_data)
        diag_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )
        story.append(diag_table)
        story.append(Spacer(1, 0.2 * inch))

    if artifact:
        story.append(Paragraph("<b>Provenance</b>", styles["Heading2"]))
        story.append(Paragraph(f"Run ID: {artifact.metadata.run_id}", styles["Normal"]))
        story.append(
            Paragraph(f"Timestamp: {artifact.metadata.timestamp}", styles["Normal"])
        )
        story.append(
            Paragraph(
                f"Package Version: {artifact.metadata.package_version}",
                styles["Normal"],
            )
        )
        story.append(Spacer(1, 0.2 * inch))

    if plot_path and Path(plot_path).exists():
        try:
            img = Image(plot_path, width=6 * inch, height=4 * inch)
            story.append(Spacer(1, 0.2 * inch))
            story.append(Paragraph("<b>Forecast Plot</b>", styles["Heading2"]))
            story.append(img)
        except Exception as e:
            logger.warning(f"Could not embed plot: {e}")

    doc.build(story)
    logger.info(f"PDF report saved to {output_path}")


def generate_field_pdf_report(
    well_results: List[Dict],
    output_path: Optional[str] = None,
    title: str = "Field Summary Report",
) -> str:
    """Generate field summary PDF report."""
    if not REPORTLAB_AVAILABLE:
        raise ImportError(
            "reportlab not available. Install with: pip install reportlab"
        )

    if output_path is None:
        output_path = "field_summary.pdf"

    df = pd.DataFrame(well_results)

    doc = SimpleDocTemplate(output_path, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()

    title_para = Paragraph(title, styles["Title"])
    story.append(title_para)
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("<b>Summary Statistics</b>", styles["Heading2"]))
    summary_data = [["Metric", "Value"]]
    summary_data.append(["Total Wells", str(len(df))])

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    sum_metrics = {"eur", "npv", "p50_eur", "p50_npv"}
    for col in numeric_cols[:10]:
        if col in sum_metrics:
            summary_data.append([f"{col} (Total)", f"{df[col].sum():,.0f}"])
            summary_data.append([f"{col} (Mean)", f"{df[col].mean():,.0f}"])

    summary_table = Table(summary_data)
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 12),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )
    story.append(summary_table)
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("<b>Well Results (Sample)</b>", styles["Heading2"]))
    well_data = [list(df.columns[:8])]
    well_data.extend(
        [
            [str(row[col])[:20] for col in df.columns[:8]]
            for _, row in df.head(20).iterrows()
        ]
    )

    well_table = Table(well_data)
    well_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("FONTSIZE", (0, 1), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )
    story.append(well_table)

    if len(df) > 20:
        story.append(Spacer(1, 0.1 * inch))
        story.append(
            Paragraph(
                f"<i>Showing first 20 of {len(df)} wells. Full data available in CSV export.</i>",
                styles["Normal"],
            )
        )

    doc.build(story)
    logger.info(f"Field PDF report saved to {output_path}")

    return output_path


def _generate_html_report(
    fit_result: FitResult,
    diagnostics: Optional[DiagnosticsResult],
    artifact: Optional[FitArtifact],
) -> str:
    """Generate HTML content for well report.

    Args:
        fit_result: Fit result object
        diagnostics: Optional diagnostics result
        artifact: Optional fit artifact

    Returns:
        HTML string
    """

    # Extract key information
    params = fit_result.params
    model_name = (
        fit_result.model.name if hasattr(fit_result.model, "name") else "Unknown"
    )

    grade = diagnostics.grade if diagnostics else "N/A"
    quality_score = diagnostics.quality_score if diagnostics else None

    # Build HTML
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>DCA Report - {model_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .section {{ margin: 20px 0; }}
        .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
        .grade {{ font-size: 24px; font-weight: bold; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Decline Curve Analysis Report</h1>

    <div class="section">
        <h2>Model Information</h2>
        <p><strong>Model:</strong> {model_name}</p>
        <p><strong>Fit Status:</strong> {"Success" if fit_result.success else "Failed"}</p>
        {f'<p><strong>Grade:</strong> <span class="grade">{grade}</span></p>' if grade != "N/A" else ""}
        {f'<p><strong>Quality Score:</strong> {quality_score:.2f}</p>' if quality_score else ""}
    </div>

    <div class="section">
        <h2>Parameters</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
    """

    for param, value in params.items():
        html += f"<tr><td>{param}</td><td>{value:.4f}</td></tr>"

    html += """
        </table>
    </div>
    """

    if diagnostics:
        html += """
    <div class="section">
        <h2>Diagnostics</h2>
        <div class="metric">
            <strong>RMSE:</strong> {:.4f}
        </div>
        <div class="metric">
            <strong>MAE:</strong> {:.4f}
        </div>
        <div class="metric">
            <strong>R²:</strong> {:.4f}
        </div>
    </div>
    """.format(
            diagnostics.metrics.get("rmse", 0),
            diagnostics.metrics.get("mae", 0),
            diagnostics.metrics.get("r_squared", 0),
        )

    if artifact:
        html += f"""
    <div class="section">
        <h2>Provenance</h2>
        <p><strong>Run ID:</strong> {artifact.metadata.run_id}</p>
        <p><strong>Timestamp:</strong> {artifact.metadata.timestamp}</p>
        <p><strong>Package Version:</strong> {artifact.metadata.package_version}</p>
    </div>
    """

    html += """
</body>
</html>
"""

    return html


def generate_field_summary(
    well_results: List[Dict],
    output_path: Optional[str] = None,
) -> str:
    """Generate field summary table.

    Args:
        well_results: List of well result dictionaries
        output_path: Output file path

    Returns:
        Path to generated summary file
    """
    if output_path is None:
        output_path = "field_summary.csv"

    import pandas as pd

    # Convert to DataFrame
    df = pd.DataFrame(well_results)

    # Save to CSV
    df.to_csv(output_path, index=False)

    logger.info(f"Generated field summary: {len(well_results)} wells -> {output_path}")

    return output_path
