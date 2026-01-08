"""Example: PDF report generation.

This example demonstrates how to generate PDF reports for wells and fields.
"""

import numpy as np
import pandas as pd

# Note: This example requires reportlab to be installed
# pip install reportlab

try:
    from decline_curve.artifacts import FitArtifact, Metadata
    from decline_curve.fit_diagnostics import DiagnosticsResult
    from decline_curve.fitting import FitResult
    from decline_curve.reports import generate_field_pdf_report, generate_well_report

    REPORTLAB_AVAILABLE = True
except ImportError as e:
    print(f"Required modules not available: {e}")
    REPORTLAB_AVAILABLE = False


def generate_example_well_results():
    """Generate example well results for demonstration."""
    well_results = []
    for i in range(20):
        well_results.append(
            {
                "well_id": f"WELL_{i:03d}",
                "eur": 50000 + np.random.normal(0, 10000),
                "npv": 2000000 + np.random.normal(0, 500000),
                "p50_eur": 50000 + np.random.normal(0, 10000),
                "p50_npv": 2000000 + np.random.normal(0, 500000),
                "operator": np.random.choice(
                    ["Operator_A", "Operator_B", "Operator_C"]
                ),
                "county": np.random.choice(["County_X", "County_Y"]),
            }
        )
    return well_results


def main():
    """Main example function."""
    print("=" * 70)
    print("PDF Report Generation Example")
    print("=" * 70)

    if not REPORTLAB_AVAILABLE:
        print("\n⚠️  reportlab not available.")
        print("   Install with: pip install reportlab")
        print("   Or: pip install decline-curve[reports]")
        return

    # Example 1: Generate field summary PDF
    print("\n1. Generating field summary PDF report...")
    well_results = generate_example_well_results()

    try:
        pdf_path = generate_field_pdf_report(
            well_results,
            output_path="example_field_report.pdf",
            title="Example Field Summary Report",
        )
        print(f"   ✓ Field PDF report saved to: {pdf_path}")
    except Exception as e:
        print(f"   ✗ Error generating field PDF: {e}")

    # Example 2: Generate well report (would need actual FitResult)
    print("\n2. Well report generation requires FitResult object.")
    print("   See decline_curve.reports.generate_well_report() for usage.")
    print("   Example:")
    print("   >>> from decline_curve.reports import generate_well_report")
    print("   >>> report_path = generate_well_report(")
    print("   ...     fit_result, diagnostics, format='pdf'")
    print("   ... )")

    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)
    print("\nNote: PDF reports include:")
    print("  - Summary statistics")
    print("  - Well-level data tables")
    print("  - Professional formatting")
    print("  - Embedded plots (if provided)")


if __name__ == "__main__":
    main()
