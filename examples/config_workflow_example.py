"""Example: Complete workflow using configuration file.

This example demonstrates how to run a complete analysis workflow
using a configuration file, making it easy for users coming from
spreadsheets to get started.
"""

import logging
import sys
from pathlib import Path

from decline_curve.config import BatchJobConfig
from decline_curve.logging_config import configure_logging
from decline_curve.runner import FieldScaleRunner, RunnerConfig
from decline_curve.schemas import ProductionInputSchema, ProductionOutputSchema

# Step 1: Load configuration
config_path = sys.argv[1] if len(sys.argv) > 1 else "examples/config_example.toml"

if not Path(config_path).exists():
    print(f"Configuration file not found: {config_path}")
    print("Creating example configuration...")
    from decline_curve.config import create_example_config

    create_example_config(config_path, format="toml")
    print(f"Example config created at {config_path}")
    print("Please edit the config file and run again.")
    sys.exit(1)

print(f"Loading configuration from: {config_path}")
config = BatchJobConfig.from_file(config_path)

# Step 2: Configure logging
configure_logging(level=getattr(logging, config.log_level), log_file=config.log_file)

# Step 3: Load data
print(f"\nLoading data from: {config.data.path}")
import pandas as pd

if config.data.format == "csv":
    df = pd.read_csv(config.data.path)
elif config.data.format == "parquet":
    df = pd.read_parquet(config.data.path)
else:
    raise ValueError(f"Unsupported format: {config.data.format}")

df[config.data.date_col] = pd.to_datetime(df[config.data.date_col])

# Validate and standardize
df_standard = ProductionInputSchema.standardize(df)
validation = ProductionInputSchema.validate(df_standard)

if not validation["valid"]:
    print(f"Validation errors: {validation['errors']}")
    sys.exit(1)

print(
    f"  Loaded {len(df_standard):,} records for {df_standard[config.data.well_id_col].nunique():,} wells"
)


# Step 4: Define forecast function
def forecast_well(df_well):
    """Forecast function for a single well."""
    from decline_curve import dca

    well_id = df_well[config.data.well_id_col].iloc[0]
    df_well = df_well.sort_values(config.data.date_col)

    # Create production series
    series = df_well.set_index(config.data.date_col)[config.data.value_col]
    series = series.asfreq("MS", fill_value=0)

    # Filter to valid production
    series = series[series > 0]

    if len(series) < 6:  # Need at least 6 months
        return {"well_id": well_id, "error": "Insufficient data"}

    try:
        forecast = dca.single_well(
            series,
            model=config.model.model,
            kind=config.model.kind,
            horizon=config.model.horizon,
        )

        return {"well_id": well_id, "forecast": forecast, "historical": series}
    except Exception as e:
        return {"well_id": well_id, "error": str(e)}


# Step 5: Run analysis
print(f"\nRunning analysis...")
print(f"  Model: {config.model.model} ({config.model.kind})")
print(f"  Horizon: {config.model.horizon} months")
print(f"  Parallel jobs: {config.n_jobs}")

runner_config = RunnerConfig(
    n_jobs=config.n_jobs,
    chunk_size=config.chunk_size,
    max_retries=config.max_retries,
    log_level=config.log_level,
    log_file=config.log_file,
)

runner = FieldScaleRunner(runner_config)

results = runner.run(
    df_standard,
    forecast_func=forecast_well,
    well_id_col=config.data.well_id_col,
)

print(f"\nAnalysis complete:")
print(f"  Successful: {results.summary.successful_wells}")
print(f"  Failed: {results.summary.failed_wells}")
print(f"  Success rate: {results.summary.success_rate:.1f}%")

# Step 6: Save outputs
print(f"\nSaving outputs to: {config.output.output_dir}")

output_schema = ProductionOutputSchema.from_runner_results(results)
parameters_df, forecasts_df, events_df = output_schema.to_dataframes()

# Save based on config
Path(config.output.output_dir).mkdir(parents=True, exist_ok=True)

if config.output.save_parameters:
    output_format = config.output.format
    if output_format == "csv":
        parameters_df.to_csv(
            Path(config.output.output_dir) / "parameters.csv", index=False
        )
    elif output_format == "parquet":
        parameters_df.to_parquet(
            Path(config.output.output_dir) / "parameters.parquet", index=False
        )

if config.output.save_forecasts:
    output_format = config.output.format
    if output_format == "csv":
        forecasts_df.to_csv(
            Path(config.output.output_dir) / "forecasts.csv", index=False
        )
    elif output_format == "parquet":
        forecasts_df.to_parquet(
            Path(config.output.output_dir) / "forecasts.parquet", index=False
        )

if config.output.save_reports:
    from decline_curve.reports import generate_field_pdf_report

    well_results = parameters_df.to_dict("records")
    report_path = generate_field_pdf_report(
        well_results,
        output_path=Path(config.output.output_dir) / "field_report.pdf",
    )
    print(f"  Report saved to: {report_path}")

print(f"\n✓ Analysis complete! Results saved to {config.output.output_dir}")
