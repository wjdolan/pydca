# CLI and Configuration File Examples

This directory contains examples for using the decline curve analysis library
via command-line interface and configuration files, making it accessible for
users coming from spreadsheets.

## Quick Start with Configuration File

1. **Create a configuration file** (or use the example):

```bash
# Generate example config
python -c "from decline_curve.config import create_example_config; create_example_config('my_config.toml')"
```

2. **Edit the configuration file** to point to your data:

```toml
[data]
path = "your_data.csv"
well_id_col = "well_id"
date_col = "date"
value_col = "oil_bbl"
```

3. **Run the analysis**:

```bash
python -m decline_curve my_config.toml
```

## Command Line Interface

### Basic Usage

```bash
# Run batch job from config (primary mode)
python -m decline_curve batch_config.toml

# Run benchmark from config
python -m decline_curve benchmark_config.toml --workflow benchmark

# Run sensitivity analysis from config
python -m decline_curve sensitivity_config.toml --workflow sensitivity
```

### With Logging

```bash
# Verbose output
python -m decline_curve batch_config.toml --verbose

# Save log to file (via config)
python -m decline_curve batch_config.toml --log-level INFO
```

## Configuration File Format

### TOML Example

See `config_example.toml` for a complete example.

### YAML Example

```yaml
data:
  path: "data/production.csv"
  format: "csv"
  well_id_col: "well_id"
  date_col: "date"
  value_col: "oil_bbl"

model:
  model: "arps"
  kind: "hyperbolic"
  horizon: 12

economics:
  price: 70.0
  opex: 15.0
  discount_rate: 0.10

output:
  output_dir: "output"
  save_forecasts: true
  save_parameters: true
```

## Workflow Examples

### Example 1: Single Well Analysis

```python
from decline_curve import dca

# Load data
df = pd.read_csv('production.csv')
well_data = df[df['well_id'] == 'WELL_001']
series = well_data.set_index('date')['oil_bbl']

# Forecast
forecast = dca.single_well(series, model='arps', horizon=12)

# Save results
forecast.to_csv('well_001_forecast.csv')
```

### Example 2: Batch Analysis with Config

```bash
# 1. Create config file
python -c "from decline_curve.config import create_example_config; create_example_config('batch_config.toml')"

# 2. Edit config to point to your data

# 3. Run batch analysis
python -m decline_curve batch_config.toml
```

### Example 3: Legacy CSV Mode (Quick Analysis)

```bash
# Quick forecast
python -m decline_curve --csv production.csv --well WELL_001 --model arps > forecast.txt

# Benchmark models
python -m decline_curve --csv production.csv --benchmark --model arps > arps_results.txt
python -m decline_curve --csv production.csv --benchmark --model arima > arima_results.txt
```

## Tips for Spreadsheet Users

1. **Export to CSV**: Save your Excel data as CSV with columns: well_id, date, oil_bbl

2. **Use Config Files**: Instead of writing Python code, edit the TOML/YAML config file

3. **Start Simple**: Begin with single well analysis, then move to batch processing

4. **Check Logs**: Review log files to understand what happened during processing

5. **Validate Data**: The library will validate your data and report any issues

## Common Workflows

### Update Forecasts Monthly

1. Add new production data to your CSV
2. Re-run the same config file
3. Compare new forecasts to previous ones

### Compare Price Scenarios

1. Create multiple config files with different prices
2. Run each config file
3. Compare output results

### Analyze Multiple Fields

1. Create separate config files for each field
2. Use different output directories
3. Aggregate results using portfolio tools

## Getting Help

- See `docs/cookbook/config_files.rst` for detailed configuration guide
- See `docs/workflows/` for complete workflow examples
- Check log files for error messages and warnings
