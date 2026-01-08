# Ecosystem Integrations

This library integrates with other tools in the ecosystem to provide comprehensive reservoir analysis workflows.

## signalplot Integration

**signalplot** provides minimalist plotting design for publication-ready figures. All plotting functions in this library use signalplot by default when available.

### Features

- **Automatic Styling**: All plots automatically use signalplot's minimalist aesthetic
- **Publication Quality**: Figures saved at 300 DPI with proper formatting
- **Consistent Design**: Black/dark gray for primary data, single accent color for emphasis
- **Graceful Fallback**: Works without signalplot, but enhanced when available

### Installation

```bash
pip install signalplot
# or
pip install decline-curve[integrations]
```

### Usage

No special code needed! All plotting functions automatically use signalplot:

```python
from decline_curve.plot import plot_forecast

# Automatically uses signalplot if available
plot_forecast(y_true, y_pred, title="Production Forecast")
```

The library will:

- Apply signalplot's minimalist style automatically
- Use black/dark gray for primary data
- Use signalplot's accent color for emphasis
- Save figures at publication quality (300 DPI)
- Fall back gracefully if signalplot isn't installed

## pygeomodeling Integration

**pygeomodeling** provides advanced Gaussian Process Regression and Kriging for spatial modeling.

### pygeomodeling Features

- **Advanced Kriging**: Uses pygeomodeling's UnifiedSPE9Toolkit for superior spatial interpolation
- **Deep GP Support**: Access to Deep Gaussian Process models for complex spatial patterns
- **Automatic Selection**: The kriging module automatically uses pygeomodeling if available

### Installing pygeomodeling

```bash
pip install pygeomodeling
# or
pip install decline-curve[integrations]
```

### Using pygeomodeling

```python
from decline_curve.spatial_kriging import krige_eur
from decline_curve.eur_estimation import calculate_eur_batch

# Calculate EUR
eur_results = calculate_eur_batch(df, well_id_col='well_id')

# Kriging will automatically use pygeomodeling if available
result = krige_eur(eur_results, method='auto')  # Uses pygeomodeling if installed
```

### Direct Integration

```python
from decline_curve.integrations import krige_eur_with_pygeomodeling

# Use pygeomodeling's advanced GP models directly
predictions, uncertainty = krige_eur_with_pygeomodeling(
    eur_df,
    lon_col='long',
    lat_col='lat',
    eur_col='eur'
)
```

## geosuite Integration

**geosuite** provides professional tools for subsurface analysis including petrophysics, geomechanics, and data loading.

### geosuite Features

- **Data Loading**: Load well data from LAS, CSV, and other formats
- **Petrophysical Analysis**: Calculate porosity, water saturation, and other properties
- **Enhanced EUR**: Combine decline curve analysis with petrophysical insights

### Installing geosuite

```bash
pip install geosuite
# or
pip install decline-curve[integrations]
```

### Using geosuite

```python
from decline_curve.integrations import (
    load_well_data_with_geosuite,
    enhanced_eur_with_geosuite
)

# Load well data using geosuite
df = load_well_data_with_geosuite('well_data.las')

# Calculate EUR with petrophysical enhancement
results = enhanced_eur_with_geosuite(
    production_df,
    well_logs=well_logs_df
)
```

## Integration Status

Check which integrations are available:

```python
from decline_curve.integrations import check_integration_availability

status = check_integration_availability()
print(f"signalplot: {status.get('signalplot', False)}")
print(f"pygeomodeling: {status['pygeomodeling']}")
print(f"geosuite: {status['geosuite']}")
```

Note: signalplot availability is checked automatically by plotting functions.

## Benefits

1. **Seamless Workflow**: Use multiple tools together without manual data conversion
2. **Best of Both Worlds**: Leverage specialized tools for their strengths
3. **Automatic Fallbacks**: Library gracefully degrades if integrations aren't available
4. **Unified API**: Consistent interface across the ecosystem

## Example Workflow

```python
# 1. Load data with geosuite
from decline_curve.integrations import load_well_data_with_geosuite
df = load_well_data_with_geosuite('production_data.las')

# 2. Calculate EUR with decline-curve
from decline_curve.eur_estimation import calculate_eur_with_kriging
eur_results = calculate_eur_with_kriging(df)  # Uses pygeomodeling if available

# 3. Enhanced analysis with geosuite
from decline_curve.integrations import enhanced_eur_with_geosuite
enhanced = enhanced_eur_with_geosuite(df, well_logs=logs_df)
```

## Installation Options

```bash
# Base installation (no integrations)
pip install decline-curve

# With spatial analysis (kriging)
pip install decline-curve[spatial]

# With ecosystem integrations
pip install decline-curve[integrations]

# Everything
pip install decline-curve[all]
```
