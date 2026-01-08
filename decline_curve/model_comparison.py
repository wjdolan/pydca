"""Model comparison helper.

This module provides a comparison helper that runs several models on the same
well or cohort and returns a compact metric table plus a blended ensemble if requested.
"""

from typing import Dict, List, Optional

import pandas as pd

from .evaluate import mae, rmse, smape
from .logging_config import get_logger
from .model_interface import ForecastModel
from .model_registry import get_registry

logger = get_logger(__name__)


def compare_models(
    series: pd.Series,
    model_names: List[str],
    horizon: int = 12,
    return_ensemble: bool = False,
    ensemble_method: str = "mean",
    **model_kwargs,
) -> Dict[str, any]:
    """
    Compare multiple models on the same well.

    Runs several models on the same production series and returns a compact
    metric table plus optional blended ensemble.

    Args:
        series: Historical production time series
        model_names: List of model names to compare (e.g., ['arps', 'arima', 'chronos'])
        horizon: Forecast horizon
        return_ensemble: If True, also return ensemble forecast
        ensemble_method: Ensemble method ('mean', 'median', 'weighted')
        **model_kwargs: Additional kwargs passed to each model (keyed by model name)

    Returns:
        Dictionary with:
        - 'metrics': DataFrame with metrics for each model
        - 'forecasts': Dictionary of forecasts by model name
        - 'ensemble': Ensemble forecast (if return_ensemble=True)
        - 'best_model': Name of best model by RMSE

    Example:
        >>> import pandas as pd
        >>> from decline_curve.model_comparison import compare_models
        >>> dates = pd.date_range('2020-01-01', periods=24, freq='MS')
        >>> production = pd.Series([...], index=dates)
        >>> results = compare_models(
        ...     production,
        ...     model_names=['arps', 'arima'],
        ...     horizon=12,
        ...     return_ensemble=True
        ... )
        >>> print(results['metrics'])
        >>> print(f"Best model: {results['best_model']}")
    """
    registry = get_registry()
    forecasts = {}
    metrics_list = []

    # Fit and forecast with each model
    for model_name in model_names:
        if not registry.is_available(model_name):
            logger.warning(f"Model '{model_name}' not available, skipping")
            continue

        try:
            # Get model-specific kwargs
            kwargs = model_kwargs.get(model_name, {})
            kwargs.setdefault("horizon", horizon)

            # Create and fit model
            model_class = registry.get(model_name)
            model = model_class(**kwargs)
            model.fit(series)

            # Generate forecast
            forecast = model.forecast(horizon=horizon)
            forecasts[model_name] = forecast

            # Calculate metrics on historical fit
            hist_forecast = forecast[: len(series)]
            common = series.index.intersection(hist_forecast.index)
            if len(common) > 0:
                y_true = series.loc[common]
                y_pred = hist_forecast.loc[common]

                metrics = {
                    "model": model_name,
                    "rmse": rmse(y_true, y_pred),
                    "mae": mae(y_true, y_pred),
                    "smape": smape(y_true, y_pred),
                }
                metrics_list.append(metrics)
            else:
                logger.warning(
                    f"No overlap between series and forecast for {model_name}"
                )

        except Exception as e:
            logger.error(f"Failed to fit/forecast with {model_name}: {e}")
            continue

    if not metrics_list:
        raise ValueError(
            "No models succeeded. Check model availability and data quality."
        )

    # Create metrics DataFrame
    metrics_df = pd.DataFrame(metrics_list)

    # Find best model
    best_model = metrics_df.loc[metrics_df["rmse"].idxmin(), "model"]

    result = {
        "metrics": metrics_df,
        "forecasts": forecasts,
        "best_model": best_model,
    }

    # Create ensemble if requested
    if return_ensemble and len(forecasts) > 1:
        ensemble_forecast = _create_ensemble(forecasts, method=ensemble_method)
        result["ensemble"] = ensemble_forecast

    logger.info(
        f"Compared {len(metrics_list)} models. Best: {best_model} "
        f"(RMSE: {metrics_df.loc[metrics_df['model'] == best_model, 'rmse'].values[0]:.2f})"
    )

    return result


def _create_ensemble(
    forecasts: Dict[str, pd.Series], method: str = "mean"
) -> pd.Series:
    """Create ensemble forecast from multiple forecasts.

    Args:
        forecasts: Dictionary of forecasts by model name
        method: Ensemble method ('mean', 'median', 'weighted')

    Returns:
        Ensemble forecast series
    """
    if not forecasts:
        raise ValueError("No forecasts provided for ensemble")

    # Align all forecasts to common index
    forecast_list = list(forecasts.values())
    common_index = forecast_list[0].index

    # Ensure all forecasts have same index
    aligned_forecasts = []
    for forecast in forecast_list:
        if not forecast.index.equals(common_index):
            forecast = forecast.reindex(common_index)
        aligned_forecasts.append(forecast.values)

    # Stack into array
    forecast_array = pd.DataFrame(
        {name: f.values for name, f in forecasts.items()}, index=common_index
    )

    # Apply ensemble method
    if method == "mean":
        ensemble_values = forecast_array.mean(axis=1)
    elif method == "median":
        ensemble_values = forecast_array.median(axis=1)
    elif method == "weighted":
        # Equal weights for now (could be improved with performance-based weights)
        weights = pd.Series(1.0 / len(forecasts), index=forecast_array.columns)
        ensemble_values = (forecast_array * weights).sum(axis=1)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")

    return pd.Series(ensemble_values, index=common_index, name="ensemble")


def compare_models_batch(
    df: pd.DataFrame,
    model_names: List[str],
    well_id_col: str = "well_id",
    date_col: str = "date",
    value_col: str = "oil",
    horizon: int = 12,
    top_n: Optional[int] = None,
    **model_kwargs,
) -> pd.DataFrame:
    """
    Compare models across multiple wells.

    Args:
        df: DataFrame with production data for multiple wells
        model_names: List of model names to compare
        well_id_col: Column name for well identifier
        date_col: Column name for date
        value_col: Column name for production values
        horizon: Forecast horizon
        top_n: Process only top N wells (None for all)
        **model_kwargs: Additional kwargs for models

    Returns:
        DataFrame with metrics for each well and model combination
    """
    well_ids = df[well_id_col].unique()
    if top_n:
        well_ids = well_ids[:top_n]

    all_results = []

    for well_id in well_ids:
        well_df = df[df[well_id_col] == well_id].copy()
        well_df = well_df[[date_col, value_col]].dropna()
        well_df[date_col] = pd.to_datetime(well_df[date_col])
        well_df = well_df.set_index(date_col).sort_index()

        if len(well_df) < 6:  # Minimum data points
            continue

        series = well_df[value_col]

        try:
            results = compare_models(
                series, model_names=model_names, horizon=horizon, **model_kwargs
            )

            # Add well_id to metrics
            metrics = results["metrics"].copy()
            metrics["well_id"] = well_id
            all_results.append(metrics)

        except Exception as e:
            logger.warning(f"Failed to compare models for well {well_id}: {e}")
            continue

    if not all_results:
        raise ValueError("No wells processed successfully")

    return pd.concat(all_results, ignore_index=True)
