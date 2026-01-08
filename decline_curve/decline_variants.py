"""Additional decline curve variants for practical applications.

This module provides decline variants that matter in practice:
- Fixed terminal decline: Transition to a fixed terminal decline rate
- Time to boundary: Constraints that prevent absurd tail behavior
"""

from typing import Optional

import numpy as np
import pandas as pd

from .logging_config import get_logger
from .models import ArpsParams, fit_arps, predict_arps

logger = get_logger(__name__)


def fixed_terminal_decline(
    series: pd.Series,
    kind: str = "hyperbolic",
    terminal_decline_rate: float = 0.05,
    transition_criteria: str = "rate",
    transition_value: Optional[float] = None,
) -> tuple[pd.Series, dict]:
    """
    Fit decline curve with transition to fixed terminal decline rate.

    This variant prevents unrealistic long-term forecasts by transitioning
    to a fixed terminal decline rate (typically 5-10% per year) when the
    hyperbolic decline rate becomes too low.

    Args:
        series: Historical production time series
        kind: Initial decline type ('exponential', 'harmonic', 'hyperbolic')
        terminal_decline_rate: Fixed annual decline rate for terminal phase
            (e.g., 0.05 = 5% per year)
        transition_criteria: When to transition:
            - 'rate': Transition when decline rate reaches threshold
            - 'time': Transition at fixed time
            - 'cumulative': Transition at cumulative volume
        transition_value: Threshold value for transition
            - If 'rate': minimum decline rate before transition (default: terminal_decline_rate)
            - If 'time': time in months (default: 60 months)
            - If 'cumulative': cumulative volume (default: None, uses rate-based)

    Returns:
        Tuple of (forecast_series, params_dict) where params_dict contains:
        - initial_params: ArpsParams for initial phase
        - terminal_decline_rate: Terminal decline rate
        - transition_time: Time of transition
        - transition_rate: Rate at transition

    Example:
        >>> import pandas as pd
        >>> from decline_curve.decline_variants import fixed_terminal_decline
        >>> dates = pd.date_range('2020-01-01', periods=36, freq='MS')
        >>> production = pd.Series([...], index=dates)
        >>> forecast, params = fixed_terminal_decline(
        ...     production,
        ...     kind='hyperbolic',
        ...     terminal_decline_rate=0.06  # 6% per year
        ... )
    """
    if transition_criteria == "rate":
        if transition_value is None:
            transition_value = terminal_decline_rate
    elif transition_criteria == "time":
        if transition_value is None:
            transition_value = 60.0  # 5 years default
    elif transition_criteria == "cumulative":
        if transition_value is None:
            transition_value = None  # Will use rate-based
    else:
        raise ValueError(
            f"Unknown transition_criteria: {transition_criteria}. "
            "Must be 'rate', 'time', or 'cumulative'"
        )

    # Fit initial decline
    t = np.arange(len(series))
    q = series.values
    initial_params = fit_arps(t, q, kind=kind)

    # Convert terminal decline rate from annual to monthly
    terminal_decline_monthly = terminal_decline_rate / 12.0

    # Determine transition point
    if transition_criteria == "time":
        transition_time = transition_value
    elif transition_criteria == "rate":
        # Find when decline rate reaches threshold
        # For hyperbolic: D(t) = di / (1 + b*di*t)
        # Solve for t when D(t) = transition_value
        di = initial_params.di
        b = initial_params.b
        if b > 0:
            transition_time = (di / transition_value - 1) / (b * di)
            transition_time = max(0, min(transition_time, len(series) * 2))
        else:
            # Exponential: constant decline, transition immediately
            transition_time = 0.0
    else:  # cumulative
        # Use rate-based transition if cumulative not specified
        di = initial_params.di
        b = initial_params.b
        if b > 0:
            transition_time = (di / terminal_decline_monthly - 1) / (b * di)
        else:
            transition_time = 0.0

    # Calculate rate at transition
    t_transition = np.array([transition_time])
    q_transition = predict_arps(t_transition, initial_params)[0]

    # Generate forecast
    horizon = 240  # 20 years
    t_full = np.arange(0, len(series) + horizon, 1.0)

    forecast_rates = np.zeros_like(t_full)

    for i, t_val in enumerate(t_full):
        if t_val <= transition_time:
            # Initial phase
            forecast_rates[i] = predict_arps(np.array([t_val]), initial_params)[0]
        else:
            # Terminal phase: exponential decline from transition point
            t_terminal = t_val - transition_time
            forecast_rates[i] = q_transition * np.exp(
                -terminal_decline_monthly * t_terminal
            )

    # Create forecast series
    dates = pd.date_range(
        series.index[0], periods=len(forecast_rates), freq=series.index.freq or "MS"
    )
    forecast_series = pd.Series(forecast_rates, index=dates, name="forecast")

    params_dict = {
        "initial_params": initial_params,
        "terminal_decline_rate": terminal_decline_rate,
        "transition_time": transition_time,
        "transition_rate": q_transition,
        "kind": kind,
    }

    return forecast_series, params_dict


def time_to_boundary(
    series: pd.Series,
    kind: str = "hyperbolic",
    max_time: float = 240.0,
    min_rate: float = 1.0,
    enforce_bounds: bool = True,
) -> tuple[pd.Series, dict]:
    """
    Fit decline curve with time-to-boundary constraints.

    This variant prevents absurd tail behavior by:
    1. Limiting forecast to maximum time horizon
    2. Setting minimum economic rate threshold
    3. Enforcing reasonable parameter bounds

    Args:
        series: Historical production time series
        kind: Decline type ('exponential', 'harmonic', 'hyperbolic')
        max_time: Maximum forecast time in months (default: 240 = 20 years)
        min_rate: Minimum economic production rate (default: 1.0)
        enforce_bounds: If True, enforce parameter bounds to prevent unrealistic forecasts

    Returns:
        Tuple of (forecast_series, params_dict) where params_dict contains:
        - params: Fitted ArpsParams (with bounds enforced if requested)
        - max_time: Maximum time used
        - min_rate: Minimum rate threshold
        - time_to_min_rate: Time when forecast reaches min_rate

    Example:
        >>> import pandas as pd
        >>> from decline_curve.decline_variants import time_to_boundary
        >>> dates = pd.date_range('2020-01-01', periods=36, freq='MS')
        >>> production = pd.Series([...], index=dates)
        >>> forecast, params = time_to_boundary(
        ...     production,
        ...     max_time=300,  # 25 years
        ...     min_rate=5.0   # 5 bbl/month minimum
        ... )
    """
    # Fit decline
    t = np.arange(len(series))
    q = series.values
    params = fit_arps(t, q, kind=kind)

    # Enforce bounds if requested
    if enforce_bounds:
        # Reasonable bounds for parameters
        params.qi = max(1.0, min(params.qi, 1e6))  # qi between 1 and 1M
        params.di = max(
            0.001, min(params.di, 10.0)
        )  # di between 0.1% and 1000% per month
        if kind == "hyperbolic":
            params.b = max(0.0, min(params.b, 2.0))  # b between 0 and 2

    # Find time when rate reaches minimum
    # For hyperbolic: q(t) = qi / (1 + b*di*t)^(1/b) = min_rate
    # Solve: (1 + b*di*t) = (qi/min_rate)^b
    # t = ((qi/min_rate)^b - 1) / (b*di)
    if kind == "exponential":
        # q(t) = qi * exp(-di*t) = min_rate
        # t = -ln(min_rate/qi) / di
        if params.qi > min_rate and params.di > 0:
            time_to_min_rate = -np.log(min_rate / params.qi) / params.di
        else:
            time_to_min_rate = max_time
    elif kind == "harmonic":
        # q(t) = qi / (1 + di*t) = min_rate
        # t = (qi/min_rate - 1) / di
        if params.qi > min_rate and params.di > 0:
            time_to_min_rate = (params.qi / min_rate - 1) / params.di
        else:
            time_to_min_rate = max_time
    else:  # hyperbolic
        if params.b > 0 and params.qi > min_rate and params.di > 0:
            time_to_min_rate = ((params.qi / min_rate) ** params.b - 1) / (
                params.b * params.di
            )
        else:
            time_to_min_rate = max_time

    # Use minimum of max_time and time_to_min_rate
    effective_max_time = min(max_time, time_to_min_rate)

    # Generate forecast up to effective max time
    t_forecast = np.arange(
        0, min(len(series) + int(effective_max_time), len(series) + 240), 1.0
    )
    forecast_rates = predict_arps(t_forecast, params)

    # Clip to minimum rate
    forecast_rates = np.maximum(forecast_rates, min_rate)

    # Create forecast series
    dates = pd.date_range(
        series.index[0], periods=len(forecast_rates), freq=series.index.freq or "MS"
    )
    forecast_series = pd.Series(forecast_rates, index=dates, name="forecast")

    params_dict = {
        "params": params,
        "max_time": max_time,
        "min_rate": min_rate,
        "time_to_min_rate": time_to_min_rate,
        "effective_max_time": effective_max_time,
    }

    return forecast_series, params_dict
