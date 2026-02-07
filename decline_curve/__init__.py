"""Decline Curve Analysis package.

Keep top-level imports lightweight and lazily load optional helpers to avoid
triggering optional dependency warnings on `import decline_curve`.
"""

from . import dca
from .logging_config import configure_logging, get_logger

__version__ = "0.2.1"

__all__ = [
    "__version__",
    "dca",
    "configure_logging",
    "get_logger",
    "calculate_confidence_intervals",
    "holt_winters_forecast",
    "linear_trend_forecast",
    "moving_average_forecast",
    "simple_exponential_smoothing",
    "data_processing",
]


def __getattr__(name: str):
    """Lazily expose selected convenience symbols for backward compatibility."""
    if name in {
        "calculate_confidence_intervals",
        "holt_winters_forecast",
        "linear_trend_forecast",
        "moving_average_forecast",
        "simple_exponential_smoothing",
    }:
        from .forecast_statistical import (
            calculate_confidence_intervals,
            holt_winters_forecast,
            linear_trend_forecast,
            moving_average_forecast,
            simple_exponential_smoothing,
        )

        return {
            "calculate_confidence_intervals": calculate_confidence_intervals,
            "holt_winters_forecast": holt_winters_forecast,
            "linear_trend_forecast": linear_trend_forecast,
            "moving_average_forecast": moving_average_forecast,
            "simple_exponential_smoothing": simple_exponential_smoothing,
        }[name]

    if name == "data_processing":
        from .utils import data_processing

        return data_processing

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
