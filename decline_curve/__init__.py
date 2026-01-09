"""Decline Curve Analysis package.

A comprehensive Python library for decline curve analysis with multi-phase
forecasting, data utilities, and ML models.
"""

from . import (  # noqa: F401
    catalog,
    config,
    dca,
    eur_estimation,
    history_matching,
    integrations,
    ipr,
    material_balance,
    model_comparison,
    model_interface,
    model_registry,
    monte_carlo,
    multiphase,
    multiphase_flow,
    panel_analysis,
    panel_analysis_sweep,
    parameter_resample,
    physics_informed,
    physics_reserves,
    portfolio,
    probabilistic_forecast,
    profiling,
    pvt,
    risk_report,
    rta,
    runner,
    scenarios,
    schemas,
    segmented_decline,
    spatial_kriging,
    uncertainty_core,
    vlp,
    well_test,
)
from .forecast_statistical import (  # noqa: F401
    calculate_confidence_intervals,
    holt_winters_forecast,
    linear_trend_forecast,
    moving_average_forecast,
    simple_exponential_smoothing,
)
from .logging_config import configure_logging, get_logger  # noqa: F401

try:
    from . import deep_learning  # noqa: F401
except ImportError:
    # Deep learning module requires PyTorch
    pass

try:
    from . import ensemble, forecast_deepar, forecast_tft  # noqa: F401
except ImportError:
    # DeepAR, ensemble, and TFT modules require PyTorch
    pass

from .utils import data_processing  # noqa: F401

try:
    from . import benchmark_factory  # noqa: F401
except ImportError:
    # Benchmark factory requires optional dependencies
    pass

__version__ = "0.2.1"
