"""Core uncertainty representation for decline curve analysis.

This module defines the core representation of uncertainty used throughout
the library. We use forecast draws by period as the primary representation,
with parameter distributions as a secondary representation for specific use cases.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .logging_config import get_logger
from .models import ArpsParams

logger = get_logger(__name__)


@dataclass
class ForecastDraws:
    """
    Core representation of uncertainty: forecast draws by period.

    This is the primary uncertainty representation used throughout the library.
    Each draw represents one possible future realization of the production forecast.

    Attributes:
        draws: Array of forecast draws [n_draws, n_periods]
        dates: Date index for periods
        quantiles: Pre-computed quantiles (P10, P50, P90) for fast access
        metadata: Additional metadata about the draws
    """

    draws: np.ndarray  # Shape: (n_draws, n_periods)
    dates: pd.DatetimeIndex
    metadata: Dict[str, any] = None

    def __post_init__(self):
        """Initialize quantiles from draws."""
        if self.metadata is None:
            self.metadata = {}

    @property
    def p10(self) -> pd.Series:
        """P10 forecast (90th percentile - optimistic)."""
        return pd.Series(
            np.percentile(self.draws, 90, axis=0),
            index=self.dates,
            name="p10",
        )

    @property
    def p50(self) -> pd.Series:
        """P50 forecast (50th percentile - median)."""
        return pd.Series(
            np.percentile(self.draws, 50, axis=0),
            index=self.dates,
            name="p50",
        )

    @property
    def p90(self) -> pd.Series:
        """P90 forecast (10th percentile - conservative)."""
        return pd.Series(
            np.percentile(self.draws, 10, axis=0),
            index=self.dates,
            name="p90",
        )

    @property
    def mean(self) -> pd.Series:
        """Mean forecast."""
        return pd.Series(
            np.mean(self.draws, axis=0),
            index=self.dates,
            name="mean",
        )

    @property
    def std(self) -> pd.Series:
        """Standard deviation of forecasts."""
        return pd.Series(
            np.std(self.draws, axis=0),
            index=self.dates,
            name="std",
        )

    def get_quantile(self, q: float) -> pd.Series:
        """Get arbitrary quantile.

        Args:
            q: Quantile value (0-100)

        Returns:
            Quantile forecast series
        """
        return pd.Series(
            np.percentile(self.draws, q, axis=0),
            index=self.dates,
            name=f"p{int(q)}",
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with quantiles.

        Returns:
            DataFrame with columns: p10, p50, p90, mean, std
        """
        return pd.DataFrame(
            {
                "p10": self.p10,
                "p50": self.p50,
                "p90": self.p90,
                "mean": self.mean,
                "std": self.std,
            },
            index=self.dates,
        )


@dataclass
class ParameterDistribution:
    """
    Secondary uncertainty representation: parameter distributions.

    Used for specific use cases where parameter-level uncertainty is needed
    (e.g., sensitivity analysis, parameter correlation studies).

    Attributes:
        qi_dist: Distribution for initial rate (qi)
        di_dist: Distribution for decline rate (di)
        b_dist: Distribution for b-factor
        correlation: Optional correlation matrix between parameters
    """

    qi_dist: Dict[str, float]  # e.g., {'type': 'lognormal', 'mean': 1200, 'std': 0.3}
    di_dist: Dict[str, float]
    b_dist: Dict[str, float]
    correlation: Optional[np.ndarray] = None

    def sample(self, n: int, seed: Optional[int] = None) -> np.ndarray:
        """Sample parameters from distributions.

        Args:
            n: Number of samples
            seed: Random seed

        Returns:
            Array of parameter samples [n, 3] where columns are [qi, di, b]
        """
        if seed is not None:
            np.random.seed(seed)

        # Sample each parameter
        qi_samples = self._sample_param(self.qi_dist, n)
        di_samples = self._sample_param(self.di_dist, n)
        b_samples = self._sample_param(self.b_dist, n)

        samples = np.column_stack([qi_samples, di_samples, b_samples])

        # Apply correlation if provided
        if self.correlation is not None:
            # Use Cholesky decomposition for correlated sampling
            from scipy.linalg import cholesky

            try:
                L = cholesky(self.correlation, lower=True)
                # Transform to correlated samples
                samples = samples @ L.T
            except Exception as e:
                logger.warning(f"Failed to apply correlation: {e}")

        return samples

    def _sample_param(self, dist: Dict[str, float], n: int) -> np.ndarray:
        """Sample from a single parameter distribution."""
        dist_type = dist.get("type", "normal")

        if dist_type == "normal":
            return np.random.normal(dist["mean"], dist["std"], n)
        elif dist_type == "lognormal":
            # For lognormal, mean and std are in log space
            log_mean = np.log(dist["mean"])
            log_std = dist.get("std", 0.3)  # Coefficient of variation
            return np.random.lognormal(log_mean, log_std, n)
        elif dist_type == "uniform":
            return np.random.uniform(dist["min"], dist["max"], n)
        elif dist_type == "triangular":
            return np.random.triangular(dist["min"], dist["mode"], dist["max"], n)
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")


def parameter_distribution_to_forecast_draws(
    param_dist: ParameterDistribution,
    dates: pd.DatetimeIndex,
    n_draws: int = 1000,
    seed: Optional[int] = None,
) -> ForecastDraws:
    """
    Convert parameter distribution to forecast draws.

    This is the bridge between parameter-level and forecast-level uncertainty.

    Args:
        param_dist: Parameter distribution
        dates: Date index for forecast
        n_draws: Number of draws
        seed: Random seed

    Returns:
        ForecastDraws object
    """
    from .models import predict_arps

    # Sample parameters
    param_samples = param_dist.sample(n_draws, seed=seed)

    # Generate forecast for each parameter sample
    n_periods = len(dates)
    draws = np.zeros((n_draws, n_periods))

    t = np.arange(n_periods)

    for i, (qi, di, b) in enumerate(param_samples):
        params = ArpsParams(qi=qi, di=di, b=b)
        forecast = predict_arps(t, params)
        draws[i] = forecast

    return ForecastDraws(
        draws=draws,
        dates=dates,
        metadata={
            "source": "parameter_distribution",
            "n_draws": n_draws,
            "seed": seed,
        },
    )
