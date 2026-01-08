"""Unified interface for all forecasting models.

This module provides a unified interface that wraps each model family behind
a consistent API. Each model gets a class with fit and forecast methods, plus
a way to set a random seed and a horizon. The caller never touches framework details.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import pandas as pd

from .logging_config import get_logger

logger = get_logger(__name__)


class ForecastModel(ABC):
    """
    Unified interface for all forecasting models.

    All forecasting models (Arps, ARIMA, Chronos, TimesFM, etc.) implement
    this interface, ensuring consistent usage across the library.

    Example:
        >>> from decline_curve.model_interface import ForecastModel
        >>> model = ForecastModel.create('arps', kind='hyperbolic')
        >>> model.fit(series)
        >>> forecast = model.forecast(horizon=12)
    """

    def __init__(
        self,
        random_seed: Optional[int] = None,
        horizon: int = 12,
        **kwargs,
    ):
        """Initialize forecast model.

        Args:
            random_seed: Random seed for reproducibility
            horizon: Default forecast horizon
            **kwargs: Model-specific parameters
        """
        self.random_seed = random_seed
        self.horizon = horizon
        self._fitted = False
        self._fitted_params: Optional[Dict[str, Any]] = None

        # Set random seed if provided
        if random_seed is not None:
            self._set_random_seed(random_seed)

    @abstractmethod
    def fit(self, series: pd.Series) -> "ForecastModel":
        """
        Fit the model to historical data.

        Args:
            series: Historical production time series with DatetimeIndex

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def forecast(self, horizon: Optional[int] = None) -> pd.Series:
        """
        Generate forecast.

        Args:
            horizon: Forecast horizon (uses self.horizon if None)

        Returns:
            Forecasted production series

        Raises:
            ValueError: If model has not been fitted
        """
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Get fitted model parameters.

        Returns:
            Dictionary of parameter names and values
        """
        pass

    def _set_random_seed(self, seed: int):
        """Set random seed for reproducibility.

        Args:
            seed: Random seed value
        """
        import numpy as np

        np.random.seed(seed)

        # Also set for other libraries if available
        try:
            import random

            random.seed(seed)
        except ImportError:
            pass

    @property
    def is_fitted(self) -> bool:
        """Check if model has been fitted."""
        return self._fitted

    @classmethod
    def create(cls, model_name: str, **kwargs) -> "ForecastModel":
        """
        Create a model instance by name.

        This is a convenience method that uses the model registry.

        Args:
            model_name: Short model name ('arps', 'arima', 'chronos', 'timesfm')
            **kwargs: Model-specific parameters

        Returns:
            ForecastModel instance

        Example:
            >>> model = ForecastModel.create('arps', kind='hyperbolic')
            >>> model = ForecastModel.create('arima', order=(1,1,1))
        """
        from .model_registry import ModelRegistry

        registry = ModelRegistry()
        model_class = registry.get(model_name)
        if model_class is None:
            raise ValueError(
                f"Unknown model: {model_name}. Available: {registry.list()}"
            )

        return model_class(**kwargs)


class ArpsForecastModel(ForecastModel):
    """Arps decline curve model wrapper."""

    def __init__(
        self,
        kind: str = "hyperbolic",
        random_seed: Optional[int] = None,
        horizon: int = 12,
    ):
        """Initialize Arps model.

        Args:
            kind: Decline type ('exponential', 'harmonic', 'hyperbolic')
            random_seed: Random seed
            horizon: Default forecast horizon
        """
        super().__init__(random_seed=random_seed, horizon=horizon)
        self.kind = kind
        self._series: Optional[pd.Series] = None
        self._params: Optional[Dict[str, Any]] = None

    def fit(self, series: pd.Series) -> "ArpsForecastModel":
        """Fit Arps model to historical data."""
        import numpy as np

        from .models import fit_arps

        self._series = series
        t = np.arange(len(series))
        q = series.values
        params = fit_arps(t, q, kind=self.kind)

        self._params = {
            "qi": params.qi,
            "di": params.di,
            "b": params.b,
            "kind": self.kind,
        }
        self._fitted = True

        return self

    def forecast(self, horizon: Optional[int] = None) -> pd.Series:
        """Generate Arps forecast."""
        if not self._fitted:
            raise ValueError("Model must be fitted before forecasting")

        import numpy as np

        from .models import ArpsParams, predict_arps

        horizon = horizon or self.horizon
        full_t = np.arange(len(self._series) + horizon)
        params = ArpsParams(
            qi=self._params["qi"],
            di=self._params["di"],
            b=self._params["b"],
        )
        yhat = predict_arps(full_t, params)

        idx = pd.date_range(
            self._series.index[0],
            periods=len(yhat),
            freq=self._series.index.freq or "MS",
        )
        return pd.Series(yhat, index=idx, name=f"arps_{self.kind}")

    def get_params(self) -> Dict[str, Any]:
        """Get fitted parameters."""
        if not self._fitted:
            raise ValueError("Model must be fitted before getting parameters")
        return self._params.copy()


class ARIMAForecastModel(ForecastModel):
    """ARIMA model wrapper."""

    def __init__(
        self,
        order: Optional[tuple[int, int, int]] = None,
        seasonal: bool = False,
        seasonal_period: int = 12,
        random_seed: Optional[int] = None,
        horizon: int = 12,
    ):
        """Initialize ARIMA model.

        Args:
            order: ARIMA order (p, d, q). If None, auto-selects.
            seasonal: Whether to use seasonal ARIMA
            seasonal_period: Seasonal period
            random_seed: Random seed
            horizon: Default forecast horizon
        """
        super().__init__(random_seed=random_seed, horizon=horizon)
        self.order = order
        self.seasonal = seasonal
        self.seasonal_period = seasonal_period
        self._series: Optional[pd.Series] = None
        self._model = None

    def fit(self, series: pd.Series) -> "ARIMAForecastModel":
        """Fit ARIMA model to historical data."""
        try:
            from .forecast_arima import forecast_arima
        except ImportError:
            raise ImportError(
                "ARIMA requires statsmodels. Install with: pip install statsmodels"
            )

        self._series = series
        # Store for forecasting - actual fitting happens in forecast_arima
        self._fitted = True
        return self

    def forecast(self, horizon: Optional[int] = None) -> pd.Series:
        """Generate ARIMA forecast."""
        if not self._fitted:
            raise ValueError("Model must be fitted before forecasting")

        from .forecast_arima import forecast_arima

        horizon = horizon or self.horizon
        return forecast_arima(
            self._series,
            horizon=horizon,
            order=self.order,
            seasonal=self.seasonal,
            seasonal_period=self.seasonal_period,
        )

    def get_params(self) -> Dict[str, Any]:
        """Get fitted parameters."""
        if not self._fitted:
            raise ValueError("Model must be fitted before getting parameters")
        return {
            "order": self.order,
            "seasonal": self.seasonal,
            "seasonal_period": self.seasonal_period,
        }


class ChronosForecastModel(ForecastModel):
    """Chronos model wrapper."""

    def __init__(
        self,
        model_size: str = "small",
        random_seed: Optional[int] = None,
        horizon: int = 12,
    ):
        """Initialize Chronos model.

        Args:
            model_size: Model size ('tiny', 'small', 'base', 'large')
            random_seed: Random seed
            horizon: Default forecast horizon
        """
        super().__init__(random_seed=random_seed, horizon=horizon)
        self.model_size = model_size
        self._series: Optional[pd.Series] = None

    def fit(self, series: pd.Series) -> "ChronosForecastModel":
        """Fit Chronos model to historical data."""
        try:
            from .forecast_chronos import check_chronos_availability

            if not check_chronos_availability():
                raise ImportError(
                    "Chronos requires transformers. Install with: pip install decline-curve[llm]"
                )
        except ImportError:
            raise ImportError(
                "Chronos requires transformers. Install with: pip install decline-curve[llm]"
            )

        self._series = series
        self._fitted = True
        return self

    def forecast(self, horizon: Optional[int] = None) -> pd.Series:
        """Generate Chronos forecast."""
        if not self._fitted:
            raise ValueError("Model must be fitted before forecasting")

        from .forecast_chronos import forecast_chronos

        horizon = horizon or self.horizon
        return forecast_chronos(
            self._series,
            horizon=horizon,
            model_size=self.model_size,
        )

    def get_params(self) -> Dict[str, Any]:
        """Get fitted parameters."""
        if not self._fitted:
            raise ValueError("Model must be fitted before getting parameters")
        return {"model_size": self.model_size}


class TimesFMForecastModel(ForecastModel):
    """TimesFM model wrapper."""

    def __init__(
        self,
        random_seed: Optional[int] = None,
        horizon: int = 12,
    ):
        """Initialize TimesFM model.

        Args:
            random_seed: Random seed
            horizon: Default forecast horizon
        """
        super().__init__(random_seed=random_seed, horizon=horizon)
        self._series: Optional[pd.Series] = None

    def fit(self, series: pd.Series) -> "TimesFMForecastModel":
        """Fit TimesFM model to historical data."""
        try:
            from .forecast_timesfm import check_timesfm_availability

            if not check_timesfm_availability():
                raise ImportError(
                    "TimesFM requires transformers. Install with: pip install decline-curve[llm]"
                )
        except ImportError:
            raise ImportError(
                "TimesFM requires transformers. Install with: pip install decline-curve[llm]"
            )

        self._series = series
        self._fitted = True
        return self

    def forecast(self, horizon: Optional[int] = None) -> pd.Series:
        """Generate TimesFM forecast."""
        if not self._fitted:
            raise ValueError("Model must be fitted before forecasting")

        from .forecast_timesfm import forecast_timesfm

        horizon = horizon or self.horizon
        return forecast_timesfm(self._series, horizon=horizon)

    def get_params(self) -> Dict[str, Any]:
        """Get fitted parameters."""
        if not self._fitted:
            raise ValueError("Model must be fitted before getting parameters")
        return {}
