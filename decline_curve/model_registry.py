"""Model registry for resolving short names to model classes.

This module provides a registry that maps short model names (e.g., 'arps', 'arima')
to their corresponding model classes. This keeps the CLI and config files clean.
"""

from typing import Dict, List, Optional, Type

from .logging_config import get_logger
from .model_interface import ForecastModel

logger = get_logger(__name__)


class ModelRegistry:
    """
    Registry for forecasting models.

    Maps short names to model classes, enabling clean CLI and config file usage.

    Example:
        >>> from decline_curve.model_registry import ModelRegistry
        >>> registry = ModelRegistry()
        >>> model_class = registry.get('arps')
        >>> model = model_class(kind='hyperbolic')
    """

    def __init__(self):
        """Initialize model registry."""
        self._models: Dict[str, Type[ForecastModel]] = {}
        self._register_defaults()

    def _register_defaults(self):
        """Register default models."""
        from .model_interface import (
            ARIMAForecastModel,
            ArpsForecastModel,
            ChronosForecastModel,
            TimesFMForecastModel,
        )

        # Classical decline (always available)
        self.register("arps", ArpsForecastModel)

        # Statistical models (require statsmodels)
        try:
            self.register("arima", ARIMAForecastModel)
        except ImportError:
            logger.debug("ARIMA not available (statsmodels not installed)")

        # LLM models (require transformers)
        try:
            self.register("chronos", ChronosForecastModel)
        except ImportError:
            logger.debug("Chronos not available (transformers not installed)")

        try:
            self.register("timesfm", TimesFMForecastModel)
        except ImportError:
            logger.debug("TimesFM not available (transformers not installed)")

    def register(self, name: str, model_class: Type[ForecastModel]):
        """Register a model class.

        Args:
            name: Short model name (e.g., 'arps', 'arima')
            model_class: Model class that implements ForecastModel interface
        """
        if not issubclass(model_class, ForecastModel):
            raise ValueError(f"Model class must inherit from ForecastModel")

        self._models[name] = model_class
        logger.debug(f"Registered model: {name} -> {model_class.__name__}")

    def get(self, name: str) -> Optional[Type[ForecastModel]]:
        """Get model class by name.

        Args:
            name: Short model name

        Returns:
            Model class or None if not found
        """
        return self._models.get(name)

    def list(self) -> List[str]:
        """List all registered model names.

        Returns:
            List of model names
        """
        return list(self._models.keys())

    def is_available(self, name: str) -> bool:
        """Check if a model is available.

        Args:
            name: Short model name

        Returns:
            True if model is registered and available
        """
        return name in self._models


# Global registry instance
_registry = None


def get_registry() -> ModelRegistry:
    """Get global model registry instance.

    Returns:
        ModelRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
