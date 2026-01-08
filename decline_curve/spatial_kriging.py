"""Kriging-based spatial interpolation for EUR estimation.

This module provides kriging-based spatial interpolation to improve EUR estimates
across thousands of wells by leveraging spatial correlation. It integrates with
the EUR estimation module to provide spatially-informed EUR predictions.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .logging_config import get_logger

logger = get_logger(__name__)

# Try to import kriging libraries
try:
    from pykrige import OrdinaryKriging, UniversalKriging

    PYKRIGE_AVAILABLE = True
except ImportError:
    PYKRIGE_AVAILABLE = False
    logger.debug("PyKrige not available. Install with: pip install pykrige")

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.gaussian_process.kernels import ConstantKernel as C

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.debug("scikit-learn not available. Install with: pip install scikit-learn")


@dataclass
class KrigingResult:
    """Result of kriging interpolation.

    Attributes:
        predicted_values: Predicted EUR values at target locations
        uncertainty: Prediction uncertainty (standard deviation)
        target_locations: Target locations (lon, lat) where predictions were made
        variogram_params: Variogram parameters used (if available)
        method: Kriging method used ('ordinary', 'universal', 'gaussian_process')
    """

    predicted_values: np.ndarray
    uncertainty: np.ndarray
    target_locations: np.ndarray
    variogram_params: Optional[Dict[str, float]] = None
    method: str = "ordinary"


def krige_eur(
    eur_df: pd.DataFrame,
    lon_col: str = "long",
    lat_col: str = "lat",
    eur_col: str = "eur",
    target_locations: Optional[np.ndarray] = None,
    method: str = "auto",
    variogram_model: str = "spherical",
    nlags: int = 6,
    weight: bool = True,
) -> KrigingResult:
    """
    Perform kriging interpolation for EUR values.

    This function uses spatial correlation to interpolate EUR values at
    unobserved locations, improving estimates for wells with limited data.

    Args:
        eur_df: DataFrame with EUR values and locations
        lon_col: Column name for longitude
        lat_col: Column name for latitude
        eur_col: Column name for EUR values
        target_locations: Target locations for interpolation [n_targets, 2] (lon, lat).
            If None, creates a grid covering the data extent.
        method: Kriging method:
            - 'auto': Automatically select best available (PyKrige > scikit-learn)
            - 'ordinary': Ordinary kriging (PyKrige)
            - 'universal': Universal kriging (PyKrige)
            - 'gaussian_process': Gaussian Process regression (scikit-learn)
        variogram_model: Variogram model type ('linear', 'power', 'gaussian',
            'spherical', 'exponential', 'hole-effect'). Only used with PyKrige.
        nlags: Number of lags for variogram estimation
        weight: Use distance-based weighting

    Returns:
        KrigingResult with predictions and uncertainty

    Example:
        >>> import pandas as pd
        >>> from decline_curve.spatial_kriging import krige_eur
        >>> # eur_df has columns: well_id, eur, long, lat
        >>> result = krige_eur(eur_df, lon_col='long', lat_col='lat', eur_col='eur')
        >>> print(f"Predicted EUR at locations: {result.predicted_values}")
    """
    # Validate input
    required_cols = [lon_col, lat_col, eur_col]
    missing = [col for col in required_cols if col not in eur_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Filter to valid data
    valid_mask = (
        eur_df[lon_col].notna()
        & eur_df[lat_col].notna()
        & eur_df[eur_col].notna()
        & (eur_df[eur_col] > 0)
    )
    eur_df = eur_df[valid_mask].copy()

    if len(eur_df) < 3:
        raise ValueError("Need at least 3 wells with valid EUR and location data")

    # Extract coordinates and values
    lon = eur_df[lon_col].values
    lat = eur_df[lat_col].values
    eur_values = eur_df[eur_col].values

    # Create target locations if not provided
    if target_locations is None:
        target_locations = _create_prediction_grid(lon, lat, resolution=50)

    # Auto-select method
    if method == "auto":
        # Check for pygeomodeling first (most advanced)
        try:
            import pygeomodeling

            return _krige_pygeomodeling(lon, lat, eur_values, target_locations)
        except ImportError:
            pass

        method_priority = [
            ("ordinary", PYKRIGE_AVAILABLE),
            ("gaussian_process", SKLEARN_AVAILABLE),
        ]

        method = next(
            (m for m, available in method_priority if available),
            None,
        )

        if method is None:
            raise ImportError(
                "No kriging library available. Install one of: "
                "pygeomodeling (pip install pygeomodeling), "
                "PyKrige (pip install pykrige), "
                "or scikit-learn (pip install scikit-learn)"
            )

    method_handlers = {
        "ordinary": lambda: _krige_pykrige(
            lon,
            lat,
            eur_values,
            target_locations,
            "ordinary",
            variogram_model,
            nlags,
            weight,
        ),
        "universal": lambda: _krige_pykrige(
            lon,
            lat,
            eur_values,
            target_locations,
            "universal",
            variogram_model,
            nlags,
            weight,
        ),
        "gaussian_process": lambda: _krige_gaussian_process(
            lon, lat, eur_values, target_locations
        ),
    }

    if method in ["ordinary", "universal"] and not PYKRIGE_AVAILABLE:
        raise ImportError(
            f"{method.capitalize()} kriging requires PyKrige. "
            "Install with: pip install pykrige"
        )

    handler = method_handlers.get(method)
    if handler is None:
        raise ValueError(
            f"Unknown kriging method: {method}. "
            f"Must be one of: {list(method_handlers.keys())}"
        )

    return handler()


def _krige_pykrige(
    lon: np.ndarray,
    lat: np.ndarray,
    values: np.ndarray,
    target_locations: np.ndarray,
    method: str,
    variogram_model: str = "spherical",
    nlags: int = 6,
    weight: bool = True,
) -> "KrigingResult":
    """Perform kriging using PyKrige.

    Args:
        lon: Longitude of observed points
        lat: Latitude of observed points
        values: Observed EUR values
        target_locations: Target locations [n_targets, 2] (lon, lat)
        method: 'ordinary' or 'universal'
        variogram_model: Variogram model type
        nlags: Number of lags
        weight: Use distance weighting

    Returns:
        KrigingResult
    """
    target_lon = target_locations[:, 0]
    target_lat = target_locations[:, 1]

    kriging_classes = {
        "ordinary": OrdinaryKriging,
        "universal": UniversalKriging,
    }

    kriging_class = kriging_classes.get(method)
    if kriging_class is None:
        raise ValueError(f"Unknown PyKrige method: {method}")

    try:
        kriging = kriging_class(
            lon,
            lat,
            values,
            variogram_model=variogram_model,
            nlags=nlags,
            weight=weight,
            verbose=False,
            enable_plotting=False,
        )
        z_pred, ss = kriging.execute("points", target_lon, target_lat)
        uncertainty = np.sqrt(ss)

        # Extract variogram parameters if available
        variogram_params = None
        if hasattr(kriging, "variogram_model_parameters"):
            params = kriging.variogram_model_parameters
            variogram_params = {
                "sill": params[0] if len(params) > 0 else None,
                "range": params[1] if len(params) > 1 else None,
                "nugget": params[2] if len(params) > 2 else None,
            }

        return KrigingResult(
            predicted_values=z_pred,
            uncertainty=uncertainty,
            target_locations=target_locations,
            variogram_params=variogram_params,
            method=method,
        )

    except Exception as e:
        logger.error(f"PyKrige kriging failed: {e}")
        raise


def _create_prediction_grid(
    lon: np.ndarray, lat: np.ndarray, resolution: int = 50
) -> np.ndarray:
    """Create a prediction grid covering the data extent.

    Args:
        lon: Longitude values
        lat: Latitude values
        resolution: Grid resolution (number of points per dimension)

    Returns:
        Array of target locations [n_points, 2] (lon, lat)
    """
    lon_min, lon_max = lon.min(), lon.max()
    lat_min, lat_max = lat.min(), lat.max()

    # Add 10% buffer
    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min
    lon_min -= lon_range * 0.1
    lon_max += lon_range * 0.1
    lat_min -= lat_range * 0.1
    lat_max += lat_range * 0.1

    # Create grid
    lon_grid = np.linspace(lon_min, lon_max, resolution)
    lat_grid = np.linspace(lat_min, lat_max, resolution)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

    # Flatten to [n_points, 2]
    target_locations = np.column_stack([lon_mesh.ravel(), lat_mesh.ravel()])

    return target_locations


def _krige_pygeomodeling(
    lon: np.ndarray,
    lat: np.ndarray,
    values: np.ndarray,
    target_locations: np.ndarray,
) -> KrigingResult:
    """Perform kriging using pygeomodeling's advanced GP models.

    Args:
        lon: Longitude of observed points
        lat: Latitude of observed points
        values: Observed EUR values
        target_locations: Target locations [n_targets, 2] (lon, lat)

    Returns:
        KrigingResult
    """
    try:
        from pygeomodeling import UnifiedSPE9Toolkit
    except ImportError:
        raise ImportError(
            "pygeomodeling not available. Install with: pip install pygeomodeling"
        )

    # Prepare training data
    X_train = np.column_stack([lon, lat])
    y_train = values

    # Use pygeomodeling's toolkit
    toolkit = UnifiedSPE9Toolkit()

    # Create sklearn GP model with combined kernel (best performance)
    model = toolkit.create_sklearn_model("gpr", kernel_type="combined")
    model.fit(X_train, y_train)

    # Predict
    z_pred, uncertainty = model.predict(target_locations, return_std=True)

    return KrigingResult(
        predicted_values=z_pred,
        uncertainty=uncertainty,
        target_locations=target_locations,
        method="pygeomodeling_gpr",
    )


def _krige_gaussian_process(
    lon: np.ndarray,
    lat: np.ndarray,
    values: np.ndarray,
    target_locations: np.ndarray,
) -> KrigingResult:
    """Perform kriging using scikit-learn Gaussian Process (fallback).

    Args:
        lon: Longitude of observed points
        lat: Latitude of observed points
        values: Observed EUR values
        target_locations: Target locations [n_targets, 2] (lon, lat)

    Returns:
        KrigingResult
    """
    from sklearn.preprocessing import StandardScaler

    # Prepare training data
    X_train = np.column_stack([lon, lat])
    y_train = values

    # Normalize coordinates
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_target_scaled = scaler.transform(target_locations)

    # Define kernel (RBF is similar to exponential variogram)
    kernel = C(1.0, (1e-3, 1e3)) * RBF([1.0, 1.0], (1e-2, 1e2))

    # Fit Gaussian Process
    gp = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=10, alpha=0.1, random_state=42
    )
    gp.fit(X_train_scaled, y_train)

    # Predict
    z_pred, uncertainty = gp.predict(X_target_scaled, return_std=True)

    return KrigingResult(
        predicted_values=z_pred,
        uncertainty=uncertainty,
        target_locations=target_locations,
        method="gaussian_process",
    )


def improve_eur_with_kriging(
    eur_df: pd.DataFrame,
    well_id_col: str = "well_id",
    lon_col: str = "long",
    lat_col: str = "lat",
    eur_col: str = "eur",
    min_wells_for_kriging: int = 10,
    kriging_method: str = "auto",
    use_leave_one_out: bool = True,
) -> pd.DataFrame:
    """
    Improve EUR estimates using kriging interpolation.

    This function uses spatial kriging to improve EUR estimates for wells,
    especially those with limited production history or uncertain decline fits.

    Args:
        eur_df: DataFrame with EUR values and well locations
        well_id_col: Column name for well identifier
        lon_col: Column name for longitude
        lat_col: Column name for latitude
        eur_col: Column name for EUR values
        min_wells_for_kriging: Minimum number of wells required for kriging
        kriging_method: Kriging method to use

    Returns:
        DataFrame with original EUR and kriged EUR estimates

    Example:
        >>> from decline_curve.eur_estimation import calculate_eur_batch
        >>> from decline_curve.spatial_kriging import improve_eur_with_kriging
        >>> # Calculate EUR for wells
        >>> eur_results = calculate_eur_batch(df, well_id_col='well_id')
        >>> # Improve with kriging
        >>> eur_improved = improve_eur_with_kriging(
        ...     eur_results,
        ...     lon_col='long',
        ...     lat_col='lat',
        ...     eur_col='eur'
        ... )
    """
    # Validate input
    required_cols = [well_id_col, lon_col, lat_col, eur_col]
    missing = [col for col in required_cols if col not in eur_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Filter to valid data
    valid_mask = (
        eur_df[lon_col].notna()
        & eur_df[lat_col].notna()
        & eur_df[eur_col].notna()
        & (eur_df[eur_col] > 0)
    )
    eur_valid = eur_df[valid_mask].copy()

    if len(eur_valid) < min_wells_for_kriging:
        logger.warning(
            f"Only {len(eur_valid)} wells with valid data. "
            f"Need at least {min_wells_for_kriging} for kriging. "
            "Returning original EUR values."
        )
        eur_df["eur_kriged"] = eur_df[eur_col]
        eur_df["kriging_uncertainty"] = np.nan
        return eur_df

    # Perform kriging at well locations
    eur_df = eur_df.copy()
    eur_df["eur_kriged"] = np.nan
    eur_df["kriging_uncertainty"] = np.nan

    try:
        if use_leave_one_out:
            # Leave-one-out cross-validation: for each well, predict using other wells
            # This is more accurate but slower
            logger.info("Using leave-one-out cross-validation for kriging")
            for idx, row in eur_valid.iterrows():
                # Create training set excluding current well
                train_mask = eur_valid.index != idx
                train_data = eur_valid[train_mask]

                if len(train_data) < 3:
                    # Not enough data for kriging, skip
                    continue

                # Target location is the current well
                target_loc = np.array([[row[lon_col], row[lat_col]]])

                # Perform kriging
                kriging_result = krige_eur(
                    train_data,
                    lon_col=lon_col,
                    lat_col=lat_col,
                    eur_col=eur_col,
                    target_locations=target_loc,
                    method=kriging_method,
                )

                # Store result
                eur_df.loc[idx, "eur_kriged"] = kriging_result.predicted_values[0]
                eur_df.loc[idx, "kriging_uncertainty"] = kriging_result.uncertainty[0]
        else:
            # Faster: single kriging pass using all wells
            # Note: This uses each well's own data in the kriging, which may
            # cause overfitting, but is much faster
            logger.info("Using single-pass kriging (faster but may overfit)")
            well_locations = eur_valid[[lon_col, lat_col]].values

            kriging_result = krige_eur(
                eur_valid,
                lon_col=lon_col,
                lat_col=lat_col,
                eur_col=eur_col,
                target_locations=well_locations,
                method=kriging_method,
            )

            # Map results back to DataFrame
            for i, idx in enumerate(eur_valid.index):
                eur_df.loc[idx, "eur_kriged"] = kriging_result.predicted_values[i]
                eur_df.loc[idx, "kriging_uncertainty"] = kriging_result.uncertainty[i]

        logger.info(
            f"Kriging interpolation complete for {len(eur_valid)} wells",
            extra={
                "n_wells": len(eur_valid),
                "method": kriging_result.method,
            },
        )

    except Exception as e:
        logger.error(f"Kriging failed: {e}")
        # Return original values on failure
        eur_df["eur_kriged"] = eur_df[eur_col]
        eur_df["kriging_uncertainty"] = np.nan

    return eur_df


def create_eur_map(
    eur_df: pd.DataFrame,
    lon_col: str = "long",
    lat_col: str = "lat",
    eur_col: str = "eur",
    resolution: int = 100,
    kriging_method: str = "auto",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create an EUR map using kriging interpolation.

    Generates a continuous EUR map by interpolating well EUR values across
    a spatial grid. Useful for visualization and identifying sweet spots.

    Args:
        eur_df: DataFrame with EUR values and locations
        lon_col: Column name for longitude
        lat_col: Column name for latitude
        eur_col: Column name for EUR values
        resolution: Grid resolution (number of points per dimension)
        kriging_method: Kriging method to use

    Returns:
        Tuple of (lon_grid, lat_grid, eur_map, uncertainty_map) where:
        - lon_grid, lat_grid: Grid coordinates
        - eur_map: Interpolated EUR values [resolution, resolution]
        - uncertainty_map: Prediction uncertainty [resolution, resolution]

    Example:
        >>> from decline_curve.spatial_kriging import create_eur_map
        >>> import matplotlib.pyplot as plt
        >>> lon_grid, lat_grid, eur_map, unc_map = create_eur_map(eur_df)
        >>> plt.contourf(lon_grid, lat_grid, eur_map)
        >>> plt.colorbar(label='EUR (bbl)')
    """
    # Extract coordinates and values
    valid_mask = (
        eur_df[lon_col].notna()
        & eur_df[lat_col].notna()
        & eur_df[eur_col].notna()
        & (eur_df[eur_col] > 0)
    )
    eur_valid = eur_df[valid_mask].copy()

    if len(eur_valid) < 3:
        raise ValueError("Need at least 3 wells with valid EUR and location data")

    lon = eur_valid[lon_col].values
    lat = eur_valid[lat_col].values

    # Create prediction grid
    lon_min, lon_max = lon.min(), lon.max()
    lat_min, lat_max = lat.min(), lat.max()

    # Add buffer
    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min
    lon_min -= lon_range * 0.1
    lon_max += lon_range * 0.1
    lat_min -= lat_range * 0.1
    lat_max += lat_range * 0.1

    # Create grid
    lon_grid = np.linspace(lon_min, lon_max, resolution)
    lat_grid = np.linspace(lat_min, lat_max, resolution)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

    # Flatten for kriging
    target_locations = np.column_stack([lon_mesh.ravel(), lat_mesh.ravel()])

    # Perform kriging
    kriging_result = krige_eur(
        eur_valid,
        lon_col=lon_col,
        lat_col=lat_col,
        eur_col=eur_col,
        target_locations=target_locations,
        method=kriging_method,
    )

    # Reshape results
    eur_map = kriging_result.predicted_values.reshape((resolution, resolution))
    uncertainty_map = kriging_result.uncertainty.reshape((resolution, resolution))

    return lon_mesh, lat_mesh, eur_map, uncertainty_map
