"""Integration with pygeomodeling and geosuite libraries.

This module provides integration points with other libraries in the ecosystem:
- pygeomodeling: Advanced Gaussian Process Regression and Kriging
- geosuite: Professional subsurface analysis tools

These integrations allow users to leverage the full ecosystem of tools
for comprehensive reservoir analysis workflows.
"""

from typing import Optional

import numpy as np
import pandas as pd

from .logging_config import get_logger

logger = get_logger(__name__)

# Try to import pygeomodeling
try:
    import pygeomodeling

    PYGEO_AVAILABLE = True
except ImportError:
    PYGEO_AVAILABLE = False
    logger.debug("pygeomodeling not available. Install with: pip install pygeomodeling")

# Try to import geosuite
try:
    import geosuite

    GEOSUITE_AVAILABLE = True
except ImportError:
    GEOSUITE_AVAILABLE = False
    logger.debug("geosuite not available. Install with: pip install geosuite")

# Try to import signalplot
try:
    import signalplot

    SIGNALPLOT_AVAILABLE = True
except ImportError:
    SIGNALPLOT_AVAILABLE = False
    logger.debug(
        "signalplot not available. Install with: pip install signalplot. "
        "Plots will use basic matplotlib styling."
    )


def krige_eur_with_pygeomodeling(
    eur_df: pd.DataFrame,
    lon_col: str = "long",
    lat_col: str = "lat",
    eur_col: str = "eur",
    target_locations: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform kriging using pygeomodeling's advanced GP models.

    This function uses pygeomodeling's UnifiedSPE9Toolkit for kriging,
    which provides advanced Gaussian Process models including Deep GP options.

    Args:
        eur_df: DataFrame with EUR values and locations
        lon_col: Column name for longitude
        lat_col: Column name for latitude
        eur_col: Column name for EUR values
        target_locations: Target locations [n_targets, 2] (lon, lat).
            If None, creates a grid covering the data extent.

    Returns:
        Tuple of (predicted_values, uncertainty)

    Example:
        >>> from decline_curve.integrations import krige_eur_with_pygeomodeling
        >>> predictions, uncertainty = krige_eur_with_pygeomodeling(eur_df)
    """
    if not PYGEO_AVAILABLE:
        raise ImportError(
            "pygeomodeling not available. Install with: pip install pygeomodeling"
        )

    try:
        from pygeomodeling import UnifiedSPE9Toolkit
    except ImportError:
        raise ImportError(
            "Could not import UnifiedSPE9Toolkit from pygeomodeling. "
            "Ensure pygeomodeling is properly installed."
        )

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
    eur_valid = eur_df[valid_mask].copy()

    if len(eur_valid) < 3:
        raise ValueError("Need at least 3 wells with valid EUR and location data")

    # Prepare data for pygeomodeling
    X_train = eur_valid[[lon_col, lat_col]].values
    y_train = eur_valid[eur_col].values

    # Create target locations if not provided
    if target_locations is None:
        from .spatial_kriging import _create_prediction_grid

        target_locations = _create_prediction_grid(
            eur_valid[lon_col].values, eur_valid[lat_col].values
        )

    # Use pygeomodeling's toolkit
    toolkit = UnifiedSPE9Toolkit()

    # Create sklearn GP model (pygeomodeling's interface)
    try:
        model = toolkit.create_sklearn_model("gpr", kernel_type="combined")
        model.fit(X_train, y_train)

        # Predict
        predictions, uncertainty = model.predict(target_locations, return_std=True)

        logger.info(
            f"Kriging with pygeomodeling complete: {len(predictions)} predictions "
            f"(method: pygeomodeling_gpr, n_wells: {len(eur_valid)})"
        )

        return predictions, uncertainty

    except Exception as e:
        logger.error(f"pygeomodeling kriging failed: {e}")
        raise


def load_well_data_with_geosuite(
    file_path: str, file_type: Optional[str] = None
) -> pd.DataFrame:
    """
    Load well data using geosuite's data import capabilities.

    This function uses geosuite to load well data from various formats
    (LAS, CSV, etc.) and converts it to the standard input schema.

    Args:
        file_path: Path to well data file
        file_type: File type ('las', 'csv', 'xlsx', etc.). If None, auto-detects.

    Returns:
        DataFrame in standard input schema format

    Example:
        >>> from decline_curve.integrations import load_well_data_with_geosuite
        >>> from decline_curve.schemas import convert_to_input_schema
        >>> df = load_well_data_with_geosuite('well_data.las')
        >>> df_standard = convert_to_input_schema(df)
    """
    if not GEOSUITE_AVAILABLE:
        raise ImportError("geosuite not available. Install with: pip install geosuite")

    try:
        file_type_map = {
            ".las": "las",
            ".csv": "csv",
            ".xlsx": "xlsx",
            ".xls": "xlsx",
        }

        if file_type is None:
            file_path_lower = file_path.lower()
            file_type = next(
                (
                    ftype
                    for ext, ftype in file_type_map.items()
                    if file_path_lower.endswith(ext)
                ),
                None,
            )

        loaders = {
            "las": lambda: _load_las_file(file_path),
            "csv": lambda: pd.read_csv(file_path),
            "xlsx": lambda: pd.read_excel(file_path),
        }

        if hasattr(geosuite, "load_well_data"):
            df = geosuite.load_well_data(file_path, file_type=file_type)
        elif hasattr(geosuite, "data") and hasattr(geosuite.data, "load"):
            df = geosuite.data.load(file_path)
        else:
            logger.warning("geosuite data loading API not found, using pandas fallback")
            loader = loaders.get(file_type)
            if loader is None:
                raise ValueError(f"Unknown file type: {file_type}")
            df = loader()

        return df

    except Exception as e:
        logger.error(f"Failed to load data with geosuite: {e}")
        raise


def calculate_petrophysical_properties_with_geosuite(
    well_logs: pd.DataFrame,
    depth_col: str = "depth",
    gr_col: str = "GR",
    nphi_col: str = "NPHI",
    rhob_col: str = "RHOB",
) -> pd.DataFrame:
    """
    Calculate petrophysical properties using geosuite.

    This function uses geosuite's petrophysics module to calculate
    properties like porosity, water saturation, etc. that can inform
    decline curve analysis.

    Args:
        well_logs: DataFrame with well log data
        depth_col: Column name for depth
        gr_col: Column name for gamma ray
        nphi_col: Column name for neutron porosity
        rhob_col: Column name for bulk density

    Returns:
        DataFrame with calculated petrophysical properties

    Example:
        >>> from decline_curve.integrations import calculate_petrophysical_properties_with_geosuite
        >>> petro_props = calculate_petrophysical_properties_with_geosuite(well_logs)
    """
    if not GEOSUITE_AVAILABLE:
        raise ImportError("geosuite not available. Install with: pip install geosuite")

    try:
        # Use geosuite's petrophysics functions
        # This is a placeholder - adjust based on actual geosuite API
        results = well_logs.copy()

        if hasattr(geosuite, "petro"):
            # Calculate water saturation if resistivity available
            if "RT" in well_logs.columns or "RES" in well_logs.columns:
                rt_col = "RT" if "RT" in well_logs.columns else "RES"
                if hasattr(geosuite.petro, "calculate_water_saturation"):
                    results["SW"] = geosuite.petro.calculate_water_saturation(
                        resistivity=well_logs[rt_col],
                        porosity=well_logs.get(nphi_col, 0.2),
                        rw=0.05,  # Default water resistivity
                    )

            # Calculate porosity if density available
            if rhob_col in well_logs.columns:
                if hasattr(geosuite.petro, "calculate_porosity"):
                    results["PHI"] = geosuite.petro.calculate_porosity(
                        density=well_logs[rhob_col],
                        matrix_density=2.65,  # Default sandstone
                    )

        logger.info("Calculated petrophysical properties with geosuite")
        return results

    except Exception as e:
        logger.error(f"Failed to calculate petrophysical properties: {e}")
        raise


def enhanced_eur_with_geosuite(
    production_df: pd.DataFrame,
    well_logs: Optional[pd.DataFrame] = None,
    well_id_col: str = "well_id",
    date_col: str = "date",
    value_col: str = "oil_bbl",
) -> pd.DataFrame:
    """
    Calculate EUR with enhanced analysis using geosuite.

    This function combines decline curve analysis with geosuite's
    petrophysical and geomechanical analysis to provide enhanced EUR estimates.

    Args:
        production_df: DataFrame with production data
        well_logs: Optional DataFrame with well log data (for petrophysical analysis)
        well_id_col: Column name for well identifier
        date_col: Column name for date
        value_col: Column name for production value

    Returns:
        DataFrame with EUR results and enhanced properties

    Example:
        >>> from decline_curve.integrations import enhanced_eur_with_geosuite
        >>> results = enhanced_eur_with_geosuite(production_df, well_logs=logs)
    """
    from .eur_estimation import calculate_eur_batch

    # Calculate standard EUR
    eur_results = calculate_eur_batch(
        production_df,
        well_id_col=well_id_col,
        date_col=date_col,
        value_col=value_col,
    )

    # Add geosuite analysis if well logs provided
    if well_logs is not None and GEOSUITE_AVAILABLE:
        try:
            petro_props = calculate_petrophysical_properties_with_geosuite(well_logs)

            # Merge petrophysical properties with EUR results
            # (assuming well_id_col matches)
            if well_id_col in petro_props.columns:
                eur_results = eur_results.merge(
                    petro_props[[well_id_col, "SW", "PHI"]].groupby(well_id_col).mean(),
                    on=well_id_col,
                    how="left",
                )
                logger.info("Added petrophysical properties to EUR results")

        except Exception as e:
            logger.warning(f"Failed to add geosuite analysis: {e}")

    return eur_results


def check_integration_availability() -> dict[str, bool]:
    """
    Check which integration libraries are available.

    Returns:
        Dictionary with availability status for each library

    Example:
        >>> from decline_curve.integrations import check_integration_availability
        >>> status = check_integration_availability()
        >>> print(f"signalplot: {status['signalplot']}")
        >>> print(f"pygeomodeling: {status['pygeomodeling']}")
        >>> print(f"geosuite: {status['geosuite']}")
    """
    return {
        "signalplot": SIGNALPLOT_AVAILABLE,
        "pygeomodeling": PYGEO_AVAILABLE,
        "geosuite": GEOSUITE_AVAILABLE,
    }
