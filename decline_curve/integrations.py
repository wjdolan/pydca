"""Integration with pygeomodeling and geosuite libraries.

This module provides integration points with other libraries in the ecosystem:
- pygeomodeling: Advanced Gaussian Process Regression and Kriging
- geosuite: Professional subsurface analysis tools
- signalplot: Professional plotting styles

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
    import pygeomodeling  # noqa: F401

    PYGEO_AVAILABLE = True
except ImportError:
    PYGEO_AVAILABLE = False
    logger.debug("pygeomodeling not available. Install with: pip install pygeomodeling")

# Try to import geosuite
try:
    import geosuite  # noqa: F401

    GEOSUITE_AVAILABLE = True
except ImportError:
    GEOSUITE_AVAILABLE = False
    logger.debug("geosuite not available. Install with: pip install geosuite")

# Try to import signalplot
try:
    import signalplot  # noqa: F401

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

    # Try multiple API patterns for creating and using models
    try:
        # Pattern 1: create_sklearn_model method
        if hasattr(toolkit, "create_sklearn_model"):
            try:
                model = toolkit.create_sklearn_model("gpr", kernel_type="combined")
                model.fit(X_train, y_train)
                predictions, uncertainty = model.predict(
                    target_locations, return_std=True
                )
                logger.info(
                    f"Kriging with pygeomodeling complete: {len(predictions)} predictions "
                    f"(method: pygeomodeling_gpr via create_sklearn_model, n_wells: {len(eur_valid)})"
                )
                return predictions, uncertainty
            except Exception as e:
                logger.debug(f"create_sklearn_model failed: {e}")

        # Pattern 2: create_model method
        if hasattr(toolkit, "create_model"):
            try:
                model = toolkit.create_model("gpr", backend="sklearn")
                model.fit(X_train, y_train)
                predictions, uncertainty = model.predict(
                    target_locations, return_std=True
                )
                logger.info(
                    f"Kriging with pygeomodeling complete: {len(predictions)} predictions "
                    f"(method: pygeomodeling_gpr via create_model, n_wells: {len(eur_valid)})"
                )
                return predictions, uncertainty
            except Exception as e:
                logger.debug(f"create_model failed: {e}")

        # Pattern 3: Direct sklearn GP import from pygeomodeling
        try:
            from pygeomodeling.models import GaussianProcessRegressor as PyGeoGPR

            model = PyGeoGPR()
            model.fit(X_train, y_train)
            predictions, uncertainty = model.predict(target_locations, return_std=True)
            logger.info(
                f"Kriging with pygeomodeling complete: {len(predictions)} predictions "
                f"(method: pygeomodeling_gpr direct, n_wells: {len(eur_valid)})"
            )
            return predictions, uncertainty
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Direct PyGeoGPR failed: {e}")

        # If all patterns fail, raise informative error
        raise RuntimeError(
            "Could not find compatible pygeomodeling API. "
            "Tried: create_sklearn_model, create_model, and direct GaussianProcessRegressor import."
        )

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

        # Try multiple geosuite API patterns
        df = None

        # Pattern 1: Direct load_well_data function
        if hasattr(geosuite, "load_well_data"):
            try:
                df = geosuite.load_well_data(file_path, file_type=file_type)
                logger.debug("Loaded data using geosuite.load_well_data")
            except Exception as e:
                logger.debug(f"geosuite.load_well_data failed: {e}")

        # Pattern 2: geosuite.data.load
        if df is None and hasattr(geosuite, "data") and hasattr(geosuite.data, "load"):
            try:
                df = geosuite.data.load(file_path)
                logger.debug("Loaded data using geosuite.data.load")
            except Exception as e:
                logger.debug(f"geosuite.data.load failed: {e}")

        # Pattern 3: geosuite.io.load_las or similar
        if df is None and hasattr(geosuite, "io"):
            if hasattr(geosuite.io, "load_las") and file_type == "las":
                try:
                    df = geosuite.io.load_las(file_path)
                    logger.debug("Loaded data using geosuite.io.load_las")
                except Exception as e:
                    logger.debug(f"geosuite.io.load_las failed: {e}")
            elif hasattr(geosuite.io, "load") and file_type:
                try:
                    df = geosuite.io.load(file_path, file_type=file_type)
                    logger.debug(f"Loaded data using geosuite.io.load({file_type})")
                except Exception as e:
                    logger.debug(f"geosuite.io.load failed: {e}")

        # Pattern 4: Fallback to pandas with LAS support
        if df is None:
            logger.warning("geosuite data loading API not found, using fallback")
            if file_type == "las":
                df = _load_las_file(file_path)
            elif file_type == "csv":
                df = pd.read_csv(file_path)
            elif file_type in ("xlsx", "xls"):
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unknown file type: {file_type}")

        return df

    except Exception as e:
        logger.error(f"Failed to load data with geosuite: {e}")
        raise


def _load_las_file(file_path: str) -> pd.DataFrame:
    """Load LAS file using available libraries.

    Tries lasio first, then falls back to other methods.

    Args:
        file_path: Path to LAS file

    Returns:
        DataFrame with well log data
    """
    # Try lasio (most common LAS library)
    try:
        import lasio

        las = lasio.read(file_path)
        df = las.df()
        # Add depth column if not present
        if df.index.name != "DEPTH" and "DEPTH" not in df.columns:
            df.reset_index(inplace=True)
        return df
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"lasio failed to load {file_path}: {e}")

    # Try geosuite if available
    if GEOSUITE_AVAILABLE:
        try:
            if hasattr(geosuite, "io") and hasattr(geosuite.io, "read_las"):
                return geosuite.io.read_las(file_path)
        except Exception as e:
            logger.warning(f"geosuite LAS loading failed: {e}")

    # Final fallback: raise error
    raise ImportError(
        "LAS file loading requires lasio. Install with: pip install lasio"
    )


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
        results = well_logs.copy()

        # Try multiple geosuite petrophysics API patterns
        calculated = False

        # Pattern 1: geosuite.petro module
        if hasattr(geosuite, "petro"):
            try:
                # Calculate water saturation if resistivity available
                if "RT" in well_logs.columns or "RES" in well_logs.columns:
                    rt_col = "RT" if "RT" in well_logs.columns else "RES"

                    # Try different method names
                    sw_methods = [
                        "calculate_water_saturation",
                        "water_saturation",
                        "sw",
                        "calculate_sw",
                    ]
                    for method_name in sw_methods:
                        if hasattr(geosuite.petro, method_name):
                            method = getattr(geosuite.petro, method_name)
                            # Try different parameter patterns
                            try:
                                if nphi_col in well_logs.columns:
                                    results["SW"] = method(
                                        resistivity=well_logs[rt_col],
                                        porosity=well_logs[nphi_col],
                                    )
                                else:
                                    results["SW"] = method(
                                        resistivity=well_logs[rt_col],
                                        porosity=0.2,  # Default
                                    )
                                calculated = True
                                logger.debug(
                                    f"Calculated SW using geosuite.petro.{method_name}"
                                )
                                break
                            except (TypeError, ValueError):
                                continue

                # Calculate porosity if density available
                if rhob_col in well_logs.columns:
                    phi_methods = [
                        "calculate_porosity",
                        "porosity",
                        "phi",
                        "calculate_phi",
                    ]
                    for method_name in phi_methods:
                        if hasattr(geosuite.petro, method_name):
                            method = getattr(geosuite.petro, method_name)
                            try:
                                results["PHI"] = method(
                                    density=well_logs[rhob_col],
                                    matrix_density=2.65,  # Default sandstone
                                )
                                calculated = True
                                logger.debug(
                                    f"Calculated PHI using geosuite.petro.{method_name}"
                                )
                                break
                            except (TypeError, ValueError):
                                continue
            except Exception as e:
                logger.debug(f"geosuite.petro methods failed: {e}")

        # Pattern 2: geosuite.petrophysics module
        if not calculated and hasattr(geosuite, "petrophysics"):
            try:
                petro = geosuite.petrophysics
                if "RT" in well_logs.columns or "RES" in well_logs.columns:
                    rt_col = "RT" if "RT" in well_logs.columns else "RES"
                    if hasattr(petro, "sw"):
                        results["SW"] = petro.sw(
                            well_logs[rt_col], well_logs.get(nphi_col, 0.2)
                        )
                        calculated = True
                if rhob_col in well_logs.columns and hasattr(petro, "phi"):
                    results["PHI"] = petro.phi(well_logs[rhob_col])
                    calculated = True
            except Exception as e:
                logger.debug(f"geosuite.petrophysics methods failed: {e}")

        # Pattern 3: Direct function calls
        if not calculated:
            # Try top-level functions
            if hasattr(geosuite, "calculate_sw") and (
                "RT" in well_logs.columns or "RES" in well_logs.columns
            ):
                rt_col = "RT" if "RT" in well_logs.columns else "RES"
                try:
                    results["SW"] = geosuite.calculate_sw(well_logs[rt_col])
                    calculated = True
                except Exception:
                    pass

            if hasattr(geosuite, "calculate_phi") and rhob_col in well_logs.columns:
                try:
                    results["PHI"] = geosuite.calculate_phi(well_logs[rhob_col])
                    calculated = True
                except Exception:
                    pass

        if calculated:
            logger.info("Calculated petrophysical properties with geosuite")
        else:
            logger.warning(
                "Could not find geosuite petrophysics API, returning original data"
            )

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
