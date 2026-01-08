"""Stable input and output schemas for production tables.

This module defines the standard schemas for production data input and output,
ensuring consistency across the library and with external systems.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from .logging_config import get_logger

logger = get_logger(__name__)


class ProductionInputSchema:
    """
    Stable input schema for production tables.

    Required columns:
    - well_id: Well identifier (string)
    - date: Production date (datetime)
    - oil: Oil production (float, bbl/month)
    - gas: Gas production (float, mcf/month, optional)
    - water: Water production (float, bbl/month, optional)
    - status: Well status flag (string, optional): 'producing', 'shut_in', 'abandoned'

    The schema is designed to be compatible with common public data formats
    (NDIC, state regulatory databases, etc.).
    """

    REQUIRED_COLUMNS = ["well_id", "date", "oil"]
    OPTIONAL_COLUMNS = ["gas", "water", "status"]
    ALL_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

    @classmethod
    def validate(cls, df: pd.DataFrame) -> bool:
        """Validate DataFrame against input schema.

        Args:
            df: DataFrame to validate

        Returns:
            True if valid, raises ValueError if invalid
        """
        # Check required columns
        missing = set(cls.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Check date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            raise ValueError("'date' column must be datetime type")

        # Check numeric columns
        for col in ["oil", "gas", "water"]:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    raise ValueError(f"'{col}' column must be numeric")

        # Check status values if present
        if "status" in df.columns:
            valid_statuses = {"producing", "shut_in", "abandoned", None}
            invalid = set(df["status"].dropna().unique()) - valid_statuses
            if invalid:
                logger.warning(f"Invalid status values found: {invalid}")

        return True

    @classmethod
    def standardize(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize DataFrame to input schema.

        Args:
            df: DataFrame to standardize

        Returns:
            Standardized DataFrame
        """
        df = df.copy()

        # Ensure date is datetime
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        # Ensure numeric columns are float
        for col in ["oil", "gas", "water"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Sort by well_id and date
        if "well_id" in df.columns and "date" in df.columns:
            df = df.sort_values(["well_id", "date"])

        return df


class ProductionOutputSchema:
    """
    Standard output schema for production analysis.

    The output consists of three tables:

    1. Parameters table: Per-well parameters and summary metrics
       Columns: well_id, qi, di, b, rmse, mae, smape, eur, ...

    2. Forecasts table: Monthly forecasts
       Columns: well_id, date, oil_forecast, gas_forecast, water_forecast, ...

    3. Events table: Event markers and errors
       Columns: well_id, date, event_type, event_message, ...

    This schema ensures consistent output format across all analysis types.
    """

    @staticmethod
    def from_runner_results(results: Any) -> "ProductionOutputSchema":
        """Create output schema from runner results.

        Args:
            results: RunnerResults object (imported here to avoid circular dependency)

        Returns:
            ProductionOutputSchema instance
        """
        from .runner import (  # Import here to avoid circular dependency
            RunnerResults,
            WellResult,
        )

        # Build parameters table
        parameters_data = []
        forecasts_data = []
        events_data = []

        for well_result in results.well_results:
            well_id = well_result.well_id

            # Parameters row
            param_row = {"well_id": well_id, "success": well_result.success}
            if well_result.params:
                param_row.update(well_result.params)
            if well_result.metrics:
                param_row.update(well_result.metrics)
            parameters_data.append(param_row)

            # Forecast rows
            if well_result.forecast is not None:
                for date, value in well_result.forecast.items():
                    forecasts_data.append(
                        {
                            "well_id": well_id,
                            "date": date,
                            "oil_forecast": value,
                        }
                    )

            # Event rows
            if not well_result.success and well_result.error:
                events_data.append(
                    {
                        "well_id": well_id,
                        "date": None,
                        "event_type": "error",
                        "event_message": well_result.error,
                    }
                )
            for warning in well_result.warnings:
                events_data.append(
                    {
                        "well_id": well_id,
                        "date": None,
                        "event_type": "warning",
                        "event_message": warning,
                    }
                )

        return ProductionOutputSchema(
            parameters=pd.DataFrame(parameters_data),
            forecasts=pd.DataFrame(forecasts_data),
            events=pd.DataFrame(events_data),
        )

    def __init__(
        self,
        parameters: pd.DataFrame,
        forecasts: pd.DataFrame,
        events: pd.DataFrame,
    ):
        """Initialize output schema.

        Args:
            parameters: Parameters DataFrame
            forecasts: Forecasts DataFrame
            events: Events DataFrame
        """
        self.parameters = parameters
        self.forecasts = forecasts
        self.events = events

    def to_dataframes(self) -> Dict[str, pd.DataFrame]:
        """Convert to dictionary of DataFrames.

        Returns:
            Dictionary with keys 'parameters', 'forecasts', 'events'
        """
        return {
            "parameters": self.parameters,
            "forecasts": self.forecasts,
            "events": self.events,
        }

    def save(self, output_dir: str, format: str = "parquet"):
        """Save output to files.

        Args:
            output_dir: Output directory
            format: File format ('parquet', 'csv', 'excel')
        """
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if format == "parquet":
            self.parameters.to_parquet(output_path / "parameters.parquet", index=False)
            self.forecasts.to_parquet(output_path / "forecasts.parquet", index=False)
            self.events.to_parquet(output_path / "events.parquet", index=False)
        elif format == "csv":
            self.parameters.to_csv(output_path / "parameters.csv", index=False)
            self.forecasts.to_csv(output_path / "forecasts.csv", index=False)
            self.events.to_csv(output_path / "events.csv", index=False)
        elif format == "excel":
            with pd.ExcelWriter(output_path / "results.xlsx") as writer:
                self.parameters.to_excel(writer, sheet_name="parameters", index=False)
                self.forecasts.to_excel(writer, sheet_name="forecasts", index=False)
                self.events.to_excel(writer, sheet_name="events", index=False)
        else:
            raise ValueError(f"Unknown format: {format}")


def convert_to_input_schema(
    df: pd.DataFrame,
    well_id_col: str = "well_id",
    date_col: str = "date",
    oil_col: str = "oil",
    gas_col: Optional[str] = "gas",
    water_col: Optional[str] = "water",
    status_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convert DataFrame to standard input schema.

    This helper function maps common column names from various data sources
    (NDIC, state databases, etc.) to the standard input schema.

    Args:
        df: Input DataFrame
        well_id_col: Column name for well identifier
        date_col: Column name for date
        oil_col: Column name for oil production
        gas_col: Column name for gas production (optional)
        water_col: Column name for water production (optional)
        status_col: Column name for well status (optional)

    Returns:
        DataFrame with standard column names

    Example:
        >>> # Convert NDIC format
        >>> df_standard = convert_to_input_schema(
        ...     df_ndic,
        ...     well_id_col='API_WELLNO',
        ...     date_col='ReportDate',
        ...     oil_col='Oil',
        ...     gas_col='Gas',
        ...     water_col='Wtr'
        ... )
    """
    df = df.copy()

    # Map columns
    column_mapping = {
        well_id_col: "well_id",
        date_col: "date",
        oil_col: "oil",
    }

    if gas_col and gas_col in df.columns:
        column_mapping[gas_col] = "gas"
    if water_col and water_col in df.columns:
        column_mapping[water_col] = "water"
    if status_col and status_col in df.columns:
        column_mapping[status_col] = "status"

    # Rename columns
    df = df.rename(columns=column_mapping)

    # Ensure required columns exist
    if "well_id" not in df.columns:
        raise ValueError(f"Column '{well_id_col}' not found in DataFrame")
    if "date" not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in DataFrame")
    if "oil" not in df.columns:
        raise ValueError(f"Column '{oil_col}' not found in DataFrame")

    # Standardize
    return ProductionInputSchema.standardize(df)


def validate_input_schema(df: pd.DataFrame) -> bool:
    """Validate DataFrame against input schema.

    Args:
        df: DataFrame to validate

    Returns:
        True if valid

    Raises:
        ValueError if invalid
    """
    return ProductionInputSchema.validate(df)
