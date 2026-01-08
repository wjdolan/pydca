"""Runner for field and state scale workflows.

This module provides a runner concept that handles chunking, parallel workers,
retries, and logging for large-scale decline curve analysis. The forecast
functions stay pure - the runner handles the infrastructure concerns.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd

from .logging_config import get_logger

logger = get_logger(__name__)

try:
    from joblib import Parallel, delayed

    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    logger.warning(
        "joblib not available. Install with: pip install joblib. "
        "Parallel processing will be unavailable."
    )


@dataclass
class RunnerConfig:
    """Configuration for the runner.

    Attributes:
        n_jobs: Number of parallel workers (-1 for all cores, 1 for sequential)
        chunk_size: Number of wells to process per chunk
        max_retries: Maximum number of retries for failed wells
        retry_delay: Delay in seconds between retries
        log_file: Optional path to log file
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        progress_bar: Show progress bar
        save_intermediate: Save intermediate results after each chunk
        intermediate_dir: Directory for intermediate results
    """

    n_jobs: int = -1
    chunk_size: int = 100
    max_retries: int = 2
    retry_delay: float = 1.0
    log_file: Optional[str] = None
    log_level: str = "INFO"
    progress_bar: bool = True
    save_intermediate: bool = False
    intermediate_dir: str = "intermediate_results"


@dataclass
class WellResult:
    """Result for a single well.

    Attributes:
        well_id: Well identifier
        success: Whether processing succeeded
        params: Fitted parameters (if successful)
        forecast: Forecast series (if successful)
        metrics: Evaluation metrics
        error: Error message (if failed)
        warnings: List of warning messages
    """

    well_id: str
    success: bool
    params: Optional[Dict[str, Any]] = None
    forecast: Optional[pd.Series] = None
    metrics: Optional[Dict[str, float]] = None
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class RunnerResults:
    """Results from runner execution.

    Attributes:
        well_results: List of WellResult objects
        summary: Summary statistics
        errors: List of error messages
        warnings: List of warning messages
    """

    well_results: List[WellResult]
    summary: Dict[str, Any]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dataframes(self) -> Dict[str, pd.DataFrame]:
        """Convert results to standard output schema DataFrames.

        Returns:
            Dictionary with keys:
            - 'parameters': Per-well parameters and metrics
            - 'forecasts': Monthly forecasts
            - 'events': Event markers and errors
        """
        from .schemas import ProductionOutputSchema

        return ProductionOutputSchema.from_runner_results(self).to_dataframes()


class FieldScaleRunner:
    """
    Runner for field and state scale decline curve analysis.

    This runner handles:
    - Chunking large datasets into manageable pieces
    - Parallel processing with configurable workers
    - Retry logic for failed wells
    - Comprehensive logging (console and file)
    - Progress tracking
    - Intermediate result saving

    The forecast functions stay pure - this runner handles all the
    infrastructure concerns.

    Example:
        >>> from decline_curve.runner import FieldScaleRunner, RunnerConfig
        >>> from decline_curve import dca
        >>>
        >>> # Load production data
        >>> df = pd.read_csv('production_data.csv')
        >>>
        >>> # Configure runner
        >>> config = RunnerConfig(n_jobs=4, chunk_size=100, max_retries=2)
        >>> runner = FieldScaleRunner(config)
        >>>
        >>> # Define forecast function
        >>> def forecast_well(well_data: pd.DataFrame) -> WellResult:
        ...     series = well_data['oil_bbl']
        ...     forecast = dca.single_well(series, model='arps', horizon=12)
        ...     return WellResult(well_id=well_data['well_id'].iloc[0], ...)
        >>>
        >>> # Run analysis
        >>> results = runner.run(df, forecast_well)
        >>> output_dfs = results.to_dataframes()
    """

    def __init__(self, config: Optional[RunnerConfig] = None):
        """Initialize runner with configuration.

        Args:
            config: Runner configuration. If None, uses defaults.
        """
        self.config = config or RunnerConfig()
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging according to configuration."""
        from .logging_config import configure_logging

        configure_logging(
            level=self.config.log_level,
            log_file=self.config.log_file,
        )

    def run(
        self,
        df: pd.DataFrame,
        forecast_func: Callable[[pd.DataFrame], WellResult],
        well_id_col: str = "well_id",
        date_col: str = "date",
    ) -> RunnerResults:
        """
        Run field-scale decline curve analysis.

        Args:
            df: Production DataFrame (will be validated and converted to input schema)
            forecast_func: Function that takes a well DataFrame and returns WellResult
            well_id_col: Column name for well identifier
            date_col: Column name for date

        Returns:
            RunnerResults with all well results and summary
        """
        logger.info(
            f"Starting field-scale analysis",
            extra={
                "n_wells": (
                    df[well_id_col].nunique() if well_id_col in df.columns else 0
                ),
                "n_records": len(df),
                "n_jobs": self.config.n_jobs,
                "chunk_size": self.config.chunk_size,
            },
        )

        # Convert to input schema
        try:
            from .schemas import convert_to_input_schema, validate_input_schema

            df_input = convert_to_input_schema(
                df, well_id_col=well_id_col, date_col=date_col
            )
            validate_input_schema(df_input)
        except Exception as e:
            logger.error(f"Input schema validation failed: {e}")
            raise

        # Get unique wells
        well_ids = df_input["well_id"].unique()
        n_wells = len(well_ids)

        logger.info(f"Processing {n_wells} wells in chunks of {self.config.chunk_size}")

        # Process in chunks
        all_results = []
        errors = []
        warnings = []

        chunks = [
            well_ids[i : i + self.config.chunk_size]
            for i in range(0, n_wells, self.config.chunk_size)
        ]

        for chunk_idx, chunk_well_ids in enumerate(chunks):
            logger.info(
                f"Processing chunk {chunk_idx + 1}/{len(chunks)} "
                f"({len(chunk_well_ids)} wells)"
            )

            chunk_results = self._process_chunk(
                df_input, chunk_well_ids, forecast_func, chunk_idx
            )

            all_results.extend(chunk_results)

            # Save intermediate results if requested
            if self.config.save_intermediate:
                self._save_intermediate(chunk_results, chunk_idx)

        # Generate summary
        summary = self._generate_summary(all_results)

        logger.info(
            f"Field-scale analysis complete",
            extra={
                "n_wells": n_wells,
                "successful": summary["n_successful"],
                "failed": summary["n_failed"],
                "success_rate": summary["success_rate"],
            },
        )

        return RunnerResults(
            well_results=all_results,
            summary=summary,
            errors=errors,
            warnings=warnings,
        )

    def _process_chunk(
        self,
        df: pd.DataFrame,
        well_ids: List[str],
        forecast_func: Callable[[pd.DataFrame], WellResult],
        chunk_idx: int,
    ) -> List[WellResult]:
        """Process a chunk of wells.

        Args:
            df: Input DataFrame
            well_ids: List of well IDs in this chunk
            forecast_func: Forecast function
            chunk_idx: Chunk index for logging

        Returns:
            List of WellResult objects
        """
        if JOBLIB_AVAILABLE and self.config.n_jobs != 1:
            # Parallel processing
            results = Parallel(n_jobs=self.config.n_jobs, verbose=0)(
                delayed(self._process_single_well_with_retry)(
                    df[df["well_id"] == well_id], well_id, forecast_func
                )
                for well_id in well_ids
            )
        else:
            # Sequential processing
            results = [
                self._process_single_well_with_retry(
                    df[df["well_id"] == well_id], well_id, forecast_func
                )
                for well_id in well_ids
            ]

        return results

    def _process_single_well_with_retry(
        self,
        well_df: pd.DataFrame,
        well_id: str,
        forecast_func: Callable[[pd.DataFrame], WellResult],
    ) -> WellResult:
        """Process a single well with retry logic.

        Args:
            well_df: DataFrame for this well
            well_id: Well identifier
            forecast_func: Forecast function

        Returns:
            WellResult
        """
        import time

        last_error = None
        for attempt in range(self.config.max_retries + 1):
            try:
                result = forecast_func(well_df)
                if result.success:
                    return result
                else:
                    last_error = result.error
            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"Well {well_id} failed (attempt {attempt + 1}/{self.config.max_retries + 1}): {e}"
                )

            if attempt < self.config.max_retries:
                time.sleep(self.config.retry_delay)

        # All retries failed
        logger.error(
            f"Well {well_id} failed after {self.config.max_retries + 1} attempts"
        )
        return WellResult(
            well_id=well_id,
            success=False,
            error=last_error or "Unknown error",
        )

    def _save_intermediate(self, results: List[WellResult], chunk_idx: int):
        """Save intermediate results.

        Args:
            results: Results from current chunk
            chunk_idx: Chunk index
        """
        intermediate_dir = Path(self.config.intermediate_dir)
        intermediate_dir.mkdir(parents=True, exist_ok=True)

        # Convert to DataFrames
        runner_results = RunnerResults(
            well_results=results, summary={}, errors=[], warnings=[]
        )
        dfs = runner_results.to_dataframes()

        # Save each DataFrame
        for name, df in dfs.items():
            output_file = intermediate_dir / f"chunk_{chunk_idx}_{name}.parquet"
            df.to_parquet(output_file, index=False)

        logger.debug(f"Saved intermediate results for chunk {chunk_idx}")

    def _generate_summary(self, results: List[WellResult]) -> Dict[str, Any]:
        """Generate summary statistics.

        Args:
            results: List of all well results

        Returns:
            Summary dictionary
        """
        n_total = len(results)
        n_successful = sum(1 for r in results if r.success)
        n_failed = n_total - n_successful

        summary = {
            "n_total": n_total,
            "n_successful": n_successful,
            "n_failed": n_failed,
            "success_rate": n_successful / n_total if n_total > 0 else 0.0,
        }

        # Add metrics summary if available
        successful_results = [r for r in results if r.success and r.metrics]
        if successful_results:
            metrics_df = pd.DataFrame([r.metrics for r in successful_results])
            summary["metrics"] = {
                "mean_rmse": (
                    metrics_df["rmse"].mean() if "rmse" in metrics_df else None
                ),
                "mean_mae": metrics_df["mae"].mean() if "mae" in metrics_df else None,
                "mean_smape": (
                    metrics_df["smape"].mean() if "smape" in metrics_df else None
                ),
            }

        return summary
