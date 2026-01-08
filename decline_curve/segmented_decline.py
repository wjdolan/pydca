"""Segmented decline curve analysis.

This module provides support for user-defined segmented decline curves where
users can define segments by date or cumulative volume. The library enforces
continuity and sensible parameter ranges.
"""

from dataclasses import dataclass
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd

from .logging_config import get_logger
from .models import ArpsParams, fit_arps, predict_arps

logger = get_logger(__name__)


@dataclass
class DeclineSegment:
    """A single segment in a segmented decline curve.

    Attributes:
        start_date: Start date of segment (if defined by date)
        end_date: End date of segment (if defined by date)
        start_cum: Start cumulative volume (if defined by volume)
        end_cum: End cumulative volume (if defined by volume)
        kind: Arps decline type for this segment
        params: Fitted Arps parameters for this segment
        t_start: Start time index (internal)
        t_end: End time index (internal)
    """

    kind: Literal["exponential", "harmonic", "hyperbolic"]
    params: ArpsParams
    t_start: float = 0.0
    t_end: Optional[float] = None
    start_date: Optional[pd.Timestamp] = None
    end_date: Optional[pd.Timestamp] = None
    start_cum: Optional[float] = None
    end_cum: Optional[float] = None

    def __post_init__(self):
        """Validate segment parameters."""
        if self.params.qi <= 0:
            raise ValueError("Initial rate (qi) must be positive")
        if self.params.di <= 0:
            raise ValueError("Decline rate (di) must be positive")
        if self.kind == "hyperbolic" and (self.params.b < 0 or self.params.b > 2.0):
            raise ValueError(f"b-factor must be between 0 and 2.0, got {self.params.b}")
        if self.kind == "exponential" and abs(self.params.b) > 1e-6:
            raise ValueError("b-factor must be 0 for exponential decline")
        if self.kind == "harmonic" and abs(self.params.b - 1.0) > 1e-6:
            raise ValueError("b-factor must be 1.0 for harmonic decline")


@dataclass
class SegmentedDeclineResult:
    """Result of segmented decline curve analysis.

    Attributes:
        segments: List of decline segments
        forecast: Complete forecast series
        continuity_errors: List of continuity violations (if any)
    """

    segments: list[DeclineSegment]
    forecast: pd.Series
    continuity_errors: list[str] = None

    def __post_init__(self):
        """Initialize continuity errors list."""
        if self.continuity_errors is None:
            self.continuity_errors = []


def _check_continuity(
    segment1: DeclineSegment, segment2: DeclineSegment, t_transition: float
) -> tuple[bool, Optional[str]]:
    """Check continuity between two segments at transition point.

    Args:
        segment1: First segment
        segment2: Second segment
        t_transition: Time of transition (relative to segment1 start)

    Returns:
        Tuple of (is_continuous, error_message)
    """
    # Calculate rate at end of segment1
    t1_local = t_transition - segment1.t_start
    q1_end = predict_arps(np.array([t1_local]), segment1.params)[0]

    # Calculate rate at start of segment2
    t2_local = 0.0
    q2_start = predict_arps(np.array([t2_local]), segment2.params)[0]

    # Check continuity (allow small tolerance for numerical errors)
    tolerance = max(q1_end, q2_start) * 0.01  # 1% tolerance
    if abs(q1_end - q2_start) > tolerance:
        return False, (
            f"Rate discontinuity at transition: "
            f"segment1 end={q1_end:.2f}, segment2 start={q2_start:.2f}"
        )

    return True, None


def _fit_segment(
    series: pd.Series,
    start_idx: int,
    end_idx: int,
    kind: Literal["exponential", "harmonic", "hyperbolic"],
) -> DeclineSegment:
    """Fit a single segment to production data.

    Args:
        series: Production time series
        start_idx: Start index (inclusive)
        end_idx: End index (exclusive)
        kind: Arps decline type

    Returns:
        Fitted DeclineSegment
    """
    segment_series = series.iloc[start_idx:end_idx]
    if len(segment_series) < 3:
        raise ValueError(
            f"Segment must have at least 3 data points, got {len(segment_series)}"
        )

    t = np.arange(len(segment_series))
    q = segment_series.values
    params = fit_arps(t, q, kind=kind)

    return DeclineSegment(
        kind=kind,
        params=params,
        t_start=start_idx,
        t_end=end_idx,
        start_date=(
            segment_series.index[0]
            if isinstance(series.index, pd.DatetimeIndex)
            else None
        ),
        end_date=(
            segment_series.index[-1]
            if isinstance(series.index, pd.DatetimeIndex)
            else None
        ),
    )


def segmented_decline_by_dates(
    series: pd.Series,
    segment_dates: list[tuple[pd.Timestamp, pd.Timestamp]],
    kinds: Optional[list[Literal["exponential", "harmonic", "hyperbolic"]]] = None,
    enforce_continuity: bool = True,
) -> SegmentedDeclineResult:
    """
    Fit segmented decline curve with segments defined by dates.

    Args:
        series: Historical production time series with DatetimeIndex
        segment_dates: List of (start_date, end_date) tuples for each segment.
            Dates should be in order and non-overlapping.
        kinds: List of Arps decline types for each segment.
            If None, uses 'hyperbolic' for all segments.
        enforce_continuity: If True, enforce rate continuity at segment boundaries.
            If False, allow discontinuities (may indicate data issues).

    Returns:
        SegmentedDeclineResult with fitted segments and forecast

    Example:
        >>> import pandas as pd
        >>> from decline_curve.segmented_decline import segmented_decline_by_dates
        >>> dates = pd.date_range('2020-01-01', periods=36, freq='MS')
        >>> production = pd.Series([...], index=dates)
        >>> segment_dates = [
        ...     (pd.Timestamp('2020-01-01'), pd.Timestamp('2020-12-01')),
        ...     (pd.Timestamp('2020-12-01'), pd.Timestamp('2022-12-01')),
        ... ]
        >>> result = segmented_decline_by_dates(
        ...     production, segment_dates, kinds=['hyperbolic', 'exponential']
        ... )
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Input series must have DatetimeIndex")

    if kinds is None:
        kinds = ["hyperbolic"] * len(segment_dates)
    if len(kinds) != len(segment_dates):
        raise ValueError("Number of kinds must match number of segments")

    # Validate and sort segment dates
    sorted_segments = sorted(segment_dates, key=lambda x: x[0])
    for i in range(len(sorted_segments) - 1):
        if sorted_segments[i][1] > sorted_segments[i + 1][0]:
            raise ValueError(
                f"Segments overlap: {sorted_segments[i]} and {sorted_segments[i+1]}"
            )

    # Fit each segment
    segments = []
    continuity_errors = []

    for i, (start_date, end_date) in enumerate(sorted_segments):
        # Find indices for this segment
        mask = (series.index >= start_date) & (series.index < end_date)
        segment_series = series[mask]

        if len(segment_series) < 3:
            logger.warning(
                f"Segment {i} has insufficient data: {len(segment_series)} points"
            )
            continue

        start_idx = series.index.get_loc(segment_series.index[0])
        end_idx = series.index.get_loc(segment_series.index[-1]) + 1

        try:
            segment = _fit_segment(series, start_idx, end_idx, kinds[i])
            segments.append(segment)
        except Exception as e:
            logger.error(f"Failed to fit segment {i}: {str(e)}")
            continuity_errors.append(f"Segment {i} fitting failed: {str(e)}")
            continue

        # Check continuity with previous segment
        if enforce_continuity and len(segments) > 1:
            prev_segment = segments[-2]
            curr_segment = segments[-1]
            t_transition = curr_segment.t_start

            is_continuous, error_msg = _check_continuity(
                prev_segment, curr_segment, t_transition
            )
            if not is_continuous:
                continuity_errors.append(error_msg)
                if enforce_continuity:
                    logger.warning(f"Continuity violation: {error_msg}")

    # Generate forecast by concatenating segment forecasts
    forecast_parts = []
    cumulative_time = 0.0

    for i, segment in enumerate(segments):
        # Calculate time range for this segment
        if i == 0:
            t_segment = np.arange(segment.t_start, segment.t_end, 1.0)
        else:
            # Continue from where previous segment ended
            t_segment = np.arange(0, segment.t_end - segment.t_start, 1.0)

        # Predict for this segment
        q_segment = predict_arps(t_segment, segment.params)

        # Create index for this segment
        if i == 0:
            segment_dates_idx = series.index[int(segment.t_start) : int(segment.t_end)]
        else:
            # Continue date index from previous segment
            last_date = forecast_parts[-1].index[-1]
            n_periods = len(q_segment)
            freq = series.index.freq or pd.DateOffset(months=1)
            segment_dates_idx = pd.date_range(
                last_date + freq, periods=n_periods, freq=freq
            )

        forecast_part = pd.Series(q_segment, index=segment_dates_idx)
        forecast_parts.append(forecast_part)
        cumulative_time += len(t_segment)

    # Concatenate all segments
    if forecast_parts:
        full_forecast = pd.concat(forecast_parts)
    else:
        full_forecast = pd.Series(dtype=float)

    return SegmentedDeclineResult(
        segments=segments, forecast=full_forecast, continuity_errors=continuity_errors
    )


def segmented_decline_by_cumulative(
    series: pd.Series,
    segment_cumulatives: list[tuple[float, float]],
    kinds: Optional[list[Literal["exponential", "harmonic", "hyperbolic"]]] = None,
    enforce_continuity: bool = True,
) -> SegmentedDeclineResult:
    """
    Fit segmented decline curve with segments defined by cumulative volume.

    Args:
        series: Historical production time series
        segment_cumulatives: List of (start_cum, end_cum) tuples for each segment.
            Cumulative volumes should be in order and non-overlapping.
        kinds: List of Arps decline types for each segment.
            If None, uses 'hyperbolic' for all segments.
        enforce_continuity: If True, enforce rate continuity at segment boundaries.

    Returns:
        SegmentedDeclineResult with fitted segments and forecast

    Example:
        >>> import pandas as pd
        >>> from decline_curve.segmented_decline import segmented_decline_by_cumulative
        >>> dates = pd.date_range('2020-01-01', periods=36, freq='MS')
        >>> production = pd.Series([...], index=dates)
        >>> # Define segments by cumulative production
        >>> segment_cumulatives = [
        ...     (0, 50000),      # First 50k bbl
        ...     (50000, 150000), # Next 100k bbl
        ... ]
        >>> result = segmented_decline_by_cumulative(
        ...     production, segment_cumulatives, kinds=['hyperbolic', 'exponential']
        ... )
    """
    if kinds is None:
        kinds = ["hyperbolic"] * len(segment_cumulatives)
    if len(kinds) != len(segment_cumulatives):
        raise ValueError("Number of kinds must match number of segments")

    # Calculate cumulative production
    cumulative = series.cumsum()

    # Validate and sort segment cumulatives
    sorted_segments = sorted(segment_cumulatives, key=lambda x: x[0])
    for i in range(len(sorted_segments) - 1):
        if sorted_segments[i][1] > sorted_segments[i + 1][0]:
            raise ValueError(
                f"Segments overlap: {sorted_segments[i]} and {sorted_segments[i+1]}"
            )

    # Find indices for each segment based on cumulative volume
    segments = []
    continuity_errors = []

    for i, (start_cum, end_cum) in enumerate(sorted_segments):
        # Find data points in this cumulative range
        mask = (cumulative >= start_cum) & (cumulative < end_cum)
        segment_series = series[mask]

        if len(segment_series) < 3:
            logger.warning(
                f"Segment {i} has insufficient data: {len(segment_series)} points"
            )
            continue

        start_idx = series.index.get_loc(segment_series.index[0])
        end_idx = series.index.get_loc(segment_series.index[-1]) + 1

        try:
            segment = _fit_segment(series, start_idx, end_idx, kinds[i])
            segment.start_cum = start_cum
            segment.end_cum = end_cum
            segments.append(segment)
        except Exception as e:
            logger.error(f"Failed to fit segment {i}: {str(e)}")
            continuity_errors.append(f"Segment {i} fitting failed: {str(e)}")
            continue

        # Check continuity with previous segment
        if enforce_continuity and len(segments) > 1:
            prev_segment = segments[-2]
            curr_segment = segments[-1]
            t_transition = curr_segment.t_start

            is_continuous, error_msg = _check_continuity(
                prev_segment, curr_segment, t_transition
            )
            if not is_continuous:
                continuity_errors.append(error_msg)
                if enforce_continuity:
                    logger.warning(f"Continuity violation: {error_msg}")

    # Generate forecast (similar to date-based version)
    forecast_parts = []

    for i, segment in enumerate(segments):
        if i == 0:
            t_segment = np.arange(segment.t_start, segment.t_end, 1.0)
        else:
            t_segment = np.arange(0, segment.t_end - segment.t_start, 1.0)

        q_segment = predict_arps(t_segment, segment.params)

        if i == 0:
            segment_dates_idx = series.index[int(segment.t_start) : int(segment.t_end)]
        else:
            last_date = forecast_parts[-1].index[-1]
            n_periods = len(q_segment)
            freq = (
                series.index.freq
                if hasattr(series.index, "freq")
                else pd.DateOffset(months=1)
            )
            segment_dates_idx = pd.date_range(
                last_date + freq, periods=n_periods, freq=freq
            )

        forecast_part = pd.Series(q_segment, index=segment_dates_idx)
        forecast_parts.append(forecast_part)

    if forecast_parts:
        full_forecast = pd.concat(forecast_parts)
    else:
        full_forecast = pd.Series(dtype=float)

    return SegmentedDeclineResult(
        segments=segments, forecast=full_forecast, continuity_errors=continuity_errors
    )
