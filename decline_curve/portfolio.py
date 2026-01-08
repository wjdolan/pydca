"""Portfolio analysis and aggregation tools.

This module provides tools for aggregating well-level results by operator,
county, field, completion year, and other categories.
"""

from typing import Optional

import pandas as pd

from .logging_config import get_logger

logger = get_logger(__name__)


def aggregate_by_category(
    results_df: pd.DataFrame,
    category_col: str,
    well_id_col: str = "well_id",
    metrics: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Aggregate well results by category (operator, county, field, etc.).

    Args:
        results_df: DataFrame with well-level results
        category_col: Column name for aggregation category
        well_id_col: Column name for well identifier
        metrics: List of metric columns to aggregate (default: common metrics)

    Returns:
        DataFrame with aggregated results by category

    Example:
        >>> from decline_curve.portfolio import aggregate_by_category
        >>> operator_summary = aggregate_by_category(
        ...     eur_results, category_col='operator', metrics=['eur', 'npv']
        ... )
    """
    if metrics is None:
        # Default metrics to aggregate
        metrics = [
            "eur",
            "npv",
            "p10_eur",
            "p50_eur",
            "p90_eur",
            "p10_npv",
            "p50_npv",
            "p90_npv",
            "payback_month",
        ]
        # Only include metrics that exist in the DataFrame
        metrics = [m for m in metrics if m in results_df.columns]

    if category_col not in results_df.columns:
        raise ValueError(f"Category column '{category_col}' not found in results")

    # Group by category
    grouped = results_df.groupby(category_col)

    # Aggregate metrics using mapping
    sum_metrics = {
        "eur",
        "npv",
        "p10_eur",
        "p50_eur",
        "p90_eur",
        "p10_npv",
        "p50_npv",
        "p90_npv",
    }
    payback_metrics = {"payback_month"}

    agg_dict = {}
    for metric in metrics:
        if metric in results_df.columns:
            if metric in sum_metrics:
                agg_dict[metric] = ["sum", "mean", "std", "min", "max"]
            elif metric in payback_metrics:
                agg_dict[metric] = ["mean", "median", "min", "max"]
            else:
                agg_dict[metric] = ["mean", "std"]

    # Handle case where no metrics are found
    if agg_dict:
        aggregated = grouped.agg(agg_dict)
        # Flatten column names using vectorized operation
        aggregated.columns = [
            "_".join(col).strip() if len(col) > 1 and col[1] else col[0]
            for col in aggregated.columns.values
        ]
    else:
        # No metrics to aggregate, just create empty DataFrame with category
        aggregated = grouped.first().iloc[:, :0]  # Empty DataFrame with correct index

    # Add well counts
    aggregated["well_count"] = grouped.size()

    # Reset index to make category a column
    aggregated = aggregated.reset_index()

    return aggregated


def portfolio_summary(
    results_df: pd.DataFrame,
    category_cols: Optional[list[str]] = None,
    well_id_col: str = "well_id",
    include_p10_p50_p90: bool = True,
) -> pd.DataFrame:
    """
    Generate comprehensive portfolio summary with multiple aggregation levels.

    Args:
        results_df: DataFrame with well-level results
        category_cols: List of category columns to aggregate by
        well_id_col: Column name for well identifier
        include_p10_p50_p90: Whether to include P10/P50/P90 metrics

    Returns:
        DataFrame with portfolio summary

    Example:
        >>> from decline_curve.portfolio import portfolio_summary
        >>> summary = portfolio_summary(
        ...     results_df,
        ...     category_cols=['operator', 'county', 'completion_year']
        ... )
    """
    if category_cols is None:
        default_categories = ["operator", "county", "field", "completion_year", "state"]
        category_cols = [col for col in default_categories if col in results_df.columns]

    if not category_cols:
        logger.warning("No category columns found, returning overall summary only")
        return _overall_summary(results_df, include_p10_p50_p90)

    summaries = []

    # Overall summary
    overall = _overall_summary(results_df, include_p10_p50_p90)
    overall["category"] = "OVERALL"
    overall["category_value"] = "ALL"
    summaries.append(overall)

    # Summary by each category
    for category_col in category_cols:
        if category_col not in results_df.columns:
            logger.warning(f"Category column '{category_col}' not found, skipping")
            continue

        cat_summary = aggregate_by_category(
            results_df, category_col, well_id_col=well_id_col
        )
        cat_summary["category"] = category_col
        cat_summary = cat_summary.rename(columns={category_col: "category_value"})
        summaries.append(cat_summary)

    # Combine all summaries
    combined = pd.concat(summaries, ignore_index=True)

    return combined


def _overall_summary(
    results_df: pd.DataFrame, include_p10_p50_p90: bool = True
) -> pd.DataFrame:
    """Generate overall portfolio summary."""
    summary = {"well_count": len(results_df)}

    # Sum metrics using vectorized operations
    sum_metrics = ["eur", "npv"]
    available_sum_metrics = [m for m in sum_metrics if m in results_df.columns]

    for metric in available_sum_metrics:
        summary[f"{metric}_sum"] = results_df[metric].sum()
        summary[f"{metric}_mean"] = results_df[metric].mean()
        summary[f"{metric}_std"] = results_df[metric].std()
        summary[f"{metric}_min"] = results_df[metric].min()
        summary[f"{metric}_max"] = results_df[metric].max()

    # P10/P50/P90 if available
    if include_p10_p50_p90:
        percentiles = ["p10", "p50", "p90"]
        for metric in sum_metrics:
            for percentile in percentiles:
                col = f"{percentile}_{metric}"
                if col in results_df.columns:
                    summary[f"{col}_sum"] = results_df[col].sum()
                    summary[f"{col}_mean"] = results_df[col].mean()

    # Payback
    if "payback_month" in results_df.columns:
        valid_payback = results_df.loc[
            results_df["payback_month"] >= 0, "payback_month"
        ]
        if len(valid_payback) > 0:
            summary["payback_mean"] = valid_payback.mean()
            summary["payback_median"] = valid_payback.median()
            summary["payback_min"] = valid_payback.min()
            summary["payback_max"] = valid_payback.max()

    return pd.DataFrame([summary])


def cross_tabulate_metrics(
    results_df: pd.DataFrame,
    row_category: str,
    col_category: str,
    metric: str = "npv",
    agg_func: str = "sum",
) -> pd.DataFrame:
    """
    Create cross-tabulation of metrics by two categories.

    Args:
        results_df: DataFrame with well-level results
        row_category: Category for rows
        col_category: Category for columns
        metric: Metric to aggregate
        agg_func: Aggregation function ('sum', 'mean', 'count')

    Returns:
        Cross-tabulation DataFrame

    Example:
        >>> from decline_curve.portfolio import cross_tabulate_metrics
        >>> crosstab = cross_tabulate_metrics(
        ...     results_df,
        ...     row_category='operator',
        ...     col_category='completion_year',
        ...     metric='npv'
        ... )
    """
    if row_category not in results_df.columns:
        raise ValueError(f"Row category '{row_category}' not found")
    if col_category not in results_df.columns:
        raise ValueError(f"Column category '{col_category}' not found")

    # For count aggregation, metric column is not needed
    if agg_func != "count" and metric not in results_df.columns:
        raise ValueError(f"Metric '{metric}' not found")

    agg_func_map = {
        "sum": lambda: pd.crosstab(
            results_df[row_category],
            results_df[col_category],
            values=results_df[metric],
            aggfunc="sum",
        ),
        "mean": lambda: pd.crosstab(
            results_df[row_category],
            results_df[col_category],
            values=results_df[metric],
            aggfunc="mean",
        ),
        "count": lambda: pd.crosstab(
            results_df[row_category],
            results_df[col_category],
        ),
    }

    if agg_func not in agg_func_map:
        raise ValueError(
            f"Unknown aggregation function: {agg_func}. "
            f"Must be one of: {list(agg_func_map.keys())}"
        )

    crosstab = agg_func_map[agg_func]()

    return crosstab.fillna(0)


def portfolio_risk_summary(
    results_df: pd.DataFrame,
    category_col: Optional[str] = None,
    npv_col: str = "npv",
    p10_col: str = "p10_npv",
    p50_col: str = "p50_npv",
    p90_col: str = "p90_npv",
) -> pd.DataFrame:
    """
    Generate portfolio risk summary with P10/P50/P90 metrics.

    Args:
        results_df: DataFrame with well-level results
        category_col: Optional category column for grouping
        npv_col: Column name for NPV
        p10_col: Column name for P10 NPV
        p50_col: Column name for P50 NPV
        p90_col: Column name for P90 NPV

    Returns:
        DataFrame with risk metrics by category

    Example:
        >>> from decline_curve.portfolio import portfolio_risk_summary
        >>> risk_summary = portfolio_risk_summary(
        ...     results_df, category_col='operator'
        ... )
    """
    if category_col is None:
        # Overall summary
        summary = _calculate_risk_metrics(
            results_df, npv_col, p10_col, p50_col, p90_col
        )
        return pd.DataFrame([summary])

    # Group by category
    grouped = results_df.groupby(category_col)
    summaries = []

    for category_value, group_df in grouped:
        summary = _calculate_risk_metrics(group_df, npv_col, p10_col, p50_col, p90_col)
        summary[category_col] = category_value
        summaries.append(summary)

    return pd.DataFrame(summaries)


def _calculate_risk_metrics(
    df: pd.DataFrame, npv_col: str, p10_col: str, p50_col: str, p90_col: str
) -> dict:
    """Calculate risk metrics for a group of wells."""
    summary = {"well_count": len(df)}

    # NPV metrics
    if npv_col in df.columns:
        summary["npv_sum"] = df[npv_col].sum()
        summary["npv_mean"] = df[npv_col].mean()
        summary["npv_std"] = df[npv_col].std()
        summary["positive_npv_count"] = (df[npv_col] > 0).sum()
        summary["positive_npv_pct"] = (df[npv_col] > 0).mean() * 100

    # P10/P50/P90 if available
    for percentile, col in [("p10", p10_col), ("p50", p50_col), ("p90", p90_col)]:
        if col in df.columns:
            summary[f"{percentile}_sum"] = df[col].sum()
            summary[f"{percentile}_mean"] = df[col].mean()

    # Risk metrics
    if all(col in df.columns for col in [p10_col, p50_col, p90_col]):
        summary["npv_range"] = df[p10_col].sum() - df[p90_col].sum()
        summary["npv_uncertainty"] = (
            (df[p10_col].sum() - df[p90_col].sum()) / abs(df[p50_col].sum()) * 100
            if df[p50_col].sum() != 0
            else 0
        )

    return summary
