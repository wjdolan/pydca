"""
Plotting utilities for decline curve analysis with minimal production-ready aesthetics.

Uses signalplot for consistent minimalist styling based on Tufte's principles.
"""

from typing import TYPE_CHECKING, Mapping, Optional, Union

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .logging_config import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from .models import ArpsParams

# Try to import signalplot
try:
    import signalplot

    SIGNALPLOT_AVAILABLE = True
except ImportError:
    SIGNALPLOT_AVAILABLE = False
    logger.debug(
        "signalplot not available. Install with: pip install signalplot. "
        "Falling back to basic matplotlib styling."
    )


def minimal_style():
    """Apply minimalist production-ready styling to matplotlib plots.

    Uses signalplot if available, otherwise falls back to basic styling.
    Clean, professional aesthetics suitable for reports and presentations.
    Based on Tufte's principles of data-ink ratio and visual clarity.
    """
    if SIGNALPLOT_AVAILABLE:
        signalplot.apply()
    else:
        # Fallback to basic minimalist styling
        plt.style.use("default")
        plt.rcParams.update(
            {
                # Minimal frame
                "axes.linewidth": 0.5,
                "axes.spines.left": True,
                "axes.spines.bottom": True,
                "axes.spines.top": False,
                "axes.spines.right": False,
                # Clean ticks
                "xtick.bottom": True,
                "xtick.top": False,
                "ytick.left": True,
                "ytick.right": False,
                "xtick.direction": "out",
                "ytick.direction": "out",
                # Subtle grid
                "axes.grid": True,
                "grid.linewidth": 0.3,
                "grid.alpha": 0.5,
                "grid.color": "#E0E0E0",
                # Typography
                "font.size": 9,
                "axes.labelsize": 10,
                "axes.titlesize": 11,
                "legend.fontsize": 8,
                "legend.frameon": False,
                # Figure
                "figure.figsize": (10, 6),
                "figure.dpi": 100,
                "figure.facecolor": "white",
                "axes.facecolor": "white",
            }
        )


def _range_markers(ax, values: np.ndarray):
    """Add subtle range markers to indicate data spread (optional)."""
    # Removed for cleaner production plots
    pass


def plot_forecast(
    y_true: pd.Series,
    y_pred: pd.Series,
    title: str = "Production Forecast",
    filename: Optional[str] = None,
    show_metrics: bool = True,
):
    """
    Plot historical production data with forecast.

    Parameters:
    - y_true: Historical production data
    - y_pred: Forecasted production data
    - title: Plot title
    - filename: Optional filename to save plot
    - show_metrics: Whether to display evaluation metrics
    """
    minimal_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    # Split historical and forecast data
    hist_end = y_true.index[-1]
    hist_data = y_pred[y_pred.index <= hist_end]
    forecast_data = y_pred[y_pred.index > hist_end]

    # Use signalplot colors if available, otherwise use dark gray/black
    if SIGNALPLOT_AVAILABLE:
        primary_color = "black"
        secondary_color = "#666666"  # Medium gray
        accent_color = signalplot.ACCENT if hasattr(signalplot, "ACCENT") else "#C73E1D"
    else:
        primary_color = "#2E86AB"
        secondary_color = "#666666"
        accent_color = "#C73E1D"

    # Plot historical data
    ax.plot(
        y_true.index,
        y_true.values,
        "o-",
        color=primary_color,
        linewidth=1.5,
        markersize=3,
        label="Historical",
        alpha=0.9,
    )

    # Plot fitted curve on historical period
    if len(hist_data) > 0:
        ax.plot(
            hist_data.index,
            hist_data.values,
            "--",
            color=secondary_color,
            linewidth=1.5,
            label="Fitted",
            alpha=0.7,
        )

    # Plot forecast
    if len(forecast_data) > 0:
        ax.plot(
            forecast_data.index,
            forecast_data.values,
            "-",
            color=accent_color if SIGNALPLOT_AVAILABLE else primary_color,
            linewidth=2,
            label="Forecast",
            alpha=0.9,
        )

    # Formatting
    ax.set_xlabel("Date")
    ax.set_ylabel("Production Rate")
    ax.set_title(title, pad=15)

    # Format x-axis dates
    if isinstance(y_true.index, pd.DatetimeIndex):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Add metrics text box if requested
    if show_metrics and len(hist_data) > 0:
        from .evaluate import mae, rmse, smape

        common_idx = y_true.index.intersection(hist_data.index)
        if len(common_idx) > 0:
            y_common = y_true.loc[common_idx]
            pred_common = hist_data.loc[common_idx]

            metrics_text = f"RMSE: {rmse(y_common, pred_common):.1f}\n"
            metrics_text += f"MAE: {mae(y_common, pred_common):.1f}\n"
            metrics_text += f"SMAPE: {smape(y_common, pred_common):.1f}%"

            ax.text(
                0.02,
                0.98,
                metrics_text,
                transform=ax.transAxes,
                verticalalignment="top",
                fontsize=8,
                bbox=dict(
                    boxstyle="square,pad=0.5",
                    facecolor="white",
                    alpha=0.9,
                    edgecolor="#CCCCCC" if not SIGNALPLOT_AVAILABLE else "#E0E0E0",
                    linewidth=0.5,
                ),
            )

    ax.legend(loc="upper right", frameon=False)
    plt.tight_layout()

    if filename:
        # Use signalplot's save if available, otherwise standard save
        if SIGNALPLOT_AVAILABLE and hasattr(signalplot, "save"):
            signalplot.save(filename)
        else:
            plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor="white")
        logger.info(f"Plot saved to {filename}")

    plt.show()


ArpsParamsLike = Union["ArpsParams", Mapping[str, float]]


def plot_decline_curve(
    t: np.ndarray,
    q: np.ndarray,
    params: ArpsParamsLike,
    title: str = "Decline Curve Analysis",
):
    """
    Plot decline curve with fitted parameters.

    Parameters:
    - t: time array
    - q: production rate array
    - params: fitted decline curve parameters
    - title: plot title
    """
    from .models import predict_arps

    minimal_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Use signalplot colors if available
    if SIGNALPLOT_AVAILABLE:
        primary_color = "black"
        accent_color = signalplot.ACCENT if hasattr(signalplot, "ACCENT") else "#C73E1D"
    else:
        primary_color = "#2E86AB"
        accent_color = "#F18F01"

    # Linear plot
    t_extended = np.linspace(0, max(t) * 2, 100)
    q_fit = predict_arps(t_extended, params)

    ax1.plot(t, q, "o", color=primary_color, markersize=4, label="Data", alpha=0.8)
    ax1.plot(t_extended, q_fit, "-", color=accent_color, linewidth=1.5, label="Fitted")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Production Rate")
    ax1.set_title("Linear Scale")
    ax1.legend(frameon=False)
    if not SIGNALPLOT_AVAILABLE:
        ax1.grid(True, alpha=0.3)

    # Log plot
    ax2.semilogy(t, q, "o", color=primary_color, markersize=4, label="Data", alpha=0.8)
    ax2.semilogy(
        t_extended, q_fit, "-", color=accent_color, linewidth=1.5, label="Fitted"
    )
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Production Rate (log scale)")
    ax2.set_title("Semi-log Scale")
    ax2.legend(frameon=False)
    if not SIGNALPLOT_AVAILABLE:
        ax2.grid(True, alpha=0.3)

    # Add parameter text
    if isinstance(params, Mapping):
        param_kind = params.get("kind", "unknown")
        param_qi = params.get("qi", float("nan"))
        param_di = params.get("di", float("nan"))
        param_b = params.get("b")
        param_r2 = params.get("r2")
    else:
        param_kind = getattr(params, "kind", "unknown")
        param_qi = params.qi
        param_di = params.di
        param_b = getattr(params, "b", None)
        param_r2 = getattr(params, "r2", None)

    param_text = f"Model: {str(param_kind).title()}\n"
    param_text += f"qi: {param_qi:.1f}\n"
    param_text += f"di: {param_di:.4f}\n"
    if param_b is not None:
        param_text += f"b: {param_b:.3f}\n"
    if param_r2 is not None:
        param_text += f"R²: {param_r2:.3f}"

    ax1.text(
        0.02,
        0.98,
        param_text,
        transform=ax1.transAxes,
        verticalalignment="top",
        fontsize=8,
        bbox=dict(
            boxstyle="square,pad=0.5",
            facecolor="white",
            alpha=0.9,
            edgecolor="#E0E0E0" if SIGNALPLOT_AVAILABLE else "#CCCCCC",
            linewidth=0.5,
        ),
    )

    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_benchmark_results(
    results_df: pd.DataFrame,
    metric: str = "rmse",
    title: str = "Model Benchmark Results",
):
    """
    Plot benchmark results across multiple wells.

    Parameters:
    - results_df: DataFrame with benchmark results
    - metric: metric to plot ('rmse', 'mae', 'smape')
    - title: plot title
    """
    minimal_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    if metric in results_df.columns:
        results_sorted = results_df.sort_values(metric)

        # Use signalplot colors if available
        if SIGNALPLOT_AVAILABLE:
            bar_color = "black"
            edge_color = "white"
        else:
            bar_color = "#2E86AB"
            edge_color = "white"

        bars = ax.bar(
            range(len(results_sorted)),
            results_sorted[metric],
            color=bar_color,
            alpha=0.7,
            edgecolor=edge_color,
            linewidth=0.5,
        )

        ax.set_xlabel("Well Index")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{title} - {metric.upper()}")

        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # Add summary statistics
        mean_val = results_sorted[metric].mean()
        median_val = results_sorted[metric].median()

        # Use signalplot accent color if available
        if SIGNALPLOT_AVAILABLE:
            accent_color = (
                signalplot.ACCENT if hasattr(signalplot, "ACCENT") else "#C73E1D"
            )
            secondary_color = "#666666"
        else:
            accent_color = "red"
            secondary_color = "orange"

        ax.axhline(
            mean_val,
            color=accent_color,
            linestyle="--",
            alpha=0.7,
            linewidth=1,
            label=f"Mean: {mean_val:.1f}",
        )
        ax.axhline(
            median_val,
            color=secondary_color,
            linestyle="--",
            alpha=0.7,
            linewidth=1,
            label=f"Median: {median_val:.1f}",
        )

        ax.legend(frameon=False)
        plt.tight_layout()
        plt.show()
    else:
        logger.warning(f"Metric '{metric}' not found in results DataFrame")
