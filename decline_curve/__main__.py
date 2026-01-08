"""Command-line interface for decline curve analysis."""

import argparse
import logging
from pathlib import Path

import pandas as pd

from . import dca
from .logging_config import configure_logging, get_logger

logger = get_logger(__name__)


def main():
    """Run the CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Decline curve analysis tool - config-driven workflows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run batch job from config (primary mode)
  python -m decline_curve config.toml

  # Run benchmark from config
  python -m decline_curve benchmark_config.toml --workflow benchmark

  # Run sensitivity analysis from config
  python -m decline_curve sensitivity_config.toml --workflow sensitivity

  # Legacy mode: direct parameters
  python -m decline_curve --csv data.csv --well WELL_001
        """,
    )
    parser.add_argument(
        "config", nargs="?", help="Path to TOML/YAML config file (primary mode)"
    )
    parser.add_argument(
        "--workflow",
        choices=["batch", "benchmark", "sensitivity"],
        default="batch",
        help="Workflow type (auto-detected from config if not specified)",
    )
    parser.add_argument("--csv", help="Input CSV file (legacy mode)")
    parser.add_argument(
        "--model", default="arps", choices=["arps", "timesfm", "chronos"]
    )
    parser.add_argument(
        "--kind",
        default="hyperbolic",
        choices=["exponential", "harmonic", "hyperbolic"],
    )
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--well", help="Well ID to forecast (legacy mode)")
    parser.add_argument(
        "--benchmark", action="store_true", help="Run benchmark (legacy mode)"
    )
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()

    # Configure logging
    log_level = getattr(logging, args.log_level)
    if args.verbose:
        log_level = logging.DEBUG
    configure_logging(level=log_level)

    # Primary mode: config-driven workflow
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            logger.info("Creating example config...")
            try:
                from .config import create_example_config

                create_example_config(str(config_path), format="toml")
                logger.info(f"Example config created at {config_path}")
                logger.info("Please edit the config file and run again.")
            except Exception as e:
                logger.error(f"Failed to create example config: {e}")
            return

        # Auto-detect workflow from config file content if not specified
        workflow = args.workflow
        if workflow == "batch":
            try:
                import tomli

                with open(config_path, "rb") as f:
                    config_dict = tomli.load(f)
                # Try to detect workflow type from config keys
                if "param_grid" in config_dict or "prices" in config_dict:
                    workflow = "sensitivity"
                elif "top_n" in config_dict or "data" in config_dict:
                    # Could be batch or benchmark, check for benchmark-specific keys
                    if "top_n" in config_dict and config_dict.get("data", {}).get(
                        "path"
                    ):
                        workflow = "benchmark"
            except Exception:
                pass  # Use default

        logger.info(f"Running {workflow} workflow from config: {config_path}")
        try:
            if workflow == "benchmark":
                results = dca.run_benchmark(config_path)
                logger.info(
                    f"✓ Benchmark complete: {len(results['results'])} wells processed"
                )
            elif workflow == "sensitivity":
                results = dca.run_sensitivity_analysis(config_path)
                logger.info(
                    f"✓ Sensitivity analysis complete: "
                    f"{len(results['results'])} combinations tested"
                )
            else:  # batch
                results = dca.run_batch_job(config_path)
                logger.info(
                    f"✓ Batch job complete: "
                    f"{results['summary'].successful_wells} successful, "
                    f"{results['summary'].failed_wells} failed"
                )
            logger.info(f"Results saved to: {results['output_dir']}")
        except Exception as e:
            logger.error(f"{workflow.title()} workflow failed: {e}", exc_info=True)
            return

    # Legacy mode: direct parameters
    elif args.csv:
        df = pd.read_csv(args.csv)

        if args.benchmark:
            result = dca.benchmark(
                df,
                model=args.model,
                kind=args.kind,
                horizon=args.horizon,
                top_n=args.top_n,
                verbose=args.verbose,
            )
            logger.info(result.to_string(index=False))
        else:
            if args.well is None:
                raise ValueError("Must provide --well when not using --benchmark")
            sub = df[df["well_id"] == args.well].copy()
            sub["date"] = pd.to_datetime(sub["date"])
            y = sub.set_index("date")["oil_bbl"].asfreq("MS")
            yhat = dca.forecast(
                y,
                model=args.model,
                kind=args.kind,
                horizon=args.horizon,
                verbose=args.verbose,
            )
            dca.plot(y, yhat, title=f"{args.well} {args.model}")
    else:
        parser.print_help()
        logger.info("\nRecommended usage: python -m decline_curve <config.toml>")
