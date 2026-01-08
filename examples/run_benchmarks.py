#!/usr/bin/env python3
"""Run benchmark suite for decline curve analysis.

This script benchmarks the library on known datasets and reports
throughput and error rates.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from decline_curve.benchmarks import print_benchmark_summary, run_benchmark_suite


def load_dataset(path: str, name: str) -> dict:
    """Load a dataset from file.

    Args:
        path: Path to dataset file
        name: Dataset name

    Returns:
        Dataset dictionary
    """
    path_obj = Path(path)

    if not path_obj.exists():
        print(f"Warning: Dataset file not found: {path}")
        return None

    try:
        if path_obj.suffix == ".csv":
            df = pd.read_csv(path)
        elif path_obj.suffix in [".parquet", ".pq"]:
            df = pd.read_parquet(path)
        else:
            print(f"Warning: Unsupported file format: {path_obj.suffix}")
            return None

        # Validate required columns
        required_cols = ["well_id", "date", "oil_bbl"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"Warning: Missing required columns: {missing}")
            return None

        return {"name": name, "df": df, "config": {}}

    except Exception as e:
        print(f"Error loading dataset {name}: {e}")
        return None


def main():
    """Main benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Run decline curve analysis benchmarks"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Dataset files to benchmark (CSV or Parquet)",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        help="Names for datasets (must match --datasets)",
    )
    parser.add_argument(
        "--output",
        help="Output CSV file for results",
        default="benchmark_results.csv",
    )
    parser.add_argument(
        "--generate-sample",
        action="store_true",
        help="Generate sample dataset for testing",
    )

    args = parser.parse_args()

    # Generate sample dataset if requested
    if args.generate_sample:
        print("Generating sample dataset...")
        import numpy as np

        np.random.seed(42)
        data = []
        for i in range(100):  # 100 wells
            well_id = f"WELL_{i:03d}"
            dates = pd.date_range("2020-01-01", periods=36, freq="MS")
            qi = 1000 + np.random.normal(0, 200)
            di = 0.1 + np.random.normal(0, 0.02)
            b = 0.5 + np.random.normal(0, 0.1)
            t = np.arange(len(dates))
            production = qi / ((1 + b * di * t) ** (1 / b))
            production = production * (1 + np.random.normal(0, 0.1, len(production)))
            production = np.maximum(production, 0)

            for date, prod in zip(dates, production):
                data.append({"well_id": well_id, "date": date, "oil_bbl": prod})

        df = pd.DataFrame(data)
        output_path = "sample_benchmark_data.csv"
        df.to_csv(output_path, index=False)
        print(f"Sample dataset saved to {output_path}")
        return

    # Load datasets
    if not args.datasets:
        print(
            "No datasets specified. Use --datasets to specify files or --generate-sample to create test data."
        )
        sys.exit(1)

    if args.names and len(args.names) != len(args.datasets):
        print("Error: Number of --names must match number of --datasets")
        sys.exit(1)

    datasets = []
    for i, dataset_path in enumerate(args.datasets):
        name = args.names[i] if args.names else Path(dataset_path).stem
        dataset = load_dataset(dataset_path, name)
        if dataset:
            datasets.append(dataset)

    if not datasets:
        print("Error: No valid datasets loaded")
        sys.exit(1)

    # Run benchmarks
    print(f"\nRunning benchmarks on {len(datasets)} dataset(s)...")
    results_df = run_benchmark_suite(datasets, output_path=args.output)

    # Print summary
    print_benchmark_summary(results_df)

    print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
