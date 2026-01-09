#!/bin/bash
# CLI Examples for Decline Curve Analysis
# These examples show how to use the command-line interface

# Example 1: Run batch job from config
echo "Example 1: Batch config run"
python -m decline_curve examples/config_example.toml

# Example 2: Legacy CSV mode (single well)
echo "Example 2: Forecast single well (legacy CSV)"
python -m decline_curve --csv data/production.csv --well WELL_001 --model arps --kind hyperbolic --horizon 12

# Example 3: Legacy CSV mode (benchmark)
echo "Example 3: Benchmark wells (legacy CSV)"
python -m decline_curve --csv data/production.csv --benchmark --model arps --top_n 10

# Benchmark/sensitivity workflows require user-provided config files:
# python -m decline_curve path/to/benchmark_config.toml --workflow benchmark
# python -m decline_curve path/to/sensitivity_config.toml --workflow sensitivity
