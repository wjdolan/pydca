#!/bin/bash
# CLI Examples for Decline Curve Analysis
# These examples show how to use the command-line interface

# Example 1: Forecast a single well from CSV
echo "Example 1: Forecast single well"
dca fit data/production.csv --well WELL_001 --model arps --kind hyperbolic --horizon 12

# Example 2: Run benchmark on all wells
echo "Example 2: Benchmark all wells"
dca fit data/production.csv --benchmark --model arps --top_n 10

# Example 3: With verbose logging
echo "Example 3: Verbose output"
dca fit data/production.csv --well WELL_001 --model arps --verbose --log-level INFO

# Example 4: Using different models
echo "Example 4: ARIMA model"
dca fit data/production.csv --well WELL_001 --model arima --horizon 24

# Example 5: Export results
echo "Example 5: Save forecast plot"
dca fit data/production.csv --well WELL_001 --model arps --output forecast.png
