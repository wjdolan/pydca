"""Example: Physics-Informed Decline Curve Analysis.

This example demonstrates the use of physics-informed models including:
1. Material balance-based decline
2. Pressure decline forecasting
3. Physics-informed constraints in neural networks
4. Reservoir simulation integration
"""

import numpy as np
import pandas as pd

from decline_curve.physics_informed import (
    MaterialBalanceParams,
    apply_physics_constraints,
    compare_dca_with_simulation,
    load_reservoir_simulation,
    material_balance_forecast,
    pressure_decline_forecast,
)


def example_1_material_balance():
    """Example 1: Material balance-based forecasting."""
    print("=" * 70)
    print("Example 1: Material Balance-Based Forecasting")
    print("=" * 70)

    # Generate sample production data
    dates = pd.date_range("2020-01-01", periods=36, freq="MS")
    production = pd.Series(
        1000 * np.exp(-0.01 * np.arange(36)), index=dates, name="oil"
    )

    # Forecast using material balance
    print("\n1. Generating material balance forecast...")
    forecast = material_balance_forecast(production, horizon=12)

    print(f"   Historical production: {len(production)} months")
    print(f"   Forecast horizon: {len(forecast)} months")
    print(f"   Last historical: {production.iloc[-1]:.1f} bbl/month")
    print(f"   First forecast: {forecast.iloc[0]:.1f} bbl/month")

    # With custom material balance parameters
    print("\n2. Material balance forecast with custom parameters...")
    mb_params = MaterialBalanceParams(
        N=1e6,  # Original oil in place (STB)
        Boi=1.2,  # Initial formation volume factor
        pi=5000.0,  # Initial pressure (psi)
    )

    forecast_custom = material_balance_forecast(
        production, material_balance_params=mb_params, horizon=12
    )

    print(f"   Custom forecast first value: {forecast_custom.iloc[0]:.1f} bbl/month")


def example_2_pressure_decline():
    """Example 2: Pressure decline-based forecasting."""
    print("\n" + "=" * 70)
    print("Example 2: Pressure Decline-Based Forecasting")
    print("=" * 70)

    # Generate sample pressure data
    dates = pd.date_range("2020-01-01", periods=24, freq="MS")
    pressure = pd.Series(
        5000 * np.exp(-0.001 * np.arange(24)), index=dates, name="pressure"
    )

    # Generate corresponding production (for calibration)
    production = pd.Series(
        1000 * np.exp(-0.01 * np.arange(24)), index=dates, name="oil"
    )

    print("\n1. Generating pressure decline forecast...")
    pressure_fcst, production_fcst = pressure_decline_forecast(
        pressure, production_data=production, horizon=12
    )

    print(f"   Historical pressure: {len(pressure)} months")
    print(f"   Forecast horizon: {len(pressure_fcst)} months")
    print(f"   Last historical pressure: {pressure.iloc[-1]:.1f} psi")
    print(f"   First forecast pressure: {pressure_fcst.iloc[0]:.1f} psi")
    print(f"   First forecast production: {production_fcst.iloc[0]:.1f} bbl/month")


def example_3_physics_constraints():
    """Example 3: Applying physics-informed constraints."""
    print("\n" + "=" * 70)
    print("Example 3: Physics-Informed Constraints")
    print("=" * 70)

    # Simulate an unconstrained forecast (might have unrealistic values)
    forecast = np.array([100, 120, 110, 90, 80, 70])  # Has increase
    historical = np.array([100, 95, 90])

    print("\n1. Unconstrained forecast:")
    print(f"   {forecast}")

    # Apply physics constraints
    print("\n2. Applying physics constraints...")
    constrained = apply_physics_constraints(
        forecast,
        historical=historical,
        min_rate=0.0,
        max_increase=0.1,  # Allow 10% increase max
        enforce_decline=True,  # Enforce monotonic decline
    )

    print(f"   Constrained forecast: {constrained}")
    print(f"   ✓ Non-negative: {all(constrained >= 0)}")
    print(
        f"   ✓ Monotonic decline: {all(constrained[i] <= constrained[i-1] for i in range(1, len(constrained)))}"
    )


def example_4_simulation_integration():
    """Example 4: Reservoir simulation integration."""
    print("\n" + "=" * 70)
    print("Example 4: Reservoir Simulation Integration")
    print("=" * 70)

    # Create sample simulation data
    dates = pd.date_range("2020-01-01", periods=12, freq="MS")
    sim_data = pd.DataFrame(
        {
            "pressure": 5000 * np.exp(-0.001 * np.arange(12)),
            "oil_rate": 1000 * np.exp(-0.01 * np.arange(12)),
            "gas_rate": 5000 * np.exp(-0.01 * np.arange(12)),
        },
        index=dates,
    )

    print("\n1. Sample simulation data:")
    print(f"   {sim_data.head()}")

    # Generate DCA forecast
    production = sim_data["oil_rate"]
    from decline_curve import dca

    dca_forecast = dca.forecast(production, model="arps", horizon=12)

    print("\n2. Comparing DCA forecast with simulation...")
    comparison = compare_dca_with_simulation(
        dca_forecast, sim_data, production_col="oil_rate"
    )

    print(f"   Comparison metrics:")
    print(f"   Mean difference: {comparison['difference'].mean():.1f} bbl/month")
    print(f"   Mean % difference: {comparison['pct_difference'].mean():.1f}%")
    print(f"   RMSE: {np.sqrt((comparison['difference']**2).mean()):.1f} bbl/month")


def example_5_physics_informed_neural_networks():
    """Example 5: Physics-informed neural networks."""
    print("\n" + "=" * 70)
    print("Example 5: Physics-Informed Neural Networks")
    print("=" * 70)

    print("\nPhysics-informed constraints are automatically applied to:")
    print("  - LSTM forecasts (EncoderDecoderLSTMForecaster)")
    print("  - Forecast post-processing constraints")
    print("  - Physical plausibility checks")
    print("\nConstraints ensure:")
    print("  ✓ Non-negative production rates")
    print("  ✓ Continuity with historical data")
    print("  ✓ Realistic forecast behavior")
    print("  ✓ Physical consistency across phases")

    print("\nExample usage:")
    print("  >>> constrained = apply_physics_constraints(raw_forecast)")
    print("  >>> # apply constraints before publishing")
    print("  >>> forecast = constrained")
    print("  # Physics constraints are automatically applied!")


def main():
    """Run all examples."""
    example_1_material_balance()
    example_2_pressure_decline()
    example_3_physics_constraints()
    example_4_simulation_integration()
    example_5_physics_informed_neural_networks()

    print("\n" + "=" * 70)
    print("Physics-Informed DCA Examples Complete!")
    print("=" * 70)
    print("\nKey Features Implemented:")
    print("  ✓ Material balance-based decline models")
    print("  ✓ Pressure decline forecasting")
    print("  ✓ Physics-informed constraints for neural networks")
    print("  ✓ Reservoir simulation integration")
    print("\nFor more information, see:")
    print("  - decline_curve.physics_informed module")
    print("  - decline_curve.dca.forecast(model='material_balance')")


if __name__ == "__main__":
    main()
