"""Example: Integration with pygeomodeling and geosuite.

This example demonstrates how to use decline-curve with other libraries
in the ecosystem for comprehensive reservoir analysis.
"""

import numpy as np
import pandas as pd


# Example data generation
def generate_example_data():
    """Generate example production and well log data."""
    np.random.seed(42)

    n_wells = 30
    well_ids = [f"WELL_{i:03d}" for i in range(n_wells)]

    production_data = []
    for well_id in well_ids:
        dates = pd.date_range("2020-01-01", periods=36, freq="MS")
        qi = 1000 + np.random.normal(0, 200)
        di = 0.1 + np.random.normal(0, 0.02)
        b = 0.5 + np.random.normal(0, 0.1)
        t = np.arange(len(dates))
        production = qi / ((1 + b * di * t) ** (1 / b))
        production = production * (1 + np.random.normal(0, 0.1, len(production)))
        production = np.maximum(production, 0)

        for date, prod in zip(dates, production):
            production_data.append(
                {
                    "well_id": well_id,
                    "date": date,
                    "oil_bbl": prod,
                    "long": -103.0 + np.random.normal(0, 0.2),
                    "lat": 47.5 + np.random.normal(0, 0.2),
                }
            )

    return pd.DataFrame(production_data)


def main():
    """Main example function."""
    print("=" * 70)
    print("Ecosystem Integration Example")
    print("=" * 70)

    # Check integration availability
    print("\n1. Checking integration library availability...")
    from decline_curve.integrations import check_integration_availability

    status = check_integration_availability()
    print(
        f"   pygeomodeling: {'✓ Available' if status['pygeomodeling'] else '✗ Not installed'}"
    )
    print(f"   geosuite: {'✓ Available' if status['geosuite'] else '✗ Not installed'}")

    # Generate example data
    print("\n2. Loading production data...")
    df = generate_example_data()
    print(f"   Loaded {len(df)} records for {df['well_id'].nunique()} wells")

    # Calculate EUR
    print("\n3. Calculating EUR...")
    from decline_curve.eur_estimation import calculate_eur_batch

    eur_results = calculate_eur_batch(
        df, well_id_col="well_id", date_col="date", value_col="oil_bbl"
    )

    # Add location information
    well_locations = df.groupby("well_id")[["long", "lat"]].first().reset_index()
    eur_results = eur_results.merge(well_locations, on="well_id", how="left")

    print(f"   Calculated EUR for {len(eur_results)} wells")
    print(f"   Mean EUR: {eur_results['eur'].mean():,.0f} bbl")

    # Try pygeomodeling integration
    if status["pygeomodeling"]:
        print("\n4. Using pygeomodeling for advanced kriging...")
        try:
            from decline_curve.integrations import krige_eur_with_pygeomodeling

            predictions, uncertainty = krige_eur_with_pygeomodeling(
                eur_results, lon_col="long", lat_col="lat", eur_col="eur"
            )
            print(f"   ✓ pygeomodeling kriging complete")
            print(f"   Predictions: {len(predictions)} points")
            print(f"   Mean prediction: {np.mean(predictions):,.0f} bbl")
            print(f"   Mean uncertainty: {np.mean(uncertainty):,.0f} bbl")
        except Exception as e:
            print(f"   ✗ pygeomodeling integration failed: {e}")
    else:
        print("\n4. pygeomodeling not available - skipping advanced kriging")
        print("   Install with: pip install pygeomodeling")

    # Try geosuite integration
    if status["geosuite"]:
        print("\n5. Using geosuite for enhanced analysis...")
        try:
            from decline_curve.integrations import enhanced_eur_with_geosuite

            enhanced_results = enhanced_eur_with_geosuite(
                df, well_id_col="well_id", date_col="date", value_col="oil_bbl"
            )
            print(f"   ✓ geosuite integration complete")
            print(f"   Enhanced results: {len(enhanced_results)} wells")
        except Exception as e:
            print(f"   ✗ geosuite integration failed: {e}")
    else:
        print("\n5. geosuite not available - skipping enhanced analysis")
        print("   Install with: pip install geosuite")

    print("\n" + "=" * 70)
    print("Integration example complete!")
    print("=" * 70)
    print("\nTo enable full integration:")
    print("  pip install decline-curve[integrations]")
    print("  # or individually:")
    print("  pip install pygeomodeling geosuite")


if __name__ == "__main__":
    main()
