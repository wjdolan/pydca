"""Example: EUR Estimation with Spatial Kriging.

This example demonstrates how to use kriging-based spatial interpolation
to improve EUR estimates across multiple wells.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Example data generation (replace with your actual data loading)
def generate_example_data():
    """Generate example production data with spatial correlation."""
    np.random.seed(42)

    # Create wells with spatial correlation
    n_wells = 50
    well_ids = [f"WELL_{i:03d}" for i in range(n_wells)]

    # Spatial coordinates (clustered)
    centers = [(47.5, -103.0), (47.3, -102.8), (47.7, -103.2)]
    locations = []
    for _ in range(n_wells):
        center = centers[np.random.randint(len(centers))]
        lon = center[1] + np.random.normal(0, 0.1)
        lat = center[0] + np.random.normal(0, 0.1)
        locations.append((lon, lat))

    # Generate production data with spatial correlation
    production_data = []
    for well_id, (lon, lat) in zip(well_ids, locations):
        # Spatial trend: higher EUR near center
        base_eur = 50000 + (47.5 - lat) * 10000 + abs(-103.0 - lon) * 5000
        eur = base_eur + np.random.normal(0, base_eur * 0.2)
        eur = max(10000, eur)  # Minimum EUR

        # Generate monthly production (simplified)
        dates = pd.date_range("2020-01-01", periods=36, freq="MS")
        qi = eur / 200  # Rough estimate
        di = 0.1
        b = 0.5
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
                    "long": lon,
                    "lat": lat,
                }
            )

    return pd.DataFrame(production_data)


def main():
    """Main example function."""
    print("=" * 70)
    print("EUR Estimation with Spatial Kriging Example")
    print("=" * 70)

    # Generate or load data
    print("\n1. Loading production data...")
    df = generate_example_data()
    print(f"   Loaded {len(df)} records for {df['well_id'].nunique()} wells")

    # Calculate EUR using standard method
    print("\n2. Calculating EUR using decline curve analysis...")
    from decline_curve.eur_estimation import calculate_eur_batch

    eur_results = calculate_eur_batch(
        df,
        well_id_col="well_id",
        date_col="date",
        value_col="oil_bbl",
        model_type="hyperbolic",
    )
    print(f"   Calculated EUR for {len(eur_results)} wells")
    print(f"   Mean EUR: {eur_results['eur'].mean():,.0f} bbl")

    # Add location information
    well_locations = df.groupby("well_id")[["long", "lat"]].first().reset_index()
    eur_results = eur_results.merge(well_locations, on="well_id", how="left")

    # Improve with kriging
    print("\n3. Improving EUR estimates with spatial kriging...")
    try:
        from decline_curve.spatial_kriging import improve_eur_with_kriging

        eur_improved = improve_eur_with_kriging(
            eur_results,
            well_id_col="well_id",
            lon_col="long",
            lat_col="lat",
            eur_col="eur",
            kriging_method="auto",
        )

        print(f"   Kriging interpolation complete")
        print(f"   Original mean EUR: {eur_improved['eur'].mean():,.0f} bbl")
        if eur_improved["eur_kriged"].notna().any():
            print(f"   Kriged mean EUR: {eur_improved['eur_kriged'].mean():,.0f} bbl")
            print(
                f"   Mean uncertainty: {eur_improved['kriging_uncertainty'].mean():,.0f} bbl"
            )

        # Create EUR map
        print("\n4. Creating EUR map...")
        from decline_curve.spatial_kriging import create_eur_map

        lon_grid, lat_grid, eur_map, unc_map = create_eur_map(
            eur_improved, lon_col="long", lat_col="lat", eur_col="eur", resolution=50
        )

        # Plot results
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # EUR map
        im1 = axes[0].contourf(lon_grid, lat_grid, eur_map, levels=20, cmap="YlOrRd")
        axes[0].scatter(
            eur_improved["long"],
            eur_improved["lat"],
            c="black",
            s=30,
            alpha=0.6,
            marker="x",
        )
        axes[0].set_xlabel("Longitude")
        axes[0].set_ylabel("Latitude")
        axes[0].set_title("Kriged EUR Map")
        plt.colorbar(im1, ax=axes[0], label="EUR (bbl)")

        # Uncertainty map
        im2 = axes[1].contourf(lon_grid, lat_grid, unc_map, levels=20, cmap="viridis")
        axes[1].scatter(
            eur_improved["long"],
            eur_improved["lat"],
            c="white",
            s=30,
            alpha=0.6,
            marker="x",
        )
        axes[1].set_xlabel("Longitude")
        axes[1].set_ylabel("Latitude")
        axes[1].set_title("Kriging Uncertainty Map")
        plt.colorbar(im2, ax=axes[1], label="Uncertainty (bbl)")

        plt.tight_layout()
        plt.savefig("eur_kriging_map.png", dpi=150, bbox_inches="tight")
        print("   Saved: eur_kriging_map.png")
        plt.close()

    except ImportError as e:
        print(f"   Kriging not available: {e}")
        print("   Install with: pip install decline-curve[spatial]")

    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
