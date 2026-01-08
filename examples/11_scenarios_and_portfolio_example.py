"""Example: Price scenarios and portfolio analysis.

This example demonstrates:
1. Running multiple price scenarios (low, base, high)
2. Comparing scenario results
3. Portfolio aggregation by operator, county, etc.
"""

import numpy as np
import pandas as pd

from decline_curve.portfolio import (
    aggregate_by_category,
    cross_tabulate_metrics,
    portfolio_risk_summary,
    portfolio_summary,
)
from decline_curve.scenarios import (
    PriceScenario,
    compare_scenarios,
    run_price_scenarios,
)


def generate_example_data():
    """Generate example production and well data."""
    np.random.seed(42)

    n_wells = 50
    well_ids = [f"WELL_{i:03d}" for i in range(n_wells)]

    production_data = []
    well_metadata = []

    for i, well_id in enumerate(well_ids):
        # Generate production
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
                }
            )

        # Generate metadata
        operators = ["Operator_A", "Operator_B", "Operator_C"]
        counties = ["County_X", "County_Y", "County_Z"]
        fields = ["Field_1", "Field_2", "Field_3"]
        completion_years = [2019, 2020, 2021, 2022]

        well_metadata.append(
            {
                "well_id": well_id,
                "operator": np.random.choice(operators),
                "county": np.random.choice(counties),
                "field": np.random.choice(fields),
                "completion_year": np.random.choice(completion_years),
            }
        )

    return pd.DataFrame(production_data), pd.DataFrame(well_metadata)


def main():
    """Main example function."""
    print("=" * 70)
    print("Price Scenarios and Portfolio Analysis Example")
    print("=" * 70)

    # Generate example data
    print("\n1. Loading production data...")
    prod_df, metadata_df = generate_example_data()
    print(f"   Loaded {len(prod_df)} records for {prod_df['well_id'].nunique()} wells")

    # Calculate EUR for each well (simplified)
    print("\n2. Calculating EUR and NPV...")
    eur_results = []
    for well_id in prod_df["well_id"].unique():
        well_prod = prod_df[prod_df["well_id"] == well_id]["oil_bbl"].values
        eur = np.sum(well_prod)
        # Simple NPV calculation
        price = 70.0
        opex = 15.0
        revenue = np.sum(well_prod * price)
        costs = np.sum(well_prod * opex) + (len(well_prod) * 5000)
        npv = revenue - costs

        eur_results.append(
            {
                "well_id": well_id,
                "eur": eur,
                "npv": npv,
                "p10_eur": eur * 0.9,
                "p50_eur": eur,
                "p90_eur": eur * 1.1,
                "p10_npv": npv * 0.9,
                "p50_npv": npv,
                "p90_npv": npv * 1.1,
            }
        )

    results_df = pd.DataFrame(eur_results)
    results_df = results_df.merge(metadata_df, on="well_id", how="left")
    print(f"   Calculated EUR and NPV for {len(results_df)} wells")

    # Price scenarios
    print("\n3. Running price scenarios...")
    # Use first well as example
    example_well = prod_df[prod_df["well_id"] == "WELL_001"]
    production_forecast = example_well["oil_bbl"].values

    scenarios = [
        PriceScenario("low", oil_price=50.0, opex=15.0),
        PriceScenario("base", oil_price=70.0, opex=15.0),
        PriceScenario("high", oil_price=90.0, opex=15.0),
    ]

    scenario_results = run_price_scenarios(production_forecast, scenarios)
    print("\n   Scenario Results:")
    print(scenario_results[["scenario", "npv", "payback_month"]])

    # Compare scenarios
    print("\n4. Comparing scenarios...")
    comparison = compare_scenarios(scenario_results)
    print("\n   Scenario Comparison:")
    print(comparison[["scenario", "npv", "npv_vs_base", "npv_pct_change"]])

    # Portfolio aggregation
    print("\n5. Portfolio aggregation by operator...")
    operator_summary = aggregate_by_category(
        results_df, category_col="operator", metrics=["eur", "npv"]
    )
    print("\n   Operator Summary:")
    print(operator_summary[["operator", "eur_sum", "npv_sum", "well_count"]])

    # Portfolio summary
    print("\n6. Comprehensive portfolio summary...")
    portfolio = portfolio_summary(
        results_df, category_cols=["operator", "county", "completion_year"]
    )
    print("\n   Portfolio Summary (first 10 rows):")
    print(portfolio.head(10))

    # Cross-tabulation
    print("\n7. Cross-tabulation: NPV by operator and completion year...")
    crosstab = cross_tabulate_metrics(
        results_df,
        row_category="operator",
        col_category="completion_year",
        metric="npv",
        agg_func="sum",
    )
    print("\n   NPV by Operator and Completion Year:")
    print(crosstab)

    # Risk summary
    print("\n8. Portfolio risk summary...")
    risk_summary = portfolio_risk_summary(results_df, category_col="operator")
    print("\n   Risk Summary by Operator:")
    print(risk_summary[["operator", "well_count", "npv_sum", "positive_npv_pct"]])

    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
