Single Unconventional Well Analysis
====================================

This workflow demonstrates a complete analysis of a single unconventional well,
from data loading through forecast, economics, and reporting.

Scenario
--------

You have a new horizontal well in the Permian Basin that has been producing for
18 months. You need to:

1. Forecast production for the next 24 months
2. Estimate ultimate recovery (EUR)
3. Evaluate economics under multiple price scenarios
4. Generate a professional report

Step 1: Load and Prepare Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   import numpy as np
   from decline_curve import dca
   from decline_curve.scenarios import PriceScenario, run_price_scenarios

   # Load production data (in practice, this would come from a database or file)
   # For this example, we'll create sample data
   dates = pd.date_range('2022-01-01', periods=18, freq='MS')

   # Realistic Permian well production profile
   qi = 1200  # Initial production (bbl/month)
   di = 0.12  # Initial decline rate
   b = 0.6    # Hyperbolic exponent

   t = np.arange(len(dates))
   production = qi / ((1 + b * di * t) ** (1/b))

   # Add realistic noise
   np.random.seed(42)
   noise = np.random.normal(0, production * 0.08)
   production = np.maximum(production + noise, 0)

   series = pd.Series(production, index=dates, name='oil_bbl')

   print(f"Historical production: {len(series)} months")
   print(f"Average monthly production: {series.mean():.0f} bbl/month")
   print(f"Cumulative to date: {series.sum():,.0f} bbl")

Step 2: Generate Forecast
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Generate 24-month forecast using hyperbolic Arps
   forecast = dca.single_well(
       series,
       model='arps',
       kind='hyperbolic',
       horizon=24,
       return_params=True
   )

   forecast_series, params = forecast

   print(f"\nForecast Parameters:")
   print(f"  Initial rate (qi): {params['qi']:.1f} bbl/month")
   print(f"  Decline rate (di): {params['di']:.4f}")
   print(f"  Hyperbolic exponent (b): {params['b']:.3f}")
   print(f"  R²: {params.get('r2', 'N/A'):.3f}")

   # Combine historical and forecast
   full_series = pd.concat([series, forecast_series])

   print(f"\nForecast Summary:")
   print(f"  Next 12 months: {forecast_series.head(12).sum():,.0f} bbl")
   print(f"  Next 24 months: {forecast_series.sum():,.0f} bbl")

Step 3: Estimate EUR
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from decline_curve.eur_estimation import calculate_eur

   # Calculate EUR with economic limit
   eur_result = calculate_eur(
       series,
       model_type='hyperbolic',
       econ_limit=10.0,  # 10 bbl/month economic limit
       min_months=6
   )

   print(f"\nEUR Estimation:")
   print(f"  Estimated Ultimate Recovery: {eur_result['eur']:,.0f} bbl")
   print(f"  Remaining reserves: {eur_result['eur'] - series.sum():,.0f} bbl")
   print(f"  Recovery factor: {(series.sum() / eur_result['eur'] * 100):.1f}%")

Step 4: Economic Analysis with Scenarios
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Define price scenarios
   scenarios = [
       PriceScenario('low', oil_price=50.0, opex=15.0, discount_rate=0.10),
       PriceScenario('base', oil_price=70.0, opex=15.0, discount_rate=0.10),
       PriceScenario('high', oil_price=90.0, opex=15.0, discount_rate=0.10),
   ]

   # Run scenarios on forecast
   scenario_results = run_price_scenarios(forecast_series, scenarios)

   print(f"\nEconomic Analysis:")
   print(scenario_results[['scenario', 'npv', 'payback_month', 'total_revenue']])

   # Compare scenarios
   from decline_curve.scenarios import compare_scenarios
   comparison = compare_scenarios(scenario_results)

   print(f"\nScenario Comparison (vs Base):")
   print(comparison[['scenario', 'npv', 'npv_vs_base', 'npv_pct_change']])

Step 5: Generate Report
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from decline_curve.plot import plot_forecast
   from decline_curve.reports import generate_well_report

   # Create plot
   plot_forecast(
       series,
       forecast_series,
       title=f'Well Forecast - EUR: {eur_result["eur"]:,.0f} bbl',
       filename='well_forecast.png'
   )

   # Generate PDF report (requires reportlab)
   # from decline_curve.fitting import FitResult
   # fit_result = FitResult(...)  # Would come from actual fitting
   # report_path = generate_well_report(
   #     fit_result,
   #     output_path='well_report.pdf',
   #     format='pdf'
   # )

Key Questions Answered
-----------------------

1. **What is the expected production over the next 24 months?**
   - Answer: Forecast shows X bbl over next 24 months

2. **What is the estimated ultimate recovery?**
   - Answer: EUR is Y bbl based on hyperbolic decline

3. **Is the well economic under current price assumptions?**
   - Answer: NPV is positive/negative under base case scenario

4. **How sensitive is economics to price changes?**
   - Answer: Scenario comparison shows X% change in NPV per $10/bbl price change

Next Steps
----------

- Update forecast as new production data becomes available
- Compare with type curves for similar wells
- Perform sensitivity analysis on decline parameters
- Evaluate completion design alternatives

See Also
--------

* :doc:`../cookbook/update_forecast` - How to update forecasts with new data
* :doc:`../cookbook/compare_completions` - Compare different completion designs
* :doc:`../workflows/field_analysis` - Small field analysis workflow
