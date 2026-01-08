Test a New Price Deck
======================

**Problem**: You have a new price forecast and want to see how it affects
well economics across your portfolio.

Solution
--------

.. code-block:: python

   import pandas as pd
   from decline_curve.scenarios import PriceScenario, run_price_scenarios, compare_scenarios
   from decline_curve.portfolio import portfolio_risk_summary

   # Load production forecasts (from previous analysis)
   forecasts_df = pd.read_csv('well_forecasts.csv', parse_dates=['date'])

   # Define new price deck
   new_price_deck = PriceScenario(
       name='new_price_deck',
       oil_price=75.0,  # New price assumption
       opex=15.0,
       discount_rate=0.10
   )

   # Compare with current price deck
   current_price_deck = PriceScenario(
       name='current_price_deck',
       oil_price=70.0,  # Current price
       opex=15.0,
       discount_rate=0.10
   )

   # Run scenarios for each well
   scenario_results = []

   for well_id in forecasts_df['well_id'].unique():
       well_forecast = forecasts_df[forecasts_df['well_id'] == well_id]['forecast'].values

       # Run both scenarios
       results = run_price_scenarios(
           well_forecast,
           scenarios=[current_price_deck, new_price_deck]
       )
       results['well_id'] = well_id
       scenario_results.append(results)

   # Combine results
   all_results = pd.concat(scenario_results, ignore_index=True)

   # Compare scenarios
   comparison = compare_scenarios(all_results)

   print("Price Deck Comparison:")
   print(comparison.groupby('scenario').agg({
       'npv': ['mean', 'sum'],
       'payback_month': 'mean'
   }))

   # Portfolio-level impact
   portfolio_comparison = all_results.groupby('scenario').agg({
       'npv': 'sum',
       'payback_month': 'mean'
   })

   print(f"\nPortfolio Impact:")
   print(f"  Current price deck NPV: ${portfolio_comparison.loc['current_price_deck', 'npv']:,.0f}")
   print(f"  New price deck NPV: ${portfolio_comparison.loc['new_price_deck', 'npv']:,.0f}")
   print(f"  Change: ${portfolio_comparison.loc['new_price_deck', 'npv'] - portfolio_comparison.loc['current_price_deck', 'npv']:,.0f}")

Time-Varying Price Deck
-----------------------

If prices change over time:

.. code-block:: python

   # Create time-varying price scenario
   dates = pd.date_range('2024-01-01', periods=24, freq='MS')

   # Price escalates over time
   prices = [70.0 + i * 0.5 for i in range(24)]  # $70 to $81.50

   # For each well, apply time-varying prices
   # (This would require custom implementation or using multi-phase scenarios)

   # Simplified: Use average price
   avg_price = sum(prices) / len(prices)
   time_varying_scenario = PriceScenario(
       name='time_varying',
       oil_price=avg_price,
       opex=15.0
   )

Multiple Price Scenarios
------------------------

Test multiple price scenarios at once:

.. code-block:: python

   scenarios = [
       PriceScenario('pessimistic', oil_price=50.0, opex=15.0),
       PriceScenario('base', oil_price=70.0, opex=15.0),
       PriceScenario('optimistic', oil_price=90.0, opex=15.0),
   ]

   # Run for representative well
   sample_well_forecast = forecasts_df[forecasts_df['well_id'] == 'WELL_001']['forecast'].values

   results = run_price_scenarios(sample_well_forecast, scenarios)
   comparison = compare_scenarios(results)

   print("Multiple Scenario Comparison:")
   print(comparison[['scenario', 'npv', 'npv_vs_base', 'npv_pct_change']])

See Also
--------

* :doc:`../workflows/single_well_workflow` - Single well analysis with scenarios
* :doc:`portfolio_analysis` - Portfolio-level analysis
