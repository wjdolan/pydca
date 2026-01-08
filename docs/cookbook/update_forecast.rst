Update Forecasts for New Months
================================

**Problem**: You have a forecast from last month, and now you have new production
data. You want to update the forecast without starting from scratch.

Solution
--------

.. code-block:: python

   import pandas as pd
   from decline_curve import dca

   # Load previous forecast
   previous_forecast = pd.read_csv('previous_forecast.csv', index_col=0, parse_dates=True)
   previous_forecast = previous_forecast['forecast']

   # Load new production data
   new_data = pd.read_csv('new_production.csv', parse_dates=['date'])
   new_data = new_data.set_index('date')['oil_bbl']

   # Combine historical data (previous + new)
   # Get historical portion from previous forecast
   historical_cutoff = previous_forecast.index[0]  # First forecast date
   historical_data = new_data[new_data.index < historical_cutoff]

   # Add new data points
   if len(historical_data) > 0:
       full_historical = pd.concat([historical_data, new_data[new_data.index >= historical_cutoff]])
   else:
       full_historical = new_data

   # Generate new forecast
   updated_forecast = dca.single_well(
       full_historical,
       model='arps',
       kind='hyperbolic',
       horizon=24  # Forecast next 24 months
   )

   # Compare with previous forecast
   print("Forecast Update Summary:")
   print(f"  Previous 12-month forecast: {previous_forecast.head(12).sum():,.0f} bbl")
   print(f"  Updated 12-month forecast: {updated_forecast.head(12).sum():,.0f} bbl")
   print(f"  Change: {((updated_forecast.head(12).sum() / previous_forecast.head(12).sum() - 1) * 100):.1f}%")

   # Save updated forecast
   updated_forecast.to_csv('updated_forecast.csv')

Alternative: Incremental Update
--------------------------------

If you only want to update the forecast parameters without re-fitting:

.. code-block:: python

   # Load previous parameters
   previous_params = {
       'qi': 1000,
       'di': 0.1,
       'b': 0.5
   }

   # Adjust based on new data (simplified approach)
   # In practice, you'd re-fit with new data
   from decline_curve.models import predict_arps
   import numpy as np

   # Use new data to adjust initial rate
   latest_rate = new_data.iloc[-1]
   time_elapsed = len(new_data)

   # Re-fit with updated data
   updated_forecast = dca.single_well(
       full_historical,
       model='arps',
       kind='hyperbolic',
       horizon=24
   )

See Also
--------

* :doc:`../workflows/single_well_workflow` - Complete single well analysis
* :doc:`compare_completions` - Compare different scenarios
