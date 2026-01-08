Small Field Analysis Workflow
==============================

This workflow demonstrates analyzing a small field with 20-50 wells,
including type curves, portfolio aggregation, and field-level economics.

Scenario
--------

You manage a small field with 30 wells across 3 operators. You need to:

1. Analyze all wells and generate forecasts
2. Create type curves by operator
3. Aggregate results by operator and completion year
4. Evaluate field-level economics
5. Generate field summary report

Step 1: Load Field Data
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   from decline_curve import dca
   from decline_curve.portfolio import portfolio_summary, aggregate_by_category

   # Load production data (in practice, from database or file)
   # df should have columns: well_id, date, oil_bbl, operator, completion_year, county
   df = pd.read_csv('field_production.csv')
   df['date'] = pd.to_datetime(df['date'])

   print(f"Field Data Summary:")
   print(f"  Total wells: {df['well_id'].nunique()}")
   print(f"  Total records: {len(df)}")
   print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
   print(f"  Operators: {df['operator'].unique().tolist()}")

Step 2: Batch Forecast All Wells
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Run batch forecast for all wells
   forecasts = dca.batch_jobs(
       df,
       well_col='well_id',
       date_col='date',
       value_col='oil_bbl',
       model='arps',
       kind='hyperbolic',
       horizon=24,
       n_jobs=-1  # Use all available cores
   )

   print(f"\nForecast Complete:")
   print(f"  Wells processed: {forecasts['well_id'].nunique()}")
   print(f"  Forecast months: {len(forecasts)}")

Step 3: Calculate EUR for All Wells
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from decline_curve.eur_estimation import calculate_eur_batch

   # Calculate EUR for all wells
   eur_results = calculate_eur_batch(
       df,
       well_id_col='well_id',
       date_col='date',
       value_col='oil_bbl',
       model_type='hyperbolic',
       n_jobs=-1
   )

   # Merge with well metadata
   well_metadata = df[['well_id', 'operator', 'completion_year', 'county']].drop_duplicates()
   eur_results = eur_results.merge(well_metadata, on='well_id', how='left')

   print(f"\nEUR Calculation Complete:")
   print(f"  Wells with EUR: {len(eur_results)}")
   print(f"  Total field EUR: {eur_results['eur'].sum():,.0f} bbl")
   print(f"  Average EUR per well: {eur_results['eur'].mean():,.0f} bbl")

Step 4: Create Type Curves by Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create type curves for each operator
   for operator in df['operator'].unique():
       operator_wells = df[df['operator'] == operator]['well_id'].unique()
       operator_data = df[df['well_id'].isin(operator_wells)]

       type_curve = dca.type_curves(
           operator_data,
           grouping_col='well_id',
           date_col='date',
           value_col='oil_bbl',
           kind='hyperbolic',
           method='median',
           normalize=True
       )

       print(f"\n{operator} Type Curve:")
       print(f"  Normalized initial rate: {type_curve.iloc[0]:.3f}")
       print(f"  12-month cumulative: {type_curve.head(12).sum():.3f}")

Step 5: Portfolio Aggregation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Aggregate by operator
   operator_summary = aggregate_by_category(
       eur_results,
       category_col='operator',
       metrics=['eur', 'p50_eur']
   )

   print(f"\nOperator Summary:")
   print(operator_summary[['operator', 'eur_sum', 'eur_mean', 'well_count']])

   # Aggregate by completion year
   year_summary = aggregate_by_category(
       eur_results,
       category_col='completion_year',
       metrics=['eur', 'p50_eur']
   )

   print(f"\nCompletion Year Summary:")
   print(year_summary[['completion_year', 'eur_sum', 'eur_mean', 'well_count']])

   # Comprehensive portfolio summary
   portfolio = portfolio_summary(
       eur_results,
       category_cols=['operator', 'completion_year', 'county']
   )

   print(f"\nPortfolio Summary:")
   print(portfolio.head(10))

Step 6: Field-Level Economics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from decline_curve.scenarios import PriceScenario, run_price_scenarios
   from decline_curve.portfolio import portfolio_risk_summary

   # Calculate economics for each well (simplified - would use actual forecasts)
   # In practice, you'd combine forecasts with price scenarios

   # Field-level risk summary
   risk_summary = portfolio_risk_summary(
       eur_results,
       category_col='operator'
   )

   print(f"\nField Risk Summary by Operator:")
   print(risk_summary[['operator', 'well_count', 'npv_sum', 'positive_npv_pct']])

Step 7: Generate Field Report
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from decline_curve.reports import generate_field_pdf_report

   # Convert EUR results to list of dictionaries for report
   well_results = eur_results.to_dict('records')

   # Generate PDF report
   report_path = generate_field_pdf_report(
       well_results,
       output_path='field_summary_report.pdf',
       title='Field Summary Report'
   )

   print(f"\nField report saved to: {report_path}")

Key Questions Answered
-----------------------

1. **What is the total field EUR?**
   - Answer: Total field EUR is X bbl across Y wells

2. **Which operator has the best performance?**
   - Answer: Operator summary shows operator Z with highest average EUR

3. **How do wells perform by completion year?**
   - Answer: Year summary shows trends in well performance over time

4. **What is the field-level economic outlook?**
   - Answer: Portfolio risk summary shows field NPV and success rates

Next Steps
----------

- Compare field performance to regional benchmarks
- Identify underperforming wells for optimization
- Plan development drilling based on type curves
- Update forecasts quarterly as new data arrives

See Also
--------

* :doc:`../workflows/single_well_workflow` - Single well analysis
* :doc:`../workflows/state_scale_workflow` - State-scale analysis
* :doc:`../cookbook/portfolio_analysis` - Advanced portfolio techniques
