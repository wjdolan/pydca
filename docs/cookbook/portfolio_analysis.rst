Portfolio Analysis Techniques
=============================

**Problem**: You want to analyze a portfolio of wells by operator, county, field,
or other categories to understand performance patterns and make strategic decisions.

Solution
--------

Basic Aggregation
-----------------

.. code-block:: python

   import pandas as pd
   from decline_curve.portfolio import aggregate_by_category, portfolio_summary
   from decline_curve.eur_estimation import calculate_eur_batch

   # Load EUR results (from previous analysis)
   eur_results = pd.read_csv('eur_results.csv')

   # Aggregate by operator
   operator_summary = aggregate_by_category(
       eur_results,
       category_col='operator',
       metrics=['eur', 'npv', 'p50_eur']
   )

   print("Operator Summary:")
   print(operator_summary[['operator', 'eur_sum', 'eur_mean', 'well_count']])

   # Aggregate by county
   county_summary = aggregate_by_category(
       eur_results,
       category_col='county',
       metrics=['eur', 'npv']
   )

   print("\nCounty Summary:")
   print(county_summary[['county', 'eur_sum', 'well_count']])

Multi-Level Portfolio Summary
------------------------------

.. code-block:: python

   # Comprehensive portfolio summary
   portfolio = portfolio_summary(
       eur_results,
       category_cols=['operator', 'county', 'completion_year']
   )

   print("Portfolio Summary:")
   print(portfolio.head(20))

   # Filter to specific category
   operator_portfolio = portfolio[portfolio['category'] == 'operator']
   print("\nOperator-Level Summary:")
   print(operator_portfolio[['category_value', 'eur_sum', 'well_count']])

Cross-Tabulation
----------------

.. code-block:: python

   from decline_curve.portfolio import cross_tabulate_metrics

   # Cross-tabulate NPV by operator and completion year
   crosstab = cross_tabulate_metrics(
       eur_results,
       row_category='operator',
       col_category='completion_year',
       metric='npv',
       agg_func='sum'
   )

   print("NPV by Operator and Completion Year:")
   print(crosstab)

   # Count wells by operator and county
   well_counts = cross_tabulate_metrics(
       eur_results,
       row_category='operator',
       col_category='county',
       metric='npv',  # Metric doesn't matter for count
       agg_func='count'
   )

   print("\nWell Counts by Operator and County:")
   print(well_counts)

Portfolio Risk Analysis
-----------------------

.. code-block:: python

   from decline_curve.portfolio import portfolio_risk_summary

   # Risk summary by operator
   risk_summary = portfolio_risk_summary(
       eur_results,
       category_col='operator'
   )

   print("Portfolio Risk Summary:")
   print(risk_summary[['operator', 'well_count', 'npv_sum', 'positive_npv_pct', 'npv_uncertainty']])

   # Overall portfolio risk
   overall_risk = portfolio_risk_summary(eur_results)
   print("\nOverall Portfolio Risk:")
   print(overall_risk[['well_count', 'npv_sum', 'positive_npv_pct']])

Identifying Top Performers
---------------------------

.. code-block:: python

   # Top 10 wells by EUR
   top_wells = eur_results.nlargest(10, 'eur')[['well_id', 'eur', 'operator', 'county']]
   print("Top 10 Wells by EUR:")
   print(top_wells)

   # Top operators by total EUR
   operator_summary = aggregate_by_category(eur_results, category_col='operator')
   top_operators = operator_summary.nlargest(5, 'eur_sum')[['operator', 'eur_sum', 'well_count']]
   print("\nTop 5 Operators by Total EUR:")
   print(top_operators)

   # Best performing counties
   county_summary = aggregate_by_category(eur_results, category_col='county')
   top_counties = county_summary.nlargest(5, 'eur_mean')[['county', 'eur_mean', 'well_count']]
   print("\nTop 5 Counties by Average EUR:")
   print(top_counties)

Comparing Time Periods
-----------------------

.. code-block:: python

   # Compare wells by completion year
   year_summary = aggregate_by_category(
       eur_results,
       category_col='completion_year',
       metrics=['eur', 'npv']
   )

   print("Performance by Completion Year:")
   print(year_summary[['completion_year', 'eur_mean', 'eur_std', 'well_count']])

   # Identify trends
   year_summary = year_summary.sort_values('completion_year')
   print("\nTrend Analysis:")
   for _, row in year_summary.iterrows():
       print(f"  {row['completion_year']}: {row['eur_mean']:,.0f} bbl/well (n={row['well_count']})")

Exporting Results
-----------------

.. code-block:: python

   # Save portfolio summaries
   portfolio.to_csv('portfolio_summary.csv', index=False)
   operator_summary.to_csv('operator_summary.csv', index=False)
   county_summary.to_csv('county_summary.csv', index=False)

   # Create Excel workbook with multiple sheets
   with pd.ExcelWriter('portfolio_analysis.xlsx') as writer:
       portfolio.to_excel(writer, sheet_name='Portfolio', index=False)
       operator_summary.to_excel(writer, sheet_name='By Operator', index=False)
       county_summary.to_excel(writer, sheet_name='By County', index=False)
       crosstab.to_excel(writer, sheet_name='Cross-Tab', index=True)

See Also
--------

* :doc:`../workflows/field_analysis` - Field-level analysis workflow
* :doc:`../workflows/state_scale_workflow` - Large-scale portfolio analysis
