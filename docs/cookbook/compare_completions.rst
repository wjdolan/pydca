Compare Two Completion Designs
===============================

**Problem**: You want to compare the performance of two different completion
designs (e.g., different proppant volumes, stage counts) to determine which
performs better.

Solution
--------

.. code-block:: python

   import pandas as pd
   from decline_curve import dca
   from decline_curve.portfolio import aggregate_by_category

   # Load production data with completion design information
   df = pd.read_csv('wells_with_completions.csv')
   df['date'] = pd.to_datetime(df['date'])

   # Define completion designs to compare
   design_a_wells = df[df['completion_design'] == 'Design_A']['well_id'].unique()
   design_b_wells = df[df['completion_design'] == 'Design_B']['well_id'].unique()

   print(f"Design A wells: {len(design_a_wells)}")
   print(f"Design B wells: {len(design_b_wells)}")

   # Calculate EUR for all wells
   from decline_curve.eur_estimation import calculate_eur_batch

   eur_results = calculate_eur_batch(
       df,
       well_id_col='well_id',
       date_col='date',
       value_col='oil_bbl'
   )

   # Merge with completion design info
   completion_info = df[['well_id', 'completion_design', 'proppant_volume', 'stage_count']].drop_duplicates()
   eur_results = eur_results.merge(completion_info, on='well_id', how='left')

   # Aggregate by completion design
   design_comparison = aggregate_by_category(
       eur_results,
       category_col='completion_design',
       metrics=['eur', 'p50_eur', 'p90_eur']
   )

   print("\nCompletion Design Comparison:")
   print(design_comparison[['completion_design', 'eur_mean', 'eur_std', 'well_count']])

   # Statistical comparison
   design_a_eur = eur_results[eur_results['completion_design'] == 'Design_A']['eur']
   design_b_eur = eur_results[eur_results['completion_design'] == 'Design_B']['eur']

   print(f"\nStatistical Comparison:")
   print(f"  Design A average EUR: {design_a_eur.mean():,.0f} bbl")
   print(f"  Design B average EUR: {design_b_eur.mean():,.0f} bbl")
   print(f"  Difference: {((design_b_eur.mean() / design_a_eur.mean() - 1) * 100):.1f}%")

   # Type curve comparison
   print("\nType Curve Comparison:")
   for design in ['Design_A', 'Design_B']:
       design_data = df[df['completion_design'] == design]
       type_curve = dca.type_curves(
           design_data,
           grouping_col='well_id',
           date_col='date',
           value_col='oil_bbl',
           kind='hyperbolic',
           method='median'
       )
       print(f"  {design} - 12-month cumulative: {type_curve.head(12).sum():.3f}")

Advanced: Normalize by Completion Parameters
----------------------------------------------

If completion designs have different proppant volumes, normalize by that:

.. code-block:: python

   # Calculate EUR per unit of proppant
   eur_results['eur_per_ton_proppant'] = eur_results['eur'] / eur_results['proppant_volume']

   # Compare normalized EUR
   design_comparison_normalized = aggregate_by_category(
       eur_results,
       category_col='completion_design',
       metrics=['eur_per_ton_proppant']
   )

   print("\nNormalized Comparison (EUR per ton proppant):")
   print(design_comparison_normalized)

See Also
--------

* :doc:`../workflows/field_analysis` - Field-level analysis
* :doc:`test_price_deck` - Testing economic scenarios
