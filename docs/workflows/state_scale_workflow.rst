State-Scale Dataset Analysis Workflow
=====================================

This workflow demonstrates analyzing large-scale datasets (thousands of wells)
using the runner framework, catalog system, and batch processing.

Scenario
--------

You need to analyze all wells in Pennsylvania (or New Mexico) from public data.
This involves:

1. Loading data from multiple sources using catalog
2. Processing thousands of wells efficiently
3. Generating standardized outputs
4. Creating state-wide summaries

Step 1: Set Up Data Catalog
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from decline_curve.catalog import DatasetCatalog, CatalogEntry
   from decline_curve.runner import FieldScaleRunner, RunnerConfig
   from decline_curve.schemas import ProductionInputSchema, ProductionOutputSchema

   # Initialize catalog
   catalog = DatasetCatalog('catalogs/')

   # Load Pennsylvania dataset entry
   pa_entry = catalog.get('pennsylvania')

   if pa_entry is None:
       print("Creating Pennsylvania catalog entry...")
       # Create entry programmatically
       pa_entry = catalog.create_entry(
           name='pennsylvania',
           description='Pennsylvania production data',
           sources=[
               {
                   'path': 'data/pa_production.csv',
                   'format': 'csv',
                   'columns': {
                       'well_id': 'API',
                       'date': 'ProductionDate',
                       'oil_bbl': 'OilBBL'
                   }
               }
           ],
           filters=[
               {'column': 'oil_bbl', 'operator': '>', 'value': 0},
               {'column': 'date', 'operator': '>=', 'value': '2020-01-01'}
           ]
       )

   print(f"Catalog entry: {pa_entry.name}")
   print(f"  Sources: {len(pa_entry.sources)}")
   print(f"  Filters: {len(pa_entry.filters)}")

Step 2: Load Data Using Catalog
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Load data using catalog
   df = catalog.load_data(pa_entry)

   # Validate and standardize
   df_standard = ProductionInputSchema.standardize(df)
   validation_result = ProductionInputSchema.validate(df_standard)

   if not validation_result['valid']:
       print(f"Validation errors: {validation_result['errors']}")
   else:
       print(f"\nData Loaded and Validated:")
       print(f"  Wells: {df_standard['well_id'].nunique():,}")
       print(f"  Records: {len(df_standard):,}")
       print(f"  Date range: {df_standard['date'].min()} to {df_standard['date'].max()}")

Step 3: Configure Runner
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Configure runner for large-scale processing
   config = RunnerConfig(
       n_jobs=-1,           # Use all available cores
       chunk_size=500,      # Process 500 wells at a time
       max_retries=2,       # Retry failed wells twice
       log_level='INFO',
       log_file='pa_analysis.log',
       save_intermediate=True,
       intermediate_dir='output/intermediate'
   )

   runner = FieldScaleRunner(config)

   print(f"Runner configured:")
   print(f"  Chunk size: {config.chunk_size}")
   print(f"  Max retries: {config.max_retries}")
   print(f"  Log file: {config.log_file}")

Step 4: Define Forecast Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from decline_curve.dca import single_well

   def forecast_well(df_well):
       """Forecast function for a single well."""
       # Group by well and sort by date
       well_id = df_well['well_id'].iloc[0]
       df_well = df_well.sort_values('date')

       # Create production series
       series = df_well.set_index('date')['oil_bbl']
       series = series.asfreq('MS', fill_value=0)  # Monthly frequency

       # Filter to valid production
       series = series[series > 0]

       if len(series) < 6:  # Need at least 6 months
           return None

       try:
           # Generate forecast
           forecast = single_well(
               series,
               model='arps',
               kind='hyperbolic',
               horizon=12
           )

           return {
               'well_id': well_id,
               'forecast': forecast,
               'historical': series
           }
       except Exception as e:
           return {'well_id': well_id, 'error': str(e)}

Step 5: Run Large-Scale Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Run analysis using runner
   results = runner.run(
       df_standard,
       forecast_func=forecast_well,
       well_id_col='well_id'
   )

   print(f"\nAnalysis Complete:")
   print(f"  Total wells processed: {results.summary.total_wells}")
   print(f"  Successful: {results.summary['n_successful']}")
   print(f"  Failed: {results.summary['n_failed']}")
   print(f"  Success rate: {results.summary.success_rate:.1f}%")
   print(f"  Total time: {results.summary.total_time:.1f} seconds")
   print(f"  Throughput: {results.summary.throughput_wells_per_sec:.1f} wells/sec")

Step 6: Generate Standardized Outputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Convert to standardized output schema
   output_schema = ProductionOutputSchema.from_runner_results(results)

   # Get output tables
   parameters_df, forecasts_df, events_df = output_schema.to_dataframes()

   print(f"\nOutput Tables:")
   print(f"  Parameters: {len(parameters_df)} wells")
   print(f"  Forecasts: {len(forecasts_df)} records")
   print(f"  Events: {len(events_df)} events")

   # Save outputs
   output_schema.save('output/pa_analysis')

   print(f"\nOutputs saved to output/pa_analysis/")

Step 7: State-Wide Summary
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from decline_curve.portfolio import portfolio_summary
   from decline_curve.eur_estimation import calculate_eur_batch

   # Calculate EUR for all wells
   eur_results = calculate_eur_batch(
       df_standard,
       well_id_col='well_id',
       date_col='date',
       value_col='oil_bbl',
       n_jobs=-1
   )

   # Add metadata if available
   # eur_results = eur_results.merge(metadata_df, on='well_id', how='left')

   # State-wide summary
   state_summary = portfolio_summary(
       eur_results,
       category_cols=['county']  # If county data available
   )

   print(f"\nState-Wide Summary:")
   print(f"  Total wells: {len(eur_results):,}")
   print(f"  Total EUR: {eur_results['eur'].sum():,.0f} bbl")
   print(f"  Average EUR: {eur_results['eur'].mean():,.0f} bbl")

   # Top counties
   if 'county' in eur_results.columns:
       county_summary = aggregate_by_category(
           eur_results,
           category_col='county',
           metrics=['eur']
       )
       print(f"\nTop 10 Counties by EUR:")
       print(county_summary.nlargest(10, 'eur_sum')[['county', 'eur_sum', 'well_count']])

Key Questions Answered
-----------------------

1. **How many wells can be analyzed and at what throughput?**
   - Answer: Runner processed X wells at Y wells/second

2. **What is the state-wide EUR estimate?**
   - Answer: Total state EUR is X bbl across Y wells

3. **Which counties have the best performance?**
   - Answer: County summary shows top performers

4. **What is the data quality and success rate?**
   - Answer: Success rate is X% with Y% of wells having sufficient data

Best Practices
--------------

1. **Use Catalog for Data Management**
   - Define data sources once in YAML
   - Reuse across multiple analyses
   - Version control catalog entries

2. **Configure Runner Appropriately**
   - Use chunk_size based on memory constraints
   - Set max_retries for transient failures
   - Enable intermediate saving for long jobs

3. **Monitor Progress**
   - Check log files regularly
   - Review intermediate results
   - Validate outputs before final summary

4. **Handle Errors Gracefully**
   - Review events table for failed wells
   - Investigate common failure modes
   - Re-run with adjusted parameters if needed

Next Steps
----------

- Compare state performance to other states
- Identify trends over time
- Generate quarterly updates
- Export results for GIS mapping

See Also
--------

* :doc:`../workflows/single_well_workflow` - Single well analysis
* :doc:`../workflows/field_analysis` - Small field analysis
* :doc:`../cookbook/config_files` - Using configuration files
