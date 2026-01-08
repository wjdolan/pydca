Using Configuration Files
===========================

**Problem**: You want to run batch analyses using configuration files instead
of writing Python code, making it easier for non-programmers and enabling
reproducible workflows.

Solution
--------

Create a TOML Configuration File
----------------------------------

.. code-block:: toml

   # config.toml
   [data]
   path = "data/production.csv"
   format = "csv"
   well_id_col = "well_id"
   date_col = "date"
   value_col = "oil_bbl"
   filters = { min_months = 6 }

   [model]
   model = "arps"
   kind = "hyperbolic"
   horizon = 12

   [economics]
   price = 70.0
   opex = 15.0
   fixed_opex = 5000.0
   discount_rate = 0.10

   [[economics.scenarios]]
   name = "low"
   price = 50.0

   [[economics.scenarios]]
   name = "base"
   price = 70.0

   [[economics.scenarios]]
   name = "high"
   price = 90.0

   [output]
   output_dir = "output"
   save_forecasts = true
   save_parameters = true
   save_plots = false
   save_reports = false
   format = "csv"

   n_jobs = -1
   chunk_size = 100
   max_retries = 2
   log_level = "INFO"
   log_file = "batch_job.log"

Load and Use Configuration
---------------------------

.. code-block:: python

   from decline_curve.config import BatchJobConfig
   from decline_curve.runner import FieldScaleRunner
   from decline_curve.logging_config import configure_logging
   import logging

   # Load configuration
   config = BatchJobConfig.from_file('config.toml')

   # Configure logging
   configure_logging(
       level=getattr(logging, config.log_level),
       log_file=config.log_file
   )

   # Load data
   import pandas as pd
   df = pd.read_csv(config.data.path)
   df[config.data.date_col] = pd.to_datetime(df[config.data.date_col])

   # Create runner
   runner_config = RunnerConfig(
       n_jobs=config.n_jobs,
       chunk_size=config.chunk_size,
       max_retries=config.max_retries
   )
   runner = FieldScaleRunner(runner_config)

   # Define forecast function
   def forecast_well(df_well):
       from decline_curve import dca
       well_id = df_well[config.data.well_id_col].iloc[0]
       series = df_well.set_index(config.data.date_col)[config.data.value_col]

       try:
           forecast = dca.single_well(
               series,
               model=config.model.model,
               kind=config.model.kind,
               horizon=config.model.horizon
           )
           return {'well_id': well_id, 'forecast': forecast}
       except Exception as e:
           return {'well_id': well_id, 'error': str(e)}

   # Run analysis
   results = runner.run(
       df,
       forecast_func=forecast_well,
       well_id_col=config.data.well_id_col
   )

   # Save outputs
   from decline_curve.schemas import ProductionOutputSchema
   output_schema = ProductionOutputSchema.from_runner_results(results)
   output_schema.save(config.output.output_dir)

Using YAML Instead
------------------

.. code-block:: yaml

   # config.yaml
   data:
     path: "data/production.csv"
     format: "csv"
     well_id_col: "well_id"
     date_col: "date"
     value_col: "oil_bbl"

   model:
     model: "arps"
     kind: "hyperbolic"
     horizon: 12

   economics:
     price: 70.0
     opex: 15.0
     discount_rate: 0.10

   output:
     output_dir: "output"
     save_forecasts: true

Command Line Usage
------------------

Create a simple CLI script:

.. code-block:: python

   # run_analysis.py
   import sys
   from decline_curve.config import BatchJobConfig

   if __name__ == '__main__':
       config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.toml'
       config = BatchJobConfig.from_file(config_path)

       # Run analysis using config
       # ... (implementation as above)

Then run from command line:

.. code-block:: bash

   python run_analysis.py config.toml

Generate Example Config
------------------------

.. code-block:: python

   from decline_curve.config import create_example_config

   # Create example TOML config
   create_example_config('example_config.toml', format='toml')

   # Create example YAML config
   create_example_config('example_config.yaml', format='yaml')

See Also
--------

* :doc:`../workflows/state_scale_workflow` - Large-scale analysis with config
* :doc:`../examples/config_example.toml` - Example configuration file
