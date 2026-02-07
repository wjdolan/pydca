# Pruning Playbook (Core-First)

This repo supports many optional features. If your goal is a smaller, easier-to-maintain fork, start with a strict core and add features back only when needed.

## Recommended Baseline

Keep first:

- `decline_curve/dca.py`
- `decline_curve/models.py`
- `decline_curve/models_arps.py`
- `decline_curve/forecast.py`
- `decline_curve/evaluate.py`
- `decline_curve/economics.py`
- `decline_curve/reserves.py`
- `decline_curve/plot.py`
- `decline_curve/logging_config.py`
- `decline_curve/utils/data_processing.py`

This gives you classic Arps DCA + metrics + economics + plotting.

## Optional Feature Groups

Remove these only when you are sure you do not need them:

- Statistical forecasting: `forecast_arima.py`, `forecast_statistical.py`, `panel_analysis.py` (depends on `statsmodels`)
- Deep learning / LLM: `deep_learning.py`, `forecast_deepar.py`, `forecast_tft.py`, `forecast_timesfm.py`, `forecast_chronos.py`, `ensemble.py` (depends on `torch`/`transformers`)
- Spatial/integrations: `spatial_kriging.py`, `integrations.py` (depends on `pykrige`, `pygeomodeling`, `geosuite`)
- Config/batch ecosystem: `config.py`, `runner.py`, `schemas.py`, `batch.py`, `catalog.py`, `benchmark_factory.py`, `panel_analysis_sweep.py`
- Reports/artifacts: `reports.py`, `artifacts.py`, `registry.py`, `risk_report.py` (depends on `reportlab`, `mlflow`)
- Physics-heavy modules: `physics_informed.py`, `physics_reserves.py`, `pvt.py`, `ipr.py`, `rta.py`, `vlp.py`, `well_test.py`, `yield_models.py`

## Current Dependency Signals

From code inspection:

- `pydantic` used in `fitting.py`, `econ_spec.py`
- `yaml` used in `config.py`, `catalog.py`, `batch.py`, `panel_analysis_sweep.py`
- `mlflow` used in `benchmark_factory.py`, `panel_analysis_sweep.py`
- `reportlab` used in `reports.py`
- `statsmodels` used in `forecast_arima.py`, `forecast_statistical.py`, `panel_analysis.py`
- `torch`/`transformers` used in deep-learning and foundation-model forecast modules
- `pykrige`/`pygeomodeling` used in `spatial_kriging.py`, `integrations.py`

## Practical 3-Phase Trim Plan

1. Freeze a known-good baseline
- Create a branch: `git checkout -b prune/core-baseline`
- Install editable with dev tools: `pip install -e ".[dev,examples]"`
- Run tests: `pytest -q`

2. Trim import surface (low risk)
- In `decline_curve/__init__.py`, avoid eager importing many submodules.
- Export only stable core symbols first (`dca`, `configure_logging`, `get_logger`).
- Re-test after this change.

3. Remove feature groups one by one (medium risk)
- Delete one group at a time.
- Remove related optional extras from `pyproject.toml`.
- Update docs/examples to avoid removed modules.
- Run tests after each group removal.

## First Safe Change to Make

If you want minimal warnings and cleaner imports, simplify `decline_curve/__init__.py` first. The current file eagerly imports many modules, which triggers optional dependency warnings even when you are not using those features.

## Decision Rule

Only keep a module if at least one of these is true:

- It is needed by your intended public API.
- It is required by an example/test you plan to keep.
- It is cheap to maintain and has no heavy optional dependency burden.
