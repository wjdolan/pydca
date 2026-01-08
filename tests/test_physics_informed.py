"""Tests for physics-informed decline curve models."""

import numpy as np
import pandas as pd
import pytest

from decline_curve.physics_informed import (
    MaterialBalanceDecline,
    MaterialBalanceParams,
    PressureDeclineModel,
    apply_physics_constraints,
    compare_dca_with_simulation,
    load_reservoir_simulation,
    material_balance_forecast,
    pressure_decline_forecast,
)


class TestMaterialBalanceDecline:
    """Test material balance decline model."""

    def test_material_balance_model_interface(self):
        """Test that MaterialBalanceDecline implements Model interface."""
        model = MaterialBalanceDecline()
        assert model.name == "MaterialBalanceDecline"
        assert hasattr(model, "rate")
        assert hasattr(model, "cum")
        assert hasattr(model, "constraints")
        assert hasattr(model, "initial_guess")

    def test_material_balance_rate(self):
        """Test material balance rate calculation."""
        model = MaterialBalanceDecline()
        t = np.array([0, 30, 60, 90, 120])
        params = {"N": 1e6, "D": 0.001, "qi": 1000.0, "Boi": 1.2, "pi": 5000.0}

        rates = model.rate(t, params)

        assert len(rates) == len(t)
        assert all(rates >= 0)
        assert rates[0] > rates[-1]  # Should decline

    def test_material_balance_cumulative(self):
        """Test material balance cumulative calculation."""
        model = MaterialBalanceDecline()
        t = np.array([0, 30, 60, 90, 120])
        params = {"N": 1e6, "D": 0.001}

        cum = model.cum(t, params)

        assert len(cum) == len(t)
        assert all(cum >= 0)
        assert cum[-1] > cum[0]  # Should increase
        assert cum[-1] <= params["N"]  # Can't exceed OOIP

    def test_material_balance_constraints(self):
        """Test material balance parameter constraints."""
        model = MaterialBalanceDecline()
        constraints = model.constraints()

        assert "N" in constraints
        assert "D" in constraints
        assert "qi" in constraints
        assert constraints["N"][0] > 0  # N must be positive

    def test_material_balance_initial_guess(self):
        """Test initial guess generation."""
        model = MaterialBalanceDecline()
        t = np.arange(0, 120, 1)
        q = 1000 * np.exp(-0.01 * t)

        guess = model.initial_guess(t, q)

        assert "N" in guess
        assert "D" in guess
        assert "qi" in guess
        assert guess["N"] > 0
        assert guess["D"] > 0


class TestPressureDeclineModel:
    """Test pressure decline model."""

    def test_pressure_decline_model_interface(self):
        """Test that PressureDeclineModel implements Model interface."""
        model = PressureDeclineModel()
        assert model.name == "PressureDeclineModel"
        assert hasattr(model, "rate")
        assert hasattr(model, "cum")
        assert hasattr(model, "constraints")
        assert hasattr(model, "initial_guess")

    def test_pressure_decline_rate(self):
        """Test pressure decline rate calculation."""
        model = PressureDeclineModel()
        t = np.array([0, 30, 60, 90, 120])
        params = {"pi": 5000.0, "D": 0.001, "b": 1.0, "J": 1.0, "pwf": 500.0}

        rates = model.rate(t, params)

        assert len(rates) == len(t)
        assert all(rates >= 0)
        # Rate should decline as pressure declines
        assert rates[0] > rates[-1] or rates[0] == 0

    def test_pressure_decline_constraints(self):
        """Test pressure decline parameter constraints."""
        model = PressureDeclineModel()
        constraints = model.constraints()

        assert "pi" in constraints
        assert "D" in constraints
        assert "J" in constraints
        assert "pwf" in constraints
        assert constraints["pi"][0] > 0
        assert constraints["J"][0] > 0


class TestPhysicsConstraints:
    """Test physics-informed constraint application."""

    def test_apply_physics_constraints_non_negative(self):
        """Test that constraints ensure non-negative production."""
        forecast = np.array([100, 50, -10, 20, 30])
        constrained = apply_physics_constraints(forecast, min_rate=0.0)

        assert all(constrained >= 0)
        assert constrained[2] == 0  # Negative value clipped

    def test_apply_physics_constraints_continuity(self):
        """Test continuity with historical data."""
        historical = np.array([100, 90, 80])
        forecast = np.array([200, 150, 100])  # Unrealistic jump

        constrained = apply_physics_constraints(
            forecast, historical=historical, max_increase=0.1
        )

        # First forecast should be near last historical (within 10% increase)
        assert constrained[0] <= historical[-1] * 1.1

    def test_apply_physics_constraints_decline(self):
        """Test enforced decline behavior."""
        forecast = np.array([100, 120, 110, 90])  # Has increase

        constrained = apply_physics_constraints(forecast, enforce_decline=True)

        # Should be monotonically decreasing
        for i in range(1, len(constrained)):
            assert constrained[i] <= constrained[i - 1] * 1.01  # Allow small tolerance

    def test_apply_physics_constraints_empty(self):
        """Test with empty forecast."""
        forecast = np.array([])
        constrained = apply_physics_constraints(forecast)

        assert len(constrained) == 0


class TestMaterialBalanceForecast:
    """Test material balance forecasting function."""

    def test_material_balance_forecast_basic(self):
        """Test basic material balance forecast."""
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        production = pd.Series(
            1000 * np.exp(-0.01 * np.arange(24)), index=dates, name="oil"
        )

        forecast = material_balance_forecast(production, horizon=12)

        assert len(forecast) == 12
        assert all(forecast >= 0)
        assert isinstance(forecast.index, pd.DatetimeIndex)

    def test_material_balance_forecast_with_params(self):
        """Test material balance forecast with custom parameters."""
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        production = pd.Series(
            1000 * np.exp(-0.01 * np.arange(24)), index=dates, name="oil"
        )

        params = MaterialBalanceParams(N=1e6, Boi=1.2, pi=5000.0)

        forecast = material_balance_forecast(
            production, material_balance_params=params, horizon=12
        )

        assert len(forecast) == 12


class TestPressureDeclineForecast:
    """Test pressure decline forecasting function."""

    def test_pressure_decline_forecast_basic(self):
        """Test basic pressure decline forecast."""
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        pressure = pd.Series(
            5000 * np.exp(-0.001 * np.arange(24)), index=dates, name="pressure"
        )

        pressure_fcst, production_fcst = pressure_decline_forecast(pressure, horizon=12)

        assert len(pressure_fcst) == 12
        assert len(production_fcst) == 12
        assert all(pressure_fcst >= 0)
        assert all(production_fcst >= 0)

    def test_pressure_decline_forecast_with_production(self):
        """Test pressure decline forecast with production calibration."""
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        pressure = pd.Series(
            5000 * np.exp(-0.001 * np.arange(24)), index=dates, name="pressure"
        )
        production = pd.Series(
            1000 * np.exp(-0.01 * np.arange(24)), index=dates, name="oil"
        )

        pressure_fcst, production_fcst = pressure_decline_forecast(
            pressure, production_data=production, horizon=12
        )

        assert len(pressure_fcst) == 12
        assert len(production_fcst) == 12


class TestReservoirSimulationIntegration:
    """Test reservoir simulation integration."""

    def test_load_reservoir_simulation_csv(self, tmp_path):
        """Test loading CSV simulation data."""
        # Create test CSV
        csv_file = tmp_path / "sim_data.csv"
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=12, freq="MS"),
                "pressure": 5000 * np.exp(-0.001 * np.arange(12)),
                "oil_rate": 1000 * np.exp(-0.01 * np.arange(12)),
                "gas_rate": 5000 * np.exp(-0.01 * np.arange(12)),
            }
        )
        df.to_csv(csv_file, index=False)

        sim_data = load_reservoir_simulation(str(csv_file))

        assert "pressure" in sim_data.columns
        assert "oil_rate" in sim_data.columns
        assert len(sim_data) == 12

    def test_compare_dca_with_simulation(self):
        """Test comparing DCA forecast with simulation."""
        dates = pd.date_range("2020-01-01", periods=12, freq="MS")
        dca_forecast = pd.Series(
            1000 * np.exp(-0.01 * np.arange(12)), index=dates, name="forecast"
        )

        sim_data = pd.DataFrame(
            {
                "oil_rate": 950 * np.exp(-0.01 * np.arange(12)),
            },
            index=dates,
        )

        comparison = compare_dca_with_simulation(
            dca_forecast, sim_data, production_col="oil_rate"
        )

        assert len(comparison) > 0
        assert "dca_forecast" in comparison.columns
        assert "simulation" in comparison.columns
        assert "difference" in comparison.columns


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_material_balance_empty_data(self):
        """Test material balance with empty data."""
        empty_series = pd.Series([], dtype=float)

        with pytest.raises((ValueError, IndexError)):
            material_balance_forecast(empty_series, horizon=12)

    def test_pressure_decline_empty_data(self):
        """Test pressure decline with empty data."""
        empty_series = pd.Series([], dtype=float)

        with pytest.raises((ValueError, IndexError)):
            pressure_decline_forecast(empty_series, horizon=12)

    def test_load_simulation_nonexistent_file(self):
        """Test loading nonexistent simulation file."""
        with pytest.raises((FileNotFoundError, pd.errors.EmptyDataError)):
            load_reservoir_simulation("nonexistent_file.csv")
