"""Tests for history matching, multi-phase flow, and physics-based reserves."""

import numpy as np
import pandas as pd
import pytest

from decline_curve.history_matching import (
    HistoryMatchResult,
    history_match_material_balance,
    quantify_parameter_uncertainty,
    sensitivity_analysis_material_balance,
)
from decline_curve.multiphase_flow import (
    FlowPattern,
    beggs_brill_correlation,
    calculate_liquid_holdup,
    hagedorn_brown_correlation,
    identify_flow_pattern,
)
from decline_curve.physics_reserves import (
    ReservesClassification,
    classify_reserves_from_material_balance,
    identify_decline_type_from_physics,
)


class TestHistoryMatching:
    """Test history matching functionality."""

    def test_history_match_material_balance_basic(self):
        """Test basic history matching."""
        time = np.array([30, 60, 90, 120, 150, 180])
        production = np.array([10000, 20000, 30000, 40000, 50000, 60000])

        result = history_match_material_balance(time, production)

        assert isinstance(result, HistoryMatchResult)
        assert "N" in result.optimized_params
        assert result.production_match["rmse"] >= 0

    def test_history_match_with_pressure(self):
        """Test history matching with pressure data."""
        time = np.array([30, 60, 90, 120])
        production = np.array([10000, 20000, 30000, 40000])
        pressure = np.array([5000, 4800, 4600, 4400])

        result = history_match_material_balance(time, production, pressure=pressure)

        assert result.pressure_match["rmse"] >= 0

    def test_quantify_parameter_uncertainty(self):
        """Test parameter uncertainty quantification."""
        result = HistoryMatchResult(
            optimized_params={"N": 1e6, "pi": 5000.0, "D": 0.001},
            objective_value=1000.0,
            success=True,
            message="",
            iterations=10,
            pressure_match={"rmse": 10.0, "mae": 8.0},
            production_match={"rmse": 100.0, "mae": 80.0},
        )

        time = np.array([30, 60, 90])
        production = np.array([10000, 20000, 30000])

        uncertainty = quantify_parameter_uncertainty(
            result, time, production, n_samples=100
        )

        assert "N" in uncertainty
        assert "p10" in uncertainty["N"]
        assert "p50" in uncertainty["N"]
        assert "p90" in uncertainty["N"]

    def test_sensitivity_analysis(self):
        """Test sensitivity analysis."""
        time = np.array([30, 60, 90])
        production = np.array([10000, 20000, 30000])
        base_params = {"N": 1e6, "D": 0.001}

        results = sensitivity_analysis_material_balance(time, production, base_params)

        assert len(results) > 0
        assert "parameter" in results.columns
        assert "rmse" in results.columns


class TestMultiphaseFlow:
    """Test multi-phase flow correlations."""

    def test_identify_flow_pattern(self):
        """Test flow pattern identification."""
        pattern = identify_flow_pattern(
            liquid_rate=1000,
            gas_rate=500,
            pipe_diameter=2.441,
            liquid_density=50.0,
            gas_density=0.1,
            liquid_viscosity=1.0,
            gas_viscosity=0.01,
        )

        assert isinstance(pattern, FlowPattern)
        assert pattern.pattern in ["bubble", "slug", "churn", "annular", "mist"]
        assert 0.0 <= pattern.liquid_holdup <= 1.0

    def test_beggs_brill_correlation(self):
        """Test Beggs-Brill correlation."""
        drop, holdup = beggs_brill_correlation(
            liquid_rate=1000,
            gas_rate=500,
            pipe_diameter=2.441,
            pipe_length=5000,
            liquid_density=50.0,
            gas_density=0.1,
            liquid_viscosity=1.0,
            gas_viscosity=0.01,
        )

        assert drop > 0
        assert 0.0 <= holdup <= 1.0

    def test_hagedorn_brown_correlation(self):
        """Test Hagedorn-Brown correlation."""
        drop, holdup = hagedorn_brown_correlation(
            liquid_rate=1000,
            gas_rate=500,
            pipe_diameter=2.441,
            pipe_length=5000,
            liquid_density=50.0,
            gas_density=0.1,
            liquid_viscosity=1.0,
            gas_viscosity=0.01,
        )

        assert drop > 0
        assert 0.0 <= holdup <= 1.0

    def test_calculate_liquid_holdup(self):
        """Test liquid holdup calculation."""
        holdup = calculate_liquid_holdup(
            liquid_rate=1000,
            gas_rate=500,
            pipe_diameter=2.441,
            liquid_density=50.0,
            gas_density=0.1,
            method="beggs_brill",
        )

        assert 0.0 <= holdup <= 1.0


class TestPhysicsReserves:
    """Test physics-based reserves classification."""

    def test_classify_reserves_from_material_balance(self):
        """Test reserves classification."""
        params = {"N": 1e6, "pi": 5000.0}
        uncertainty = {
            "N": {"p10": 8e5, "p50": 1e6, "p90": 1.2e6},
            "pi": {"p10": 4500, "p50": 5000, "p90": 5500},
        }

        reserves = classify_reserves_from_material_balance(
            params, uncertainty, n_simulations=100
        )

        assert isinstance(reserves, ReservesClassification)
        assert reserves.p1_reserves > 0
        assert reserves.p2_reserves > 0
        assert reserves.p3_reserves > 0
        # P1 (P90, conservative) >= P2 (P50, best) >= P3 (P10, optimistic)
        # Actually: P90 >= P50 >= P10, so p1 >= p2 >= p3
        assert reserves.p1_reserves >= reserves.p2_reserves >= reserves.p3_reserves

    def test_identify_decline_type_from_physics(self):
        """Test decline type identification from physics."""
        dates = pd.date_range("2020-01-01", periods=12, freq="MS")
        production = pd.Series(1000 * np.exp(-0.01 * np.arange(12)), index=dates)

        decline_type = identify_decline_type_from_physics(production)

        assert decline_type in ["exponential", "harmonic", "hyperbolic"]

    def test_identify_decline_type_with_drive_mechanism(self):
        """Test decline type identification with drive mechanism."""
        dates = pd.date_range("2020-01-01", periods=12, freq="MS")
        production = pd.Series(1000 * np.exp(-0.01 * np.arange(12)), index=dates)

        decline_type = identify_decline_type_from_physics(
            production, drive_mechanism="water_drive"
        )

        assert decline_type in ["exponential", "harmonic", "hyperbolic"]


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_history_match_insufficient_data(self):
        """Test history matching with insufficient data."""
        time = np.array([30])
        production = np.array([10000])

        # Should still work but may not converge
        result = history_match_material_balance(time, production)

        assert isinstance(result, HistoryMatchResult)

    def test_multiphase_flow_zero_rates(self):
        """Test multi-phase flow with zero rates."""
        pattern = identify_flow_pattern(
            liquid_rate=0,
            gas_rate=0,
            pipe_diameter=2.441,
            liquid_density=50.0,
            gas_density=0.1,
            liquid_viscosity=1.0,
            gas_viscosity=0.01,
        )

        assert pattern.liquid_holdup >= 0
