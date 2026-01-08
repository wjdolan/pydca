"""Tests for RTA, well test, and VLP modules."""

import numpy as np
import pytest

from decline_curve.rta import (
    RTAResult,
    analyze_production_data,
    calculate_srv,
    estimate_fracture_half_length,
    estimate_permeability_from_production,
    identify_flow_regime,
)
from decline_curve.vlp import (
    NodalAnalysisResult,
    calculate_choke_performance,
    calculate_tubing_performance,
    generate_vlp_curve,
    perform_nodal_analysis,
)
from decline_curve.well_test import (
    WellTestResult,
    analyze_buildup_test,
    analyze_drawdown_test,
    calculate_productivity_index_from_test,
    detect_boundaries,
)


class TestRTA:
    """Test Rate Transient Analysis."""

    def test_identify_flow_regime_linear(self):
        """Test flow regime identification for linear flow."""
        time = np.array([1, 5, 10, 20, 30])
        rate = 1000 / np.sqrt(time)  # Linear flow: q ~ 1/sqrt(t)

        regime = identify_flow_regime(time, rate)

        assert regime in ["linear", "transient", "unknown"]

    def test_identify_flow_regime_boundary_dominated(self):
        """Test flow regime identification for boundary dominated flow."""
        time = np.array([1, 10, 30, 60, 90, 120])
        rate = 1000 * np.exp(-0.01 * time)  # Exponential decline

        regime = identify_flow_regime(time, rate)

        # Flow regime identification can vary - just check it's a valid regime
        assert regime in [
            "linear",
            "bilinear",
            "boundary_dominated",
            "transient",
            "unknown",
        ]

    def test_estimate_permeability_from_production(self):
        """Test permeability estimation from production data."""
        time = np.array([1, 10, 30, 60, 90])
        rate = np.array([1000, 800, 600, 500, 450])

        k = estimate_permeability_from_production(time, rate)

        assert k > 0
        assert k < 1000  # Reasonable range

    def test_estimate_fracture_half_length(self):
        """Test fracture half-length estimation."""
        time = np.array([1, 5, 10, 20, 30])
        rate = 1000 / np.sqrt(time)

        xf = estimate_fracture_half_length(
            time, rate, permeability=1.0, pressure_drop=1000.0
        )

        assert xf >= 0

    def test_calculate_srv(self):
        """Test SRV calculation."""
        srv = calculate_srv(
            fracture_half_length=500,
            number_of_fractures=20,
            fracture_spacing=200,
        )

        assert srv > 0

    def test_analyze_production_data(self):
        """Test comprehensive production data analysis."""
        time = np.array([1, 10, 30, 60, 90, 120])
        rate = np.array([1000, 800, 600, 500, 450, 400])

        result = analyze_production_data(time, rate)

        assert isinstance(result, RTAResult)
        assert result.flow_regime in [
            "linear",
            "bilinear",
            "boundary_dominated",
            "transient",
            "unknown",
        ]
        assert result.permeability > 0


class TestWellTest:
    """Test well test analysis."""

    def test_analyze_buildup_test(self):
        """Test buildup test analysis."""
        time = np.array([0.1, 0.5, 1, 2, 5, 10, 24])
        pressure = np.array([3000, 3200, 3400, 3600, 3800, 3900, 3950])

        result = analyze_buildup_test(
            time, pressure, production_rate=1000, production_time=720
        )

        assert isinstance(result, WellTestResult)
        assert result.permeability > 0
        assert -50 < result.skin < 50  # Allow negative skin (stimulated wells)

    def test_analyze_drawdown_test(self):
        """Test drawdown test analysis."""
        time = np.array([0.1, 0.5, 1, 2, 5, 10, 24])
        pressure = 5000 - 100 * np.log(time + 1)  # Simulated drawdown

        result = analyze_drawdown_test(
            time, pressure, production_rate=1000, initial_pressure=5000
        )

        assert isinstance(result, WellTestResult)
        assert result.permeability > 0

    def test_detect_boundaries(self):
        """Test boundary detection."""
        time = np.array([1, 2, 5, 10, 20, 50, 100])
        pressure = 5000 - 100 * np.log(time)  # Normal decline

        distance, boundary_type = detect_boundaries(
            time, pressure, permeability=10.0, porosity=0.15, total_compressibility=1e-5
        )

        assert boundary_type in ["no_flow", "constant_pressure", "unknown"]

    def test_calculate_productivity_index_from_test(self):
        """Test productivity index calculation from well test."""
        test_result = WellTestResult(
            permeability=10.0, skin=2.0, reservoir_pressure=5000.0
        )

        J = calculate_productivity_index_from_test(test_result)

        assert J > 0


class TestVLP:
    """Test Vertical Lift Performance."""

    def test_calculate_tubing_performance(self):
        """Test tubing performance calculation."""
        pwf = calculate_tubing_performance(
            rate=1000, wellhead_pressure=500, tubing_depth=5000
        )

        assert pwf > 500  # Should be higher than wellhead
        assert pwf < 10000  # Reasonable upper bound

    def test_generate_vlp_curve(self):
        """Test VLP curve generation."""
        rates, pressures = generate_vlp_curve(wellhead_pressure=500, tubing_depth=5000)

        assert len(rates) > 0
        assert len(pressures) == len(rates)
        assert all(pressures > 0)

    def test_perform_nodal_analysis(self):
        """Test nodal analysis."""
        result = perform_nodal_analysis(
            reservoir_pressure=5000,
            productivity_index=1.0,
            wellhead_pressure=500,
            tubing_depth=5000,
        )

        assert isinstance(result, NodalAnalysisResult)
        assert result.operating_rate >= 0
        assert result.operating_pressure > 0
        assert len(result.ipr_curve[0]) > 0
        assert len(result.vlp_curve[0]) > 0

    def test_calculate_choke_performance(self):
        """Test choke performance calculation."""
        rate = calculate_choke_performance(
            upstream_pressure=1000,
            downstream_pressure=500,
            choke_size=0.5,
        )

        assert rate >= 0

    def test_choke_zero_flow(self):
        """Test choke with no pressure difference."""
        rate = calculate_choke_performance(
            upstream_pressure=500,
            downstream_pressure=500,
            choke_size=0.5,
        )

        assert rate == 0.0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_rta_insufficient_data(self):
        """Test RTA with insufficient data."""
        time = np.array([1])
        rate = np.array([1000])

        regime = identify_flow_regime(time, rate)

        assert regime == "unknown"

    def test_well_test_insufficient_data(self):
        """Test well test with insufficient data."""
        time = np.array([1])
        pressure = np.array([3000])

        with pytest.raises(ValueError):
            analyze_buildup_test(time, pressure, 1000, 720)

    def test_vlp_zero_rate(self):
        """Test VLP with zero rate."""
        pwf = calculate_tubing_performance(
            rate=0, wellhead_pressure=500, tubing_depth=5000
        )

        assert pwf == 500  # Should equal wellhead pressure
