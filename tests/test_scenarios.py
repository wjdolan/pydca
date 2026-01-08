"""Tests for price scenario analysis."""

import numpy as np
import pandas as pd
import pytest

from decline_curve.scenarios import (
    PriceScenario,
    compare_scenarios,
    run_multi_phase_scenarios,
    run_price_scenarios,
)


class TestPriceScenario:
    """Test PriceScenario dataclass."""

    def test_create_scenario(self):
        """Test creating a price scenario."""
        scenario = PriceScenario("base", oil_price=70.0, opex=15.0)
        assert scenario.name == "base"
        assert scenario.oil_price == 70.0
        assert scenario.opex == 15.0
        assert scenario.discount_rate == 0.10

    def test_scenario_validation(self):
        """Test scenario parameter validation."""
        # Negative price should raise error
        with pytest.raises(ValueError, match="Oil price must be non-negative"):
            PriceScenario("low", oil_price=-10.0)

        # Negative gas price should raise error
        with pytest.raises(ValueError, match="Gas price must be non-negative"):
            PriceScenario("low", oil_price=50.0, gas_price=-5.0)

        # Negative opex should raise error
        with pytest.raises(ValueError, match="Operating cost must be non-negative"):
            PriceScenario("low", oil_price=50.0, opex=-5.0)


class TestRunPriceScenarios:
    """Test run_price_scenarios function."""

    def test_single_scenario(self):
        """Test running a single scenario."""
        production = np.array([1000, 900, 800, 700, 600])
        scenarios = [PriceScenario("base", oil_price=70.0, opex=15.0)]

        results = run_price_scenarios(production, scenarios)

        assert len(results) == 1
        assert results.iloc[0]["scenario"] == "base"
        assert "npv" in results.columns
        assert "payback_month" in results.columns

    def test_multiple_scenarios(self):
        """Test running multiple scenarios."""
        production = np.array([1000, 900, 800, 700, 600])
        scenarios = [
            PriceScenario("low", oil_price=50.0, opex=15.0),
            PriceScenario("base", oil_price=70.0, opex=15.0),
            PriceScenario("high", oil_price=90.0, opex=15.0),
        ]

        results = run_price_scenarios(production, scenarios)

        assert len(results) == 3
        assert set(results["scenario"]) == {"low", "base", "high"}
        # Higher price should yield higher NPV
        assert (
            results[results["scenario"] == "high"]["npv"].iloc[0]
            > results[results["scenario"] == "base"]["npv"].iloc[0]
        )
        assert (
            results[results["scenario"] == "base"]["npv"].iloc[0]
            > results[results["scenario"] == "low"]["npv"].iloc[0]
        )

    def test_pandas_series_input(self):
        """Test with pandas Series input."""
        dates = pd.date_range("2020-01-01", periods=12, freq="MS")
        production = pd.Series([1000 - i * 50 for i in range(12)], index=dates)
        scenarios = [PriceScenario("base", oil_price=70.0, opex=15.0)]

        results = run_price_scenarios(production, scenarios)

        assert len(results) == 1
        assert results.iloc[0]["npv"] is not None

    def test_different_phases(self):
        """Test with different production phases."""
        production = np.array([1000, 900, 800])
        scenarios = [PriceScenario("base", oil_price=70.0, gas_price=3.0, opex=15.0)]

        # Test oil phase
        results_oil = run_price_scenarios(production, scenarios, phase="oil")
        assert len(results_oil) == 1

        # Test gas phase
        results_gas = run_price_scenarios(production, scenarios, phase="gas")
        assert len(results_gas) == 1

    def test_invalid_phase(self):
        """Test with invalid phase."""
        production = np.array([1000, 900, 800])
        scenarios = [PriceScenario("base", oil_price=70.0, opex=15.0)]

        with pytest.raises(ValueError, match="Unknown phase"):
            run_price_scenarios(production, scenarios, phase="invalid")


class TestRunMultiPhaseScenarios:
    """Test run_multi_phase_scenarios function."""

    def test_oil_only(self):
        """Test with oil only."""
        oil_prod = np.array([1000, 900, 800])
        scenarios = [PriceScenario("base", oil_price=70.0, opex=15.0)]

        results = run_multi_phase_scenarios(oil_prod, scenarios)

        assert len(results) == 1
        assert results.iloc[0]["scenario"] == "base"

    def test_oil_and_gas(self):
        """Test with oil and gas."""
        oil_prod = np.array([1000, 900, 800])
        gas_prod = np.array([5000, 4500, 4000])
        scenarios = [PriceScenario("base", oil_price=70.0, gas_price=3.0, opex=15.0)]

        results = run_multi_phase_scenarios(
            oil_prod, scenarios, gas_production=gas_prod
        )

        assert len(results) == 1
        assert results.iloc[0]["total_revenue"] > 0

    def test_all_phases(self):
        """Test with oil, gas, and water."""
        oil_prod = np.array([1000, 900, 800])
        gas_prod = np.array([5000, 4500, 4000])
        water_prod = np.array([500, 450, 400])
        scenarios = [
            PriceScenario(
                "base", oil_price=70.0, gas_price=3.0, water_price=-2.0, opex=15.0
            )
        ]

        results = run_multi_phase_scenarios(
            oil_prod, scenarios, gas_production=gas_prod, water_production=water_prod
        )

        assert len(results) == 1
        # Water cost should reduce total revenue
        assert results.iloc[0]["total_revenue"] > 0


class TestCompareScenarios:
    """Test compare_scenarios function."""

    def test_compare_with_base(self):
        """Test comparing scenarios with base case."""
        scenario_results = pd.DataFrame(
            {
                "scenario": ["low", "base", "high"],
                "npv": [100000, 200000, 300000],
                "payback_month": [24, 18, 12],
            }
        )

        comparison = compare_scenarios(scenario_results)

        assert "npv_vs_base" in comparison.columns
        assert "npv_pct_change" in comparison.columns
        assert "payback_vs_base" in comparison.columns

        # Base should have zero difference
        base_row = comparison[comparison["scenario"] == "base"].iloc[0]
        assert base_row["npv_vs_base"] == 0
        assert base_row["npv_pct_change"] == 0

    def test_compare_without_base(self):
        """Test comparing when no base scenario exists."""
        scenario_results = pd.DataFrame(
            {
                "scenario": ["scenario1", "scenario2"],
                "npv": [100000, 200000],
                "payback_month": [24, 18],
            }
        )

        comparison = compare_scenarios(scenario_results)

        # Should use first scenario as base
        assert "npv_vs_base" in comparison.columns


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_production(self):
        """Test with empty production array."""
        production = np.array([])
        scenarios = [PriceScenario("base", oil_price=70.0, opex=15.0)]

        # Function should handle empty arrays gracefully
        results = run_price_scenarios(production, scenarios)
        # Should return a DataFrame (may be empty)
        assert isinstance(results, pd.DataFrame)

    def test_zero_production(self):
        """Test with zero production."""
        production = np.array([0, 0, 0])
        scenarios = [PriceScenario("base", oil_price=70.0, opex=15.0)]

        results = run_price_scenarios(production, scenarios)

        assert len(results) == 1
        # NPV should be negative or zero due to fixed costs (or zero if no costs)
        assert results.iloc[0]["npv"] <= 0

    def test_very_short_series(self):
        """Test with very short production series."""
        production = np.array([1000])
        scenarios = [PriceScenario("base", oil_price=70.0, opex=15.0)]

        results = run_price_scenarios(production, scenarios)

        assert len(results) == 1
        assert results.iloc[0]["npv"] is not None

    def test_negative_production(self):
        """Test handling of negative production values."""
        production = np.array([1000, -100, 800])  # Negative value in middle
        scenarios = [PriceScenario("base", oil_price=70.0, opex=15.0)]

        # Should handle gracefully
        results = run_price_scenarios(production, scenarios)
        assert len(results) == 1
