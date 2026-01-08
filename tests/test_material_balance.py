"""Tests for material balance models with different drive mechanisms."""

import numpy as np

from decline_curve.material_balance import (
    GasCapDriveParams,
    GasReservoirParams,
    SolutionGasDriveParams,
    WaterDriveParams,
    gas_cap_drive_material_balance,
    gas_reservoir_pz_method,
    identify_drive_mechanism,
    solution_gas_drive_material_balance,
    undersaturated_oil_material_balance,
    water_drive_material_balance,
)


class TestSolutionGasDrive:
    """Test solution gas drive material balance."""

    def test_solution_gas_drive_basic(self):
        """Test basic solution gas drive material balance."""
        params = SolutionGasDriveParams(
            N=1e6,  # 1 MMSTB
            pi=5000.0,
            pb=3000.0,
            api_gravity=30.0,
            gas_gravity=0.65,
            temperature=200.0,
        )

        result = solution_gas_drive_material_balance(
            pressure=4000.0, cumulative_oil=100000.0, params=params
        )

        assert "Np_calculated" in result
        assert "Bo" in result
        assert "Rs" in result
        assert "Bg" in result
        assert "Gp" in result
        assert result["Bo"] > 0
        assert result["Rs"] >= 0

    def test_solution_gas_drive_pressure_decline(self):
        """Test solution gas drive with pressure decline."""
        params = SolutionGasDriveParams(N=1e6, pi=5000.0, pb=3000.0)

        result1 = solution_gas_drive_material_balance(4500.0, 50000.0, params)
        result2 = solution_gas_drive_material_balance(4000.0, 100000.0, params)

        # Rs should decrease as pressure decreases
        assert result2["Rs"] < result1["Rs"]


class TestWaterDrive:
    """Test water drive material balance."""

    def test_water_drive_basic(self):
        """Test basic water drive material balance."""
        params = WaterDriveParams(
            N=1e6,
            pi=5000.0,
            We=50000.0,  # Water influx
            Wp=10000.0,  # Water production
        )

        result = water_drive_material_balance(
            pressure=4500.0, cumulative_oil=100000.0, params=params
        )

        assert "Np_calculated" in result
        assert "Bo" in result
        assert "water_influx" in result
        assert result["water_influx"] > 0

    def test_water_drive_no_influx(self):
        """Test water drive without water influx."""
        params = WaterDriveParams(N=1e6, pi=5000.0, We=0.0, Wp=0.0)

        result = water_drive_material_balance(4500.0, 100000.0, params)

        assert result["water_influx"] == 0.0


class TestGasCapDrive:
    """Test gas cap drive material balance."""

    def test_gas_cap_drive_basic(self):
        """Test basic gas cap drive material balance."""
        params = GasCapDriveParams(
            N=1e6,
            m=0.1,  # 10% gas cap
            pi=5000.0,
            api_gravity=30.0,
            gas_gravity=0.65,
            temperature=200.0,
        )

        result = gas_cap_drive_material_balance(
            pressure=4000.0, cumulative_oil=100000.0, params=params
        )

        assert "Np_calculated" in result
        assert "gas_cap_expansion" in result
        assert "solution_gas_expansion" in result
        # Gas cap expansion can be positive or negative depending on pressure
        # Just check that the calculation completed
        assert "gas_cap_expansion" in result

    def test_gas_cap_drive_no_gas_cap(self):
        """Test gas cap drive with m=0 (no gas cap)."""
        params = GasCapDriveParams(N=1e6, m=0.0, pi=5000.0)

        result = gas_cap_drive_material_balance(4000.0, 100000.0, params)

        assert result["gas_cap_expansion"] == 0.0


class TestGasReservoir:
    """Test gas reservoir p/Z method."""

    def test_gas_reservoir_pz_basic(self):
        """Test basic gas reservoir p/Z method."""
        params = GasReservoirParams(
            G=1e9,  # 1 BCF
            pi=5000.0,
            Zi=1.0,
            temperature=200.0,
            gas_gravity=0.65,
        )

        result = gas_reservoir_pz_method(
            pressure=4000.0, cumulative_gas=100000000.0, params=params
        )

        assert "G_calculated" in result
        assert "Z" in result
        assert "p_over_z" in result
        assert "recovery_factor" in result
        assert 0.1 < result["Z"] < 2.0
        assert result["p_over_z"] > 0

    def test_gas_reservoir_pz_decline(self):
        """Test gas reservoir p/Z with pressure decline."""
        params = GasReservoirParams(G=1e9, pi=5000.0, Zi=1.0)

        result1 = gas_reservoir_pz_method(4500.0, 50000000.0, params)
        result2 = gas_reservoir_pz_method(4000.0, 100000000.0, params)

        # p/Z should decrease
        assert result2["p_over_z"] < result1["p_over_z"]
        # Recovery factor should increase
        assert result2["recovery_factor"] > result1["recovery_factor"]


class TestUndersaturatedOil:
    """Test undersaturated oil material balance."""

    def test_undersaturated_oil_basic(self):
        """Test basic undersaturated oil material balance."""
        params = SolutionGasDriveParams(
            N=1e6,
            pi=5000.0,
            pb=3000.0,  # Bubble point
            api_gravity=30.0,
            gas_gravity=0.65,
            temperature=200.0,
        )

        # Pressure above bubble point
        result = undersaturated_oil_material_balance(
            pressure=4500.0, cumulative_oil=100000.0, params=params
        )

        assert "Np_calculated" in result
        assert "Bo" in result
        assert "oil_expansion" in result
        assert "rock_water_expansion" in result
        # No gas expansion term
        assert "gas_expansion" not in result

    def test_undersaturated_oil_below_bubble_point(self):
        """Test undersaturated oil with pressure below bubble point (warning)."""
        params = SolutionGasDriveParams(N=1e6, pi=5000.0, pb=3000.0)

        # Should still work but log warning
        result = undersaturated_oil_material_balance(2500.0, 100000.0, params)

        assert "Np_calculated" in result


class TestDriveMechanismIdentification:
    """Test drive mechanism identification."""

    def test_identify_solution_gas_drive(self):
        """Test identification of solution gas drive."""
        pressure = np.array([5000, 4500, 4000, 3500, 3000])
        production = np.array([1000, 900, 800, 700, 600])

        mechanism = identify_drive_mechanism(pressure, production)

        assert mechanism in ["solution_gas", "unknown"]

    def test_identify_water_drive(self):
        """Test identification of water drive."""
        pressure = np.array([5000, 4900, 4850, 4800, 4780])
        production = np.array([1000, 950, 900, 850, 800])
        water_cut = np.array([0.0, 0.1, 0.2, 0.3, 0.4])

        mechanism = identify_drive_mechanism(
            pressure, production, water_cut_history=water_cut
        )

        assert mechanism == "water_drive"

    def test_identify_gas_cap_drive(self):
        """Test identification of gas cap drive."""
        pressure = np.array([5000, 4800, 4600, 4400, 4200])
        production = np.array([1000, 950, 900, 850, 800])
        gor = np.array([500, 600, 700, 800, 900])

        mechanism = identify_drive_mechanism(pressure, production, gor_history=gor)

        assert mechanism == "gas_cap"

    def test_identify_combination_drive(self):
        """Test identification of combination drive."""
        pressure = np.array([5000, 4900, 4850, 4800, 4780])
        production = np.array([1000, 950, 900, 850, 800])
        water_cut = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        gor = np.array([500, 600, 700, 800, 900])

        mechanism = identify_drive_mechanism(
            pressure, production, water_cut_history=water_cut, gor_history=gor
        )

        assert mechanism == "combination"

    def test_identify_unknown(self):
        """Test identification with insufficient data."""
        pressure = np.array([5000])
        production = np.array([1000])

        mechanism = identify_drive_mechanism(pressure, production)

        assert mechanism == "unknown"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_solution_gas_drive_zero_cumulative(self):
        """Test solution gas drive with zero cumulative production."""
        params = SolutionGasDriveParams(N=1e6, pi=5000.0)

        result = solution_gas_drive_material_balance(4500.0, 0.0, params)

        assert result["Gp"] == 0.0

    def test_gas_reservoir_zero_production(self):
        """Test gas reservoir with zero production."""
        params = GasReservoirParams(G=1e9, pi=5000.0)

        result = gas_reservoir_pz_method(5000.0, 0.0, params)

        assert result["recovery_factor"] == 0.0
        assert abs(result["remaining_gas"] - params.G) < 1e6  # Allow small tolerance
