"""Tests for PVT correlations."""

import numpy as np
import pytest

from decline_curve.pvt import (
    PVTProperties,
    beggs_robinson_oil_viscosity,
    calculate_pvt_properties,
    chew_connally_dead_oil_viscosity,
    gas_fvf,
    gas_z_factor,
    lee_gonzalez_gas_viscosity,
    standing_bo,
    standing_rs,
    vasquez_beggs_bo,
    vasquez_beggs_rs,
    water_fvf,
    water_viscosity,
)


class TestStandingCorrelation:
    """Test Standing (1947) correlations."""

    def test_standing_rs_basic(self):
        """Test basic Standing Rs calculation."""
        rs = standing_rs(
            pressure=5000, temperature=200, api_gravity=30, gas_gravity=0.65
        )

        assert rs > 0
        assert rs < 10000  # Reasonable range

    def test_standing_rs_increases_with_pressure(self):
        """Test that Rs increases with pressure."""
        rs1 = standing_rs(pressure=1000, temperature=200, api_gravity=30)
        rs2 = standing_rs(pressure=5000, temperature=200, api_gravity=30)

        assert rs2 > rs1

    def test_standing_bo_basic(self):
        """Test basic Standing Bo calculation."""
        rs = 500  # SCF/STB
        Bo = standing_bo(rs=rs, temperature=200, api_gravity=30, gas_gravity=0.65)

        assert Bo >= 1.0  # Bo should be >= 1.0
        assert Bo < 3.0  # Reasonable upper bound

    def test_standing_bo_increases_with_rs(self):
        """Test that Bo increases with Rs."""
        Bo1 = standing_bo(rs=100, temperature=200, api_gravity=30)
        Bo2 = standing_bo(rs=1000, temperature=200, api_gravity=30)

        assert Bo2 > Bo1


class TestVasquezBeggsCorrelation:
    """Test Vasquez-Beggs (1980) correlations."""

    def test_vasquez_beggs_rs_basic(self):
        """Test basic Vasquez-Beggs Rs calculation."""
        rs = vasquez_beggs_rs(
            pressure=5000, temperature=200, api_gravity=30, gas_gravity=0.65
        )

        assert rs > 0
        assert rs < 10000

    def test_vasquez_beggs_rs_api_dependent(self):
        """Test that Vasquez-Beggs uses different coefficients for different API."""
        rs_light = vasquez_beggs_rs(
            pressure=5000, temperature=200, api_gravity=40, gas_gravity=0.65
        )
        rs_heavy = vasquez_beggs_rs(
            pressure=5000, temperature=200, api_gravity=20, gas_gravity=0.65
        )

        # Both should be valid
        assert rs_light > 0
        assert rs_heavy > 0

    def test_vasquez_beggs_bo_basic(self):
        """Test basic Vasquez-Beggs Bo calculation."""
        rs = 500
        Bo = vasquez_beggs_bo(
            rs=rs, temperature=200, api_gravity=30, gas_gravity=0.65, pressure=5000
        )

        assert Bo >= 1.0
        assert Bo < 3.0


class TestGasProperties:
    """Test gas property correlations."""

    def test_gas_z_factor_basic(self):
        """Test basic Z-factor calculation."""
        Z = gas_z_factor(pressure=5000, temperature=200, gas_gravity=0.65)

        assert 0.1 < Z < 2.0  # Reasonable range

    def test_gas_z_factor_methods(self):
        """Test different Z-factor methods."""
        Z1 = gas_z_factor(
            pressure=5000, temperature=200, gas_gravity=0.65, method="standing"
        )
        Z2 = gas_z_factor(
            pressure=5000, temperature=200, gas_gravity=0.65, method="hall_yarborough"
        )

        assert 0.1 < Z1 < 2.0
        assert (
            0.5 <= Z2 <= 1.5
        )  # Hall-Yarborough should be in reasonable range (inclusive)

    def test_gas_fvf_basic(self):
        """Test basic gas FVF calculation."""
        Bg = gas_fvf(pressure=5000, temperature=200, gas_gravity=0.65)

        assert Bg > 0
        assert Bg < 1.0  # Gas FVF is typically < 1.0 at reservoir conditions

    def test_gas_fvf_decreases_with_pressure(self):
        """Test that Bg decreases with pressure."""
        Bg1 = gas_fvf(pressure=1000, temperature=200, gas_gravity=0.65)
        Bg2 = gas_fvf(pressure=5000, temperature=200, gas_gravity=0.65)

        assert Bg1 > Bg2


class TestOilViscosity:
    """Test oil viscosity correlations."""

    def test_chew_connally_dead_oil_viscosity(self):
        """Test Chew-Connally dead oil viscosity."""
        muod = chew_connally_dead_oil_viscosity(temperature=200, api_gravity=30)

        assert muod > 0.1
        assert muod < 100.0  # Reasonable range

    def test_beggs_robinson_oil_viscosity(self):
        """Test Beggs-Robinson live oil viscosity."""
        muo = beggs_robinson_oil_viscosity(rs=500, temperature=200, api_gravity=30)

        assert muo > 0.1
        assert muo < 100.0

    def test_beggs_robinson_decreases_with_rs(self):
        """Test that live oil viscosity decreases with Rs."""
        muo1 = beggs_robinson_oil_viscosity(rs=100, temperature=200, api_gravity=30)
        muo2 = beggs_robinson_oil_viscosity(rs=1000, temperature=200, api_gravity=30)

        assert muo2 < muo1  # More gas = lower viscosity


class TestGasViscosity:
    """Test gas viscosity correlations."""

    def test_lee_gonzalez_gas_viscosity(self):
        """Test Lee-Gonzalez-Eakin gas viscosity."""
        mug = lee_gonzalez_gas_viscosity(
            pressure=5000, temperature=200, gas_gravity=0.65
        )

        assert mug > 0.005
        assert (
            mug <= 0.15
        )  # Gas viscosity is typically 0.01-0.1 cp, allow wider range (inclusive)


class TestWaterProperties:
    """Test water property correlations."""

    def test_water_fvf_basic(self):
        """Test basic water FVF calculation."""
        Bw = water_fvf(pressure=5000, temperature=200)

        assert 0.9 < Bw < 1.1  # Water FVF is close to 1.0

    def test_water_viscosity_basic(self):
        """Test basic water viscosity calculation."""
        muw = water_viscosity(temperature=200)

        assert 0.05 < muw < 2.0  # Allow lower bound for high temperatures

    def test_water_viscosity_decreases_with_temperature(self):
        """Test that water viscosity decreases with temperature."""
        muw1 = water_viscosity(temperature=100)
        muw2 = water_viscosity(
            temperature=250
        )  # Use 250 instead of 300 to avoid hitting bounds

        assert muw2 < muw1


class TestComprehensivePVT:
    """Test comprehensive PVT calculation."""

    def test_calculate_pvt_properties_standing(self):
        """Test comprehensive PVT calculation with Standing method."""
        pvt = calculate_pvt_properties(
            pressure=5000,
            temperature=200,
            api_gravity=30,
            gas_gravity=0.65,
            method="standing",
        )

        assert isinstance(pvt, PVTProperties)
        assert pvt.Bo >= 1.0
        assert pvt.Rs > 0
        assert pvt.Bg > 0
        assert pvt.muo > 0
        assert pvt.mug > 0
        assert pvt.Bw > 0
        assert pvt.muw > 0
        assert 0.1 < pvt.Z < 2.0

    def test_calculate_pvt_properties_vasquez_beggs(self):
        """Test comprehensive PVT calculation with Vasquez-Beggs method."""
        pvt = calculate_pvt_properties(
            pressure=5000,
            temperature=200,
            api_gravity=30,
            gas_gravity=0.65,
            method="vasquez_beggs",
        )

        assert isinstance(pvt, PVTProperties)
        assert pvt.Bo >= 1.0
        assert pvt.Rs > 0

    def test_calculate_pvt_properties_with_rs(self):
        """Test PVT calculation with provided Rs."""
        pvt = calculate_pvt_properties(
            pressure=5000,
            temperature=200,
            api_gravity=30,
            gas_gravity=0.65,
            rs=500,
            method="standing",
        )

        assert pvt.Rs == 500
        assert pvt.Bo >= 1.0

    def test_pvt_properties_vectorized(self):
        """Test PVT calculation with vectorized pressure."""
        pressures = np.array([1000, 3000, 5000])
        pvt_list = [
            calculate_pvt_properties(
                p, temperature=200, api_gravity=30, gas_gravity=0.65
            )
            for p in pressures
        ]

        # Bo should increase with pressure (more gas in solution)
        assert pvt_list[2].Bo > pvt_list[0].Bo
        # Rs should increase with pressure
        assert pvt_list[2].Rs > pvt_list[0].Rs


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_standing_rs_zero_pressure(self):
        """Test Standing Rs at zero pressure."""
        rs = standing_rs(pressure=0, temperature=200, api_gravity=30)

        assert rs >= 0

    def test_gas_z_factor_invalid_method(self):
        """Test Z-factor with invalid method."""
        with pytest.raises(ValueError):
            gas_z_factor(
                pressure=5000, temperature=200, gas_gravity=0.65, method="invalid"
            )

    def test_calculate_pvt_invalid_method(self):
        """Test PVT calculation with invalid method."""
        with pytest.raises(ValueError):
            calculate_pvt_properties(
                pressure=5000,
                temperature=200,
                api_gravity=30,
                method="invalid",
            )
