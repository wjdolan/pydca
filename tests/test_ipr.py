"""Tests for IPR (Inflow Performance Relationship) models."""

import pytest

from decline_curve.ipr import (
    calculate_productivity_index,
    cinco_ley_fractured_ipr,
    composite_ipr,
    fetkovich_ipr,
    generate_ipr_curve,
    joshi_horizontal_ipr,
    linear_ipr,
    vogel_ipr,
)


class TestLinearIPR:
    """Test linear IPR model."""

    def test_linear_ipr_basic(self):
        """Test basic linear IPR calculation."""
        rate = linear_ipr(
            reservoir_pressure=5000, flowing_pressure=3000, productivity_index=1.0
        )

        assert rate == 2000.0  # J * (p - pwf) = 1.0 * (5000 - 3000)

    def test_linear_ipr_zero_rate(self):
        """Test linear IPR when pwf >= p."""
        rate1 = linear_ipr(5000, 5000, 1.0)
        rate2 = linear_ipr(5000, 6000, 1.0)

        assert rate1 == 0.0
        assert rate2 == 0.0

    def test_linear_ipr_decreases_with_pwf(self):
        """Test that rate decreases as flowing pressure increases."""
        rate1 = linear_ipr(5000, 2000, 1.0)
        rate2 = linear_ipr(5000, 3000, 1.0)
        rate3 = linear_ipr(5000, 4000, 1.0)

        assert rate1 > rate2 > rate3


class TestVogelIPR:
    """Test Vogel IPR model."""

    def test_vogel_ipr_basic(self):
        """Test basic Vogel IPR calculation."""
        rate = vogel_ipr(
            reservoir_pressure=5000,
            flowing_pressure=2000,
            productivity_index=1.0,
        )

        assert rate > 0
        assert rate < 10000  # Reasonable range

    def test_vogel_ipr_with_max_rate(self):
        """Test Vogel IPR with provided max_rate."""
        rate = vogel_ipr(
            reservoir_pressure=5000,
            flowing_pressure=2000,
            max_rate=1000,
        )

        assert rate > 0
        assert rate <= 1000

    def test_vogel_ipr_undersaturated(self):
        """Test Vogel IPR for undersaturated oil."""
        rate = vogel_ipr(
            reservoir_pressure=5000,
            flowing_pressure=4000,
            productivity_index=1.0,
            bubble_point_pressure=3000,
        )

        # Should use linear IPR (both pressures above bubble point)
        expected = linear_ipr(5000, 4000, 1.0)
        assert abs(rate - expected) < 1.0

    def test_vogel_ipr_saturated(self):
        """Test Vogel IPR for saturated oil."""
        rate = vogel_ipr(
            reservoir_pressure=5000,
            flowing_pressure=2000,
            productivity_index=1.0,
            bubble_point_pressure=3000,
        )

        # Should have contribution from both linear and Vogel regions
        assert rate > 0

    def test_vogel_ipr_zero_at_reservoir_pressure(self):
        """Test that Vogel IPR gives zero rate at reservoir pressure."""
        rate = vogel_ipr(
            reservoir_pressure=5000,
            flowing_pressure=5000,
            productivity_index=1.0,
        )

        assert rate == 0.0


class TestFetkovichIPR:
    """Test Fetkovich IPR model."""

    def test_fetkovich_ipr_basic(self):
        """Test basic Fetkovich IPR calculation."""
        rate = fetkovich_ipr(
            reservoir_pressure=5000,
            flowing_pressure=3000,
            max_rate=1000,
            n=1.0,
        )

        assert rate > 0
        assert rate <= 1000

    def test_fetkovich_ipr_n_equals_one(self):
        """Test Fetkovich with n=1."""
        rate_fet = fetkovich_ipr(5000, 3000, 2000, n=1.0)

        # With n=1: q = q_max * (1 - pwf/p) = 2000 * (1 - 3000/5000) = 2000 * 0.4 = 800
        assert abs(rate_fet - 800.0) < 1.0

    def test_fetkovich_ipr_n_less_than_one(self):
        """Test Fetkovich with n < 1 (typical for solution gas drive)."""
        rate1 = fetkovich_ipr(5000, 3000, 1000, n=0.5)
        rate2 = fetkovich_ipr(5000, 3000, 1000, n=1.0)

        # With n=0.5, rate should be lower than n=1.0
        # n=1: q = 1000 * (1 - 0.6) = 400
        # n=0.5: q = 1000 * (1 - sqrt(0.6)) ≈ 1000 * 0.225 = 225
        assert rate1 < rate2
        assert rate1 > 0
        assert rate2 > 0

    def test_fetkovich_ipr_invalid_n(self):
        """Test Fetkovich with invalid exponent."""
        with pytest.raises(ValueError):
            fetkovich_ipr(5000, 3000, 1000, n=0)


class TestCompositeIPR:
    """Test composite IPR for layered reservoirs."""

    def test_composite_ipr_basic(self):
        """Test basic composite IPR calculation."""
        rate = composite_ipr(
            reservoir_pressure=5000,
            flowing_pressure=3000,
            layer_pressures=[5000, 4500, 4000],
            layer_productivity_indices=[0.5, 0.3, 0.2],
        )

        assert rate > 0
        # Should be sum of individual layer rates
        rate1 = linear_ipr(5000, 3000, 0.5)
        rate2 = linear_ipr(4500, 3000, 0.3)
        rate3 = linear_ipr(4000, 3000, 0.2)
        expected = rate1 + rate2 + rate3
        assert abs(rate - expected) < 1.0

    def test_composite_ipr_mismatched_lengths(self):
        """Test composite IPR with mismatched layer arrays."""
        with pytest.raises(ValueError):
            composite_ipr(
                5000,
                3000,
                layer_pressures=[5000, 4500],
                layer_productivity_indices=[0.5, 0.3, 0.2],
            )


class TestHorizontalWellIPR:
    """Test Joshi horizontal well IPR."""

    def test_joshi_horizontal_ipr_basic(self):
        """Test basic Joshi horizontal IPR calculation."""
        rate = joshi_horizontal_ipr(
            reservoir_pressure=5000,
            flowing_pressure=3000,
            horizontal_length=2000,
            reservoir_thickness=50,
            permeability=10,
            oil_viscosity=1.0,
            formation_volume_factor=1.2,
        )

        assert rate > 0

    def test_joshi_horizontal_ipr_zero_at_reservoir_pressure(self):
        """Test Joshi IPR gives zero when pwf >= p."""
        rate = joshi_horizontal_ipr(
            reservoir_pressure=5000,
            flowing_pressure=5000,
            horizontal_length=2000,
            reservoir_thickness=50,
            permeability=10,
            oil_viscosity=1.0,
            formation_volume_factor=1.2,
        )

        assert rate == 0.0


class TestFracturedWellIPR:
    """Test Cinco-Ley fractured well IPR."""

    def test_cinco_ley_fractured_ipr_basic(self):
        """Test basic Cinco-Ley fractured IPR calculation."""
        rate = cinco_ley_fractured_ipr(
            reservoir_pressure=5000,
            flowing_pressure=3000,
            fracture_half_length=500,
            reservoir_thickness=50,
            permeability=1.0,
            oil_viscosity=1.0,
            formation_volume_factor=1.2,
        )

        assert rate > 0

    def test_cinco_ley_fractured_ipr_high_conductivity(self):
        """Test Cinco-Ley with high fracture conductivity."""
        rate = cinco_ley_fractured_ipr(
            reservoir_pressure=5000,
            flowing_pressure=3000,
            fracture_half_length=500,
            reservoir_thickness=50,
            permeability=1.0,
            oil_viscosity=1.0,
            formation_volume_factor=1.2,
            fracture_conductivity=10000,  # High conductivity
        )

        assert rate > 0


class TestProductivityIndex:
    """Test productivity index calculation from well test data."""

    def test_calculate_productivity_index_linear(self):
        """Test PI calculation for linear IPR."""
        J = calculate_productivity_index(
            test_rate=2000,
            reservoir_pressure=5000,
            flowing_pressure=3000,
            ipr_type="linear",
        )

        assert abs(J - 1.0) < 0.01  # Should be 2000 / (5000 - 3000) = 1.0

    def test_calculate_productivity_index_vogel(self):
        """Test PI calculation for Vogel IPR."""
        J = calculate_productivity_index(
            test_rate=500,
            reservoir_pressure=5000,
            flowing_pressure=2000,
            ipr_type="vogel",
        )

        assert J > 0

    def test_calculate_productivity_index_fetkovich(self):
        """Test PI calculation for Fetkovich IPR."""
        J = calculate_productivity_index(
            test_rate=500,
            reservoir_pressure=5000,
            flowing_pressure=3000,
            ipr_type="fetkovich",
            n=0.5,
        )

        assert J > 0

    def test_calculate_productivity_index_invalid(self):
        """Test PI calculation with invalid parameters."""
        with pytest.raises(ValueError):
            calculate_productivity_index(
                test_rate=0,
                reservoir_pressure=5000,
                flowing_pressure=3000,
            )

        with pytest.raises(ValueError):
            calculate_productivity_index(
                test_rate=500,
                reservoir_pressure=5000,
                flowing_pressure=6000,  # pwf > p
            )


class TestIPRCurveGeneration:
    """Test IPR curve generation."""

    def test_generate_ipr_curve_linear(self):
        """Test generating linear IPR curve."""
        pwf_array, rate_array = generate_ipr_curve(
            reservoir_pressure=5000,
            productivity_index=1.0,
            ipr_type="linear",
        )

        assert len(pwf_array) == 50
        assert len(rate_array) == 50
        assert rate_array[0] > 0  # Rate at pwf=0
        assert rate_array[-1] == 0.0  # Rate at pwf=p

    def test_generate_ipr_curve_vogel(self):
        """Test generating Vogel IPR curve."""
        pwf_array, rate_array = generate_ipr_curve(
            reservoir_pressure=5000,
            productivity_index=1.0,
            ipr_type="vogel",
        )

        assert len(pwf_array) == 50
        assert len(rate_array) == 50
        assert all(rate_array >= 0)

    def test_generate_ipr_curve_fetkovich(self):
        """Test generating Fetkovich IPR curve."""
        pwf_array, rate_array = generate_ipr_curve(
            reservoir_pressure=5000,
            max_rate=1000,
            ipr_type="fetkovich",
            n=0.5,
        )

        assert len(pwf_array) == 50
        assert len(rate_array) == 50
        assert all(rate_array >= 0)
        assert rate_array[0] <= 1000  # Max rate

    def test_generate_ipr_curve_invalid_type(self):
        """Test generating IPR curve with invalid type."""
        with pytest.raises(ValueError):
            generate_ipr_curve(
                reservoir_pressure=5000,
                ipr_type="invalid",
            )


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_vogel_ipr_missing_parameters(self):
        """Test Vogel IPR with missing required parameters."""
        with pytest.raises(ValueError):
            vogel_ipr(
                reservoir_pressure=5000,
                flowing_pressure=3000,
                # Neither max_rate nor productivity_index provided
            )

    def test_fetkovich_ipr_edge_cases(self):
        """Test Fetkovich IPR edge cases."""
        # At reservoir pressure
        rate1 = fetkovich_ipr(5000, 5000, 1000, n=0.5)
        assert rate1 == 0.0

        # At zero pressure
        rate2 = fetkovich_ipr(5000, 0, 1000, n=0.5)
        assert rate2 == 1000.0
