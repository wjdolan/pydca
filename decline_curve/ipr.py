"""Inflow Performance Relationship (IPR) models for well deliverability.

This module provides industry-standard IPR models for calculating well production
rates as a function of bottomhole flowing pressure.

IPR Models Implemented:
- Linear IPR: q = J * (p - pwf) - for single-phase flow
- Vogel IPR: Solution gas drive reservoirs (two-phase flow)
- Fetkovich IPR: Two-phase flow with decline
- Composite IPR: Layered reservoirs
- Horizontal Well IPR: Joshi model
- Fractured Well IPR: Cinco-Ley model

References:
- Vogel, J.V., "Inflow Performance Relationships for Solution-Gas Drive Wells,"
  JPT, January 1968.
- Fetkovich, M.J., "The Isochronal Testing of Oil Wells," SPE 4529, 1973.
- Joshi, S.D., "Augmentation of Well Productivity with Slant and Horizontal Wells,"
  JPT, June 1988.
- Cinco-Ley, H. and Samaniego-V., F., "Transient Pressure Analysis for Fractured
  Wells," JPT, September 1981.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class IPRResult:
    """Container for IPR calculation results.

    Attributes:
        rate: Production rate (STB/day or MCF/day)
        pwf: Bottomhole flowing pressure (psi)
        productivity_index: Productivity index (STB/day/psi or MCF/day/psi)
        max_rate: Maximum rate at zero flowing pressure (STB/day or MCF/day)
    """

    rate: float
    pwf: float
    productivity_index: float
    max_rate: float


def linear_ipr(
    reservoir_pressure: float,
    flowing_pressure: float,
    productivity_index: float,
) -> float:
    """Calculate production rate using linear IPR.

    Linear IPR is valid for single-phase flow (oil above bubble point or gas).

    Formula:
        q = J * (p - pwf)

    Args:
        reservoir_pressure: Average reservoir pressure (psi)
        flowing_pressure: Bottomhole flowing pressure (psi)
        productivity_index: Productivity index (STB/day/psi or MCF/day/psi)

    Returns:
        Production rate (STB/day or MCF/day)

    Example:
        >>> rate = linear_ipr(reservoir_pressure=5000, flowing_pressure=3000, productivity_index=1.0)
        >>> print(f"Rate: {rate:.1f} STB/day")
    """
    if flowing_pressure >= reservoir_pressure:
        return 0.0

    rate = productivity_index * (reservoir_pressure - flowing_pressure)
    return max(0.0, rate)


def vogel_ipr(
    reservoir_pressure: float,
    flowing_pressure: float,
    max_rate: Optional[float] = None,
    productivity_index: Optional[float] = None,
    bubble_point_pressure: Optional[float] = None,
) -> float:
    """Calculate production rate using Vogel IPR.

    Vogel IPR is used for solution gas drive reservoirs (two-phase flow).
    Valid when reservoir pressure is above bubble point but flowing pressure
    may be below bubble point.

    Formula (Vogel, 1968):
        q = q_max * [1 - 0.2 * (pwf/p) - 0.8 * (pwf/p)^2]

    For undersaturated oil (p > pb, pwf > pb):
        q = J * (p - pwf)

    For saturated oil (p > pb, pwf < pb):
        q = qb + q_max_vogel * [1 - 0.2 * (pwf/pb) - 0.8 * (pwf/pb)^2]

    Args:
        reservoir_pressure: Average reservoir pressure (psi)
        flowing_pressure: Bottomhole flowing pressure (psi)
        max_rate: Maximum rate at zero flowing pressure (STB/day).
            If None, calculated from productivity_index.
        productivity_index: Productivity index for single-phase region (STB/day/psi).
            Required if max_rate is None.
        bubble_point_pressure: Bubble point pressure (psi). If None, assumes
            reservoir is at or below bubble point.

    Returns:
        Production rate (STB/day)

    Reference:
        Vogel, J.V., "Inflow Performance Relationships for Solution-Gas Drive Wells,"
        JPT, January 1968.

    Example:
        >>> rate = vogel_ipr(
        ...     reservoir_pressure=5000,
        ...     flowing_pressure=2000,
        ...     productivity_index=1.0,
        ...     bubble_point_pressure=3000
        ... )
    """
    if flowing_pressure >= reservoir_pressure:
        return 0.0

    # If no bubble point specified, assume fully saturated
    if bubble_point_pressure is None:
        bubble_point_pressure = reservoir_pressure

    # Calculate max rate if not provided
    if max_rate is None:
        if productivity_index is None:
            raise ValueError("Either max_rate or productivity_index must be provided")
        # For Vogel, max_rate occurs at pwf=0
        # Approximate: q_max ≈ J * p / 1.8 (Vogel relationship)
        max_rate = productivity_index * reservoir_pressure / 1.8

    # Case 1: Undersaturated oil (both pressures above bubble point)
    if (
        reservoir_pressure > bubble_point_pressure
        and flowing_pressure >= bubble_point_pressure
    ):
        if productivity_index is None:
            # Estimate J from max_rate
            productivity_index = max_rate * 1.8 / reservoir_pressure
        return productivity_index * (reservoir_pressure - flowing_pressure)

    # Case 2: Saturated oil (flowing pressure below bubble point)
    if flowing_pressure < bubble_point_pressure:
        # Rate at bubble point
        if productivity_index is None:
            productivity_index = max_rate * 1.8 / reservoir_pressure

        if reservoir_pressure > bubble_point_pressure:
            qb = productivity_index * (reservoir_pressure - bubble_point_pressure)
        else:
            qb = 0.0

        # Vogel equation for two-phase region
        pwf_norm = flowing_pressure / bubble_point_pressure
        q_vogel = max_rate * (1 - 0.2 * pwf_norm - 0.8 * pwf_norm**2)

        return qb + q_vogel

    # Case 3: Fully saturated (both below or at bubble point)
    # Standard Vogel equation
    pwf_norm = flowing_pressure / reservoir_pressure
    rate = max_rate * (1 - 0.2 * pwf_norm - 0.8 * pwf_norm**2)

    return max(0.0, rate)


def fetkovich_ipr(
    reservoir_pressure: float,
    flowing_pressure: float,
    max_rate: float,
    n: float = 1.0,
) -> float:
    """Calculate production rate using Fetkovich IPR.

    Fetkovich IPR is a generalized form that can represent both single-phase
    and two-phase flow. It's particularly useful for wells with decline.

    Formula:
        q = q_max * [1 - (pwf/p)^n]

    Where:
        n = 1.0: Linear IPR (single-phase)
        n = 0.5: Typical for solution gas drive
        n varies: Depends on flow regime

    Args:
        reservoir_pressure: Average reservoir pressure (psi)
        flowing_pressure: Bottomhole flowing pressure (psi)
        max_rate: Maximum rate at zero flowing pressure (STB/day)
        n: Fetkovich exponent (typically 0.5 to 1.0), default 1.0

    Returns:
        Production rate (STB/day)

    Reference:
        Fetkovich, M.J., "The Isochronal Testing of Oil Wells," SPE 4529, 1973.

    Example:
        >>> rate = fetkovich_ipr(
        ...     reservoir_pressure=5000,
        ...     flowing_pressure=3000,
        ...     max_rate=1000,
        ...     n=0.5
        ... )
    """
    if flowing_pressure >= reservoir_pressure:
        return 0.0

    if n <= 0:
        raise ValueError("Fetkovich exponent n must be > 0")

    pwf_norm = flowing_pressure / reservoir_pressure
    rate = max_rate * (1 - pwf_norm**n)

    return max(0.0, rate)


def composite_ipr(
    reservoir_pressure: float,
    flowing_pressure: float,
    layer_pressures: list[float],
    layer_productivity_indices: list[float],
    layer_bubble_points: Optional[list[Optional[float]]] = None,
) -> float:
    """Calculate production rate using composite IPR for layered reservoirs.

    Composite IPR sums the contribution from multiple layers, each with its
    own pressure and productivity index.

    Formula:
        q_total = Σ q_i
        where q_i is calculated using appropriate IPR for each layer

    Args:
        reservoir_pressure: Average reservoir pressure (psi)
        flowing_pressure: Bottomhole flowing pressure (psi)
        layer_pressures: List of layer pressures (psi)
        layer_productivity_indices: List of productivity indices (STB/day/psi)
        layer_bubble_points: Optional list of bubble point pressures (psi).
            If None, uses linear IPR for all layers.

    Returns:
        Total production rate (STB/day)

    Example:
        >>> rate = composite_ipr(
        ...     reservoir_pressure=5000,
        ...     flowing_pressure=3000,
        ...     layer_pressures=[5000, 4500, 4000],
        ...     layer_productivity_indices=[0.5, 0.3, 0.2]
        ... )
    """
    if len(layer_pressures) != len(layer_productivity_indices):
        raise ValueError(
            "layer_pressures and layer_productivity_indices must have same length"
        )

    total_rate = 0.0

    for i, (p_layer, J_layer) in enumerate(
        zip(layer_pressures, layer_productivity_indices)
    ):
        if layer_bubble_points is not None:
            pb_layer = layer_bubble_points[i]
            if pb_layer is not None and flowing_pressure < pb_layer:
                # Use Vogel for this layer
                q_max_layer = J_layer * p_layer / 1.8
                rate_layer = vogel_ipr(
                    p_layer,
                    flowing_pressure,
                    max_rate=q_max_layer,
                    bubble_point_pressure=pb_layer,
                )
            else:
                # Use linear IPR
                rate_layer = linear_ipr(p_layer, flowing_pressure, J_layer)
        else:
            # Use linear IPR
            rate_layer = linear_ipr(p_layer, flowing_pressure, J_layer)

        total_rate += rate_layer

    return max(0.0, total_rate)


def joshi_horizontal_ipr(
    reservoir_pressure: float,
    flowing_pressure: float,
    horizontal_length: float,
    reservoir_thickness: float,
    permeability: float,
    oil_viscosity: float,
    formation_volume_factor: float,
    wellbore_radius: float = 0.25,
    anisotropy: float = 1.0,
    skin: float = 0.0,
) -> float:
    """Calculate production rate for horizontal well using Joshi model.

    Joshi IPR accounts for horizontal well geometry and anisotropy.

    Formula:
        J = (2π * k * h) / (μ * Bo) * (1 / (ln(re/rw) + S))
        where re accounts for horizontal well geometry

    Args:
        reservoir_pressure: Average reservoir pressure (psi)
        flowing_pressure: Bottomhole flowing pressure (psi)
        horizontal_length: Horizontal well length (ft)
        reservoir_thickness: Reservoir thickness (ft)
        permeability: Horizontal permeability (md)
        oil_viscosity: Oil viscosity (cp)
        formation_volume_factor: Oil FVF (RB/STB)
        wellbore_radius: Wellbore radius (ft), default 0.25
        anisotropy: Permeability anisotropy (kv/kh), default 1.0
        skin: Skin factor (dimensionless), default 0.0

    Returns:
        Production rate (STB/day)

    Reference:
        Joshi, S.D., "Augmentation of Well Productivity with Slant and Horizontal Wells,"
        JPT, June 1988.

    Example:
        >>> rate = joshi_horizontal_ipr(
        ...     reservoir_pressure=5000,
        ...     flowing_pressure=3000,
        ...     horizontal_length=2000,
        ...     reservoir_thickness=50,
        ...     permeability=10,
        ...     oil_viscosity=1.0,
        ...     formation_volume_factor=1.2
        ... )
    """
    if flowing_pressure >= reservoir_pressure:
        return 0.0

    # Convert permeability from md to darcy
    k_darcy = permeability / 1000.0

    # Effective wellbore radius for horizontal well
    beta = np.sqrt(anisotropy)
    _ = horizontal_length / 2.0  # noqa: F841

    # Drainage radius (assume square drainage area)
    # For horizontal well: re ≈ sqrt(A/π) where A = L * 2 * re_vertical
    re_vertical = 745.0  # Typical vertical drainage radius (ft)
    re_horizontal = np.sqrt(horizontal_length * 2 * re_vertical / np.pi)

    # Joshi geometric factor
    L_half = horizontal_length / 2.0
    h_half = reservoir_thickness / 2.0

    # Calculate equivalent radius
    if L_half > beta * h_half:
        # Long horizontal well
        r_equiv = (
            L_half
            * (1 + np.sqrt(1 - (beta * h_half / L_half) ** 2))
            / (beta * h_half / L_half)
        )
    else:
        # Short horizontal well
        r_equiv = beta * h_half + L_half

    # Productivity index
    # J = (2π * k * h) / (μ * Bo) * (1 / (ln(re/rw) + S))
    # Convert to field units: 0.00708 for oil
    J = (
        0.00708
        * k_darcy
        * reservoir_thickness
        / (oil_viscosity * formation_volume_factor)
        * (1.0 / (np.log(re_horizontal / wellbore_radius) + skin))
    )

    # Apply geometric correction for horizontal well
    geometric_factor = 1.0 + (beta * h_half / L_half) ** 2
    J *= geometric_factor

    # Calculate rate
    rate = J * (reservoir_pressure - flowing_pressure)

    return max(0.0, rate)


def cinco_ley_fractured_ipr(
    reservoir_pressure: float,
    flowing_pressure: float,
    fracture_half_length: float,
    reservoir_thickness: float,
    permeability: float,
    oil_viscosity: float,
    formation_volume_factor: float,
    fracture_conductivity: float = 100.0,
    wellbore_radius: float = 0.25,
    skin: float = 0.0,
) -> float:
    """Calculate production rate for fractured well using Cinco-Ley model.

    Cinco-Ley model accounts for finite conductivity fractures.

    Args:
        reservoir_pressure: Average reservoir pressure (psi)
        flowing_pressure: Bottomhole flowing pressure (psi)
        fracture_half_length: Fracture half-length (ft)
        reservoir_thickness: Reservoir thickness (ft)
        permeability: Formation permeability (md)
        oil_viscosity: Oil viscosity (cp)
        formation_volume_factor: Oil FVF (RB/STB)
        fracture_conductivity: Fracture conductivity (md-ft), default 100
        wellbore_radius: Wellbore radius (ft), default 0.25
        skin: Skin factor (dimensionless), default 0.0

    Returns:
        Production rate (STB/day)

    Reference:
        Cinco-Ley, H. and Samaniego-V., F., "Transient Pressure Analysis for Fractured
        Wells," JPT, September 1981.

    Example:
        >>> rate = cinco_ley_fractured_ipr(
        ...     reservoir_pressure=5000,
        ...     flowing_pressure=3000,
        ...     fracture_half_length=500,
        ...     reservoir_thickness=50,
        ...     permeability=1.0,
        ...     oil_viscosity=1.0,
        ...     formation_volume_factor=1.2
        ... )
    """
    if flowing_pressure >= reservoir_pressure:
        return 0.0

    # Convert permeability from md to darcy
    k_darcy = permeability / 1000.0

    # Dimensionless fracture conductivity
    kf_w = fracture_conductivity  # md-ft
    xf = fracture_half_length  # ft
    CfD = kf_w / (k_darcy * 1000 * xf)  # Dimensionless

    # Effective wellbore radius for fractured well
    if CfD > 100:
        # Infinite conductivity fracture
        rw_eff = xf / 2.0
    elif CfD > 0.1:
        # Finite conductivity fracture (Cinco-Ley correlation)
        rw_eff = 0.5 * xf * np.exp(-0.72 * (CfD ** (-0.55)))
    else:
        # Low conductivity
        rw_eff = wellbore_radius

    # Drainage radius (assume circular)
    re = 745.0  # Typical drainage radius (ft)

    # Productivity index
    # J = (2π * k * h) / (μ * Bo) * (1 / (ln(re/rw_eff) + S))
    J = (
        0.00708
        * k_darcy
        * reservoir_thickness
        / (oil_viscosity * formation_volume_factor)
        * (1.0 / (np.log(re / rw_eff) + skin))
    )

    # Calculate rate
    rate = J * (reservoir_pressure - flowing_pressure)

    return max(0.0, rate)


def calculate_productivity_index(
    test_rate: float,
    reservoir_pressure: float,
    flowing_pressure: float,
    ipr_type: str = "linear",
    **kwargs,
) -> float:
    """Calculate productivity index from well test data.

    Args:
        test_rate: Production rate from well test (STB/day)
        reservoir_pressure: Reservoir pressure during test (psi)
        flowing_pressure: Flowing pressure during test (psi)
        ipr_type: IPR type ('linear', 'vogel', 'fetkovich'), default 'linear'
        **kwargs: Additional parameters for specific IPR types

    Returns:
        Productivity index (STB/day/psi)

    Example:
        >>> J = calculate_productivity_index(
        ...     test_rate=500,
        ...     reservoir_pressure=5000,
        ...     flowing_pressure=3000,
        ...     ipr_type="linear"
        ... )
    """
    if test_rate <= 0:
        raise ValueError("test_rate must be > 0")

    if flowing_pressure >= reservoir_pressure:
        raise ValueError("flowing_pressure must be < reservoir_pressure")

    if ipr_type == "linear":
        J = test_rate / (reservoir_pressure - flowing_pressure)
        return J

    elif ipr_type == "vogel":
        # For Vogel, need to solve for J or q_max
        # Approximate: J ≈ q_test * 1.8 / (p - pwf * (1 - 0.2*pwf/p - 0.8*(pwf/p)^2))
        pwf_norm = flowing_pressure / reservoir_pressure
        vogel_factor = 1 - 0.2 * pwf_norm - 0.8 * pwf_norm**2
        q_max = test_rate / vogel_factor
        J = q_max * 1.8 / reservoir_pressure
        return J

    elif ipr_type == "fetkovich":
        n = kwargs.get("n", 1.0)
        pwf_norm = flowing_pressure / reservoir_pressure
        fetkovich_factor = 1 - pwf_norm**n
        q_max = test_rate / fetkovich_factor
        # Approximate J from q_max
        J = q_max / reservoir_pressure  # Simplified
        return J

    else:
        raise ValueError(f"Unknown IPR type: {ipr_type}")


def generate_ipr_curve(
    reservoir_pressure: float,
    productivity_index: Optional[float] = None,
    max_rate: Optional[float] = None,
    bubble_point_pressure: Optional[float] = None,
    ipr_type: str = "vogel",
    n: float = 1.0,
    num_points: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate IPR curve (rate vs. flowing pressure).

    Args:
        reservoir_pressure: Average reservoir pressure (psi)
        productivity_index: Productivity index (STB/day/psi)
        max_rate: Maximum rate at zero flowing pressure (STB/day)
        bubble_point_pressure: Bubble point pressure (psi)
        ipr_type: IPR type ('linear', 'vogel', 'fetkovich'), default 'vogel'
        n: Fetkovich exponent (for fetkovich IPR), default 1.0
        num_points: Number of points in curve, default 50

    Returns:
        Tuple of (flowing_pressures, rates) arrays

    Example:
        >>> pwf_array, rate_array = generate_ipr_curve(
        ...     reservoir_pressure=5000,
        ...     productivity_index=1.0,
        ...     ipr_type="vogel"
        ... )
    """
    pwf_array = np.linspace(0, reservoir_pressure, num_points)
    rate_array = np.zeros_like(pwf_array)

    for i, pwf in enumerate(pwf_array):
        if ipr_type == "linear":
            if productivity_index is None:
                raise ValueError("productivity_index required for linear IPR")
            rate_array[i] = linear_ipr(reservoir_pressure, pwf, productivity_index)

        elif ipr_type == "vogel":
            rate_array[i] = vogel_ipr(
                reservoir_pressure,
                pwf,
                max_rate=max_rate,
                productivity_index=productivity_index,
                bubble_point_pressure=bubble_point_pressure,
            )

        elif ipr_type == "fetkovich":
            if max_rate is None:
                raise ValueError("max_rate required for fetkovich IPR")
            rate_array[i] = fetkovich_ipr(reservoir_pressure, pwf, max_rate, n=n)

        else:
            raise ValueError(f"Unknown IPR type: {ipr_type}")

    return pwf_array, rate_array
