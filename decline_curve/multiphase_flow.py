"""Multi-phase flow correlations for pressure drop and liquid holdup.

This module provides industry-standard multi-phase flow correlations:
- Beggs-Brill correlation
- Hagedorn-Brown correlation
- Flow pattern identification
- Liquid holdup calculations
- Pressure drop in pipes

References:
- Beggs, H.D. and Brill, J.P., "A Study of Two-Phase Flow in Inclined Pipes,"
  JPT, May 1973.
- Hagedorn, A.R. and Brown, K.E., "Experimental Study of Pressure Gradients
  Occurring During Continuous Two-Phase Flow in Small-Diameter Vertical Conduits,"
  JPT, April 1965.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class FlowPattern:
    """Flow pattern identification result.

    Attributes:
        pattern: Flow pattern ('bubble', 'slug', 'churn', 'annular', 'mist')
        liquid_holdup: Liquid holdup (fraction)
        pressure_drop: Pressure drop (psi/ft)
    """

    pattern: str
    liquid_holdup: float
    pressure_drop: float


def identify_flow_pattern(
    liquid_rate: float,
    gas_rate: float,
    pipe_diameter: float,
    liquid_density: float,
    gas_density: float,
    liquid_viscosity: float,
    gas_viscosity: float,
    surface_tension: float = 30.0,
    pipe_angle: float = 0.0,
) -> FlowPattern:
    """Identify flow pattern in two-phase flow.

    Uses simplified flow pattern map based on superficial velocities.

    Args:
        liquid_rate: Liquid flow rate (STB/day)
        gas_rate: Gas flow rate (MCF/day)
        pipe_diameter: Pipe diameter (inches)
        liquid_density: Liquid density (lb/ft³)
        gas_density: Gas density (lb/ft³)
        liquid_viscosity: Liquid viscosity (cp)
        gas_viscosity: Gas viscosity (cp)
        surface_tension: Surface tension (dyne/cm)
        pipe_angle: Pipe angle from horizontal (degrees)

    Returns:
        FlowPattern with identified pattern and properties

    Reference:
        Beggs, H.D. and Brill, J.P., "A Study of Two-Phase Flow in Inclined Pipes,"
        JPT, May 1973.
    """
    # Convert to ft³/s
    pipe_area_ft2 = np.pi * (pipe_diameter / 12.0) ** 2 / 4.0

    # Superficial velocities
    vsl = (
        (liquid_rate / 86400.0) * 5.615 / pipe_area_ft2
    )  # Liquid superficial velocity (ft/s)
    vsg = (gas_rate * 1000 / 86400.0) / (
        gas_density * pipe_area_ft2
    )  # Gas superficial velocity (ft/s)

    # Mixture velocity
    vm = vsl + vsg

    # Flow pattern identification (simplified)
    if vm < 0.1:
        pattern = "bubble"
    elif vsg / vm < 0.1:
        pattern = "slug"
    elif vsg / vm < 0.5:
        pattern = "churn"
    elif vsg / vm < 0.9:
        pattern = "annular"
    else:
        pattern = "mist"

    # Calculate liquid holdup (simplified)
    if pattern == "bubble":
        holdup = 0.9
    elif pattern == "slug":
        holdup = 0.7
    elif pattern == "churn":
        holdup = 0.5
    elif pattern == "annular":
        holdup = 0.2
    else:
        holdup = 0.1

    # Estimate pressure drop (simplified)
    mixture_density = holdup * liquid_density + (1 - holdup) * gas_density
    friction_factor = 0.02  # Simplified
    pressure_drop = (
        friction_factor
        * mixture_density
        * vm**2
        / (2 * 32.2 * (pipe_diameter / 12.0) * 144.0)
    )  # psi/ft

    return FlowPattern(
        pattern=pattern,
        liquid_holdup=holdup,
        pressure_drop=pressure_drop,
    )


def beggs_brill_correlation(
    liquid_rate: float,
    gas_rate: float,
    pipe_diameter: float,
    pipe_length: float,
    liquid_density: float,
    gas_density: float,
    liquid_viscosity: float,
    gas_viscosity: float,
    pipe_angle: float = 0.0,
    surface_tension: float = 30.0,
) -> Tuple[float, float]:
    """Calculate pressure drop and liquid holdup using Beggs-Brill correlation.

    Args:
        liquid_rate: Liquid flow rate (STB/day)
        gas_rate: Gas flow rate (MCF/day)
        pipe_diameter: Pipe diameter (inches)
        pipe_length: Pipe length (ft)
        liquid_density: Liquid density (lb/ft³)
        gas_density: Gas density (lb/ft³)
        liquid_viscosity: Liquid viscosity (cp)
        gas_viscosity: Gas viscosity (cp)
        pipe_angle: Pipe angle from horizontal (degrees)
        surface_tension: Surface tension (dyne/cm)

    Returns:
        Tuple of (pressure_drop_psi, liquid_holdup)

    Reference:
        Beggs, H.D. and Brill, J.P., "A Study of Two-Phase Flow in Inclined Pipes,"
        JPT, May 1973.
    """
    # Convert to ft³/s
    pipe_area_ft2 = np.pi * (pipe_diameter / 12.0) ** 2 / 4.0

    # Superficial velocities
    vsl = (liquid_rate / 86400.0) * 5.615 / pipe_area_ft2
    vsg = (gas_rate * 1000 / 86400.0) / (gas_density * pipe_area_ft2)
    vm = vsl + vsg

    # Input liquid content
    lambda_l = vsl / vm if vm > 0 else 0.0

    # Froude number
    fr = vm**2 / (32.2 * pipe_diameter / 12.0)

    # Dimensionless numbers
    n_lv = 1.938 * vsl * (liquid_density / surface_tension) ** 0.25
    n_gv = 1.938 * vsg * (liquid_density / surface_tension) ** 0.25

    # Flow pattern (simplified Beggs-Brill)
    angle_rad = np.radians(pipe_angle)

    if lambda_l < 0.01 and fr < 0.01:
        _pattern = "segregated"
        c = 0.0
        _d = 0.0
    elif lambda_l >= 0.01 and fr < 0.01:
        _pattern = "transition"
        c = 0.0
        _d = 0.0
    else:
        _pattern = "distributed"
        c = 0.0
        _d = 0.0

    # Liquid holdup (simplified Beggs-Brill)
    # H0 = a * lambda_l^b / Fr^c
    a = 0.98
    b = 0.4846
    c = 0.0868

    h0 = a * lambda_l**b / (fr**c) if fr > 0 else lambda_l

    # Inclination correction
    psi = 1.0 + c * (
        np.sin(1.8 * angle_rad) - (1.0 / 3.0) * np.sin(1.8 * angle_rad) ** 3
    )

    holdup = h0 * psi
    holdup = np.clip(holdup, 0.0, 1.0)

    # Mixture properties
    mixture_density = holdup * liquid_density + (1 - holdup) * gas_density
    mixture_viscosity = holdup * liquid_viscosity + (1 - holdup) * gas_viscosity

    # Reynolds number
    reynolds = (
        mixture_density * vm * (pipe_diameter / 12.0) / (mixture_viscosity * 0.000672)
    )

    # Friction factor (Moody)
    if reynolds > 0:
        if reynolds < 2100:
            friction_factor = 64.0 / reynolds
        else:
            friction_factor = 0.316 / (reynolds**0.25) if reynolds < 100000 else 0.014
    else:
        friction_factor = 0.02

    # Pressure drop components
    # Friction
    friction_drop = (
        friction_factor
        * mixture_density
        * vm**2
        * pipe_length
        / (2 * 32.2 * (pipe_diameter / 12.0) * 144.0)
    )

    # Hydrostatic
    hydrostatic_drop = mixture_density * pipe_length * np.sin(angle_rad) / 144.0

    # Total pressure drop
    total_drop = friction_drop + hydrostatic_drop

    return max(0.0, total_drop), holdup


def hagedorn_brown_correlation(
    liquid_rate: float,
    gas_rate: float,
    pipe_diameter: float,
    pipe_length: float,
    liquid_density: float,
    gas_density: float,
    liquid_viscosity: float,
    gas_viscosity: float,
    surface_tension: float = 30.0,
) -> Tuple[float, float]:
    """Calculate pressure drop and liquid holdup using Hagedorn-Brown correlation.

    Primarily for vertical flow.

    Args:
        liquid_rate: Liquid flow rate (STB/day)
        gas_rate: Gas flow rate (MCF/day)
        pipe_diameter: Pipe diameter (inches)
        pipe_length: Pipe length (ft)
        liquid_density: Liquid density (lb/ft³)
        gas_density: Gas density (lb/ft³)
        liquid_viscosity: Liquid viscosity (cp)
        gas_viscosity: Gas viscosity (cp)
        surface_tension: Surface tension (dyne/cm)

    Returns:
        Tuple of (pressure_drop_psi, liquid_holdup)

    Reference:
        Hagedorn, A.R. and Brown, K.E., "Experimental Study of Pressure Gradients
        Occurring During Continuous Two-Phase Flow in Small-Diameter Vertical Conduits,"
        JPT, April 1965.
    """
    # Convert to ft³/s
    pipe_area_ft2 = np.pi * (pipe_diameter / 12.0) ** 2 / 4.0

    # Superficial velocities
    vsl = (liquid_rate / 86400.0) * 5.615 / pipe_area_ft2
    vsg = (gas_rate * 1000 / 86400.0) / (gas_density * pipe_area_ft2)
    vm = vsl + vsg

    # Dimensionless numbers
    n_lv = 1.938 * vsl * (liquid_density / surface_tension) ** 0.25
    n_gv = 1.938 * vsg * (liquid_density / surface_tension) ** 0.25
    _n_d = 120.872 * (pipe_diameter / 12.0) * (liquid_density / surface_tension) ** 0.5

    # Liquid viscosity number
    n_l = (
        0.15726
        * liquid_viscosity
        * (1.0 / (liquid_density * surface_tension**3)) ** 0.25
    )

    # Hagedorn-Brown correlation (simplified)
    # Liquid holdup
    cn_l = 10 ** (
        -2.69851
        + 0.15841 * np.log10(n_lv)
        - 0.55187 * (np.log10(n_lv)) ** 2
        + 0.54785 * (np.log10(n_lv)) ** 3
        - 0.12195 * (np.log10(n_lv)) ** 4
    )

    holdup = cn_l * (n_lv / n_gv) ** 0.575 * (liquid_density / gas_density) ** 0.05
    holdup = np.clip(holdup, 0.0, 1.0)

    # Mixture properties
    mixture_density = holdup * liquid_density + (1 - holdup) * gas_density

    # Pressure drop (simplified)
    reynolds = (
        mixture_density * vm * (pipe_diameter / 12.0) / (liquid_viscosity * 0.000672)
    )
    friction_factor = (
        0.316 / (reynolds**0.25) if reynolds > 2100 and reynolds < 100000 else 0.014
    )

    # Friction drop
    friction_drop = (
        friction_factor
        * mixture_density
        * vm**2
        * pipe_length
        / (2 * 32.2 * (pipe_diameter / 12.0) * 144.0)
    )

    # Hydrostatic (vertical)
    hydrostatic_drop = mixture_density * pipe_length / 144.0

    total_drop = friction_drop + hydrostatic_drop

    return max(0.0, total_drop), holdup


def calculate_liquid_holdup(
    liquid_rate: float,
    gas_rate: float,
    pipe_diameter: float,
    liquid_density: float,
    gas_density: float,
    method: str = "beggs_brill",
    **kwargs,
) -> float:
    """Calculate liquid holdup using specified correlation.

    Args:
        liquid_rate: Liquid flow rate (STB/day)
        gas_rate: Gas flow rate (MCF/day)
        pipe_diameter: Pipe diameter (inches)
        liquid_density: Liquid density (lb/ft³)
        gas_density: Gas density (lb/ft³)
        method: Correlation method ('beggs_brill' or 'hagedorn_brown')
        **kwargs: Additional parameters for correlations

    Returns:
        Liquid holdup (fraction)
    """
    if method == "beggs_brill":
        _, holdup = beggs_brill_correlation(
            liquid_rate,
            gas_rate,
            pipe_diameter,
            pipe_length=1.0,  # Dummy length for holdup calculation
            liquid_density=liquid_density,
            gas_density=gas_density,
            liquid_viscosity=kwargs.get("liquid_viscosity", 1.0),
            gas_viscosity=kwargs.get("gas_viscosity", 0.01),
            **kwargs,
        )
        return holdup
    elif method == "hagedorn_brown":
        _, holdup = hagedorn_brown_correlation(
            liquid_rate,
            gas_rate,
            pipe_diameter,
            pipe_length=1.0,
            liquid_density=liquid_density,
            gas_density=gas_density,
            liquid_viscosity=kwargs.get("liquid_viscosity", 1.0),
            gas_viscosity=kwargs.get("gas_viscosity", 0.01),
            **kwargs,
        )
        return holdup
    else:
        raise ValueError(f"Unknown method: {method}")
