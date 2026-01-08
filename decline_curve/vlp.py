"""Vertical Lift Performance (VLP) and Nodal Analysis.

This module provides VLP capabilities including:
- Tubing performance curves
- IPR-VLP intersection (nodal analysis)
- Artificial lift performance
- Choke performance
- Well deliverability optimization

References:
- Brown, K.E., "The Technology of Artificial Lift Methods," Vol. 1-4, 1980.
- Beggs, H.D., "Production Optimization Using Nodal Analysis," 2nd Ed., 2003.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class VLPResult:
    """Container for VLP calculation results.

    Attributes:
        rate: Production rate (STB/day)
        pressure: Flowing pressure (psi)
        liquid_holdup: Liquid holdup (fraction)
        pressure_drop: Pressure drop in tubing (psi)
    """

    rate: float
    pressure: float
    liquid_holdup: float = 0.0
    pressure_drop: float = 0.0


@dataclass
class NodalAnalysisResult:
    """Container for nodal analysis results.

    Attributes:
        operating_rate: Operating rate at intersection (STB/day)
        operating_pressure: Operating pressure at intersection (psi)
        ipr_curve: IPR curve data
        vlp_curve: VLP curve data
    """

    operating_rate: float
    operating_pressure: float
    ipr_curve: Tuple[np.ndarray, np.ndarray]
    vlp_curve: Tuple[np.ndarray, np.ndarray]


def calculate_tubing_performance(
    rate: float,
    wellhead_pressure: float,
    tubing_depth: float,
    tubing_diameter: float = 2.441,  # inches
    gas_liquid_ratio: float = 500.0,  # SCF/STB
    water_cut: float = 0.0,
    oil_gravity: float = 30.0,  # API
    gas_gravity: float = 0.65,
    temperature_gradient: float = 0.015,  # °F/ft
    surface_temperature: float = 60.0,
) -> float:
    """Calculate bottomhole flowing pressure from tubing performance.

    Uses simplified two-phase flow correlation (Beggs-Brill simplified).

    Args:
        rate: Production rate (STB/day)
        wellhead_pressure: Wellhead pressure (psi)
        tubing_depth: Tubing depth (ft)
        tubing_diameter: Tubing inside diameter (inches)
        gas_liquid_ratio: Gas-liquid ratio (SCF/STB)
        water_cut: Water cut (fraction)
        oil_gravity: Oil API gravity (°API)
        gas_gravity: Gas specific gravity (air=1.0)
        temperature_gradient: Temperature gradient (°F/ft)
        surface_temperature: Surface temperature (°F)

    Returns:
        Bottomhole flowing pressure (psi)

    Reference:
        Beggs, H.D., "Production Optimization Using Nodal Analysis," 2nd Ed., 2003.

    Example:
        >>> pwf = calculate_tubing_performance(
        ...     rate=1000,
        ...     wellhead_pressure=500,
        ...     tubing_depth=5000
        ... )
    """
    if rate <= 0:
        return wellhead_pressure

    # Simplified two-phase flow calculation
    # Uses average properties and simplified pressure drop correlation

    # Average temperature
    avg_temperature = surface_temperature + temperature_gradient * tubing_depth / 2.0

    # Calculate average fluid properties
    # Simplified: assume average pressure
    avg_pressure = wellhead_pressure + 0.5 * tubing_depth * 0.433  # Hydrostatic head

    # Liquid density (simplified)
    oil_density = 141.5 / (oil_gravity + 131.5) * 62.4  # lb/ft³
    water_density = 62.4  # lb/ft³
    liquid_density = (1 - water_cut) * oil_density + water_cut * water_density

    # Gas density (simplified)
    z_factor = 1.0  # Simplified
    gas_density = (
        2.7 * gas_gravity * avg_pressure / (z_factor * (avg_temperature + 460.0))
    )  # lb/ft³

    # Mixture density
    # Simplified: use average holdup
    liquid_holdup = 0.5  # Simplified assumption
    mixture_density = liquid_holdup * liquid_density + (1 - liquid_holdup) * gas_density

    # Friction factor (simplified)
    # Use Moody correlation approximation
    reynolds = (
        rate
        * liquid_density
        * tubing_diameter
        / (1.0 * 0.000672)  # Simplified viscosity
    )  # Simplified

    if reynolds > 0:
        friction_factor = 0.316 / (reynolds**0.25) if reynolds < 100000 else 0.014
    else:
        friction_factor = 0.02

    # Pressure drop components
    # 1. Hydrostatic head
    hydrostatic = mixture_density * tubing_depth / 144.0  # Convert to psi

    # 2. Friction
    velocity = rate / (86400.0 * np.pi * (tubing_diameter / 12.0) ** 2 / 4.0)  # ft/s
    friction = (
        friction_factor
        * mixture_density
        * velocity**2
        * tubing_depth
        / (2.0 * 32.2 * (tubing_diameter / 12.0) * 144.0)
    )  # psi

    # Total pressure drop
    pressure_drop = hydrostatic + friction

    # Bottomhole pressure
    bottomhole_pressure = wellhead_pressure + pressure_drop

    return max(wellhead_pressure, bottomhole_pressure)


def generate_vlp_curve(
    wellhead_pressure: float,
    tubing_depth: float,
    rates: Optional[np.ndarray] = None,
    num_points: int = 50,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate VLP curve (bottomhole pressure vs. rate).

    Args:
        wellhead_pressure: Wellhead pressure (psi)
        tubing_depth: Tubing depth (ft)
        rates: Optional rate array. If None, generated automatically.
        num_points: Number of points in curve
        **kwargs: Additional parameters for tubing performance

    Returns:
        Tuple of (rates, bottomhole_pressures) arrays

    Example:
        >>> rates, pressures = generate_vlp_curve(
        ...     wellhead_pressure=500,
        ...     tubing_depth=5000
        ... )
    """
    if rates is None:
        rates = np.linspace(100, 5000, num_points)

    pressures = np.array(
        [
            calculate_tubing_performance(
                rate=r,
                wellhead_pressure=wellhead_pressure,
                tubing_depth=tubing_depth,
                **kwargs,
            )
            for r in rates
        ]
    )

    return rates, pressures


def perform_nodal_analysis(
    reservoir_pressure: float,
    productivity_index: float,
    wellhead_pressure: float,
    tubing_depth: float,
    ipr_type: str = "linear",
    bubble_point_pressure: Optional[float] = None,
    **kwargs,
) -> NodalAnalysisResult:
    """Perform nodal analysis (IPR-VLP intersection).

    Finds the operating point where IPR and VLP curves intersect.

    Args:
        reservoir_pressure: Average reservoir pressure (psi)
        productivity_index: Productivity index (STB/day/psi)
        wellhead_pressure: Wellhead pressure (psi)
        tubing_depth: Tubing depth (ft)
        ipr_type: IPR type ('linear' or 'vogel')
        bubble_point_pressure: Bubble point pressure (for Vogel IPR)
        **kwargs: Additional parameters for IPR and VLP

    Returns:
        NodalAnalysisResult with operating point and curves

    Example:
        >>> result = perform_nodal_analysis(
        ...     reservoir_pressure=5000,
        ...     productivity_index=1.0,
        ...     wellhead_pressure=500,
        ...     tubing_depth=5000
        ... )
        >>> print(f"Operating rate: {result.operating_rate:.1f} STB/day")
    """
    # Generate IPR curve
    rates_ipr = np.linspace(0, 5000, 100)
    pressures_ipr = np.zeros_like(rates_ipr)

    for i, rate in enumerate(rates_ipr):
        if ipr_type == "linear":
            # For IPR, we need pwf from rate
            # Reverse: pwf = p - q/J
            pressures_ipr[i] = (
                reservoir_pressure - rate / productivity_index
                if productivity_index > 0
                else reservoir_pressure
            )
        elif ipr_type == "vogel":
            # For Vogel, need to solve iteratively or use approximation
            # Simplified: use linear approximation
            pressures_ipr[i] = (
                reservoir_pressure - rate / productivity_index
                if productivity_index > 0
                else reservoir_pressure
            )
        else:
            pressures_ipr[i] = (
                reservoir_pressure - rate / productivity_index
                if productivity_index > 0
                else reservoir_pressure
            )

    pressures_ipr = np.maximum(0, pressures_ipr)

    # Generate VLP curve
    rates_vlp, pressures_vlp = generate_vlp_curve(
        wellhead_pressure=wellhead_pressure,
        tubing_depth=tubing_depth,
        rates=rates_ipr,
        **kwargs,
    )

    # Find intersection
    # Operating point: where IPR pressure = VLP pressure
    pressure_diff = pressures_ipr - pressures_vlp

    # Find where difference crosses zero
    operating_rate = 0.0
    operating_pressure = reservoir_pressure

    for i in range(len(pressure_diff) - 1):
        if pressure_diff[i] * pressure_diff[i + 1] <= 0:
            # Interpolate to find exact intersection
            if abs(pressure_diff[i + 1] - pressure_diff[i]) > 1e-6:
                t = -pressure_diff[i] / (pressure_diff[i + 1] - pressure_diff[i])
                operating_rate = rates_ipr[i] + t * (rates_ipr[i + 1] - rates_ipr[i])
                operating_pressure = pressures_ipr[i] + t * (
                    pressures_ipr[i + 1] - pressures_ipr[i]
                )
            else:
                operating_rate = rates_ipr[i]
                operating_pressure = pressures_ipr[i]
            break

    # If no intersection found, use closest point
    if operating_rate == 0.0:
        min_idx = np.argmin(np.abs(pressure_diff))
        operating_rate = rates_ipr[min_idx]
        operating_pressure = pressures_ipr[min_idx]

    return NodalAnalysisResult(
        operating_rate=max(0.0, operating_rate),
        operating_pressure=max(0.0, operating_pressure),
        ipr_curve=(rates_ipr, pressures_ipr),
        vlp_curve=(rates_vlp, pressures_vlp),
    )


def calculate_choke_performance(
    upstream_pressure: float,
    downstream_pressure: float,
    choke_size: float,  # inches
    gas_liquid_ratio: float = 500.0,
    oil_gravity: float = 30.0,
    gas_gravity: float = 0.65,
) -> float:
    """Calculate flow rate through choke.

    Uses simplified choke equation (critical flow).

    Args:
        upstream_pressure: Upstream pressure (psi)
        downstream_pressure: Downstream pressure (psi)
        choke_size: Choke diameter (inches)
        gas_liquid_ratio: Gas-liquid ratio (SCF/STB)
        oil_gravity: Oil API gravity (°API)
        gas_gravity: Gas specific gravity (air=1.0)

    Returns:
        Flow rate (STB/day)

    Reference:
        Brown, K.E., "The Technology of Artificial Lift Methods," Vol. 1-4, 1980.
    """
    if upstream_pressure <= downstream_pressure:
        return 0.0

    # Critical flow through choke
    # Simplified: q = C * d^2 * sqrt(P1^2 - P2^2) / sqrt(γ)
    # where C is choke coefficient, d is diameter, γ is specific gravity

    # Choke coefficient (simplified)
    C = 0.85  # Typical value

    # Specific gravity of mixture (simplified)
    oil_sg = 141.5 / (oil_gravity + 131.5)
    mixture_sg = (oil_sg + gas_gravity * gas_liquid_ratio / 1000.0) / (
        1 + gas_liquid_ratio / 1000.0
    )

    # Flow rate
    pressure_diff = upstream_pressure**2 - downstream_pressure**2
    if pressure_diff > 0:
        rate = C * choke_size**2 * np.sqrt(pressure_diff) / np.sqrt(mixture_sg)
        return max(0.0, rate)
    return 0.0


def optimize_artificial_lift(
    ipr_curve: Tuple[np.ndarray, np.ndarray],
    vlp_curve: Tuple[np.ndarray, np.ndarray],
    lift_type: str = "esp",
    efficiency: float = 0.6,
) -> Dict[str, float]:
    """Optimize artificial lift performance.

    Args:
        ipr_curve: IPR curve (rates, pressures)
        vlp_curve: VLP curve (rates, pressures)
        lift_type: Lift type ('esp', 'gas_lift', 'rod_pump')
        efficiency: Lift efficiency (fraction)

    Returns:
        Dictionary with optimization results
    """
    rates_ipr, pressures_ipr = ipr_curve
    rates_vlp, pressures_vlp = vlp_curve

    # Find operating point
    nodal_result = perform_nodal_analysis(
        reservoir_pressure=pressures_ipr[0],
        productivity_index=1.0,  # Will be calculated
        wellhead_pressure=pressures_vlp[0],
        tubing_depth=5000,  # Default
    )

    operating_rate = nodal_result.operating_rate
    operating_pressure = nodal_result.operating_pressure

    # Calculate power requirements (simplified)
    if lift_type == "esp":
        # ESP power = (q * Δp * 0.000017) / efficiency
        power_hp = (operating_rate * operating_pressure * 0.000017) / efficiency
    elif lift_type == "gas_lift":
        # Gas lift: gas injection rate
        gas_rate = operating_rate * 0.5  # Simplified
        power_hp = gas_rate * 0.001  # Simplified
    else:
        power_hp = 0.0

    return {
        "operating_rate": operating_rate,
        "operating_pressure": operating_pressure,
        "power_required": power_hp,
        "efficiency": efficiency,
    }
