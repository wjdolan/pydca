"""Rate Transient Analysis (RTA) for production data analysis.

This module provides RTA capabilities including:
- Flow regime identification
- Permeability estimation from production data
- Fracture half-length estimation
- Stimulated Reservoir Volume (SRV) calculation
- Production data analysis

References:
- Wattenbarger, R.A., et al., "Gas Reservoir Engineering," SPE Textbook Series, 1998.
- Ilk, D., et al., "Production Data Analysis - Challenges, Pitfalls, Diagnostics,"
  SPE 116231, 2008.
- Anderson, D.M., et al., "Analysis of Production Data from Fractured Shale Gas Wells,"
  SPE 131787, 2010.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class RTAResult:
    """Container for RTA analysis results.

    Attributes:
        flow_regime: Identified flow regime
        permeability: Estimated permeability (md)
        fracture_half_length: Estimated fracture half-length (ft)
        srv: Stimulated reservoir volume (acre-ft)
        skin: Skin factor (dimensionless)
        drainage_area: Drainage area (acres)
    """

    flow_regime: str
    permeability: float
    fracture_half_length: Optional[float] = None
    srv: Optional[float] = None
    skin: float = 0.0
    drainage_area: Optional[float] = None


def identify_flow_regime(
    time: np.ndarray,
    rate: np.ndarray,
    cumulative: Optional[np.ndarray] = None,
) -> str:
    """Identify flow regime from production data.

    Flow regimes:
    - 'linear': Linear flow (fracture dominated)
    - 'bilinear': Bilinear flow (fracture + matrix)
    - 'boundary_dominated': Boundary dominated flow
    - 'transient': Transient flow
    - 'unknown': Cannot identify

    Args:
        time: Production time (days)
        rate: Production rate (STB/day or MCF/day)
        cumulative: Optional cumulative production

    Returns:
        Identified flow regime string

    Example:
        >>> time = np.array([1, 10, 30, 60, 90, 120])
        >>> rate = np.array([1000, 800, 600, 500, 450, 400])
        >>> regime = identify_flow_regime(time, rate)
    """
    if len(time) < 3 or len(rate) < 3:
        return "unknown"

    # Filter out zeros
    valid_mask = (rate > 0) & (time > 0)
    if np.sum(valid_mask) < 3:
        return "unknown"

    time_valid = time[valid_mask]
    rate_valid = rate[valid_mask]

    # Calculate log-log derivative for flow regime identification
    # Linear flow: q vs sqrt(t) should be linear
    # Boundary dominated: q vs t should show exponential decline

    # Check for linear flow (early time, fracture dominated)
    # Linear flow: q proportional to 1/sqrt(t)
    if len(time_valid) >= 3:
        early_indices = np.arange(min(5, len(time_valid)))
        sqrt_t_early = np.sqrt(time_valid[early_indices])
        q_early = rate_valid[early_indices]

        # Check if q vs sqrt(t) is approximately linear
        if len(q_early) >= 3:
            # Calculate correlation
            correlation = np.corrcoef(sqrt_t_early, q_early)[0, 1]
            if correlation < -0.7:  # Strong negative correlation
                return "linear"

    # Check for boundary dominated flow (late time)
    # BDF: exponential decline, q vs t shows constant decline rate
    if len(time_valid) >= 5:
        late_indices = np.arange(max(0, len(time_valid) - 5), len(time_valid))
        t_late = time_valid[late_indices]
        q_late = rate_valid[late_indices]

        # Check for exponential decline
        log_q_late = np.log(q_late[q_late > 0])
        if len(log_q_late) >= 3:
            # Fit exponential: log(q) = a + b*t
            coeffs = np.polyfit(t_late[: len(log_q_late)], log_q_late, 1)
            # If decline rate is relatively constant, it's BDF
            if abs(coeffs[0]) < 0.01 and coeffs[0] < 0:  # Negative slope
                return "boundary_dominated"

    # Check for bilinear flow (intermediate)
    # Bilinear: q proportional to t^(-1/4)
    if len(time_valid) >= 4:
        mid_indices = np.arange(2, min(6, len(time_valid)))
        t_mid = time_valid[mid_indices]
        q_mid = rate_valid[mid_indices]
        t_power = t_mid ** (-0.25)

        if len(q_mid) >= 3:
            correlation = np.corrcoef(t_power, q_mid)[0, 1]
            if correlation > 0.7:
                return "bilinear"

    # Default to transient if we can't identify
    return "transient"


def estimate_permeability_from_production(
    time: np.ndarray,
    rate: np.ndarray,
    pressure: Optional[np.ndarray] = None,
    initial_pressure: float = 5000.0,
    wellbore_radius: float = 0.25,
    reservoir_thickness: float = 50.0,
    porosity: float = 0.15,
    oil_viscosity: float = 1.0,
    formation_volume_factor: float = 1.2,
    total_compressibility: float = 1e-5,
    flow_regime: Optional[str] = None,
) -> float:
    """Estimate permeability from production data.

    Uses different methods depending on flow regime:
    - Linear flow: Uses square root of time plot
    - Boundary dominated: Uses decline curve analysis
    - Transient: Uses radial flow equations

    Args:
        time: Production time (days)
        rate: Production rate (STB/day)
        pressure: Optional flowing pressure (psi)
        initial_pressure: Initial reservoir pressure (psi)
        wellbore_radius: Wellbore radius (ft)
        reservoir_thickness: Reservoir thickness (ft)
        porosity: Porosity (fraction)
        oil_viscosity: Oil viscosity (cp)
        formation_volume_factor: Oil FVF (RB/STB)
        total_compressibility: Total compressibility (1/psi)
        flow_regime: Optional flow regime (if None, identified automatically)

    Returns:
        Estimated permeability (md)

    Example:
        >>> time = np.array([1, 10, 30, 60, 90])
        >>> rate = np.array([1000, 800, 600, 500, 450])
        >>> k = estimate_permeability_from_production(time, rate)
    """
    if flow_regime is None:
        flow_regime = identify_flow_regime(time, rate)

    # Filter valid data
    valid_mask = (rate > 0) & (time > 0)
    if np.sum(valid_mask) < 3:
        return 0.0

    time_valid = time[valid_mask]
    rate_valid = rate[valid_mask]

    if flow_regime == "linear":
        # Linear flow: q = (4.064 * k * h * Δp) / (μ * Bo * sqrt(π * φ * μ * ct * xf^2 / k) * sqrt(t))
        # Simplified: k estimated from early time data
        early_indices = np.arange(min(5, len(time_valid)))
        sqrt_t = np.sqrt(time_valid[early_indices])
        q = rate_valid[early_indices]

        if len(q) >= 3:
            # Estimate from slope of q vs 1/sqrt(t)
            # k ≈ (slope * μ * Bo) / (4.064 * h * Δp)
            if pressure is not None and len(pressure) > 0:
                pressure_valid = pressure[valid_mask]
                if len(pressure_valid) >= len(early_indices):
                    dp = initial_pressure - pressure_valid[early_indices]
                    dp_avg = (
                        np.mean(dp)
                        if isinstance(dp, np.ndarray) and len(dp) > 0
                        else initial_pressure * 0.2
                    )
                else:
                    dp_avg = initial_pressure * 0.2
            else:
                dp_avg = initial_pressure * 0.2

            # Simplified estimation
            # For linear flow: k ≈ 162.6 * q * μ * Bo / (h * Δp * m_sqrt_t)
            # where m_sqrt_t is slope of q vs 1/sqrt(t)
            inv_sqrt_t = 1.0 / sqrt_t
            if len(q) >= 2:
                slope = np.polyfit(inv_sqrt_t, q, 1)[0]
                if slope > 0 and dp_avg > 0:
                    k = (
                        162.6
                        * abs(slope)
                        * oil_viscosity
                        * formation_volume_factor
                        / (reservoir_thickness * dp_avg)
                    )
                    return max(0.001, min(1000.0, k))  # Reasonable bounds

    elif flow_regime == "boundary_dominated":
        # BDF: Use decline curve analysis
        # k estimated from Arps parameters
        if len(rate_valid) >= 3:
            # Estimate decline rate
            decline_rate = (rate_valid[0] - rate_valid[-1]) / (
                rate_valid[0] * time_valid[-1]
            )
            decline_rate = max(0.0001, min(1.0, decline_rate))

            # Simplified: k ≈ (141.2 * q * μ * Bo) / (h * Δp * ln(re/rw))
            dp = initial_pressure * 0.2  # Assume 20% pressure drop
            re = 745.0  # Typical drainage radius
            k = (
                141.2
                * rate_valid[0]
                * oil_viscosity
                * formation_volume_factor
                / (reservoir_thickness * dp * np.log(re / wellbore_radius))
            )
            return max(0.001, min(1000.0, k))

    else:
        # Transient flow: Use radial flow equation
        # k = (162.6 * q * μ * Bo) / (h * Δp * m)
        # where m is slope of log-log plot
        if len(rate_valid) >= 3:
            log_t = np.log(time_valid)
            log_q = np.log(rate_valid)

            if len(log_q) >= 2:
                slope = np.polyfit(log_t, log_q, 1)[0]
                dp = initial_pressure * 0.2  # Assume 20% pressure drop

                if slope < 0 and dp > 0:
                    # Estimate from early time data
                    k = (
                        162.6
                        * rate_valid[0]
                        * oil_viscosity
                        * formation_volume_factor
                        / (reservoir_thickness * dp * abs(slope))
                    )
                    return max(0.001, min(1000.0, k))

    # Default fallback
    return 0.1  # Default permeability estimate


def estimate_fracture_half_length(
    time: np.ndarray,
    rate: np.ndarray,
    permeability: float,
    porosity: float = 0.15,
    total_compressibility: float = 1e-5,
    oil_viscosity: float = 1.0,
    formation_volume_factor: float = 1.2,
    reservoir_thickness: float = 50.0,
    pressure_drop: float = 1000.0,
) -> float:
    """Estimate fracture half-length from production data.

    Uses linear flow analysis for fractured wells.

    Args:
        time: Production time (days)
        rate: Production rate (STB/day)
        permeability: Formation permeability (md)
        porosity: Porosity (fraction)
        total_compressibility: Total compressibility (1/psi)
        oil_viscosity: Oil viscosity (cp)
        formation_volume_factor: Oil FVF (RB/STB)
        reservoir_thickness: Reservoir thickness (ft)
        pressure_drop: Pressure drop (psi)

    Returns:
        Estimated fracture half-length (ft)

    Reference:
        Wattenbarger, R.A., et al., "Gas Reservoir Engineering," SPE Textbook Series, 1998.
    """
    # Filter valid data
    valid_mask = (rate > 0) & (time > 0)
    if np.sum(valid_mask) < 3:
        return 0.0

    time_valid = time[valid_mask]
    rate_valid = rate[valid_mask]

    # Linear flow analysis
    # For linear flow: q = (4.064 * k * h * Δp) / (μ * Bo * sqrt(π * φ * μ * ct * xf^2 / k) * sqrt(t))
    # Solving for xf:
    # xf = sqrt((4.064 * k * h * Δp) / (μ * Bo * q * sqrt(π * φ * μ * ct / k) * sqrt(t)))

    # Use early time data (linear flow period)
    early_indices = np.arange(min(5, len(time_valid)))
    sqrt_t = np.sqrt(time_valid[early_indices])
    q = rate_valid[early_indices]

    if len(q) >= 2:
        # Average rate during linear flow
        q_avg = np.mean(q)

        # Calculate xf from linear flow equation
        # Simplified: xf ≈ sqrt((k * h * Δp) / (q * μ * Bo * sqrt(φ * μ * ct / k)))
        k_darcy = permeability / 1000.0  # Convert to darcy

        numerator = k_darcy * reservoir_thickness * pressure_drop
        denominator = (
            q_avg
            * oil_viscosity
            * formation_volume_factor
            * np.sqrt(porosity * oil_viscosity * total_compressibility / k_darcy)
        )

        if denominator > 0:
            xf_squared = numerator / denominator
            xf = np.sqrt(xf_squared) if xf_squared > 0 else 0.0
            return max(10.0, min(5000.0, xf))  # Reasonable bounds

    return 0.0


def calculate_srv(
    fracture_half_length: float,
    number_of_fractures: int = 1,
    fracture_spacing: float = 200.0,
    reservoir_thickness: float = 50.0,
    well_length: float = 5000.0,
) -> float:
    """Calculate Stimulated Reservoir Volume (SRV).

    SRV is the volume of reservoir that has been effectively stimulated
    by hydraulic fracturing.

    Args:
        fracture_half_length: Fracture half-length (ft)
        number_of_fractures: Number of fractures
        fracture_spacing: Spacing between fractures (ft)
        reservoir_thickness: Reservoir thickness (ft)
        well_length: Horizontal well length (ft)

    Returns:
        SRV in acre-ft

    Example:
        >>> srv = calculate_srv(fracture_half_length=500, number_of_fractures=20)
        >>> print(f"SRV: {srv:.1f} acre-ft")
    """
    # SRV calculation
    # For multiple fractures: SRV = 2 * xf * spacing * n_fractures * h
    # For single well: SRV = 2 * xf * well_length * h

    if number_of_fractures > 1:
        # Multiple fractures
        srv_ft3 = (
            2
            * fracture_half_length
            * fracture_spacing
            * number_of_fractures
            * reservoir_thickness
        )
    else:
        # Single fracture or use well length
        srv_ft3 = 2 * fracture_half_length * well_length * reservoir_thickness

    # Convert to acre-ft (1 acre-ft = 43,560 ft³)
    srv_acre_ft = srv_ft3 / 43560.0

    return max(0.0, srv_acre_ft)


def analyze_production_data(
    time: np.ndarray,
    rate: np.ndarray,
    pressure: Optional[np.ndarray] = None,
    initial_pressure: float = 5000.0,
    reservoir_thickness: float = 50.0,
    porosity: float = 0.15,
    oil_viscosity: float = 1.0,
    formation_volume_factor: float = 1.2,
) -> RTAResult:
    """Comprehensive RTA analysis of production data.

    Args:
        time: Production time (days)
        rate: Production rate (STB/day)
        pressure: Optional flowing pressure (psi)
        initial_pressure: Initial reservoir pressure (psi)
        reservoir_thickness: Reservoir thickness (ft)
        porosity: Porosity (fraction)
        oil_viscosity: Oil viscosity (cp)
        formation_volume_factor: Oil FVF (RB/STB)

    Returns:
        RTAResult with all calculated parameters

    Example:
        >>> time = np.array([1, 10, 30, 60, 90, 120])
        >>> rate = np.array([1000, 800, 600, 500, 450, 400])
        >>> result = analyze_production_data(time, rate)
        >>> print(f"Flow regime: {result.flow_regime}")
        >>> print(f"Permeability: {result.permeability:.3f} md")
    """
    # Identify flow regime
    flow_regime = identify_flow_regime(time, rate)

    # Estimate permeability
    permeability = estimate_permeability_from_production(
        time=time,
        rate=rate,
        pressure=pressure,
        initial_pressure=initial_pressure,
        reservoir_thickness=reservoir_thickness,
        porosity=porosity,
        oil_viscosity=oil_viscosity,
        formation_volume_factor=formation_volume_factor,
        flow_regime=flow_regime,
    )

    # Estimate fracture half-length if linear flow
    fracture_half_length = None
    srv = None

    if flow_regime in ("linear", "bilinear"):
        fracture_half_length = estimate_fracture_half_length(
            time=time,
            rate=rate,
            permeability=permeability,
            porosity=porosity,
            reservoir_thickness=reservoir_thickness,
            oil_viscosity=oil_viscosity,
            formation_volume_factor=formation_volume_factor,
        )

        if fracture_half_length > 0:
            # Estimate number of fractures from production behavior
            # Simplified: assume 1 fracture per 200 ft for horizontal wells
            well_length_estimate = len(time) * 10.0  # Rough estimate
            n_fractures = max(1, int(well_length_estimate / 200.0))
            srv = calculate_srv(
                fracture_half_length=fracture_half_length,
                number_of_fractures=n_fractures,
                reservoir_thickness=reservoir_thickness,
            )

    return RTAResult(
        flow_regime=flow_regime,
        permeability=permeability,
        fracture_half_length=fracture_half_length,
        srv=srv,
        skin=0.0,  # Would need well test data for accurate skin
    )
