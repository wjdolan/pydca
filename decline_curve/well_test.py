"""Well test analysis (Pressure Transient Analysis - PTA).

This module provides well test analysis capabilities including:
- Permeability estimation from pressure buildup/drawdown
- Skin factor calculation
- Boundary detection
- Well test data interpretation

References:
- Horne, R.N., "Modern Well Test Analysis," 5th Ed., 2019.
- Lee, J., Rollins, J.B., and Spivey, J.P., "Pressure Transient Testing," SPE Textbook Series, 2003.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class WellTestResult:
    """Container for well test analysis results.

    Attributes:
        permeability: Estimated permeability (md)
        skin: Skin factor (dimensionless)
        wellbore_storage: Wellbore storage coefficient (bbl/psi)
        boundary_distance: Distance to boundary (ft)
        boundary_type: Type of boundary ('no_flow', 'constant_pressure', 'unknown')
        reservoir_pressure: Estimated reservoir pressure (psi)
    """

    permeability: float
    skin: float
    wellbore_storage: float = 0.0
    boundary_distance: Optional[float] = None
    boundary_type: str = "unknown"
    reservoir_pressure: Optional[float] = None


def analyze_buildup_test(
    time: np.ndarray,
    pressure: np.ndarray,
    production_rate: float,
    production_time: float,
    reservoir_thickness: float = 50.0,
    porosity: float = 0.15,
    oil_viscosity: float = 1.0,
    formation_volume_factor: float = 1.2,
    total_compressibility: float = 1e-5,
    wellbore_radius: float = 0.25,
) -> WellTestResult:
    """Analyze pressure buildup test.

    Uses Horner plot method for buildup analysis.

    Args:
        time: Shut-in time (hours)
        pressure: Pressure during buildup (psi)
        production_rate: Production rate before shut-in (STB/day)
        production_time: Total production time before shut-in (hours)
        reservoir_thickness: Reservoir thickness (ft)
        porosity: Porosity (fraction)
        oil_viscosity: Oil viscosity (cp)
        formation_volume_factor: Oil FVF (RB/STB)
        total_compressibility: Total compressibility (1/psi)
        wellbore_radius: Wellbore radius (ft)

    Returns:
        WellTestResult with permeability, skin, etc.

    Reference:
        Horne, R.N., "Modern Well Test Analysis," 5th Ed., 2019.

    Example:
        >>> time = np.array([0.1, 0.5, 1, 2, 5, 10, 24])
        >>> pressure = np.array([3000, 3200, 3400, 3600, 3800, 3900, 3950])
        >>> result = analyze_buildup_test(time, pressure, 1000, 720)
    """
    # Filter valid data
    valid_mask = (time > 0) & (pressure > 0)
    if np.sum(valid_mask) < 3:
        raise ValueError("Insufficient data for buildup analysis")

    time_valid = time[valid_mask]
    pressure_valid = pressure[valid_mask]

    # Horner time: (tp + Δt) / Δt
    horner_time = (production_time + time_valid) / time_valid

    # Horner plot: pressure vs log(horner_time)
    log_horner = np.log(horner_time)

    # Fit straight line to middle time region (radial flow)
    # Use middle 50% of data
    mid_start = len(log_horner) // 4
    mid_end = 3 * len(log_horner) // 4

    if mid_end - mid_start < 3:
        mid_start = 0
        mid_end = len(log_horner)

    log_horner_mid = log_horner[mid_start:mid_end]
    pressure_mid = pressure_valid[mid_start:mid_end]

    # Fit line: p = m * log((tp + Δt)/Δt) + b
    coeffs = np.polyfit(log_horner_mid, pressure_mid, 1)
    slope = coeffs[0]  # m (psi/log cycle)
    intercept = coeffs[1]  # b

    # Calculate permeability from slope
    # k = (162.6 * q * μ * Bo) / (m * h)
    if abs(slope) > 0:
        permeability = (
            162.6
            * production_rate
            * oil_viscosity
            * formation_volume_factor
            / (abs(slope) * reservoir_thickness)
        )
    else:
        permeability = 0.0

    # Calculate skin factor
    # S = 1.151 * [(p1hr - pwf) / m - log(k / (φ * μ * ct * rw^2)) - 3.23]
    # Find pressure at 1 hour (or extrapolate)
    if len(time_valid) > 0:
        # Find closest to 1 hour
        idx_1hr = np.argmin(np.abs(time_valid - 1.0))
        p1hr = (
            pressure_valid[idx_1hr]
            if idx_1hr < len(pressure_valid)
            else pressure_valid[0]
        )

        # Initial flowing pressure (first point)
        pwf = pressure_valid[0]

        if abs(slope) > 0:
            log_term = np.log(
                permeability
                / (
                    porosity
                    * oil_viscosity
                    * total_compressibility
                    * wellbore_radius**2
                )
            )
            skin = 1.151 * ((p1hr - pwf) / abs(slope) - log_term - 3.23)
        else:
            skin = 0.0

        # Estimate reservoir pressure (extrapolate to infinite time)
        # p* = intercept (pressure at Horner time = 1, i.e., infinite shut-in time)
        reservoir_pressure = intercept
    else:
        skin = 0.0
        reservoir_pressure = None

    # Detect boundaries (simplified)
    boundary_distance, boundary_type = detect_boundaries(
        time_valid, pressure_valid, permeability, porosity, total_compressibility
    )

    return WellTestResult(
        permeability=max(0.001, min(1000.0, permeability)),
        skin=skin,
        boundary_distance=boundary_distance,
        boundary_type=boundary_type,
        reservoir_pressure=reservoir_pressure,
    )


def analyze_drawdown_test(
    time: np.ndarray,
    pressure: np.ndarray,
    production_rate: float,
    reservoir_thickness: float = 50.0,
    porosity: float = 0.15,
    oil_viscosity: float = 1.0,
    formation_volume_factor: float = 1.2,
    total_compressibility: float = 1e-5,
    wellbore_radius: float = 0.25,
    initial_pressure: Optional[float] = None,
) -> WellTestResult:
    """Analyze pressure drawdown test.

    Uses semilog plot method for drawdown analysis.

    Args:
        time: Production time (hours)
        pressure: Flowing pressure (psi)
        production_rate: Production rate (STB/day)
        reservoir_thickness: Reservoir thickness (ft)
        porosity: Porosity (fraction)
        oil_viscosity: Oil viscosity (cp)
        formation_volume_factor: Oil FVF (RB/STB)
        total_compressibility: Total compressibility (1/psi)
        wellbore_radius: Wellbore radius (ft)
        initial_pressure: Initial reservoir pressure (psi)

    Returns:
        WellTestResult with permeability, skin, etc.
    """
    # Filter valid data
    valid_mask = (time > 0) & (pressure > 0)
    if np.sum(valid_mask) < 3:
        raise ValueError("Insufficient data for drawdown analysis")

    time_valid = time[valid_mask]
    pressure_valid = pressure[valid_mask]

    # Semilog plot: pressure vs log(time)
    log_time = np.log(time_valid)

    # Fit straight line to middle time region (radial flow)
    mid_start = len(log_time) // 4
    mid_end = 3 * len(log_time) // 4

    if mid_end - mid_start < 3:
        mid_start = 0
        mid_end = len(log_time)

    log_time_mid = log_time[mid_start:mid_end]
    pressure_mid = pressure_valid[mid_start:mid_end]

    # Fit line: p = m * log(t) + b
    coeffs = np.polyfit(log_time_mid, pressure_mid, 1)
    slope = coeffs[0]  # m (psi/log cycle)
    intercept = coeffs[1]  # b

    # Calculate permeability from slope
    if abs(slope) > 0:
        permeability = (
            162.6
            * production_rate
            * oil_viscosity
            * formation_volume_factor
            / (abs(slope) * reservoir_thickness)
        )
    else:
        permeability = 0.0

    # Calculate skin factor
    if initial_pressure is not None and abs(slope) > 0:
        # Find pressure at 1 hour
        idx_1hr = np.argmin(np.abs(time_valid - 1.0))
        p1hr = (
            pressure_valid[idx_1hr]
            if idx_1hr < len(pressure_valid)
            else pressure_valid[0]
        )

        log_term = np.log(
            permeability
            / (porosity * oil_viscosity * total_compressibility * wellbore_radius**2)
        )
        skin = 1.151 * ((initial_pressure - p1hr) / abs(slope) - log_term - 3.23)
    else:
        skin = 0.0

    # Detect boundaries
    boundary_distance, boundary_type = detect_boundaries(
        time_valid, pressure_valid, permeability, porosity, total_compressibility
    )

    return WellTestResult(
        permeability=max(0.001, min(1000.0, permeability)),
        skin=skin,
        boundary_distance=boundary_distance,
        boundary_type=boundary_type,
        reservoir_pressure=initial_pressure,
    )


def detect_boundaries(
    time: np.ndarray,
    pressure: np.ndarray,
    permeability: float,
    porosity: float,
    total_compressibility: float,
) -> Tuple[Optional[float], str]:
    """Detect reservoir boundaries from pressure data.

    Detects:
    - No-flow boundaries (faults, pinchouts)
    - Constant pressure boundaries (aquifers, gas caps)

    Args:
        time: Time array (hours)
        pressure: Pressure array (psi)
        permeability: Permeability (md)
        porosity: Porosity (fraction)
        total_compressibility: Total compressibility (1/psi)

    Returns:
        Tuple of (boundary_distance, boundary_type)
    """
    if len(time) < 5:
        return None, "unknown"

    # Look for pressure derivative doubling (no-flow boundary)
    # or pressure stabilization (constant pressure boundary)

    # Calculate pressure derivative
    dp_dt = np.gradient(pressure, time)

    # Check for doubling of derivative (no-flow boundary indicator)
    if len(dp_dt) >= 5:
        early_derivative = np.mean(dp_dt[: len(dp_dt) // 3])
        late_derivative = np.mean(dp_dt[-len(dp_dt) // 3 :])

        if early_derivative > 0 and late_derivative > 0:
            ratio = late_derivative / early_derivative
            if ratio > 1.5:  # Significant increase
                # Estimate distance to boundary
                # r_boundary ≈ sqrt(0.00105 * k * t_boundary / (φ * μ * ct))
                # Use time where derivative doubles
                t_boundary = time[np.argmax(dp_dt > early_derivative * 1.5)]
                k_darcy = permeability / 1000.0
                distance = np.sqrt(
                    0.00105 * k_darcy * t_boundary / (porosity * total_compressibility)
                )
                return max(10.0, min(10000.0, distance)), "no_flow"

    # Check for pressure stabilization (constant pressure boundary)
    if len(pressure) >= 5:
        late_pressure = pressure[-len(pressure) // 3 :]
        pressure_change = np.max(late_pressure) - np.min(late_pressure)
        pressure_range = np.max(pressure) - np.min(pressure)

        if pressure_range > 0 and pressure_change / pressure_range < 0.05:
            # Pressure has stabilized
            return None, "constant_pressure"

    return None, "unknown"


def calculate_productivity_index_from_test(
    test_result: WellTestResult,
    reservoir_thickness: float = 50.0,
    wellbore_radius: float = 0.25,
    drainage_radius: float = 745.0,
) -> float:
    """Calculate productivity index from well test results.

    J = (0.00708 * k * h) / (μ * Bo * (ln(re/rw) + S))

    Args:
        test_result: WellTestResult from buildup or drawdown analysis
        reservoir_thickness: Reservoir thickness (ft)
        wellbore_radius: Wellbore radius (ft)
        drainage_radius: Drainage radius (ft)

    Returns:
        Productivity index (STB/day/psi)
    """
    # This is a simplified calculation
    # Full calculation would require viscosity and FVF
    # For now, return a normalized PI
    if test_result.permeability > 0:
        # Simplified: J proportional to k / (ln(re/rw) + S)
        skin_factor = max(0.0, test_result.skin)  # Positive skin reduces PI
        ln_term = np.log(drainage_radius / wellbore_radius) + skin_factor
        if ln_term > 0:
            J = test_result.permeability * reservoir_thickness / ln_term
            return max(0.1, J)
    return 0.0
