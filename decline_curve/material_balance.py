"""Material balance models for different reservoir drive mechanisms.

This module provides drive mechanism-specific material balance equations:

1. Solution Gas Drive (Depletion Drive)
   - For oil reservoirs producing by solution gas expansion
   - Accounts for gas coming out of solution

2. Water Drive
   - For oil reservoirs with active aquifer support
   - Accounts for water influx

3. Gas Cap Drive
   - For oil reservoirs with gas cap expansion
   - Accounts for gas cap expansion and solution gas

4. Combination Drive
   - Multiple drive mechanisms acting simultaneously

5. Gas Reservoir (p/Z method)
   - For dry gas or wet gas reservoirs
   - Uses p/Z vs cumulative production relationship

6. Undersaturated Oil
   - Oil above bubble point pressure
   - Single-phase flow

7. Saturated Oil
   - Oil at or below bubble point pressure
   - Two-phase flow (oil + gas)

References:
- Havlena, D. and Odeh, A.S., "The Material Balance as an Equation of a Straight Line,"
  JPT, August 1963.
- Dake, L.P., "Fundamentals of Reservoir Engineering," 1978.
- Craft, B.C. and Hawkins, M.F., "Applied Petroleum Reservoir Engineering," 1991.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .logging_config import get_logger
from .pvt import calculate_pvt_properties

logger = get_logger(__name__)


@dataclass
class SolutionGasDriveParams:
    """Parameters for solution gas drive material balance.

    Attributes:
        N: Original oil in place (STB)
        Boi: Initial oil FVF (RB/STB)
        Rsi: Initial solution gas-oil ratio (SCF/STB)
        Swi: Initial water saturation (fraction)
        cw: Water compressibility (1/psi)
        cf: Formation compressibility (1/psi)
        pi: Initial reservoir pressure (psi)
        pb: Bubble point pressure (psi)
        api_gravity: Oil API gravity (°API)
        gas_gravity: Gas specific gravity (air=1.0)
        temperature: Reservoir temperature (°F)
    """

    N: float
    Boi: float = 1.2
    Rsi: float = 500.0
    Swi: float = 0.2
    cw: float = 3e-6
    cf: float = 5e-6
    pi: float = 5000.0
    pb: float = 3000.0
    api_gravity: float = 30.0
    gas_gravity: float = 0.65
    temperature: float = 200.0


@dataclass
class WaterDriveParams:
    """Parameters for water drive material balance.

    Attributes:
        N: Original oil in place (STB)
        Boi: Initial oil FVF (RB/STB)
        Swi: Initial water saturation (fraction)
        cw: Water compressibility (1/psi)
        cf: Formation compressibility (1/psi)
        pi: Initial reservoir pressure (psi)
        We: Cumulative water influx (STB)
        Wp: Cumulative water production (STB)
        Bw: Water FVF (RB/STB)
    """

    N: float
    Boi: float = 1.2
    Swi: float = 0.2
    cw: float = 3e-6
    cf: float = 5e-6
    pi: float = 5000.0
    We: float = 0.0
    Wp: float = 0.0
    Bw: float = 1.0


@dataclass
class GasCapDriveParams:
    """Parameters for gas cap drive material balance.

    Attributes:
        N: Original oil in place (STB)
        m: Gas cap size (ratio of gas cap volume to oil volume)
        Boi: Initial oil FVF (RB/STB)
        Bgi: Initial gas FVF (RB/SCF)
        Rsi: Initial solution gas-oil ratio (SCF/STB)
        Swi: Initial water saturation (fraction)
        pi: Initial reservoir pressure (psi)
        api_gravity: Oil API gravity (°API)
        gas_gravity: Gas specific gravity (air=1.0)
        temperature: Reservoir temperature (°F)
    """

    N: float
    m: float = 0.1  # Gas cap size
    Boi: float = 1.2
    Bgi: float = 0.001
    Rsi: float = 500.0
    Swi: float = 0.2
    pi: float = 5000.0
    api_gravity: float = 30.0
    gas_gravity: float = 0.65
    temperature: float = 200.0


@dataclass
class GasReservoirParams:
    """Parameters for gas reservoir material balance (p/Z method).

    Attributes:
        G: Original gas in place (SCF)
        pi: Initial reservoir pressure (psi)
        Zi: Initial gas Z-factor
        temperature: Reservoir temperature (°F)
        gas_gravity: Gas specific gravity (air=1.0)
        We: Cumulative water influx (SCF)
        Wp: Cumulative water production (STB)
        Bw: Water FVF (RB/STB)
    """

    G: float
    pi: float = 5000.0
    Zi: float = 1.0
    temperature: float = 200.0
    gas_gravity: float = 0.65
    We: float = 0.0
    Wp: float = 0.0
    Bw: float = 1.0


def fetkovich_water_influx(
    time: float,
    aquifer_pressure: float,
    reservoir_pressure: float,
    aquifer_volume: float,
    aquifer_compressibility: float = 3e-6,
    aquifer_productivity_index: float = 0.1,
) -> float:
    """Calculate water influx using Fetkovich method.

    Fetkovich method for finite aquifer.

    Args:
        time: Time (days)
        aquifer_pressure: Initial aquifer pressure (psi)
        reservoir_pressure: Current reservoir pressure (psi)
        aquifer_volume: Aquifer volume (STB)
        aquifer_compressibility: Aquifer compressibility (1/psi)
        aquifer_productivity_index: Aquifer productivity index (STB/day/psi)

    Returns:
        Cumulative water influx (STB)

    Reference:
        Fetkovich, M.J., "A Simplified Approach to Water Influx Calculations,"
        JPT, June 1971.
    """
    # Fetkovich method
    # We = Wi * (1 - exp(-J * pi * t / Wi)) * (pi - p) / pi
    # where Wi is initial aquifer volume

    Wi = aquifer_volume
    pi = aquifer_pressure
    p = reservoir_pressure

    if Wi > 0 and pi > 0:
        # Exponential term
        exp_term = np.exp(-aquifer_productivity_index * pi * time / Wi)
        # Water influx
        We = Wi * (1 - exp_term) * (pi - p) / pi
        return max(0.0, We)
    return 0.0


def carter_tracy_water_influx(
    time: float,
    aquifer_pressure: float,
    reservoir_pressure: float,
    aquifer_radius: float,
    reservoir_radius: float,
    aquifer_permeability: float,
    aquifer_thickness: float,
    aquifer_porosity: float,
    total_compressibility: float = 1e-5,
) -> float:
    """Calculate water influx using Carter-Tracy method.

    Carter-Tracy method for infinite aquifer.

    Args:
        time: Time (days)
        aquifer_pressure: Initial aquifer pressure (psi)
        reservoir_pressure: Current reservoir pressure (psi)
        aquifer_radius: Aquifer outer radius (ft)
        reservoir_radius: Reservoir radius (ft)
        aquifer_permeability: Aquifer permeability (md)
        aquifer_thickness: Aquifer thickness (ft)
        aquifer_porosity: Aquifer porosity (fraction)
        total_compressibility: Total compressibility (1/psi)

    Returns:
        Cumulative water influx (STB)

    Reference:
        Carter, R.D. and Tracy, G.W., "An Improved Method for Calculating Water Influx,"
        JPT, December 1960.
    """
    # Carter-Tracy method (simplified)
    # Uses dimensionless time and pressure functions

    # Dimensionless time
    k_darcy = aquifer_permeability / 1000.0
    tD = (
        0.00634
        * k_darcy
        * time
        / (aquifer_porosity * total_compressibility * reservoir_radius**2)
    )

    # Dimensionless pressure drop
    pd = (
        (aquifer_pressure - reservoir_pressure) / aquifer_pressure
        if aquifer_pressure > 0
        else 0.0
    )

    # Water influx constant
    U = (
        1.119
        * aquifer_porosity
        * total_compressibility
        * aquifer_thickness
        * reservoir_radius**2
    )

    # Water influx (simplified)
    We = U * pd * np.sqrt(tD) if tD > 0 else 0.0

    return max(0.0, We)


def solution_gas_drive_material_balance(
    pressure: float,
    cumulative_oil: float,
    params: SolutionGasDriveParams,
) -> Dict[str, float]:
    """Calculate material balance for solution gas drive reservoir.

    Material balance equation for depletion drive:
    Np * Bo = N * (Bo - Boi) + N * Boi * (cw * Swi + cf) / (1 - Swi) * (pi - p)
            + Np * (Rsi - Rs) * Bg

    Args:
        pressure: Current reservoir pressure (psi)
        cumulative_oil: Cumulative oil production (STB)
        params: Solution gas drive parameters

    Returns:
        Dictionary with material balance results:
        - Np_calculated: Calculated cumulative oil (STB)
        - Bo: Current oil FVF (RB/STB)
        - Rs: Current solution GOR (SCF/STB)
        - Bg: Current gas FVF (RB/SCF)
        - Gp: Cumulative gas production (SCF)
    """
    # Calculate PVT properties at current pressure
    pvt = calculate_pvt_properties(
        pressure=pressure,
        temperature=params.temperature,
        api_gravity=params.api_gravity,
        gas_gravity=params.gas_gravity,
        method="standing",
    )

    Bo = pvt.Bo
    Rs = pvt.Rs
    Bg = pvt.Bg

    # Calculate PVT at initial conditions
    pvt_initial = calculate_pvt_properties(
        pressure=params.pi,
        temperature=params.temperature,
        api_gravity=params.api_gravity,
        gas_gravity=params.gas_gravity,
        method="standing",
    )

    Boi = pvt_initial.Bo
    Rsi = pvt_initial.Rs

    # Material balance equation
    # Left side: Np * Bo
    left_side = cumulative_oil * Bo

    # Right side components
    # 1. Oil expansion: N * (Bo - Boi)
    oil_expansion = params.N * (Bo - Boi)

    # 2. Rock and water expansion
    rock_water_expansion = (
        params.N
        * Boi
        * (params.cw * params.Swi + params.cf)
        / (1 - params.Swi)
        * (params.pi - pressure)
    )

    # 3. Gas expansion: Np * (Rsi - Rs) * Bg
    gas_expansion = cumulative_oil * (Rsi - Rs) * Bg

    # Solve for Np (iterative or use provided cumulative)
    # For now, calculate what Np should be
    right_side = oil_expansion + rock_water_expansion + gas_expansion
    Np_calculated = right_side / Bo if Bo > 0 else 0.0

    # Calculate cumulative gas production
    Gp = cumulative_oil * (Rsi - Rs)

    return {
        "Np_calculated": Np_calculated,
        "Bo": Bo,
        "Rs": Rs,
        "Bg": Bg,
        "Gp": Gp,
        "oil_expansion": oil_expansion,
        "rock_water_expansion": rock_water_expansion,
        "gas_expansion": gas_expansion,
    }


def water_drive_material_balance(
    pressure: float,
    cumulative_oil: float,
    params: WaterDriveParams,
    water_influx_model: str = "fetkovich",
    water_influx_params: Optional[Dict] = None,
) -> Dict[str, float]:
    """Calculate material balance for water drive reservoir.

    Material balance equation for water drive:
    Np * Bo = N * (Bo - Boi) + N * Boi * (cw * Swi + cf) / (1 - Swi) * (pi - p)
            + We - Wp * Bw

    Args:
        pressure: Current reservoir pressure (psi)
        cumulative_oil: Cumulative oil production (STB)
        params: Water drive parameters
        water_influx_model: Water influx model ('fetkovich' or 'carter_tracy')
        water_influx_params: Additional parameters for water influx model

    Returns:
        Dictionary with material balance results
    """
    # Calculate PVT properties at current pressure
    pvt = calculate_pvt_properties(
        pressure=pressure,
        temperature=200.0,  # Default temperature
        api_gravity=30.0,  # Default API
        gas_gravity=0.65,
        method="standing",
    )

    Bo = pvt.Bo

    # Calculate water influx if model specified
    We = params.We
    if water_influx_model == "fetkovich" and water_influx_params:
        We = fetkovich_water_influx(
            time=water_influx_params.get("time", 365.0),
            aquifer_pressure=water_influx_params.get("aquifer_pressure", params.pi),
            reservoir_pressure=pressure,
            aquifer_volume=water_influx_params.get("aquifer_volume", 1e6),
            **{
                k: v
                for k, v in water_influx_params.items()
                if k not in ["time", "aquifer_pressure", "aquifer_volume"]
            },
        )
    elif water_influx_model == "carter_tracy" and water_influx_params:
        We = carter_tracy_water_influx(
            time=water_influx_params.get("time", 365.0),
            aquifer_pressure=water_influx_params.get("aquifer_pressure", params.pi),
            reservoir_pressure=pressure,
            **{
                k: v
                for k, v in water_influx_params.items()
                if k not in ["time", "aquifer_pressure"]
            },
        )

    # Material balance equation
    # Left side: Np * Bo
    left_side = cumulative_oil * Bo

    # Right side components
    # 1. Oil expansion
    oil_expansion = params.N * (Bo - params.Boi)

    # 2. Rock and water expansion
    rock_water_expansion = (
        params.N
        * params.Boi
        * (params.cw * params.Swi + params.cf)
        / (1 - params.Swi)
        * (params.pi - pressure)
    )

    # 3. Water influx
    water_influx = We - params.Wp * params.Bw

    # Solve material balance
    right_side = oil_expansion + rock_water_expansion + water_influx
    Np_calculated = right_side / Bo if Bo > 0 else 0.0

    return {
        "Np_calculated": Np_calculated,
        "Bo": Bo,
        "oil_expansion": oil_expansion,
        "rock_water_expansion": rock_water_expansion,
        "water_influx": water_influx,
    }


def gas_cap_drive_material_balance(
    pressure: float,
    cumulative_oil: float,
    params: GasCapDriveParams,
) -> Dict[str, float]:
    """Calculate material balance for gas cap drive reservoir.

    Material balance equation for gas cap drive:
    Np * Bo = N * (Bo - Boi) + m * N * Boi * (Bg - Bgi) / Bgi
            + Np * (Rsi - Rs) * Bg

    Args:
        pressure: Current reservoir pressure (psi)
        cumulative_oil: Cumulative oil production (STB)
        params: Gas cap drive parameters

    Returns:
        Dictionary with material balance results
    """
    # Calculate PVT properties at current pressure
    pvt = calculate_pvt_properties(
        pressure=pressure,
        temperature=params.temperature,
        api_gravity=params.api_gravity,
        gas_gravity=params.gas_gravity,
        method="standing",
    )

    Bo = pvt.Bo
    Rs = pvt.Rs
    Bg = pvt.Bg

    # Calculate PVT at initial conditions
    pvt_initial = calculate_pvt_properties(
        pressure=params.pi,
        temperature=params.temperature,
        api_gravity=params.api_gravity,
        gas_gravity=params.gas_gravity,
        method="standing",
    )

    Rsi = pvt_initial.Rs

    # Material balance equation
    # Left side: Np * Bo
    left_side = cumulative_oil * Bo

    # Right side components
    # 1. Oil expansion
    oil_expansion = params.N * (Bo - params.Boi)

    # 2. Gas cap expansion
    # Gas cap expands as pressure decreases (Bg increases)
    # Use absolute value to handle cases where Bg might be less than Bgi initially
    if params.Bgi > 0:
        gas_cap_expansion = (
            params.m * params.N * params.Boi * (Bg - params.Bgi) / params.Bgi
        )
    else:
        gas_cap_expansion = 0.0

    # 3. Solution gas expansion
    solution_gas_expansion = cumulative_oil * (Rsi - Rs) * Bg

    # Solve material balance
    right_side = oil_expansion + gas_cap_expansion + solution_gas_expansion
    Np_calculated = right_side / Bo if Bo > 0 else 0.0

    # Calculate cumulative gas production
    Gp = cumulative_oil * (Rsi - Rs)

    return {
        "Np_calculated": Np_calculated,
        "Bo": Bo,
        "Rs": Rs,
        "Bg": Bg,
        "Gp": Gp,
        "oil_expansion": oil_expansion,
        "gas_cap_expansion": gas_cap_expansion,
        "solution_gas_expansion": solution_gas_expansion,
    }


def gas_reservoir_pz_method(
    pressure: float,
    cumulative_gas: float,
    params: GasReservoirParams,
) -> Dict[str, float]:
    """Calculate material balance for gas reservoir using p/Z method.

    p/Z method for gas reservoirs:
    p/Z = (pi/Zi) * (1 - Gp/G)

    Rearranged:
    G = Gp / (1 - (p/Z) / (pi/Zi))

    Args:
        pressure: Current reservoir pressure (psi)
        cumulative_gas: Cumulative gas production (SCF)
        params: Gas reservoir parameters

    Returns:
        Dictionary with material balance results:
        - G_calculated: Calculated original gas in place (SCF)
        - Z: Current gas Z-factor
        - p_over_z: Current p/Z ratio
        - pi_over_zi: Initial p/Z ratio
    """
    from .pvt import gas_z_factor

    # Calculate Z-factor at current pressure
    Z = gas_z_factor(
        pressure=pressure,
        temperature=params.temperature,
        gas_gravity=params.gas_gravity,
        method="standing",
    )

    # Calculate p/Z ratios
    p_over_z = pressure / Z
    pi_over_zi = params.pi / params.Zi

    # p/Z method equation
    # p/Z = (pi/Zi) * (1 - Gp/G)
    # Solving for G:
    if cumulative_gas == 0.0:
        # No production yet, use initial estimate
        G_calculated = params.G
    elif p_over_z < pi_over_zi and (1 - p_over_z / pi_over_zi) > 1e-6:
        G_calculated = cumulative_gas / (1 - p_over_z / pi_over_zi)
    else:
        G_calculated = params.G  # Use initial estimate

    # Calculate remaining gas
    remaining_gas = G_calculated - cumulative_gas

    return {
        "G_calculated": G_calculated,
        "Z": Z,
        "p_over_z": p_over_z,
        "pi_over_zi": pi_over_zi,
        "remaining_gas": remaining_gas,
        "recovery_factor": cumulative_gas / G_calculated if G_calculated > 0 else 0.0,
    }


def undersaturated_oil_material_balance(
    pressure: float,
    cumulative_oil: float,
    params: SolutionGasDriveParams,
) -> Dict[str, float]:
    """Calculate material balance for undersaturated oil reservoir.

    Undersaturated oil: pressure > bubble point
    Single-phase flow, no free gas

    Material balance:
    Np * Bo = N * (Bo - Boi) + N * Boi * (cw * Swi + cf) / (1 - Swi) * (pi - p)

    Args:
        pressure: Current reservoir pressure (psi)
        cumulative_oil: Cumulative oil production (STB)
        params: Material balance parameters

    Returns:
        Dictionary with material balance results
    """
    if pressure < params.pb:
        logger.warning(
            f"Pressure ({pressure} psi) is below bubble point ({params.pb} psi). "
            "Use solution_gas_drive_material_balance instead."
        )

    # Calculate PVT properties (above bubble point, Rs constant)
    pvt = calculate_pvt_properties(
        pressure=pressure,
        temperature=params.temperature,
        api_gravity=params.api_gravity,
        gas_gravity=params.gas_gravity,
        method="standing",
    )

    Bo = pvt.Bo

    # Material balance equation (no gas expansion term)
    # Left side: Np * Bo
    left_side = cumulative_oil * Bo

    # Right side components
    # 1. Oil expansion
    oil_expansion = params.N * (Bo - params.Boi)

    # 2. Rock and water expansion
    rock_water_expansion = (
        params.N
        * params.Boi
        * (params.cw * params.Swi + params.cf)
        / (1 - params.Swi)
        * (params.pi - pressure)
    )

    # Solve material balance
    right_side = oil_expansion + rock_water_expansion
    Np_calculated = right_side / Bo if Bo > 0 else 0.0

    return {
        "Np_calculated": Np_calculated,
        "Bo": Bo,
        "oil_expansion": oil_expansion,
        "rock_water_expansion": rock_water_expansion,
    }


def identify_drive_mechanism(
    pressure_history: np.ndarray,
    production_history: np.ndarray,
    water_cut_history: Optional[np.ndarray] = None,
    gor_history: Optional[np.ndarray] = None,
) -> str:
    """Identify reservoir drive mechanism from production data.

    Uses heuristics based on:
    - Pressure decline rate
    - Water cut trends
    - GOR trends
    - Production behavior

    Args:
        pressure_history: Reservoir pressure over time (psi)
        production_history: Oil production rate over time (STB/day)
        water_cut_history: Optional water cut over time (fraction)
        gor_history: Optional GOR over time (SCF/STB)

    Returns:
        Identified drive mechanism: 'solution_gas', 'water_drive', 'gas_cap',
        'combination', or 'unknown'
    """
    if len(pressure_history) < 2:
        return "unknown"

    # Calculate pressure decline rate
    pressure_decline = (pressure_history[0] - pressure_history[-1]) / len(
        pressure_history
    )

    # Analyze water cut trend
    water_drive_indicator = False
    if water_cut_history is not None and len(water_cut_history) > 1:
        water_cut_trend = water_cut_history[-1] - water_cut_history[0]
        if water_cut_trend > 0.1:  # Significant increase
            water_drive_indicator = True

    # Analyze GOR trend
    gas_drive_indicator = False
    if gor_history is not None and len(gor_history) > 1:
        gor_trend = gor_history[-1] - gor_history[0]
        if gor_trend > 100:  # Significant increase in GOR
            gas_drive_indicator = True

    # Identify drive mechanism
    if water_drive_indicator and gas_drive_indicator:
        return "combination"
    elif water_drive_indicator:
        return "water_drive"
    elif gas_drive_indicator:
        return "gas_cap"
    elif pressure_decline > 100:  # Rapid pressure decline
        return "solution_gas"
    else:
        return "solution_gas"  # Default to solution gas drive
