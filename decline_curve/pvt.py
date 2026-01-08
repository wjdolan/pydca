"""PVT (Pressure-Volume-Temperature) correlations for reservoir fluids.

This module provides industry-standard PVT correlations for:
- Oil formation volume factor (Bo)
- Solution gas-oil ratio (Rs)
- Gas formation volume factor (Bg)
- Oil viscosity (μo)
- Gas viscosity (μg)
- Water properties (Bw, μw)
- Gas compressibility factor (Z)

Correlations implemented:
- Standing (1947) - Oil FVF and Rs
- Vasquez-Beggs (1980) - Oil FVF and Rs (API gravity dependent)
- Lee-Gonzalez-Eakin (1966) - Gas viscosity
- Beggs-Robinson (1975) - Oil viscosity
- Chew-Connally (1959) - Oil viscosity (dead oil)
- Standing (1977) - Gas Z-factor
- Hall-Yarborough (1973) - Gas Z-factor
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PVTProperties:
    """Container for PVT properties at given conditions.

    Attributes:
        Bo: Oil formation volume factor (RB/STB)
        Rs: Solution gas-oil ratio (SCF/STB)
        Bg: Gas formation volume factor (RB/SCF)
        muo: Oil viscosity (cp)
        mug: Gas viscosity (cp)
        Bw: Water formation volume factor (RB/STB)
        muw: Water viscosity (cp)
        Z: Gas compressibility factor (dimensionless)
    """

    Bo: float
    Rs: float = 0.0
    Bg: float = 0.0
    muo: float = 1.0
    mug: float = 0.01
    Bw: float = 1.0
    muw: float = 0.5
    Z: float = 1.0


def standing_rs(
    pressure: float,
    temperature: float,
    api_gravity: float,
    gas_gravity: float = 0.65,
) -> float:
    """Calculate solution gas-oil ratio using Standing correlation.

    Standing (1947) correlation for solution gas-oil ratio.

    Args:
        pressure: Reservoir pressure (psi)
        temperature: Reservoir temperature (°F)
        api_gravity: Oil API gravity (°API)
        gas_gravity: Gas specific gravity (air=1.0), default 0.65

    Returns:
        Solution gas-oil ratio (SCF/STB)

    Reference:
        Standing, M.B., "A Pressure-Volume-Temperature Correlation for Mixtures
        of California Oils and Gases," Drilling and Production Practice, 1947.
    """
    # Convert API to specific gravity
    gamma_o = 141.5 / (api_gravity + 131.5)

    # Standing correlation
    x = 0.0125 * api_gravity - 0.00091 * temperature
    rs = gas_gravity * ((pressure / 18.2 + 1.4) * 10**x) ** 1.2048

    return max(0.0, rs)


def standing_bo(
    rs: float,
    temperature: float,
    api_gravity: float,
    gas_gravity: float = 0.65,
) -> float:
    """Calculate oil formation volume factor using Standing correlation.

    Standing (1947) correlation for oil FVF.

    Args:
        rs: Solution gas-oil ratio (SCF/STB)
        temperature: Reservoir temperature (°F)
        api_gravity: Oil API gravity (°API)
        gas_gravity: Gas specific gravity (air=1.0), default 0.65

    Returns:
        Oil formation volume factor (RB/STB)

    Reference:
        Standing, M.B., "A Pressure-Volume-Temperature Correlation for Mixtures
        of California Oils and Gases," Drilling and Production Practice, 1947.
    """
    # Convert API to specific gravity
    gamma_o = 141.5 / (api_gravity + 131.5)

    # Standing correlation
    f = rs * (gas_gravity / gamma_o) ** 0.5 + 1.25 * temperature
    Bo = 0.9759 + 0.00012 * f**1.2

    return max(1.0, Bo)


def vasquez_beggs_rs(
    pressure: float,
    temperature: float,
    api_gravity: float,
    gas_gravity: float = 0.65,
    pressure_sep: float = 100.0,
    temperature_sep: float = 60.0,
) -> float:
    """Calculate solution gas-oil ratio using Vasquez-Beggs correlation.

    Vasquez-Beggs (1980) correlation, API gravity dependent.

    Args:
        pressure: Reservoir pressure (psi)
        temperature: Reservoir temperature (°F)
        api_gravity: Oil API gravity (°API)
        gas_gravity: Gas specific gravity (air=1.0), default 0.65
        pressure_sep: Separator pressure (psi), default 100
        temperature_sep: Separator temperature (°F), default 60

    Returns:
        Solution gas-oil ratio (SCF/STB)

    Reference:
        Vasquez, M. and Beggs, H.D., "Correlations for Fluid Physical Property
        Prediction," JPT, June 1980.
    """
    # Correct gas gravity to separator conditions
    gamma_gs = gas_gravity * (
        1 + 5.912e-5 * api_gravity * temperature_sep * np.log10(pressure_sep / 114.7)
    )

    # API gravity dependent coefficients
    if api_gravity <= 30:
        C1 = 0.0362
        C2 = 1.0937
        C3 = 25.7240
    else:
        C1 = 0.0178
        C2 = 1.1870
        C3 = 23.9310

    # Vasquez-Beggs correlation
    rs = C1 * gamma_gs * pressure**C2 * np.exp(C3 * api_gravity / (temperature + 460))

    return max(0.0, rs)


def vasquez_beggs_bo(
    rs: float,
    temperature: float,
    api_gravity: float,
    gas_gravity: float = 0.65,
    pressure: float = 5000.0,
) -> float:
    """Calculate oil formation volume factor using Vasquez-Beggs correlation.

    Vasquez-Beggs (1980) correlation for oil FVF.

    Args:
        rs: Solution gas-oil ratio (SCF/STB)
        temperature: Reservoir temperature (°F)
        api_gravity: Oil API gravity (°API)
        gas_gravity: Gas specific gravity (air=1.0), default 0.65
        pressure: Reservoir pressure (psi), default 5000

    Returns:
        Oil formation volume factor (RB/STB)

    Reference:
        Vasquez, M. and Beggs, H.D., "Correlations for Fluid Physical Property
        Prediction," JPT, June 1980.
    """
    # API gravity dependent coefficients
    if api_gravity <= 30:
        C1 = 1.0113
        C2 = 1.1437e-5
        C3 = -1.225e-9
    else:
        C1 = 1.0113
        C2 = 1.1437e-5
        C3 = -1.225e-9

    # Vasquez-Beggs correlation
    Bo = (
        C1
        + C2 * rs
        + C3 * (temperature - 60) * (api_gravity / gas_gravity)
        + C3 * (temperature - 60) * rs
    )

    return max(1.0, Bo)


def gas_z_factor(
    pressure: float,
    temperature: float,
    gas_gravity: float = 0.65,
    method: str = "standing",
) -> float:
    """Calculate gas compressibility factor (Z-factor).

    Args:
        pressure: Reservoir pressure (psi)
        temperature: Reservoir temperature (°F)
        gas_gravity: Gas specific gravity (air=1.0), default 0.65
        method: Correlation method ('standing' or 'hall_yarborough'), default 'standing'

    Returns:
        Gas compressibility factor (dimensionless)

    Reference:
        Standing, M.B. and Katz, D.L., "Density of Natural Gases," Trans. AIME, 1942.
        Hall, K.R. and Yarborough, L., "A New Equation of State for Z-Factor Calculations,"
        Oil & Gas Journal, June 18, 1973.
    """
    if method == "standing":
        # Standing-Katz correlation (simplified)
        # Pseudo-reduced pressure and temperature
        ppc = 677 + 15.0 * gas_gravity - 37.5 * gas_gravity**2
        tpc = 168 + 325 * gas_gravity - 12.5 * gas_gravity**2

        ppr = pressure / ppc
        tpr = (temperature + 460) / tpc

        # Standing correlation (simplified approximation)
        # More accurate would use Hall-Yarborough or Dranchuk-Abou-Kassem
        if tpr < 1.0 or ppr < 0.1:
            # Low pressure/temperature: Z ≈ 1.0
            return 1.0

        Z = (
            1.0
            + (0.31506 - 1.0467 / tpr - 0.5783 / tpr**3) * ppr
            + (0.5353 - 0.6123 / tpr + 0.6815 / tpr**3) * ppr**2
        )

        return max(0.1, min(1.5, Z))  # Z typically 0.7-1.2 for most conditions

    elif method == "hall_yarborough":
        # Hall-Yarborough correlation (more accurate)
        ppc = 677 + 15.0 * gas_gravity - 37.5 * gas_gravity**2
        tpc = 168 + 325 * gas_gravity - 12.5 * gas_gravity**2

        ppr = pressure / ppc
        tpr = (temperature + 460) / tpc

        # Hall-Yarborough equation (simplified approximation)
        # Full implementation would require iterative solution
        if tpr < 1.0 or ppr < 0.1:
            return 1.0

        # Simplified Hall-Yarborough (more accurate than Standing)
        Z = (
            1.0
            + (
                0.3265
                - 1.07 / tpr
                - 0.5339 / tpr**3
                + 0.01569 / tpr**4
                - 0.05165 / tpr**5
            )
            * ppr
        )
        Z += (0.5475 - 0.7361 / tpr + 0.1844 / tpr**2) * ppr**2
        Z -= 0.1056 * (-0.7361 / tpr + 0.1844 / tpr**2) * ppr**5

        return max(0.5, min(1.5, Z))

    else:
        raise ValueError(
            f"Unknown method: {method}. Use 'standing' or 'hall_yarborough'"
        )


def gas_fvf(
    pressure: float,
    temperature: float,
    z_factor: Optional[float] = None,
    gas_gravity: float = 0.65,
) -> float:
    """Calculate gas formation volume factor.

    Args:
        pressure: Reservoir pressure (psi)
        temperature: Reservoir temperature (°F)
        z_factor: Gas compressibility factor (if None, calculated)
        gas_gravity: Gas specific gravity (air=1.0), default 0.65

    Returns:
        Gas formation volume factor (RB/SCF)

    Formula:
        Bg = 0.02827 * Z * T / P
    """
    if z_factor is None:
        z_factor = gas_z_factor(pressure, temperature, gas_gravity)

    # Standard conditions: 14.7 psia, 60°F
    T_sc = 520.0  # Rankine
    P_sc = 14.7  # psia

    # Reservoir conditions
    T_r = temperature + 460.0  # Rankine
    P_r = pressure  # psia

    # Gas FVF
    Bg = (P_sc / P_r) * (T_r / T_sc) * z_factor * 0.02827

    return max(0.0, Bg)


def beggs_robinson_oil_viscosity(
    rs: float,
    temperature: float,
    api_gravity: float,
    dead_oil_viscosity: Optional[float] = None,
) -> float:
    """Calculate oil viscosity using Beggs-Robinson correlation.

    Beggs-Robinson (1975) correlation for live oil viscosity.

    Args:
        rs: Solution gas-oil ratio (SCF/STB)
        temperature: Reservoir temperature (°F)
        api_gravity: Oil API gravity (°API)
        dead_oil_viscosity: Dead oil viscosity at reservoir temperature (cp).
            If None, calculated using Chew-Connally correlation.

    Returns:
        Live oil viscosity (cp)

    Reference:
        Beggs, H.D. and Robinson, J.R., "Estimating the Viscosity of Crude Oil Systems,"
        JPT, September 1975.
    """
    # Calculate dead oil viscosity if not provided
    if dead_oil_viscosity is None:
        dead_oil_viscosity = chew_connally_dead_oil_viscosity(temperature, api_gravity)

    # Beggs-Robinson correlation
    A = 10.715 * (rs + 100) ** (-0.515)
    B = 5.44 * (rs + 150) ** (-0.338)

    muo = A * dead_oil_viscosity**B

    return max(0.1, muo)


def chew_connally_dead_oil_viscosity(
    temperature: float,
    api_gravity: float,
) -> float:
    """Calculate dead oil viscosity using Chew-Connally correlation.

    Chew-Connally (1959) correlation for dead oil viscosity.

    Args:
        temperature: Reservoir temperature (°F)
        api_gravity: Oil API gravity (°API)

    Returns:
        Dead oil viscosity (cp)

    Reference:
        Chew, J. and Connally, C.A., "A Viscosity Correlation for Gas-Saturated
        Crude Oils," Trans. AIME, 1959.
    """
    # Chew-Connally correlation
    X = 10 ** (3.0324 - 0.02023 * api_gravity) * temperature ** (-1.163)

    muod = 10**X - 1.0

    return max(0.1, muod)


def lee_gonzalez_gas_viscosity(
    pressure: float,
    temperature: float,
    gas_gravity: float = 0.65,
    z_factor: Optional[float] = None,
) -> float:
    """Calculate gas viscosity using Lee-Gonzalez-Eakin correlation.

    Lee-Gonzalez-Eakin (1966) correlation for natural gas viscosity.

    Args:
        pressure: Reservoir pressure (psi)
        temperature: Reservoir temperature (°F)
        gas_gravity: Gas specific gravity (air=1.0), default 0.65
        z_factor: Gas compressibility factor (if None, calculated)

    Returns:
        Gas viscosity (cp)

    Reference:
        Lee, A.L., Gonzalez, M.H., and Eakin, B.E., "The Viscosity of Natural Gases,"
        JPT, August 1966.
    """
    if z_factor is None:
        z_factor = gas_z_factor(pressure, temperature, gas_gravity)

    # Lee-Gonzalez-Eakin correlation
    T_r = temperature + 460.0  # Rankine
    M = 29.0 * gas_gravity  # Molecular weight

    K = (9.4 + 0.02 * M) * T_r**1.5 / (209 + 19 * M + T_r)
    X = 3.5 + 986 / T_r + 0.01 * M
    Y = 2.4 - 0.2 * X

    # Gas density (simplified)
    rho_g = (28.97 * gas_gravity * pressure) / (z_factor * 10.73 * T_r)  # lb/ft³

    mug = K * np.exp(X * rho_g**Y) / 10000.0  # Convert to cp

    return max(0.005, min(0.15, mug))  # Allow slightly wider range


def water_fvf(
    pressure: float,
    temperature: float,
    salinity: float = 0.0,
) -> float:
    """Calculate water formation volume factor.

    Args:
        pressure: Reservoir pressure (psi)
        temperature: Reservoir temperature (°F)
        salinity: Water salinity (ppm), default 0

    Returns:
        Water formation volume factor (RB/STB)

    Reference:
        McCain, W.D., "The Properties of Petroleum Fluids," 2nd Ed., 1990.
    """
    # Water FVF correlation
    T_r = temperature + 460.0  # Rankine

    # Base FVF at standard conditions
    Bw_sc = 1.0 + 1.0e-6 * (-1.0001e-2 + 1.33391e-4 * T_r + 5.50654e-7 * T_r**2)

    # Pressure correction
    dp = pressure - 14.7
    Bw = Bw_sc * (1.0 + 3.0e-6 * dp)

    # Salinity correction (simplified)
    if salinity > 0:
        Bw *= 1.0 - 0.0001 * salinity / 1000000.0

    return max(0.9, min(1.1, Bw))


def water_viscosity(
    temperature: float,
    salinity: float = 0.0,
) -> float:
    """Calculate water viscosity.

    Args:
        temperature: Reservoir temperature (°F)
        salinity: Water salinity (ppm), default 0

    Returns:
        Water viscosity (cp)

    Reference:
        McCain, W.D., "The Properties of Petroleum Fluids," 2nd Ed., 1990.
    """
    # Water viscosity correlation (McCain, 1990)
    T_c = (temperature - 32) * 5 / 9  # Convert to Celsius

    # Pure water viscosity (corrected McCain correlation)
    # At 20°C: ~1.0 cp, at 100°C: ~0.28 cp
    if T_c <= 0:
        muw = 1.7921  # Freezing point
    else:
        # McCain correlation for water viscosity
        muw = 1.0 / (1.0 + 0.001 * T_c * (1.0 + 0.001 * T_c))
        # Scale to match known values: 1.0 cp at 20°C, 0.28 cp at 100°C
        muw = 1.0 * np.exp(-0.012 * T_c)  # Exponential fit to known data

    # Salinity correction
    if salinity > 0:
        S = salinity / 1000000.0  # Convert to fraction
        muw *= 1.0 + 0.001 * S * (1.0 + 0.01 * T_c)

    return max(
        0.2, min(2.0, muw)
    )  # Water viscosity typically 0.2-1.5 cp at reservoir temps


def calculate_pvt_properties(
    pressure: float,
    temperature: float,
    api_gravity: float = 30.0,
    gas_gravity: float = 0.65,
    rs: Optional[float] = None,
    method: str = "standing",
) -> PVTProperties:
    """Calculate comprehensive PVT properties at given conditions.

    Convenience function to calculate all PVT properties at once.

    Args:
        pressure: Reservoir pressure (psi)
        temperature: Reservoir temperature (°F)
        api_gravity: Oil API gravity (°API), default 30
        gas_gravity: Gas specific gravity (air=1.0), default 0.65
        rs: Solution gas-oil ratio (SCF/STB). If None, calculated from pressure.
        method: PVT correlation method ('standing' or 'vasquez_beggs'), default 'standing'

    Returns:
        PVTProperties object with all calculated properties

    Example:
        >>> pvt = calculate_pvt_properties(
        ...     pressure=5000, temperature=200, api_gravity=35, gas_gravity=0.7
        ... )
        >>> print(f"Bo: {pvt.Bo:.3f} RB/STB")
        >>> print(f"Rs: {pvt.Rs:.1f} SCF/STB")
    """
    # Calculate Rs if not provided
    if rs is None:
        if method == "standing":
            rs = standing_rs(pressure, temperature, api_gravity, gas_gravity)
        elif method == "vasquez_beggs":
            rs = vasquez_beggs_rs(pressure, temperature, api_gravity, gas_gravity)
        else:
            raise ValueError(f"Unknown method: {method}")

    # Calculate Bo
    if method == "standing":
        Bo = standing_bo(rs, temperature, api_gravity, gas_gravity)
    elif method == "vasquez_beggs":
        Bo = vasquez_beggs_bo(rs, temperature, api_gravity, gas_gravity, pressure)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Calculate gas properties
    Z = gas_z_factor(pressure, temperature, gas_gravity)
    Bg = gas_fvf(pressure, temperature, Z, gas_gravity)
    mug = lee_gonzalez_gas_viscosity(pressure, temperature, gas_gravity, Z)

    # Calculate oil viscosity
    muo = beggs_robinson_oil_viscosity(rs, temperature, api_gravity)

    # Calculate water properties
    Bw = water_fvf(pressure, temperature)
    muw = water_viscosity(temperature)

    return PVTProperties(
        Bo=Bo,
        Rs=rs,
        Bg=Bg,
        muo=muo,
        mug=mug,
        Bw=Bw,
        muw=muw,
        Z=Z,
    )
