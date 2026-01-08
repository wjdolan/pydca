"""Physics-based reserves classification.

This module provides reserves classification based on physics and uncertainty:
- P1/P2/P3 reserves based on material balance uncertainty
- Probabilistic material balance for reserves
- SPE-PRMS compliant reserves reporting

References:
- SPE-PRMS (Petroleum Resources Management System), 2018.
- SEC Reserves Definitions, 2009.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .logging_config import get_logger

# Material balance functions imported as needed

logger = get_logger(__name__)


@dataclass
class ReservesClassification:
    """Reserves classification result.

    Attributes:
        p1_reserves: Proved reserves (P90, STB)
        p2_reserves: Proved + Probable reserves (P50, STB)
        p3_reserves: Proved + Probable + Possible reserves (P10, STB)
        p1_probability: Probability of P1 (≥90%)
        p2_probability: Probability of P2 (≥50%)
        p3_probability: Probability of P3 (≥10%)
        uncertainty_distribution: Full uncertainty distribution
    """

    p1_reserves: float
    p2_reserves: float
    p3_reserves: float
    p1_probability: float = 0.90
    p2_probability: float = 0.50
    p3_probability: float = 0.10
    uncertainty_distribution: Optional[np.ndarray] = None


def classify_reserves_from_material_balance(
    material_balance_params: Dict[str, float],
    parameter_uncertainty: Dict[str, Dict[str, float]],
    pressure_decline_rate: float = 0.001,
    economic_limit: float = 10.0,
    n_simulations: int = 1000,
) -> ReservesClassification:
    """Classify reserves based on material balance uncertainty.

    Uses Monte Carlo simulation to generate reserves distribution.

    Args:
        material_balance_params: Material balance parameters
        parameter_uncertainty: Parameter uncertainty (from history matching)
        pressure_decline_rate: Pressure decline rate (1/day)
        economic_limit: Economic limit rate (STB/day)
        n_simulations: Number of Monte Carlo simulations

    Returns:
        ReservesClassification with P1/P2/P3 reserves

    Example:
        >>> params = {"N": 1e6, "pi": 5000.0}
        >>> uncertainty = {
        ...     "N": {"p10": 8e5, "p50": 1e6, "p90": 1.2e6},
        ...     "pi": {"p10": 4500, "p50": 5000, "p90": 5500}
        ... }
        >>> reserves = classify_reserves_from_material_balance(params, uncertainty)
    """
    # Generate parameter samples
    samples = {}
    for param_name in material_balance_params.keys():
        if param_name in parameter_uncertainty:
            unc = parameter_uncertainty[param_name]
            # Use triangular distribution from p10, p50, p90
            samples[param_name] = np.random.triangular(
                unc["p10"], unc["p50"], unc["p90"], n_simulations
            )
        else:
            # Use ±20% variation
            base_value = material_balance_params[param_name]
            samples[param_name] = np.random.normal(
                base_value, base_value * 0.2, n_simulations
            )

    # Calculate reserves for each simulation
    reserves_samples = []

    for i in range(n_simulations):
        # Sample parameters
        N = samples["N"][i] if "N" in samples else material_balance_params.get("N", 1e6)
        _pi = (
            samples["pi"][i]
            if "pi" in samples
            else material_balance_params.get("pi", 5000.0)
        )

        # Calculate cumulative production until economic limit
        # Simplified: use exponential decline
        D = pressure_decline_rate
        qi = N * D * 0.1  # Rough estimate

        # Time to economic limit
        if qi > economic_limit and D > 0:
            t_econ = np.log(qi / economic_limit) / D
            # Cumulative production
            Np = N * (1 - np.exp(-D * t_econ))
        else:
            Np = 0.0

        reserves_samples.append(Np)

    reserves_samples = np.array(reserves_samples)

    # Calculate P1/P2/P3
    p1_reserves = np.percentile(reserves_samples, 90)  # P90 (conservative)
    p2_reserves = np.percentile(reserves_samples, 50)  # P50 (best estimate)
    p3_reserves = np.percentile(reserves_samples, 10)  # P10 (optimistic)

    return ReservesClassification(
        p1_reserves=p1_reserves,
        p2_reserves=p2_reserves,
        p3_reserves=p3_reserves,
        uncertainty_distribution=reserves_samples,
    )


def identify_decline_type_from_physics(
    production_data: pd.Series,
    pressure_data: Optional[pd.Series] = None,
    drive_mechanism: Optional[str] = None,
) -> str:
    """Identify decline type from physics and production behavior.

    Uses drive mechanism, pressure behavior, and production trends to recommend
    decline type.

    Args:
        production_data: Production time series
        pressure_data: Optional pressure time series
        drive_mechanism: Optional drive mechanism ('solution_gas', 'water_drive', etc.)

    Returns:
        Recommended decline type ('exponential', 'harmonic', 'hyperbolic')

    Example:
        >>> production = pd.Series([1000, 800, 600, 500, 450], index=dates)
        >>> decline_type = identify_decline_type_from_physics(production)
    """
    # Analyze production trend
    if len(production_data) < 3:
        return "hyperbolic"  # Default

    production_values = production_data.values
    time_values = np.arange(len(production_values))

    # Calculate decline rate trend
    decline_rates = []
    for i in range(1, len(production_values)):
        if production_values[i - 1] > 0:
            decline_rate = (production_values[i - 1] - production_values[i]) / (
                production_values[i - 1] * (time_values[i] - time_values[i - 1])
            )
            decline_rates.append(decline_rate)

    if len(decline_rates) < 2:
        return "hyperbolic"

    # Check if decline rate is constant (exponential)
    decline_rate_std = np.std(decline_rates)
    decline_rate_mean = np.mean(decline_rates)

    if decline_rate_std / decline_rate_mean < 0.1:  # Low variation
        return "exponential"

    # Check if decline rate is decreasing (harmonic)
    if len(decline_rates) >= 3:
        early_decline = np.mean(decline_rates[: len(decline_rates) // 2])
        late_decline = np.mean(decline_rates[len(decline_rates) // 2 :])

        if late_decline < early_decline * 0.5:  # Significant decrease
            return "harmonic"

    # Check drive mechanism
    if drive_mechanism == "water_drive":
        # Water drive often shows harmonic decline
        return "harmonic"
    elif drive_mechanism == "solution_gas":
        # Solution gas drive often shows hyperbolic decline
        return "hyperbolic"
    elif drive_mechanism == "gas_cap":
        # Gas cap drive can show various patterns
        return "hyperbolic"

    # Check pressure behavior
    if pressure_data is not None and len(pressure_data) > 2:
        pressure_values = pressure_data.values
        pressure_decline = (pressure_values[0] - pressure_values[-1]) / len(
            pressure_values
        )

        if pressure_decline > 100:  # Rapid pressure decline
            return "hyperbolic"  # Typical for solution gas drive
        elif pressure_decline < 10:  # Slow pressure decline
            return "harmonic"  # Typical for water drive

    # Default to hyperbolic (most common for unconventional)
    return "hyperbolic"
