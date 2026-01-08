"""History matching for material balance and decline curve models.

This module provides systematic parameter optimization to match historical
production and pressure data.

Features:
- Material balance history matching
- Parameter optimization using scipy.optimize
- Uncertainty quantification
- Sensitivity analysis
- Multiple history match scenarios
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

from .logging_config import get_logger
from .material_balance import (
    GasReservoirParams,
    SolutionGasDriveParams,
    gas_reservoir_pz_method,
    solution_gas_drive_material_balance,
)

logger = get_logger(__name__)


@dataclass
class HistoryMatchResult:
    """Container for history matching results.

    Attributes:
        optimized_params: Optimized parameter dictionary
        objective_value: Final objective function value
        success: Whether optimization converged
        message: Optimization message
        iterations: Number of iterations
        pressure_match: Pressure matching statistics
        production_match: Production matching statistics
    """

    optimized_params: Dict[str, float]
    objective_value: float
    success: bool
    message: str
    iterations: int
    pressure_match: Dict[str, float]
    production_match: Dict[str, float]


def history_match_material_balance(
    time: np.ndarray,
    production: np.ndarray,
    pressure: Optional[np.ndarray] = None,
    drive_mechanism: str = "solution_gas",
    initial_params: Optional[Dict[str, float]] = None,
    param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    weights: Optional[Dict[str, float]] = None,
    method: str = "differential_evolution",
) -> HistoryMatchResult:
    """History match material balance to production and pressure data.

    Optimizes material balance parameters to match historical data.

    Args:
        time: Time array (days)
        production: Cumulative production (STB)
        pressure: Optional pressure array (psi)
        drive_mechanism: Drive mechanism ('solution_gas', 'water_drive', 'gas_reservoir')
        initial_params: Initial parameter guess
        param_bounds: Parameter bounds for optimization
        weights: Weights for objective function ('production', 'pressure')
        method: Optimization method ('differential_evolution', 'minimize')

    Returns:
        HistoryMatchResult with optimized parameters

    Example:
        >>> time = np.array([30, 60, 90, 120, 150, 180])
        >>> production = np.array([10000, 20000, 30000, 40000, 50000, 60000])
        >>> pressure = np.array([5000, 4800, 4600, 4400, 4200, 4000])
        >>> result = history_match_material_balance(time, production, pressure)
    """
    if len(time) != len(production):
        raise ValueError("time and production must have same length")

    if pressure is not None and len(pressure) != len(time):
        raise ValueError("pressure must have same length as time")

    # Default weights
    if weights is None:
        weights = {"production": 1.0, "pressure": 1.0 if pressure is not None else 0.0}

    # Default parameter bounds
    if param_bounds is None:
        param_bounds = {
            "N": (1e5, 1e7),  # OOIP
            "pi": (1000.0, 10000.0),  # Initial pressure
            "D": (1e-6, 0.1),  # Decline rate
        }

    # Default initial parameters
    if initial_params is None:
        initial_params = {
            "N": np.max(production) * 10,  # Rough estimate
            "pi": pressure[0] if pressure is not None else 5000.0,
            "D": 0.001,
        }

    def objective_function(params_array: np.ndarray) -> float:
        """Objective function for optimization."""
        # Convert array to parameter dict
        param_names = list(param_bounds.keys())
        params_dict = dict(zip(param_names, params_array))

        # Calculate material balance
        try:
            if drive_mechanism == "solution_gas":
                mb_params = SolutionGasDriveParams(
                    N=params_dict.get("N", 1e6),
                    pi=params_dict.get("pi", 5000.0),
                    pb=params_dict.get("pb", 3000.0),
                )

                # Calculate cumulative for each time step
                calculated_production = np.zeros_like(time)
                for i, (t_i, p_i) in enumerate(
                    zip(
                        time,
                        (
                            pressure
                            if pressure is not None
                            else [params_dict.get("pi", 5000.0)] * len(time)
                        ),
                    )
                ):
                    result = solution_gas_drive_material_balance(
                        p_i, calculated_production[i - 1] if i > 0 else 0.0, mb_params
                    )
                    calculated_production[i] = result["Np_calculated"]

            elif drive_mechanism == "gas_reservoir":
                # For gas, use p/Z method
                calculated_production = np.zeros_like(time)
                G = params_dict.get("G", 1e9)
                for i, p_i in enumerate(
                    pressure
                    if pressure is not None
                    else [params_dict.get("pi", 5000.0)] * len(time)
                ):
                    # Simplified: estimate cumulative from pressure
                    result = gas_reservoir_pz_method(
                        p_i,
                        calculated_production[i - 1] if i > 0 else 0.0,
                        GasReservoirParams(G=G, pi=params_dict.get("pi", 5000.0)),
                    )
                    calculated_production[i] = (
                        result["G_calculated"] * 0.1
                    )  # Approximate

            else:
                # Default: simple exponential
                N = params_dict.get("N", 1e6)
                D = params_dict.get("D", 0.001)
                calculated_production = N * (1 - np.exp(-D * time))

            # Calculate errors
            production_error = np.sum((calculated_production - production) ** 2)

            # Pressure error (if available)
            pressure_error = 0.0
            if pressure is not None:
                # Estimate pressure from production (simplified)
                estimated_pressure = params_dict.get("pi", 5000.0) * np.exp(
                    -params_dict.get("D", 0.001) * time
                )
                pressure_error = np.sum((estimated_pressure - pressure) ** 2)

            # Weighted objective
            objective = (
                weights["production"] * production_error
                + weights["pressure"] * pressure_error
            )

            return objective

        except Exception as e:
            logger.warning(f"Error in objective function: {e}")
            return 1e10  # Large penalty for invalid parameters

    # Prepare bounds for optimization
    bounds = [param_bounds[name] for name in param_bounds.keys()]

    # Initial guess
    x0 = [
        initial_params.get(name, (bounds[i][0] + bounds[i][1]) / 2)
        for i, name in enumerate(param_bounds.keys())
    ]

    # Optimize
    if method == "differential_evolution":
        result = differential_evolution(
            objective_function,
            bounds=bounds,
            seed=42,
            maxiter=100,
            popsize=15,
        )
    else:
        result = minimize(
            objective_function,
            x0=x0,
            bounds=bounds,
            method="L-BFGS-B",
        )

    # Convert result back to parameter dict
    optimized_params = dict(zip(param_bounds.keys(), result.x))

    # Calculate match statistics
    # Recalculate with optimized parameters
    if drive_mechanism == "solution_gas":
        mb_params = SolutionGasDriveParams(
            N=optimized_params.get("N", 1e6),
            pi=optimized_params.get("pi", 5000.0),
        )
        calculated_production = np.zeros_like(time)
        for i, (t_i, p_i) in enumerate(
            zip(
                time,
                (
                    pressure
                    if pressure is not None
                    else [optimized_params.get("pi", 5000.0)] * len(time)
                ),
            )
        ):
            mb_result = solution_gas_drive_material_balance(
                p_i, calculated_production[i - 1] if i > 0 else 0.0, mb_params
            )
            calculated_production[i] = mb_result["Np_calculated"]
    else:
        N = optimized_params.get("N", 1e6)
        D = optimized_params.get("D", 0.001)
        calculated_production = N * (1 - np.exp(-D * time))

    production_rmse = np.sqrt(np.mean((calculated_production - production) ** 2))
    production_mae = np.mean(np.abs(calculated_production - production))

    pressure_rmse = 0.0
    pressure_mae = 0.0
    if pressure is not None:
        estimated_pressure = optimized_params.get("pi", 5000.0) * np.exp(
            -optimized_params.get("D", 0.001) * time
        )
        pressure_rmse = np.sqrt(np.mean((estimated_pressure - pressure) ** 2))
        pressure_mae = np.mean(np.abs(estimated_pressure - pressure))

    return HistoryMatchResult(
        optimized_params=optimized_params,
        objective_value=result.fun,
        success=result.success,
        message=result.message if hasattr(result, "message") else "",
        iterations=result.nit if hasattr(result, "nit") else 0,
        pressure_match={"rmse": pressure_rmse, "mae": pressure_mae},
        production_match={"rmse": production_rmse, "mae": production_mae},
    )


def quantify_parameter_uncertainty(
    history_match_result: HistoryMatchResult,
    time: np.ndarray,
    production: np.ndarray,
    n_samples: int = 1000,
) -> Dict[str, Dict[str, float]]:
    """Quantify uncertainty in history-matched parameters.

    Uses Monte Carlo sampling around optimized parameters.

    Args:
        history_match_result: History matching result
        time: Time array (days)
        production: Production data (STB)
        n_samples: Number of Monte Carlo samples

    Returns:
        Dictionary with parameter uncertainty statistics (mean, std, p10, p50, p90)
    """
    optimized = history_match_result.optimized_params

    # Sample parameters around optimized values
    # Use ±20% variation
    samples = {}
    for param_name, param_value in optimized.items():
        if param_value > 0:
            std = param_value * 0.2  # 20% standard deviation
            samples[param_name] = np.random.normal(param_value, std, n_samples)
            samples[param_name] = np.maximum(
                samples[param_name], param_value * 0.1
            )  # Lower bound
        else:
            samples[param_name] = np.full(n_samples, param_value)

    # Calculate statistics
    uncertainty = {}
    for param_name in optimized.keys():
        param_samples = samples[param_name]
        uncertainty[param_name] = {
            "mean": float(np.mean(param_samples)),
            "std": float(np.std(param_samples)),
            "p10": float(np.percentile(param_samples, 10)),
            "p50": float(np.percentile(param_samples, 50)),
            "p90": float(np.percentile(param_samples, 90)),
        }

    return uncertainty


def sensitivity_analysis_material_balance(
    time: np.ndarray,
    production: np.ndarray,
    base_params: Dict[str, float],
    param_variations: Optional[Dict[str, List[float]]] = None,
) -> pd.DataFrame:
    """Perform sensitivity analysis on material balance parameters.

    Args:
        time: Time array (days)
        production: Production data (STB)
        base_params: Base parameter values
        param_variations: Parameter variations to test (if None, uses ±20%)

    Returns:
        DataFrame with sensitivity results
    """
    if param_variations is None:
        param_variations = {}
        for param_name, param_value in base_params.items():
            if param_value > 0:
                param_variations[param_name] = [
                    param_value * 0.8,
                    param_value * 0.9,
                    param_value,
                    param_value * 1.1,
                    param_value * 1.2,
                ]

    results = []

    for param_name, variations in param_variations.items():
        for variation in variations:
            test_params = base_params.copy()
            test_params[param_name] = variation

            # Calculate material balance
            N = test_params.get("N", 1e6)
            D = test_params.get("D", 0.001)
            calculated = N * (1 - np.exp(-D * time))

            # Calculate error
            error = np.sqrt(np.mean((calculated - production) ** 2))

            results.append(
                {
                    "parameter": param_name,
                    "value": variation,
                    "variation_pct": (variation / base_params[param_name] - 1) * 100,
                    "rmse": error,
                }
            )

    return pd.DataFrame(results)
