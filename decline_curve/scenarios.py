"""Price scenario analysis for economic evaluation.

This module provides tools for running multiple price scenarios (low, base, high)
and comparing economic results across scenarios.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .economics import economic_metrics
from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PriceScenario:
    """Price scenario definition.

    Attributes:
        name: Scenario name (e.g., 'low', 'base', 'high')
        oil_price: Oil price ($/bbl)
        gas_price: Gas price ($/mcf)
        water_price: Water disposal cost ($/bbl, typically negative)
        opex: Operating cost per unit ($/bbl or $/mcf)
        fixed_opex: Fixed operating expenses ($/month)
        discount_rate: Annual discount rate (fraction)
    """

    name: str
    oil_price: float
    gas_price: Optional[float] = None
    water_price: Optional[float] = None
    opex: float = 15.0
    fixed_opex: float = 5000.0
    discount_rate: float = 0.10

    def __post_init__(self):
        """Validate scenario parameters."""
        if self.oil_price < 0:
            raise ValueError("Oil price must be non-negative")
        if self.gas_price is not None and self.gas_price < 0:
            raise ValueError("Gas price must be non-negative")
        if self.opex < 0:
            raise ValueError("Operating cost must be non-negative")


@dataclass
class ScenarioResult:
    """Result from a single price scenario.

    Attributes:
        scenario_name: Name of the scenario
        npv: Net present value ($)
        cash_flow: Monthly cash flow array
        payback_month: Payback period (months, -1 if never)
        cumulative_cash_flow: Cumulative cash flow array
        total_revenue: Total revenue ($)
        total_opex: Total operating expenses ($)
    """

    scenario_name: str
    npv: float
    cash_flow: np.ndarray
    payback_month: int
    cumulative_cash_flow: np.ndarray
    total_revenue: float
    total_opex: float


def run_price_scenarios(
    production: pd.Series | np.ndarray,
    scenarios: list[PriceScenario],
    phase: str = "oil",
) -> pd.DataFrame:
    """
    Run multiple price scenarios on a production forecast.

    Args:
        production: Production forecast (monthly rates)
        scenarios: List of PriceScenario objects
        phase: Production phase ('oil', 'gas', 'water')

    Returns:
        DataFrame with scenario results including NPV, payback, and cash flow metrics

    Example:
        >>> from decline_curve.scenarios import PriceScenario, run_price_scenarios
        >>> scenarios = [
        ...     PriceScenario('low', oil_price=50.0, opex=15.0),
        ...     PriceScenario('base', oil_price=70.0, opex=15.0),
        ...     PriceScenario('high', oil_price=90.0, opex=15.0),
        ... ]
        >>> results = run_price_scenarios(production_forecast, scenarios)
        >>> print(results[['scenario', 'npv', 'payback_month']])
    """
    production_array = (
        production.values if isinstance(production, pd.Series) else production
    )

    phase_price_map = {
        "oil": lambda s: s.oil_price,
        "gas": lambda s: s.gas_price if s.gas_price is not None else 0.0,
        "water": lambda s: s.water_price if s.water_price is not None else 0.0,
    }

    if phase not in phase_price_map:
        raise ValueError(
            f"Unknown phase: {phase}. Must be one of: {list(phase_price_map.keys())}"
        )

    get_price = phase_price_map[phase]
    results = []

    for scenario in scenarios:
        price = get_price(scenario)

        # Calculate economics
        econ = economic_metrics(
            production_array,
            price=price,
            opex=scenario.opex,
            discount_rate=scenario.discount_rate,
        )

        # Calculate additional metrics
        total_revenue = np.sum(production_array * price)
        total_opex = np.sum(production_array * scenario.opex) + (
            len(production_array) * scenario.fixed_opex
        )
        cumulative_cf = np.cumsum(econ["cash_flow"])

        result = ScenarioResult(
            scenario_name=scenario.name,
            npv=econ["npv"],
            cash_flow=econ["cash_flow"],
            payback_month=econ["payback_month"],
            cumulative_cash_flow=cumulative_cf,
            total_revenue=total_revenue,
            total_opex=total_opex,
        )

        results.append(result)

    # Convert to DataFrame
    df = pd.DataFrame(
        [
            {
                "scenario": r.scenario_name,
                "npv": r.npv,
                "payback_month": r.payback_month,
                "total_revenue": r.total_revenue,
                "total_opex": r.total_opex,
                "net_cash_flow": r.total_revenue - r.total_opex,
                "irr_approx": _estimate_irr(r.cash_flow, scenario.discount_rate),
            }
            for r, scenario in zip(results, scenarios)
        ]
    )

    return df


def run_multi_phase_scenarios(
    oil_production: pd.Series | np.ndarray,
    scenarios: list[PriceScenario],
    gas_production: Optional[pd.Series | np.ndarray] = None,
    water_production: Optional[pd.Series | np.ndarray] = None,
) -> pd.DataFrame:
    """
    Run price scenarios on multi-phase production (oil, gas, water).

    Args:
        oil_production: Oil production forecast
        scenarios: List of PriceScenario objects
        gas_production: Optional gas production forecast
        water_production: Optional water production forecast

    Returns:
        DataFrame with scenario results for multi-phase production

    Example:
        >>> scenarios = [
        ...     PriceScenario('low', oil_price=50.0, gas_price=2.5, water_price=-2.0),
        ...     PriceScenario('base', oil_price=70.0, gas_price=3.0, water_price=-2.0),
        ...     PriceScenario('high', oil_price=90.0, gas_price=3.5, water_price=-2.0),
        ... ]
        >>> results = run_multi_phase_scenarios(
        ...     oil_prod, scenarios, gas_prod, water_prod
        ... )
    """
    oil_array = (
        oil_production.values
        if isinstance(oil_production, pd.Series)
        else oil_production
    )
    n_periods = len(oil_production)

    gas_array = (
        (
            gas_production.values
            if isinstance(gas_production, pd.Series)
            else gas_production
        )
        if gas_production is not None
        else np.zeros(n_periods)
    )

    water_array = (
        (
            water_production.values
            if isinstance(water_production, pd.Series)
            else water_production
        )
        if water_production is not None
        else np.zeros(n_periods)
    )

    results = []

    for scenario in scenarios:
        # Calculate cash flow for each phase
        oil_revenue = oil_array * scenario.oil_price
        gas_revenue = gas_array * (
            scenario.gas_price if scenario.gas_price is not None else 0.0
        )
        water_cost = water_array * (
            scenario.water_price if scenario.water_price is not None else 0.0
        )

        # Operating costs
        oil_opex = oil_array * scenario.opex
        gas_opex = gas_array * (scenario.opex * 0.1)  # Assume gas opex is 10% of oil
        fixed_opex = np.full(n_periods, scenario.fixed_opex)

        # Total cash flow
        cash_flow = (
            oil_revenue + gas_revenue + water_cost - oil_opex - gas_opex - fixed_opex
        )

        # Calculate NPV
        monthly_rate = scenario.discount_rate / 12
        npv_val = np.sum(
            cash_flow / ((1 + monthly_rate) ** np.arange(1, n_periods + 1))
        )

        # Payback
        cum_cf = np.cumsum(cash_flow)
        payback_month = int(np.argmax(cum_cf > 0)) if np.any(cum_cf > 0) else -1

        # Totals
        total_revenue = np.sum(oil_revenue + gas_revenue + water_cost)
        total_opex = np.sum(oil_opex + gas_opex + fixed_opex)

        result = ScenarioResult(
            scenario_name=scenario.name,
            npv=npv_val,
            cash_flow=cash_flow,
            payback_month=payback_month,
            cumulative_cash_flow=cum_cf,
            total_revenue=total_revenue,
            total_opex=total_opex,
        )

        results.append(result)

    # Convert to DataFrame using vectorized operations
    scenario_names = np.array([r.scenario_name for r in results])
    npvs = np.array([r.npv for r in results])
    paybacks = np.array([r.payback_month for r in results])
    revenues = np.array([r.total_revenue for r in results])
    opexes = np.array([r.total_opex for r in results])
    discount_rates = np.array([s.discount_rate for s in scenarios])
    cash_flows = [r.cash_flow for r in results]

    df = pd.DataFrame(
        {
            "scenario": scenario_names,
            "npv": npvs,
            "payback_month": paybacks,
            "total_revenue": revenues,
            "total_opex": opexes,
            "net_cash_flow": revenues - opexes,
            "irr_approx": [
                _estimate_irr(cf, dr) for cf, dr in zip(cash_flows, discount_rates)
            ],
        }
    )

    return df


def compare_scenarios(
    scenario_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare multiple scenarios and calculate differences.

    Args:
        scenario_results: DataFrame from run_price_scenarios or run_multi_phase_scenarios

    Returns:
        DataFrame with scenario comparison including differences from base case

    Example:
        >>> results = run_price_scenarios(production, scenarios)
        >>> comparison = compare_scenarios(results)
        >>> print(comparison)
    """
    # Find base scenario using vectorized operations
    scenario_values = scenario_results["scenario"].values
    base_candidates = ["base", "Base", "BASE", "baseline"]
    base_mask = np.isin(scenario_values, base_candidates)

    if base_mask.any():
        base_scenario = scenario_values[base_mask][0]
    else:
        base_scenario = scenario_values[0]
        logger.warning(
            f"No 'base' scenario found, using '{base_scenario}' as reference"
        )

    base_mask = scenario_results["scenario"] == base_scenario
    base_npv = scenario_results.loc[base_mask, "npv"].iloc[0]
    base_payback = scenario_results.loc[base_mask, "payback_month"].iloc[0]

    # Calculate differences using vectorized operations
    comparison = scenario_results.copy()
    npv_diff = comparison["npv"] - base_npv
    comparison["npv_vs_base"] = npv_diff
    comparison["npv_pct_change"] = np.where(
        base_npv != 0, (npv_diff / abs(base_npv)) * 100, 0.0
    )
    comparison["payback_vs_base"] = comparison["payback_month"] - base_payback

    return comparison


def _estimate_irr(cash_flow: np.ndarray, initial_guess: float = 0.10) -> float:
    """Estimate internal rate of return using Newton-Raphson method.

    Args:
        cash_flow: Cash flow array
        initial_guess: Initial guess for IRR

    Returns:
        Estimated IRR (annual rate)
    """
    try:
        from scipy.optimize import newton

        def npv_func(rate):
            monthly_rate = rate / 12
            return np.sum(
                cash_flow / ((1 + monthly_rate) ** np.arange(1, len(cash_flow) + 1))
            )

        irr = newton(npv_func, initial_guess, maxiter=100)
        return irr * 12  # Convert to annual
    except ImportError:
        logger.warning("scipy not available, skipping IRR calculation")
        return np.nan
    except Exception:
        return np.nan
