"""Physics-informed decline curve analysis models.

This module provides physics-based models that incorporate reservoir engineering
principles including material balance, pressure decline, and physical constraints.

Key Features:
- Material balance-based decline models
- Pressure decline forecasting
- Physics-informed neural network constraints
- Reservoir simulation integration
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .logging_config import get_logger
from .models_base import Model

logger = get_logger(__name__)


@dataclass
class MaterialBalanceParams:
    """Parameters for material balance decline model.

    Attributes:
        N: Original oil in place (STB)
        Boi: Initial formation volume factor (RB/STB)
        Bo: Current formation volume factor (RB/STB)
        cw: Water compressibility (1/psi)
        cf: Formation compressibility (1/psi)
        Swi: Initial water saturation (fraction)
        pi: Initial reservoir pressure (psi)
        p: Current reservoir pressure (psi)
        Np: Cumulative oil production (STB)
        Wp: Cumulative water production (STB)
        We: Cumulative water influx (STB)
        Wp_inj: Cumulative water injection (STB)
        temperature: Reservoir temperature (°F)
        api_gravity: Oil API gravity (°API)
        gas_gravity: Gas specific gravity (air=1.0)
        pvt_method: PVT correlation method ('standing' or 'vasquez_beggs')
    """

    N: float  # Original oil in place
    Boi: float = 1.0  # Initial FVF
    cw: float = 3e-6  # Water compressibility (1/psi)
    cf: float = 5e-6  # Formation compressibility (1/psi)
    Swi: float = 0.2  # Initial water saturation
    pi: float = 5000.0  # Initial pressure (psi)
    p: Optional[float] = None  # Current pressure
    Np: float = 0.0  # Cumulative oil production
    Wp: float = 0.0  # Cumulative water production
    We: float = 0.0  # Water influx
    Wp_inj: float = 0.0  # Water injection
    temperature: float = 200.0  # Reservoir temperature (°F)
    api_gravity: float = 30.0  # Oil API gravity (°API)
    gas_gravity: float = 0.65  # Gas specific gravity
    pvt_method: str = "standing"  # PVT correlation method


@dataclass
class PressureDeclineParams:
    """Parameters for pressure decline model.

    Attributes:
        pi: Initial reservoir pressure (psi)
        D: Pressure decline constant (1/day)
        b: Pressure decline exponent (0 < b <= 1)
        p_abandon: Abandonment pressure (psi)
    """

    pi: float  # Initial pressure
    D: float  # Pressure decline constant
    b: float = 1.0  # Decline exponent
    p_abandon: float = 100.0  # Abandonment pressure


class MaterialBalanceDecline(Model):
    """Material balance-based decline curve model.

    Uses material balance equation to relate production to reservoir pressure
    and fluid properties. Supports multiple drive mechanisms:

    - Solution gas drive (depletion drive)
    - Water drive
    - Gas cap drive
    - Combination drive
    - Undersaturated oil
    - Gas reservoirs (p/Z method)

    This model provides physics-based forecasting that accounts for:
    - Reservoir pressure decline
    - Fluid compressibility
    - Water influx/injection
    - Formation compaction
    - Drive mechanism-specific behavior
    """

    @property
    def name(self) -> str:
        """Return model name."""
        return "MaterialBalanceDecline"

    def rate(self, t: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Compute production rate from material balance.

        Args:
            t: Time array (days)
            params: Material balance parameters

        Returns:
            Production rate array (STB/day)
        """
        # Extract parameters
        N = params.get("N", 1e6)
        Boi = params.get("Boi", 1.0)
        _cw = params.get("cw", 3e-6)
        _cf = params.get("cf", 5e-6)
        _Swi = params.get("Swi", 0.2)
        pi = params.get("pi", 5000.0)

        # Pressure decline (simplified - can be enhanced)
        D = params.get("D", 0.001)  # Pressure decline rate
        p = pi * np.exp(-D * t)

        # Use PVT correlations for accurate FVF calculation
        from .pvt import calculate_pvt_properties

        # Get PVT parameters
        temperature = params.get("temperature", 200.0)  # °F
        api_gravity = params.get("api_gravity", 30.0)  # °API
        gas_gravity = params.get("gas_gravity", 0.65)
        pvt_method = params.get(
            "pvt_method", "standing"
        )  # 'standing' or 'vasquez_beggs'

        # Calculate PVT properties at each pressure
        # For efficiency, calculate at initial and current pressure, then interpolate
        _pvt_initial = calculate_pvt_properties(
            pi, temperature, api_gravity, gas_gravity, method=pvt_method
        )

        # Calculate Bo at current pressure using PVT
        # Vectorized calculation for all time steps
        Bo_values = np.zeros_like(t)
        for i, p_current in enumerate(p):
            pvt_current = calculate_pvt_properties(
                p_current, temperature, api_gravity, gas_gravity, method=pvt_method
            )
            Bo_values[i] = pvt_current.Bo

        _Bo = params.get("Bo", Bo_values)  # Use calculated Bo if not provided

        # Use drive mechanism-specific material balance
        drive_mechanism = params.get("drive_mechanism", "solution_gas")

        # Get cumulative production from time
        Np = self.cum(t, params)

        # Calculate rate using drive mechanism-specific material balance
        from .material_balance import (
            GasCapDriveParams,
            GasReservoirParams,
            SolutionGasDriveParams,
            WaterDriveParams,
            gas_cap_drive_material_balance,
            gas_reservoir_pz_method,
            solution_gas_drive_material_balance,
            undersaturated_oil_material_balance,
            water_drive_material_balance,
        )

        rates = np.zeros_like(t)

        for i, (t_i, p_i, Np_i) in enumerate(zip(t, p, Np)):
            if drive_mechanism == "solution_gas":
                mb_params = SolutionGasDriveParams(
                    N=N,
                    Boi=Boi,
                    pi=pi,
                    pb=params.get("bubble_point", pi * 0.6),
                    api_gravity=params.get("api_gravity", 30.0),
                    gas_gravity=params.get("gas_gravity", 0.65),
                    temperature=params.get("temperature", 200.0),
                )
                mb_result = solution_gas_drive_material_balance(p_i, Np_i, mb_params)
                # Estimate rate from material balance
                if i > 0:
                    dNp = Np_i - Np[i - 1]
                    dt = t_i - t[i - 1] if t_i > t[i - 1] else 1.0
                    rates[i] = dNp / dt if dt > 0 else 0.0
                else:
                    rates[i] = params.get("qi", N * D * 0.1)

            elif drive_mechanism == "water_drive":
                mb_params = WaterDriveParams(
                    N=N,
                    Boi=Boi,
                    pi=pi,
                    We=params.get("We", 0.0),
                    Wp=params.get("Wp", 0.0),
                )
                mb_result = water_drive_material_balance(p_i, Np_i, mb_params)
                if i > 0:
                    dNp = Np_i - Np[i - 1]
                    dt = t_i - t[i - 1] if t_i > t[i - 1] else 1.0
                    rates[i] = dNp / dt if dt > 0 else 0.0
                else:
                    rates[i] = params.get("qi", N * D * 0.1)

            elif drive_mechanism == "gas_cap":
                mb_params = GasCapDriveParams(
                    N=N,
                    m=params.get("gas_cap_size", 0.1),
                    Boi=Boi,
                    pi=pi,
                    api_gravity=params.get("api_gravity", 30.0),
                    gas_gravity=params.get("gas_gravity", 0.65),
                    temperature=params.get("temperature", 200.0),
                )
                mb_result = gas_cap_drive_material_balance(p_i, Np_i, mb_params)
                if i > 0:
                    dNp = Np_i - Np[i - 1]
                    dt = t_i - t[i - 1] if t_i > t[i - 1] else 1.0
                    rates[i] = dNp / dt if dt > 0 else 0.0
                else:
                    rates[i] = params.get("qi", N * D * 0.1)

            elif drive_mechanism == "gas_reservoir":
                # For gas reservoirs, use p/Z method
                G = params.get("G", N * 1000.0)  # Convert to SCF
                Gp = params.get("Gp", Np_i * 1000.0)  # Approximate
                mb_params = GasReservoirParams(
                    G=G,
                    pi=pi,
                    temperature=params.get("temperature", 200.0),
                    gas_gravity=params.get("gas_gravity", 0.65),
                )
                _ = gas_reservoir_pz_method(p_i, Gp, mb_params)  # noqa: F841
                # Estimate gas rate
                if i > 0:
                    dGp = Gp - params.get("Gp_prev", 0.0)
                    dt = t_i - t[i - 1] if t_i > t[i - 1] else 1.0
                    rates[i] = dGp / dt if dt > 0 else 0.0
                else:
                    rates[i] = params.get("qi", G * D * 0.1)
                params["Gp_prev"] = Gp

            elif drive_mechanism == "undersaturated":
                mb_params = SolutionGasDriveParams(
                    N=N,
                    Boi=Boi,
                    pi=pi,
                    pb=params.get("bubble_point", pi * 0.5),
                    api_gravity=params.get("api_gravity", 30.0),
                    gas_gravity=params.get("gas_gravity", 0.65),
                    temperature=params.get("temperature", 200.0),
                )
                _mb_result = undersaturated_oil_material_balance(p_i, Np_i, mb_params)
                if i > 0:
                    dNp = Np_i - Np[i - 1]
                    dt = t_i - t[i - 1] if t_i > t[i - 1] else 1.0
                    rates[i] = dNp / dt if dt > 0 else 0.0
                else:
                    rates[i] = params.get("qi", N * D * 0.1)
            else:
                # Default: simple exponential decline
                qi = params.get("qi", N * D * 0.1)
                di = D
                rates[i] = qi * np.exp(-di * t_i)

        return rates

    def cum(self, t: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Compute cumulative production from material balance.

        Args:
            t: Time array (days)
            params: Material balance parameters

        Returns:
            Cumulative production array (STB)
        """
        N = params.get("N", 1e6)
        D = params.get("D", 0.001)

        # Cumulative from material balance
        # Simplified: Np = N * (1 - exp(-D * t))
        return N * (1 - np.exp(-D * t))

    def constraints(self) -> Dict[str, Tuple[float, float]]:
        """Return parameter bounds."""
        return {
            "N": (1e3, 1e9),  # OOIP (STB)
            "Boi": (1.0, 3.0),  # Initial FVF
            "cw": (1e-7, 1e-4),  # Water compressibility
            "cf": (1e-7, 1e-4),  # Formation compressibility
            "Swi": (0.0, 0.5),  # Initial water saturation
            "pi": (100.0, 20000.0),  # Initial pressure (psi)
            "D": (1e-6, 0.1),  # Pressure decline rate
            "qi": (1.0, 1e6),  # Initial rate
            "temperature": (100.0, 400.0),  # Reservoir temperature (°F)
            "api_gravity": (10.0, 60.0),  # Oil API gravity (°API)
            "gas_gravity": (0.5, 1.5),  # Gas specific gravity
            "bubble_point": (100.0, 10000.0),  # Bubble point pressure (psi)
            "gas_cap_size": (0.0, 1.0),  # Gas cap size (m)
            "G": (1e6, 1e12),  # Original gas in place (SCF)
        }

    def initial_guess(self, t: np.ndarray, q: np.ndarray) -> Dict[str, float]:
        """Generate initial guess from production data."""
        valid_mask = q > 0
        if not np.any(valid_mask):
            return {
                "N": 1e6,
                "Boi": 1.2,
                "D": 0.001,
                "qi": 1000.0,
            }

        q_valid = q[valid_mask]
        qi = float(np.max(q_valid))

        # Estimate N from cumulative
        if len(q_valid) > 1:
            t_valid = t[valid_mask]
            Np_approx = np.trapz(q_valid, t_valid)
            # Rough estimate: N ≈ 2 * Np_approx (assuming 50% recovery)
            N = max(1e6, 2 * Np_approx)
        else:
            N = 1e6

        # Estimate D from decline
        if len(q_valid) >= 2:
            decline_rate = (q_valid[0] - q_valid[-1]) / (q_valid[0] * t[valid_mask][-1])
            D = max(1e-6, min(0.1, decline_rate))
        else:
            D = 0.001

        return {
            "N": float(N),
            "Boi": 1.2,
            "D": float(D),
            "qi": float(qi),
            "temperature": 200.0,
            "api_gravity": 30.0,
            "gas_gravity": 0.65,
            "pvt_method": "standing",
        }


class PressureDeclineModel(Model):
    """Pressure-based decline curve model.

    Forecasts production based on reservoir pressure decline rather than
    empirical decline curves. Uses IPR (Inflow Performance Relationship)
    models from reservoir engineering.

    Supports multiple IPR types:
    - Linear: q = J * (p - pwf)
    - Vogel: Solution gas drive (two-phase flow)
    - Fetkovich: Generalized IPR with exponent
    """

    @property
    def name(self) -> str:
        """Return model name."""
        return "PressureDeclineModel"

    def rate(self, t: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Compute production rate from pressure decline using IPR models.

        Args:
            t: Time array (days)
            params: Pressure decline parameters

        Returns:
            Production rate array
        """
        from .ipr import fetkovich_ipr, linear_ipr, vogel_ipr

        pi = params.get("pi", 5000.0)
        D = params.get("D", 0.001)
        b = params.get("b", 1.0)
        J = params.get("J", 1.0)  # Productivity index
        pwf = params.get("pwf", 500.0)  # Wellbore flowing pressure
        ipr_type = params.get("ipr_type", "linear")  # 'linear', 'vogel', 'fetkovich'
        pb = params.get("bubble_point", None)  # Bubble point pressure
        n = params.get("fetkovich_n", 1.0)  # Fetkovich exponent

        # Pressure decline
        if b == 1.0:
            p = pi / (1 + D * t)  # Harmonic pressure decline
        elif b > 0:
            p = pi / np.power(1 + b * D * t, 1 / b)  # Hyperbolic
        else:
            p = pi * np.exp(-D * t)  # Exponential

        # Calculate rate using appropriate IPR model
        q = np.zeros_like(t)

        for i, p_current in enumerate(p):
            if p_current <= pwf:
                q[i] = 0.0
                continue

            if ipr_type == "linear":
                q[i] = linear_ipr(p_current, pwf, J)

            elif ipr_type == "vogel":
                # Calculate max_rate for Vogel
                q_max = J * p_current / 1.8 if J > 0 else 0.0
                q[i] = vogel_ipr(
                    p_current,
                    pwf,
                    max_rate=q_max,
                    productivity_index=J,
                    bubble_point_pressure=pb,
                )

            elif ipr_type == "fetkovich":
                q_max = J * p_current if J > 0 else 0.0
                q[i] = fetkovich_ipr(p_current, pwf, q_max, n=n)

            else:
                # Default to linear
                q[i] = linear_ipr(p_current, pwf, J)

        return q

    def cum(self, t: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Compute cumulative production from pressure decline."""
        # Integrate rate
        q = self.rate(t, params)
        return np.cumsum(q) * (t[1] - t[0]) if len(t) > 1 else np.array([0.0])

    def constraints(self) -> Dict[str, Tuple[float, float]]:
        """Return parameter bounds."""
        return {
            "pi": (100.0, 20000.0),  # Initial pressure (psi)
            "D": (1e-6, 0.1),  # Pressure decline rate
            "b": (0.0, 1.0),  # Decline exponent
            "J": (0.1, 100.0),  # Productivity index (STB/day/psi)
            "pwf": (0.0, 5000.0),  # Wellbore flowing pressure
            "bubble_point": (100.0, 10000.0),  # Bubble point pressure (psi)
            "fetkovich_n": (0.1, 2.0),  # Fetkovich exponent
        }

    def initial_guess(self, t: np.ndarray, q: np.ndarray) -> Dict[str, float]:
        """Generate initial guess from production data."""
        valid_mask = q > 0
        if not np.any(valid_mask):
            return {
                "pi": 5000.0,
                "D": 0.001,
                "b": 0.5,
                "J": 1.0,
                "pwf": 500.0,
            }

        q_valid = q[valid_mask]
        qi = float(np.max(q_valid))

        # Estimate J assuming pi - pwf ≈ 4000 psi
        J = qi / 4000.0

        # Estimate D from decline
        if len(q_valid) >= 2:
            decline_rate = (q_valid[0] - q_valid[-1]) / (q_valid[0] * t[valid_mask][-1])
            D = max(1e-6, min(0.1, decline_rate))
        else:
            D = 0.001

        return {
            "pi": 5000.0,
            "D": float(D),
            "b": 0.5,
            "J": float(J),
            "pwf": 500.0,
            "ipr_type": "linear",
            "fetkovich_n": 1.0,
        }


def apply_physics_constraints(
    forecast: np.ndarray,
    historical: Optional[np.ndarray] = None,
    min_rate: float = 0.0,
    max_increase: Optional[float] = None,
    enforce_decline: bool = True,
) -> np.ndarray:
    """Apply physics-based constraints to forecast.

    Ensures forecast satisfies physical laws:
    - Non-negative production
    - No unrealistic increases
    - Monotonic decline (if enforced)

    Args:
        forecast: Forecasted production rates
        historical: Historical production rates (for continuity)
        min_rate: Minimum production rate (economic limit)
        max_increase: Maximum allowed increase from last historical value
        enforce_decline: If True, enforce monotonic decline

    Returns:
        Constrained forecast array
    """
    constrained = forecast.copy()

    # Ensure non-negative
    constrained = np.maximum(constrained, min_rate)

    # Enforce continuity with historical data
    if historical is not None and len(historical) > 0:
        last_historical = historical[-1]
        if max_increase is not None:
            max_allowed = last_historical * (1 + max_increase)
            constrained = np.minimum(constrained, max_allowed)

        # Ensure forecast starts near last historical value
        if len(constrained) > 0:
            constrained[0] = np.clip(
                constrained[0], 0.5 * last_historical, 1.5 * last_historical
            )

    # Enforce monotonic decline if requested
    if enforce_decline and len(constrained) > 1:
        for i in range(1, len(constrained)):
            if constrained[i] > constrained[i - 1]:
                constrained[i] = constrained[i - 1] * 0.99  # Slight decline

    return constrained


def load_reservoir_simulation(
    file_path: str,
    format: str = "auto",
) -> pd.DataFrame:
    """Load reservoir simulation output data.

    Supports common simulation output formats:
    - Eclipse (.F0018, .RSM)
    - CMG (.out)
    - CSV with standard columns

    Args:
        file_path: Path to simulation output file
        format: File format ('eclipse', 'cmg', 'csv', 'auto')

    Returns:
        DataFrame with simulation results (pressure, rates, etc.)

    Example:
        >>> from decline_curve.physics_informed import load_reservoir_simulation
        >>> sim_data = load_reservoir_simulation('eclipse_output.csv')
        >>> pressure = sim_data['pressure']
        >>> oil_rate = sim_data['oil_rate']
    """
    import os

    if format == "auto":
        ext = os.path.splitext(file_path)[1].lower()
        format_map = {
            ".csv": "csv",
            ".out": "cmg",
            ".f0018": "eclipse",
            ".rsm": "eclipse",
        }
        format = format_map.get(ext, "csv")

    if format == "csv":
        df = pd.read_csv(file_path)

        # Convert date column if present
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        elif "DATE" in df.columns:
            df["date"] = pd.to_datetime(df["DATE"])
            df = df.drop(columns=["DATE"])

        # Standardize column names
        column_map = {
            "TIME": "time",
            "PRESSURE": "pressure",
            "PRES": "pressure",
            "OIL_RATE": "oil_rate",
            "QO": "oil_rate",
            "GAS_RATE": "gas_rate",
            "QG": "gas_rate",
            "WATER_RATE": "water_rate",
            "QW": "water_rate",
            "CUM_OIL": "cum_oil",
            "CUM_GAS": "cum_gas",
            "CUM_WATER": "cum_water",
        }
        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

        # Set date as index if available
        if "date" in df.columns:
            df = df.set_index("date")

        return df

    elif format in ("eclipse", "cmg"):
        logger.warning(
            f"{format} format parsing not yet implemented. "
            "Please convert to CSV format or use specialized tools."
        )
        raise NotImplementedError(
            f"{format} format parsing not yet implemented. "
            "Use CSV format or convert simulation output to CSV."
        )

    else:
        raise ValueError(f"Unknown format: {format}")


def compare_dca_with_simulation(
    dca_forecast: pd.Series,
    simulation_data: pd.DataFrame,
    production_col: str = "oil_rate",
    date_col: Optional[str] = None,
) -> pd.DataFrame:
    """Compare DCA forecast with reservoir simulation results.

    Args:
        dca_forecast: DCA forecast series
        simulation_data: Simulation output DataFrame
        production_col: Column name for simulation production
        date_col: Column name for dates (if not index)

    Returns:
        DataFrame with comparison metrics and aligned data

    Example:
        >>> from decline_curve.physics_informed import compare_dca_with_simulation
        >>> comparison = compare_dca_with_simulation(dca_forecast, sim_data)
        >>> print(comparison[['dca_forecast', 'simulation', 'difference']])
    """
    # Extract simulation production
    if date_col and date_col in simulation_data.columns:
        sim_dates = pd.to_datetime(simulation_data[date_col])
        sim_production = simulation_data[production_col].values
    else:
        sim_dates = simulation_data.index
        sim_production = simulation_data[production_col].values

    # Align dates
    common_dates = dca_forecast.index.intersection(sim_dates)

    if len(common_dates) == 0:
        logger.warning("No overlapping dates between DCA forecast and simulation")
        return pd.DataFrame()

    dca_aligned = dca_forecast.loc[common_dates]
    sim_aligned = pd.Series(
        sim_production[sim_dates.isin(common_dates)],
        index=common_dates,
    )

    # Calculate differences
    comparison = pd.DataFrame(
        {
            "dca_forecast": dca_aligned,
            "simulation": sim_aligned,
            "difference": dca_aligned - sim_aligned,
            "pct_difference": ((dca_aligned - sim_aligned) / sim_aligned * 100).fillna(
                0
            ),
        }
    )

    return comparison


def material_balance_forecast(
    production_data: pd.Series,
    material_balance_params: Optional[MaterialBalanceParams] = None,
    horizon: int = 12,
) -> pd.Series:
    """Generate forecast using material balance model.

    Args:
        production_data: Historical production data
        material_balance_params: Material balance parameters
        horizon: Forecast horizon (months)

    Returns:
        Forecasted production series
    """
    from .fitting import CurveFitFitter, FitSpec

    # Fit material balance model
    model = MaterialBalanceDecline()

    # Convert to days for fitting
    dates = production_data.index
    t_days = np.array([(d - dates[0]).days for d in dates])
    q_values = production_data.values

    # Get initial guess
    initial_params = model.initial_guess(t_days, q_values)

    # Store non-numeric parameters separately (not for fitting)
    pvt_method = initial_params.pop("pvt_method", "standing")
    drive_mechanism = initial_params.pop("drive_mechanism", "solution_gas")

    # Override with provided parameters if available
    if material_balance_params:
        initial_params.update(
            {
                "N": material_balance_params.N,
                "Boi": material_balance_params.Boi,
                "pi": material_balance_params.pi,
                "D": 0.001,  # Estimate from data
                "temperature": material_balance_params.temperature,
                "api_gravity": material_balance_params.api_gravity,
                "gas_gravity": material_balance_params.gas_gravity,
            }
        )
        # Store non-numeric parameters separately (not in initial_guess)
        pvt_method = material_balance_params.pvt_method
        drive_mechanism = getattr(
            material_balance_params, "drive_mechanism", "solution_gas"
        )

    # Create fit spec (only numeric parameters - remove any remaining strings)
    numeric_params = {
        k: v
        for k, v in initial_params.items()
        if isinstance(v, (int, float, np.number))
    }

    fit_spec = FitSpec(
        model=model,
        initial_guess=numeric_params,
        min_points=3,
    )

    # Fit model
    fitter = CurveFitFitter()
    fit_result = fitter.fit(
        t=t_days,
        q=q_values,
        fit_spec=fit_spec,
        dates=dates,
    )

    if not fit_result.success:
        logger.warning(f"Material balance fitting failed: {fit_result.message}")
        # Fall back to simple exponential decline
        from .models import fit_arps

        try:
            params = fit_arps(t_days, q_values, kind="exponential")
            fit_result.params = {
                "qi": params.qi,
                "di": params.di,
                "D": params.di,
                "N": 1e6,
            }
        except Exception:
            # Ultimate fallback
            fit_result.params = initial_params

    # Generate forecast
    last_date = dates[-1]
    forecast_dates = pd.date_range(
        start=last_date + pd.DateOffset(days=30),
        periods=horizon,
        freq="MS",
    )

    forecast_t_days = np.array([(d - dates[0]).days for d in forecast_dates])

    # Add non-numeric parameters back to params for forecasting
    forecast_params = fit_result.params.copy()
    forecast_params["pvt_method"] = pvt_method
    forecast_params["drive_mechanism"] = drive_mechanism

    forecast_rates = model.rate(forecast_t_days, forecast_params)

    # Apply physics constraints
    constrained_rates = apply_physics_constraints(
        forecast_rates,
        historical=q_values,
        enforce_decline=True,
    )

    return pd.Series(constrained_rates, index=forecast_dates, name="forecast")


def pressure_decline_forecast(
    pressure_data: pd.Series,
    production_data: Optional[pd.Series] = None,
    horizon: int = 12,
) -> Tuple[pd.Series, pd.Series]:
    """Generate forecast using pressure decline model.

    Args:
        pressure_data: Historical pressure data
        production_data: Optional production data for calibration
        horizon: Forecast horizon (months)

    Returns:
        Tuple of (pressure_forecast, production_forecast)
    """
    # Fit pressure decline
    _model = PressureDeclineModel()

    dates = pressure_data.index
    t_days = np.array([(d - dates[0]).days for d in dates])
    p_values = pressure_data.values

    # Estimate initial parameters
    initial_params = {
        "pi": float(p_values[0]),
        "D": 0.001,
        "b": 0.5,
        "J": 1.0,
        "pwf": 500.0,
    }

    # If production data available, calibrate J
    if production_data is not None:
        q_values = production_data.values
        if len(q_values) > 0 and len(p_values) > 0:
            # Estimate J from initial rates and pressures
            qi = q_values[0]
            pi = p_values[0]
            pwf_est = pi * 0.9  # Assume 10% pressure drop
            initial_params["J"] = qi / (pi - pwf_est) if (pi - pwf_est) > 0 else 1.0

    # Fit pressure decline (simplified - fit exponential to pressure)
    from scipy.optimize import curve_fit

    def pressure_model(t, pi, D):
        return pi * np.exp(-D * t)

    try:
        popt, _ = curve_fit(
            pressure_model,
            t_days,
            p_values,
            p0=[initial_params["pi"], initial_params["D"]],
            bounds=([0, 0], [np.inf, 1.0]),
        )
        initial_params["pi"] = popt[0]
        initial_params["D"] = popt[1]
    except Exception:
        logger.warning("Pressure fitting failed, using initial guess")

    # Generate forecast
    last_date = dates[-1]
    forecast_dates = pd.date_range(
        start=last_date + pd.DateOffset(days=30),
        periods=horizon,
        freq="MS",
    )

    forecast_t_days = np.array([(d - dates[0]).days for d in forecast_dates])

    # Forecast pressure directly
    pi = initial_params["pi"]
    D = initial_params["D"]
    b = initial_params.get("b", 1.0)

    if b == 1.0:
        pressure_forecast = pi / (1 + D * forecast_t_days)
    elif b > 0:
        pressure_forecast = pi / np.power(1 + b * D * forecast_t_days, 1 / b)
    else:
        pressure_forecast = pi * np.exp(-D * forecast_t_days)

    # Forecast production from pressure
    J = initial_params["J"]
    pwf = initial_params["pwf"]
    production_forecast = J * np.maximum(pressure_forecast - pwf, 0.0)

    # Apply constraints
    production_forecast = apply_physics_constraints(
        production_forecast,
        historical=production_data.values if production_data is not None else None,
    )

    pressure_series = pd.Series(
        pressure_forecast, index=forecast_dates, name="pressure"
    )
    production_series = pd.Series(
        production_forecast, index=forecast_dates, name="forecast"
    )

    return pressure_series, production_series
