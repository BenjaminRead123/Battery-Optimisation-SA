"""Battery arbitrage strategies with explicit units and information timing."""

from dataclasses import asdict, dataclass
from pathlib import Path
import warnings

import cvxpy as cp
import numpy as np
import pandas as pd

import config
from dynamic_programming import GridOptimizer


@dataclass(frozen=True)
class Battery:
    capacity_mwh: float = config.BATTERY_CAPACITY_MWH
    power_mw: float = config.BATTERY_POWER_MW
    charge_efficiency: float = config.CHARGE_EFFICIENCY
    discharge_efficiency: float = config.DISCHARGE_EFFICIENCY
    initial_fraction: float = config.INITIAL_SOC_MWH / config.BATTERY_CAPACITY_MWH
    min_fraction: float = config.MIN_SOC_MWH / config.BATTERY_CAPACITY_MWH
    max_fraction: float = config.MAX_SOC_MWH / config.BATTERY_CAPACITY_MWH
    interval_hours: float = config.INTERVAL_HOURS
    cycle_cost_per_mwh: float = config.CYCLE_COST_PER_MWH
    soc_step_mwh: float = config.SOC_STEP_MWH

    def __post_init__(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Battery assumptions must be finite.")
        if self.capacity_mwh <= 0 or self.power_mw <= 0:
            raise ValueError("Capacity and power must be positive.")
        if not 0 <= self.min_fraction <= self.initial_fraction <= self.max_fraction <= 1:
            raise ValueError("Require 0 <= minimum <= initial <= maximum SOC fraction <= 1.")
        if self.min_fraction == self.max_fraction:
            raise ValueError("The usable SOC range must be positive.")
        if not (0 < self.charge_efficiency <= 1 and 0 < self.discharge_efficiency <= 1):
            raise ValueError("Efficiencies must be in (0, 1].")
        if self.interval_hours <= 0 or self.cycle_cost_per_mwh < 0:
            raise ValueError("Interval must be positive; cycling cost must be nonnegative.")
        if not np.isclose(24 / self.interval_hours, round(24 / self.interval_hours)):
            raise ValueError("The interval must divide a day exactly.")
        if self.soc_step_mwh <= 0:
            raise ValueError("SOC grid step must be positive.")
        for energy in [self.usable_mwh, self.initial_mwh - self.min_mwh]:
            if not np.isclose(energy / self.soc_step_mwh, round(energy / self.soc_step_mwh), atol=1e-6):
                raise ValueError("Initial SOC and limits must align with the SOC grid.")
        if self.charge_efficiency * self.max_grid_mwh < self.soc_step_mwh:
            raise ValueError("SOC step too coarse for one charging interval; reduce SOC_STEP_MWH.")

    @property
    def initial_mwh(self):
        return self.initial_fraction * self.capacity_mwh

    @property
    def min_mwh(self):
        return self.min_fraction * self.capacity_mwh

    @property
    def max_mwh(self):
        return self.max_fraction * self.capacity_mwh

    @property
    def usable_mwh(self):
        return self.max_mwh - self.min_mwh

    @property
    def max_grid_mwh(self):
        return self.power_mw * self.interval_hours


def load_prices(data_dir, interval_hours, region=None):
    """Load standard AEMO price CSVs; reject gaps, mixed regions and conflicts.

    Timestamps are interval END times, in the source's unambiguous fixed time
    basis (normally NEM time, AEST year-round). No timezone conversion is made.
    """
    files = sorted(Path(data_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSVs in {data_dir}. Add your price CSVs or use --demo explicitly."
        )
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        frame.columns = frame.columns.str.strip().str.upper()
        if not {"SETTLEMENTDATE", "RRP"} <= set(frame.columns):
            raise ValueError(f"{path.name}: expected SETTLEMENTDATE and RRP columns.")
        if "REGIONID" in frame.columns and "REGION" not in frame.columns:
            frame = frame.rename(columns={"REGIONID": "REGION"})
        if "PERIODTYPE" in frame.columns:
            trade = frame["PERIODTYPE"].astype(str).str.upper().str.strip().eq("TRADE")
            if not trade.all():
                warnings.warn(f"{path.name}: retained TRADE rows only.", stacklevel=2)
                frame = frame.loc[trade].copy()
        cols = ["SETTLEMENTDATE", "RRP"] + (["REGION"] if "REGION" in frame else [])
        frames.append(frame[cols])
    prices = pd.concat(frames, ignore_index=True)
    if prices.empty:
        raise ValueError("No trading observations remain.")
    if "REGION" in prices:
        if prices["REGION"].isna().any():
            raise ValueError("Some files lack region identifiers while others contain them.")
        prices["REGION"] = prices["REGION"].astype(str).str.strip().str.upper()
        if region:
            prices = prices.loc[prices["REGION"].eq(region.upper())].copy()
        if prices.empty:
            raise ValueError(f"No observations for region {region}.")
        if prices["REGION"].nunique() != 1:
            raise ValueError("Multiple regions found. Choose one with --region VIC1, for example.")
    elif region:
        raise ValueError("Cannot filter a region: the input has no REGION/REGIONID column.")
    prices["SETTLEMENTDATE"] = pd.to_datetime(
        prices["SETTLEMENTDATE"], format="mixed", dayfirst=True, errors="raise"
    )
    if prices["SETTLEMENTDATE"].isna().any():
        raise ValueError("Missing timestamps are not permitted.")
    if not pd.api.types.is_datetime64_any_dtype(prices["SETTLEMENTDATE"]):
        raise ValueError("Use a consistent fixed timezone/time basis for all timestamps.")
    prices["RRP"] = pd.to_numeric(prices["RRP"], errors="raise")
    if not np.isfinite(prices["RRP"]).all():
        raise ValueError("Prices contain missing or non-finite values.")
    if prices.groupby("SETTLEMENTDATE")["RRP"].nunique().gt(1).any():
        raise ValueError("Conflicting prices at the same timestamp; resolve the source files.")
    original_n = len(prices)
    prices = prices.drop_duplicates("SETTLEMENTDATE").sort_values("SETTLEMENTDATE")
    if len(prices) < original_n:
        warnings.warn(f"Removed {original_n - len(prices)} identical duplicate observations.", stacklevel=2)
    delta = pd.Timedelta(hours=interval_hours)
    differences = prices["SETTLEMENTDATE"].diff().iloc[1:]
    if not differences.eq(delta).all():
        raise ValueError(
            "Missing intervals or a frequency mismatch. Every row must be one "
            f"{delta} interval; no automatic filling or resampling is performed."
        )
    return prices.reset_index(drop=True)


def synthetic_prices(days=14, interval_hours=config.INTERVAL_HOURS, seed=42):
    """Deterministic demonstration ONLY: invented prices, not an AEMO dataset."""
    if days < 1:
        raise ValueError("Demo days must be positive.")
    rng = np.random.default_rng(seed)
    n = round(days * 24 / interval_hours)
    time = pd.date_range("2025-01-01", periods=n, freq=pd.Timedelta(hours=interval_hours))
    hour = time.hour.to_numpy() + time.minute.to_numpy() / 60
    day = np.arange(n) // round(24 / interval_hours)
    profile = (70 + 135 * np.exp(-((hour - 18.5) / 1.7) ** 2)
               - 100 * np.exp(-((hour - 12.5) / 2.8) ** 2))
    daily_noise = rng.normal(0, 13, days)[day]
    prices = profile + daily_noise + rng.normal(0, 7, n)
    # Unpredictable, clustered positive spikes and occasional negative events.
    for d in range(days):
        if rng.random() < 0.5:
            k = d * round(24 / interval_hours) + round(rng.uniform(16, 21) / interval_hours)
            prices[k:k + 3] += rng.uniform(300, 900)
        if rng.random() < 0.35:
            k = d * round(24 / interval_hours) + round(rng.uniform(10, 14) / interval_hours)
            prices[k:k + 5] -= rng.uniform(70, 180)
    return pd.DataFrame({"SETTLEMENTDATE": time + pd.Timedelta(hours=interval_hours),
                         "RRP": prices.round(2), "REGION": "SYNTHETIC"})


def build_result(prices, battery, charge, discharge, name, initial=None, forecast=None):
    """Grid-side charge/discharge in MWh; SOC and throughput on battery side."""
    initial = battery.initial_mwh if initial is None else initial
    charge, discharge = np.asarray(charge), np.asarray(discharge)
    internal_charge = battery.charge_efficiency * charge
    internal_discharge = discharge / battery.discharge_efficiency
    start = initial + np.r_[0.0, np.cumsum(internal_charge - internal_discharge)[:-1]]
    result = prices.copy().reset_index(drop=True)
    result["Interval Start"] = result["SETTLEMENTDATE"] - pd.Timedelta(hours=battery.interval_hours)
    result["Strategy"] = name
    result["Charge MWh"] = charge
    result["Discharge MWh"] = discharge
    result["EGrid"] = discharge - charge  # Positive = exported; negative = imported.
    result["Net Power MW"] = result["EGrid"] / battery.interval_hours
    result["SOC Start MWh"] = start
    result["SOC"] = start + internal_charge - internal_discharge
    result["Internal Charge MWh"] = internal_charge
    result["Internal Discharge MWh"] = internal_discharge
    result["Action"] = np.select([charge > 1e-5, discharge > 1e-5], ["Charge", "Discharge"], "Idle")
    result["Cash Flow"] = result["RRP"] * result["EGrid"]
    result["Cycling Cost"] = battery.cycle_cost_per_mwh * (internal_charge + internal_discharge) / 2
    result["Net Margin"] = result["Cash Flow"] - result["Cycling Cost"]
    result["Cumulative Margin"] = result["Net Margin"].cumsum()
    result["EFC"] = (internal_charge + internal_discharge) / (2 * battery.usable_mwh)
    if forecast is not None:
        result["Forecast RRP"] = forecast
    validate_dispatch(result, battery, initial)
    return result


def validate_dispatch(result, battery, initial, final=None):
    """Fail loudly on infeasible outputs rather than clipping away model errors."""
    tol = 1e-4
    for column in ["Charge MWh", "Discharge MWh"]:
        values = result[column].to_numpy()
        if not np.isfinite(values).all() or values.min() < -tol or values.max() > battery.max_grid_mwh + tol:
            raise AssertionError(f"Invalid grid flow in {column}.")
    if ((result["Charge MWh"] > tol) & (result["Discharge MWh"] > tol)).any():
        raise AssertionError("Simultaneous charging and discharging.")
    if result["SOC"].min() < battery.min_mwh - tol or result["SOC"].max() > battery.max_mwh + tol:
        raise AssertionError("SOC outside battery limits.")
    expected = initial + result["Internal Charge MWh"].sum() - result["Internal Discharge MWh"].sum()
    if not np.isclose(result["SOC"].iloc[-1], expected, atol=tol, rtol=0):
        raise AssertionError("Energy conservation failed.")
    if final is not None and not np.isclose(result["SOC"].iloc[-1], final, atol=tol, rtol=0):
        raise AssertionError("Terminal SOC target not met.")


def threshold_strategy(history, start, stop, battery, buy_below=config.BUY_BELOW,
                       sell_above=config.SELL_ABOVE):
    """Use the PREVIOUS realised price, avoiding knowledge of the current outcome.

    Retains the user's $30/$150 rules, with a terminal-reachability safeguard.
    Near the end that safeguard can override the thresholds to restore SOC.
    """
    if start < 1 or buy_below >= sell_above:
        raise ValueError("Need a prior price and buy_below < sell_above.")
    n = stop - start
    charge, discharge = np.zeros(n), np.zeros(n)
    soc = battery.initial_mwh
    eta_c, eta_d, limit = battery.charge_efficiency, battery.discharge_efficiency, battery.max_grid_mwh
    step = battery.soc_step_mwh
    charge_step = np.floor(eta_c * limit / step + 1e-9) * step
    discharge_step = np.floor(limit / eta_d / step + 1e-9) * step
    for j, signal in enumerate(history["RRP"].iloc[start - 1:stop - 1].to_numpy()):
        next_soc = soc
        if signal < buy_below:
            next_soc = min(battery.max_mwh, soc + charge_step)
        elif signal > sell_above:
            next_soc = max(battery.min_mwh, soc - discharge_step)
        remaining = n - j - 1
        lower = max(battery.min_mwh, battery.initial_mwh - remaining * charge_step)
        upper = min(battery.max_mwh, battery.initial_mwh + remaining * discharge_step)
        next_soc = float(np.clip(next_soc, lower, upper))
        if next_soc >= soc:
            charge[j] = (next_soc - soc) / eta_c
        else:
            discharge[j] = (soc - next_soc) * eta_d
        soc = next_soc
    result = build_result(history.iloc[start:stop], battery, charge, discharge, "Threshold (lagged)")
    validate_dispatch(result, battery, battery.initial_mwh, battery.initial_mwh)
    return result


def forecast_prices(history, decision_index, horizon, battery,
                    lookback_days=config.FORECAST_LOOKBACK_DAYS):
    """Median of the same five-minute slot over prior days, using only index < t.

    No current/future realised price or future-derived scaler enters the forecast.
    At cold start, use the most recent observed price for unobserved slots.
    """
    if decision_index < 1 or horizon < 1 or lookback_days < 1:
        raise ValueError("Forecast needs past observations, a positive horizon and lookback.")
    per_day = round(24 / battery.interval_hours)
    past = history.iloc[max(0, decision_index - lookback_days * per_day):decision_index]
    # Interval-ending timestamps: midnight corresponds to the final slot of a day.
    time = past["SETTLEMENTDATE"]
    seconds = time.dt.hour * 3600 + time.dt.minute * 60 + time.dt.second
    slot = np.rint(seconds / (battery.interval_hours * 3600)).astype(int) % per_day
    profile = pd.Series(past["RRP"].to_numpy()).groupby(slot.to_numpy()).median()
    last_time = past["SETTLEMENTDATE"].iloc[-1]
    future = pd.date_range(last_time + pd.Timedelta(hours=battery.interval_hours),
                           periods=horizon, freq=pd.Timedelta(hours=battery.interval_hours))
    future_slot = np.rint((future.hour * 3600 + future.minute * 60 + future.second)
                          / (battery.interval_hours * 3600)).astype(int) % per_day
    return profile.reindex(future_slot).fillna(past["RRP"].iloc[-1]).to_numpy(dtype=float)


class DispatchOptimizer:
    """Reusable CVXPY mixed-integer linear programme, solved with HiGHS.

    A binary operating mode forbids charge/discharge at once, including when
    prices are negative. All selected intervals are solved together for the
    perfect-foresight case; it is never silently split into daily optimisations.
    """

    def __init__(self, battery, enforce_soc_grid=False):
        self.battery = battery
        self.enforce_soc_grid = enforce_soc_grid
        self.problems = {}

    def _problem(self, n):
        b = self.battery
        charge = cp.Variable(n, nonneg=True)
        discharge = cp.Variable(n, nonneg=True)
        soc = cp.Variable(n + 1)
        charging = cp.Variable(n, boolean=True)
        prices = cp.Parameter(n)
        initial, final = cp.Parameter(), cp.Parameter()
        constraints = [
            soc[0] == initial, soc[-1] == final,
            soc >= b.min_mwh, soc <= b.max_mwh,
            soc[1:] == soc[:-1] + b.charge_efficiency * charge - discharge / b.discharge_efficiency,
            charge <= b.max_grid_mwh * charging,
            discharge <= b.max_grid_mwh * (1 - charging),
        ]
        if self.enforce_soc_grid:
            states = cp.Variable(n + 1, integer=True)
            constraints.append(soc == b.min_mwh + b.soc_step_mwh * states)
        throughput = (b.charge_efficiency * charge + discharge / b.discharge_efficiency) / 2
        objective = cp.Maximize(prices @ (discharge - charge) - b.cycle_cost_per_mwh * cp.sum(throughput))
        return cp.Problem(objective, constraints), prices, initial, final, charge, discharge

    def solve(self, forecast, initial, final):
        forecast = np.asarray(forecast, dtype=float)
        if forecast.ndim != 1 or not len(forecast) or not np.isfinite(forecast).all():
            raise ValueError("Optimisation prices must be a finite, nonempty vector.")
        n = len(forecast)
        if n not in self.problems:
            self.problems[n] = self._problem(n)
        problem, prices, initial_param, final_param, charge, discharge = self.problems[n]
        prices.value, initial_param.value, final_param.value = forecast, initial, final
        if "HIGHS" not in cp.installed_solvers():
            raise RuntimeError("Install the MILP solver with: python -m pip install highspy")
        problem.solve(solver=cp.HIGHS, warm_start=False,
                      highs_options={"time_limit": config.SOLVER_TIME_LIMIT_SECONDS,
                                     "mip_rel_gap": config.SOLVER_RELATIVE_GAP,
                                     "threads": 1})
        if problem.status != cp.OPTIMAL:
            raise RuntimeError(
                f"Solver status {problem.status}. No claim of perfection is made. "
                "Increase the time limit or explicitly analyse a shorter period."
            )
        c = np.asarray(charge.value).ravel().copy()
        d = np.asarray(discharge.value).ravel().copy()
        c[np.abs(c) < 1e-8], d[np.abs(d) < 1e-8] = 0.0, 0.0
        return c, d


def perfect_foresight(prices, battery):
    """Full-period hindsight optimum on the selected SOC grid."""
    charge, discharge = GridOptimizer(battery).solve(
        prices["RRP"].to_numpy(), battery.initial_mwh, battery.initial_mwh
    )
    result = build_result(prices, battery, charge, discharge, "Perfect foresight")
    validate_dispatch(result, battery, battery.initial_mwh, battery.initial_mwh)
    return result


def rolling_optimisation(history, start, stop, battery,
                         horizon_hours=config.FORECAST_HORIZON_HOURS,
                         replan_hours=config.REPLAN_HOURS,
                         lookback_days=config.FORECAST_LOOKBACK_DAYS, schedule=None):
    """Reforecast hourly, optimise 24 hours, execute only the first hour.

    Every solve targets the initial SOC at the end of its planning horizon.
    This explicit boundary policy limits end effects; it is a modelling choice.
    """
    horizon_n = round(horizon_hours / battery.interval_hours)
    replan_n = round(replan_hours / battery.interval_hours)
    if not 1 <= replan_n <= horizon_n:
        raise ValueError("Require 1 <= replan intervals <= horizon intervals.")
    if start < 1 or stop <= start or stop > len(history):
        raise ValueError("Invalid simulation range.")
    n = stop - start
    charge, discharge, forecasts = np.zeros(n), np.zeros(n), np.zeros(n)
    soc = battery.initial_mwh
    optimiser = GridOptimizer(battery)
    if schedule is None:
        schedule = forecast_schedule(history, start, stop, battery, horizon_hours, replan_hours, lookback_days)
    for offset, forecast in schedule:
        c, d = optimiser.solve(forecast, soc, battery.initial_mwh)
        take = min(replan_n, n - offset)
        charge[offset:offset + take] = c[:take]
        discharge[offset:offset + take] = d[:take]
        forecasts[offset:offset + take] = forecast[:take]
        soc += battery.charge_efficiency * c[:take].sum() - d[:take].sum() / battery.discharge_efficiency
    result = build_result(history.iloc[start:stop], battery, charge, discharge,
                          "Forecast optimisation", forecast=forecasts)
    validate_dispatch(result, battery, battery.initial_mwh, battery.initial_mwh)
    return result


def forecast_schedule(history, start, stop, battery,
                      horizon_hours=config.FORECAST_HORIZON_HOURS,
                      replan_hours=config.REPLAN_HOURS,
                      lookback_days=config.FORECAST_LOOKBACK_DAYS):
    """Build causal forecasts once and reuse them across equipment scenarios."""
    horizon_n = round(horizon_hours / battery.interval_hours)
    replan_n = round(replan_hours / battery.interval_hours)
    if not 1 <= replan_n <= horizon_n:
        raise ValueError("Invalid forecast/replanning horizon.")
    return [(offset, forecast_prices(history, start + offset,
             min(horizon_n, stop - start - offset), battery, lookback_days))
            for offset in range(0, stop - start, replan_n)]


def indicators(data, battery, year=None):
    """Optional year filter. EFC uses internal throughput / twice usable capacity."""
    selected = data if year is None else data.loc[data["Interval Start"].dt.year.eq(year)]
    if selected.empty:
        raise ValueError("No observations in this reporting period.")
    bought = float(selected["Charge MWh"].sum())
    sold = float(selected["Discharge MWh"].sum())
    cash = float(selected["Cash Flow"].sum())
    cost = float(selected["Cycling Cost"].sum())
    return {
        "Strategy": selected["Strategy"].iloc[0],
        "Gross Margin AUD": cash,
        "Cycling Cost AUD": cost,
        "Net Margin AUD": cash - cost,
        "Energy Bought MWh": bought,
        "Energy Sold MWh": sold,
        "Internal Charged MWh": bought * battery.charge_efficiency,
        "Internal Discharged MWh": sold / battery.discharge_efficiency,
        "EFC": float(selected["EFC"].sum()),
        "Initial SOC MWh": float(selected["SOC Start MWh"].iloc[0]),
        "Final SOC MWh": float(selected["SOC"].iloc[-1]),
        "Net AUD per MW": (cash - cost) / battery.power_mw,
        "Net AUD per Usable MWh": (cash - cost) / battery.usable_mwh,
        "Hours": len(selected) * battery.interval_hours,
    }


def run_strategies(history, start, stop, battery, schedule=None):
    if not 1 <= start < stop <= len(history):
        raise ValueError("Need past data and a nonempty evaluation period.")
    runs = [threshold_strategy(history, start, stop, battery),
            rolling_optimisation(history, start, stop, battery, schedule=schedule),
            perfect_foresight(history.iloc[start:stop], battery)]
    results = {run["Strategy"].iloc[0]: run for run in runs}
    metrics = pd.DataFrame([indicators(run, battery) for run in runs])
    oracle = float(metrics.loc[metrics["Strategy"].eq("Perfect foresight"), "Net Margin AUD"].iloc[0])
    tolerance = max(0.1, abs(oracle) * 2 * config.SOLVER_RELATIVE_GAP)
    if metrics["Net Margin AUD"].max() > oracle + tolerance:
        raise AssertionError("A feasible strategy exceeded the oracle; investigate the comparison.")
    return results, metrics
