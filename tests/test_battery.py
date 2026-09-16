"""Independent checks of causality, optimality and physical accounting."""

from dataclasses import replace
from itertools import product

import numpy as np
import pandas as pd
import pytest

from battery import (Battery, DispatchOptimizer, build_result, forecast_prices, indicators,
                     load_prices, perfect_foresight, rolling_optimisation, run_strategies,
                     synthetic_prices, threshold_strategy)
from dynamic_programming import GridOptimizer


def tiny_battery(**changes):
    return replace(Battery(), capacity_mwh=2, power_mw=1, min_fraction=0,
                   max_fraction=1, initial_fraction=0.5, interval_hours=1,
                   soc_step_mwh=0.5, **changes)


@pytest.mark.parametrize("prices", [[-100, -80, 300, 20], [50, 50, 50, 50], [100, -100, -100, 100]])
def test_dp_matches_exhaustive_enumeration(prices):
    b = tiny_battery(cycle_cost_per_mwh=7)
    states = np.arange(0, 2.01, 0.5)
    best = -np.inf
    for inside in product(states, repeat=len(prices) - 1):
        path = np.r_[b.initial_mwh, inside, b.initial_mwh]
        change = np.diff(path)
        charge = np.maximum(change, 0) / b.charge_efficiency
        discharge = np.maximum(-change, 0) * b.discharge_efficiency
        if max(charge.max(), discharge.max()) > b.max_grid_mwh + 1e-9:
            continue
        value = np.dot(prices, discharge - charge) - b.cycle_cost_per_mwh * np.abs(change).sum() / 2
        best = max(best, value)
    c, d = GridOptimizer(b).solve(prices, 1, 1)
    value = np.dot(prices, d - c) - b.cycle_cost_per_mwh * (b.charge_efficiency*c.sum()+d.sum()/b.discharge_efficiency)/2
    assert value == pytest.approx(best, abs=1e-7)


def test_dp_matches_independent_cvxpy_integer_model():
    b = tiny_battery()
    prices = np.array([-50, -100, 10, 300, 20, 5.])
    c1, d1 = GridOptimizer(b).solve(prices, 1, 1)
    c2, d2 = DispatchOptimizer(b, enforce_soc_grid=True).solve(prices, 1, 1)
    assert prices @ (d1-c1) == pytest.approx(prices @ (d2-c2), abs=1e-5)
    assert not np.any((c1 > 1e-7) & (d1 > 1e-7))


def test_energy_and_efc_by_hand():
    b = tiny_battery()
    data = pd.DataFrame({"SETTLEMENTDATE": pd.date_range("2023-01-01 01:00", periods=2, freq="h"), "RRP": [0.,100.]})
    result = build_result(data, b, [0.5/0.94,0], [0,0.5*0.94], "test")
    m = indicators(result,b)
    assert m["Internal Charged MWh"] == pytest.approx(0.5)
    assert m["EFC"] == pytest.approx(0.25)
    assert m["Final SOC MWh"] == pytest.approx(1)
    assert m["Gross Margin AUD"] == pytest.approx(47)


def test_causality_and_no_mutation():
    b = replace(Battery(), interval_hours=1)
    data = synthetic_prices(days=10,interval_hours=1)
    original = data.copy(deep=True)
    changed = data.copy()
    start, stop = 7*24, 9*24
    changed.loc[start:, "RRP"] += 10000
    assert np.array_equal(forecast_prices(data,start,24,b),forecast_prices(changed,start,24,b))
    first = rolling_optimisation(data,start,stop,b,replan_hours=6)
    other = rolling_optimisation(changed,start,stop,b,replan_hours=6)
    assert np.allclose(first["EGrid"].iloc[:6],other["EGrid"].iloc[:6])
    t1 = threshold_strategy(data,start,stop,b)
    t2 = threshold_strategy(changed,start,stop,b)
    assert t1.EGrid.iloc[0] == t2.EGrid.iloc[0]
    pd.testing.assert_frame_equal(data, original)


def test_oracle_dominates_and_terminal_soc_matches():
    b=replace(Battery(), interval_hours=1)
    data=synthetic_prices(days=9,interval_hours=1)
    runs, metrics=run_strategies(data,7*24,9*24,b)
    assert np.allclose(metrics["Initial SOC MWh"],metrics["Final SOC MWh"])
    assert metrics.iloc[-1]["Net Margin AUD"] >= metrics["Net Margin AUD"].max()-1e-5


def test_nested_grids_improve_oracle():
    b=tiny_battery()
    p=np.array([-80,-20,100,200,0,30.])
    c1,d1=GridOptimizer(b).solve(p,1,1)
    c2,d2=GridOptimizer(replace(b,soc_step_mwh=0.25)).solve(p,1,1)
    assert p@(d2-c2) >= p@(d1-c1)-1e-5


def test_loader_sorts_deduplicates_and_rejects_gaps(tmp_path):
    data=synthetic_prices(days=1)
    data.iloc[::-1].to_csv(tmp_path/'a.csv',index=False)
    data.iloc[:1].to_csv(tmp_path/'b.csv',index=False)
    with pytest.warns(UserWarning):
        clean=load_prices(tmp_path,5/60)
    assert clean.SETTLEMENTDATE.is_monotonic_increasing and len(clean)==288
    (tmp_path/'b.csv').unlink()
    data.drop(index=10).to_csv(tmp_path/'a.csv',index=False)
    with pytest.raises(ValueError,match='Missing intervals'):
        load_prices(tmp_path,5/60)


def test_loader_rejects_conflicts_and_mixed_regions(tmp_path):
    data=synthetic_prices(days=1)
    data.to_csv(tmp_path/'a.csv',index=False)
    bad=data.iloc[:1].copy();bad.RRP+=1;bad.to_csv(tmp_path/'b.csv',index=False)
    with pytest.raises(ValueError,match='Conflicting'):
        load_prices(tmp_path,5/60)
    bad=data.copy();bad.REGION='OTHER';bad.to_csv(tmp_path/'b.csv',index=False)
    with pytest.raises(ValueError,match='Multiple regions'):
        load_prices(tmp_path,5/60)


def test_end_of_month_belongs_to_interval_start_month():
    b=tiny_battery()
    prices=pd.DataFrame({'SETTLEMENTDATE':pd.to_datetime(['2022-12-31 23:00','2023-01-01 00:00']), 'RRP':[0,100]})
    r=build_result(prices,b,[0.5/0.94,0],[0,0.47],'test')
    assert indicators(r,b,year=2022)['Gross Margin AUD']==pytest.approx(47)
