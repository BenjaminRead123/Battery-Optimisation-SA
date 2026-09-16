# Battery arbitrage: South Australia

Three strategies for the same physical battery: a lagged threshold rule, causal
forecast optimisation, and a perfect-information benchmark. The supplied
configuration is 100 MW / 200 MWh, 94% efficiency each way, 20–180 MWh SOC,
100 MWh initial SOC and five-minute settlement intervals.

## Run it

Use Python 3.11 or later. From this directory:

```bash
python -m pip install -r requirements.txt
python main.py --data data --region SA1 --sweep --holdout-start 2023-01-01 --validation-days 90
```

The supplied price files are included in `data/`. They cover October 2021 to
May 2023. Results are written to `results/`. The first seven days are used only
to initialise the forecast. The default evaluation therefore starts on
8 October 2021. The validation window for configuration selection is the
90 days immediately before 1 January 2023. January–May 2023 is held out.

For a shorter first run:

```bash
python main.py --data data --region SA1 --start 2023-01-01 --end 2023-02-01 --output january_results
```

Edit `config.py` and rerun to change the battery or forecasting assumptions.
The SOC limits are derived from capacity, so changing nameplate capacity also
changes the stored-energy limits consistently. The scenario sweep changes
copies of the configuration and leaves the original unchanged.

The optional `--demo` flag explicitly creates synthetic prices. Synthetic
prices are never substituted when real input files are missing.

## Start reading here

- `results/INSIGHTS.md`: measured results, charts and interpretation.
- `results/Battery_Analysis.html`: a self-contained visual report; open in a browser.
- `config.py`: editable assumptions.
- `battery.py`: loading, forecasts, strategies, accounting, and a CVXPY model.
- `dynamic_programming.py`: the faster optimisation used for long histories.
- `main.py`: execution, comparisons and charts.
- `report.py`: the standalone visual report.
- `tests/test_battery.py`: independent optimality, causality and accounting checks.

Each result CSV uses explicit units. The three compressed dispatch CSVs are
ordinary gzip-compressed CSV files, readable with `pd.read_csv(path)`. PNG
charts are ready to view, and SVG copies allow high-quality resizing.

## The optimisation model

Let `c[t]` be MWh imported from the grid, `d[t]` MWh exported, `s[t]` stored MWh,
`p[t]` the price in AUD/MWh, and `dt` the interval length in hours.

```text
maximise sum(p[t] * (d[t] - c[t])
             - cycling_cost * (eta_c*c[t] + d[t]/eta_d) / 2)

s[t+1] = s[t] + eta_c*c[t] - d[t]/eta_d
minimum_soc <= s[t] <= maximum_soc
0 <= c[t] <= power_mw * dt
0 <= d[t] <= power_mw * dt
c[t] = 0 OR d[t] = 0
s[0] = initial_soc
s[T] = initial_soc
```

The objective is an energy trading margin after an optional throughput cost.
It is not a project NPV. No construction cost, fixed operating expense,
financing, tax, FCAS or price impact is included.

Simultaneous charging and discharging must be forbidden. With negative prices,
a model that permits both can collect cash by dissipating energy through
losses without representing the intended operating mode of the battery.

### Dynamic programming

The reported strategies operate on a 0.5 MWh grid of stored-energy states.
This is 321 states for the original battery. Every grid state is inside the
physical SOC bounds, and each transition observes the true efficiency and
power constraints. The grid slightly reduces the maximum usable power in an
individual interval when the unconstrained energy movement is not an exact
multiple of 0.5 MWh.

For each time and current SOC, the algorithm compares every feasible next SOC
implicitly through sliding maxima. The Bellman recursion is:

```text
V[t, current_soc] = max over feasible next_soc of
                   (cash_flow(current_soc, next_soc, price[t])
                    - cycling_cost(current_soc, next_soc)
                    + V[t+1, next_soc])
```

Only the required final SOC has value zero at the terminal time; other
terminal states are infeasible. Working backwards produces the optimum and
records the next-state policy. Following that policy forwards gives dispatch.
The sliding-maxima implementation exploits linear revenue in each charge or
discharge direction, reducing complexity to O(intervals × SOC states).
Numba compiles the loops. No neural network, training or approximate learning
algorithm is involved.

**“Perfect foresight” means all future prices are known and the entire selected
period is jointly optimised on this SOC grid.** It is the exact finite-state
optimum to floating-point precision. It is not a certificate of the continuous
SOC optimum. There are no artificial daily resets in the full-period benchmark.
The parameter sweep includes a 1.0/0.5/0.25 MWh holdout precision comparison.

### CVXPY alternative

`DispatchOptimizer` implements the continuous-SOC formulation as a
mixed-integer linear programme with a binary operating mode, solved by HiGHS.
No nonlinear optimisation is needed. It can optionally enforce the same SOC
grid, which is used to cross-check dynamic programming on small cases.

```python
from battery import Battery, DispatchOptimizer, load_prices, build_result

battery = Battery()
prices = load_prices('data', battery.interval_hours, region='SA1').iloc[:288]
charge, discharge = DispatchOptimizer(battery).solve(
    prices['RRP'].to_numpy(), battery.initial_mwh, battery.initial_mwh
)
result = build_result(prices, battery, charge, discharge, 'CVXPY continuous SOC')
```

A 20-month, five-minute mixed-integer problem can be expensive. A solver that
hits the time limit raises an error rather than labelling an unproven result
“perfect”. For this reason, the long-history pipeline uses finite-state dynamic
programming. Reduce the CVXPY horizon or adjust the time limit deliberately.

References: [CVXPY mixed-integer constraints](https://www.cvxpy.org/tutorial/constraints/index.html),
[CVXPY solver options](https://www.cvxpy.org/tutorial/solvers/index.html).

## Information timing and comparisons

1. **Threshold (lagged)** retains the $30 and $150 thresholds, but uses the
   previous completed interval's price. The original loop traded at the price
   it was simultaneously using as its signal, an idealised information advantage.
2. **Forecast optimisation** takes the median of matching five-minute slots
   over the previous seven days, solves a 24-hour optimisation and executes
   the first hour. It then updates the forecast using newly observed prices.
   Data at or after the decision time is excluded. The final SOC of each
   planning horizon is set to the original SOC; this is an explicit boundary
   policy that can sacrifice value and could be improved with a terminal value.
3. **Perfect foresight** uses actual prices over the full evaluation period.
   It is a hindsight benchmark, not an executable forecast or a revenue promise.

The full backtest starts and ends all strategies at the same SOC. The
threshold strategy has a reachability safeguard near the end that restores
SOC even if its thresholds would not. This avoids rewarding an algorithm for
selling initial inventory and leaving the battery empty. Terminal-restoration
costs are included in the results.

The configuration sweep ranks designs only on the validation-period forecast
strategy margin, refines the leading three on a grid with half the SOC step,
then evaluates the selected design against the original on the later holdout
at the same finer precision. The original and selected configuration files
are provided as `config.py` and `results/config_selected.py`. The latter can be
copied over `config.py` to try the selected case; it is not a claim of best
investment economics. Each validation/holdout run starts and finishes at its own
specified initial SOC. Only historical observations preceding a decision enter
its forecast; observations from earlier in the holdout may be used as time
passes. The holdout is not used to choose the design.

The largest margin is not necessarily the best investment. The grid compares
50, 100 and 150 MW with 100, 200 and 400 MWh; it has no construction budget or
equipment prices. Higher efficiency or a wider SOC window may also cost more
or affect warranties and degradation. The assumptions sweep is a sensitivity
analysis, not a recommendation to change equipment specifications freely.

## Data and accounting checks

- CSVs must have `SETTLEMENTDATE` and `RRP`. Standard `REGION`/`REGIONID`
  identifiers are respected, and multiple regions require an explicit selection.
- Rows are sorted. Identical duplicate timestamps are removed with a warning.
  Conflicting duplicates, missing intervals, missing prices and frequency
  mismatches stop execution. No filling, resampling or price clipping occurs.
- The CSV timestamps are interval-ending NEM time (AEST year-round). A midnight
  timestamp belongs to the preceding interval, so monthly and yearly reporting
  use `Interval Start = SETTLEMENTDATE - interval_hours`.
- `EGrid = discharge - charge`. Cash flow is `price * EGrid`; charging at a
  negative price produces positive cash flow.
- Battery-side energy charged is `grid_imports * charge_efficiency`.
  Battery-side energy discharged is `grid_exports / discharge_efficiency`.
- EFC is total internal charge/discharge throughput divided by twice usable
  capacity. Counts use this definition consistently, not nameplate capacity.
- Internal energy, SOC, power, non-simultaneity and final SOC are checked.
- Dynamic programming is compared with exhaustive enumeration and independent
  CVXPY optimisation on small cases. Forecast causality is checked by changing
  unseen future prices and verifying that earlier decisions are unchanged.

Run checks with:

```bash
python -m pytest -q
```

NumPy arrays are intentionally retained for sequential simulation and solver
work. They are more suitable than growing DataFrames inside a time loop. Pandas
handles input validation, reporting and export. There are no global price
DataFrames and no function mutates its caller's price data.

## Limits that matter for an advisory discussion

The model assumes a price-taking battery can transact its scheduled quantities
at observed SA regional prices. It does not simulate bidding, dispatch targets,
network constraints, marginal loss factors, auxiliary consumption or outages.
A 100–150 MW asset could materially affect the local prices it trades against.
The uploaded 2021–2023 period is historical and includes the June 2022 market
suspension; intervention directions and compensation are not modelled.

Prices, forecast quality and battery competition have changed since this
period. The results demonstrate methods and historical scenarios, not current
revenue potential. Improving the forecast, including realistic costs and
constraints, and validating on later data are separate steps.
