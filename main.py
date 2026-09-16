"""Run all strategies, compare configurations and produce charts and CSVs.

Examples:
    python main.py --data data --region SA1 --sweep
    python main.py --data data --start 2023-01-01 --end 2023-02-01
    python main.py --demo --sweep
"""

import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

import config
from battery import (Battery, forecast_schedule, indicators, load_prices, perfect_foresight,
                     rolling_optimisation, run_strategies, synthetic_prices)
from report import write_html_report

COLORS = {"Threshold (lagged)": "#64748b", "Forecast optimisation": "#007f86", "Perfect foresight": "#8051b5"}


def log(message):
    print(message, flush=True)


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.labelcolor": "#334155",
                         "axes.edgecolor": "#cbd5e1", "text.color": "#0f172a",
                         "axes.grid": True, "grid.alpha": 0.20,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})


def save_figure(fig, folder, name):
    fig.savefig(folder / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(folder / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_results(runs, metrics, battery, folder, label):
    style()
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.5), gridspec_kw={"height_ratios": [1.3, 1]}, layout="constrained")
    first=next(iter(runs.values()))
    period=f"{first['Interval Start'].iloc[0]:%d %b %Y}–{first['Interval Start'].iloc[-1]:%d %b %Y}"
    fig.suptitle(f"Battery strategy comparison · {label}\n{period}", fontsize=16, fontweight="bold")
    monthly = []
    for name, data in runs.items():
        axes[0].plot(data["Interval Start"], data["Cumulative Margin"] / 1e6, label=name,
                     color=COLORS[name], linewidth=1.8)
        totals = data.groupby(data["Interval Start"].dt.to_period("M"))["Net Margin"].sum()
        monthly.append(totals.rename(name))
    axes[0].set_ylabel("Cumulative net trading margin (AUD million)")
    axes[0].axhline(0, color="#94a3b8", linewidth=.7)
    axes[0].legend(loc="upper left", frameon=False, ncol=3)
    axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    month_table = pd.concat(monthly, axis=1)
    month_table.to_csv(folder / "monthly_margins.csv", index_label="Month")
    x = np.arange(len(month_table))
    for i, name in enumerate(runs):
        axes[1].bar(x + (i-1)*.25, month_table[name]/1e6, width=.24, color=COLORS[name])
    axes[1].set_xticks(x, [str(m) for m in month_table.index], rotation=45, ha="right")
    axes[1].set_ylabel("Monthly net trading margin (AUD million)")
    axes[1].axhline(0, color="#94a3b8", linewidth=.7)
    axes[1].set_title(f"{battery.power_mw:g} MW / {battery.capacity_mwh:g} MWh; {battery.usable_mwh:g} MWh usable", loc="left", fontsize=12)
    fig.supxlabel(f"Same {battery.soc_step_mwh:g} MWh SOC grid and end-of-period SOC for all strategies. "
                  "Energy only; excludes capital and fixed operating costs.", fontsize=9)
    save_figure(fig, folder, "strategy_comparison")

    data = runs["Perfect foresight"]
    day = data.groupby(data["Interval Start"].dt.normalize())["RRP"].std().idxmax()
    mask = data["Interval Start"].between(day-pd.Timedelta(days=1), day+pd.Timedelta(days=2), inclusive="left")
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True, layout="constrained")
    fig.suptitle("How the strategies respond to a volatile period", fontsize=17, fontweight="bold")
    t = data.loc[mask, "SETTLEMENTDATE"]
    axes[0].plot(t, data.loc[mask, "RRP"], color="#1e293b", linewidth=1)
    axes[0].set_ylabel("Actual price\n(AUD/MWh)")
    axes[0].set_title(f"{day:%d %B %Y}: day with greatest price standard deviation, plus adjacent days", fontsize=10, loc="left")
    for ax, (name, frame) in zip(axes[1:], runs.items()):
        ax.step(t, frame.loc[mask, "SOC"], where="post", color=COLORS[name], linewidth=1.1)
        ax.set_ylabel("Stored energy\n(MWh)")
        ax.set_ylim(battery.min_mwh-5, battery.max_mwh+5)
        ax.set_title(name, loc="left", fontsize=11, color=COLORS[name])
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=12))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    fig.supxlabel("Source time basis: NEM time (AEST year-round). SOC is measured at each interval end.", fontsize=9)
    save_figure(fig, folder, "dispatch_example")


def plot_sweep(table, folder, period):
    selected = table.loc[table.Strategy.eq("Forecast optimisation")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3), layout="constrained")
    for ax, field, scale, title in zip(axes, ["Net Margin AUD", "Net AUD per Usable MWh"], [1e6, 1],
                                     ["Total margin (AUD million)", "Margin per usable MWh (AUD)"]):
        pivot=selected.pivot(index="Power MW", columns="Capacity MWh", values=field).sort_index()
        values=pivot.to_numpy()/scale
        ax.imshow(values, cmap="YlGnBu", aspect="auto")
        for (i,j), v in np.ndenumerate(values):
            span=values.max()-values.min()
            dark=(v-values.min())/span > .6 if span else False
            ax.text(j,i,f"{v:,.2f}" if scale==1e6 else f"{v:,.0f}",ha="center",va="center",
                    color="white" if dark else "#0f172a",fontsize=13,fontweight="bold")
        ax.set_xticks(range(len(pivot.columns)), [f"{c:g}" for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)), [f"{p:g}" for p in pivot.index])
        ax.set_xlabel("Nameplate capacity (MWh)"); ax.set_ylabel("Grid power (MW)")
        ax.set_title(title,loc="left",fontsize=12); ax.grid(False)
    fig.suptitle(f"Battery size comparison · forecast optimisation\nValidation: {period}",fontsize=16,fontweight="bold")
    fig.supxlabel("Validation period only. Prices are treated as unaffected by battery size. Construction costs excluded.",fontsize=9)
    save_figure(fig,folder,"configuration_comparison")


def run_sweep(history, start, stop, battery, folder, holdout_start, validation_days):
    times = history.SETTLEMENTDATE - pd.Timedelta(hours=battery.interval_hours)
    if holdout_start:
        split = int(times.searchsorted(pd.Timestamp(holdout_start)))
    else:
        split = start + int((stop-start)*config.VALIDATION_FRACTION)
        per_day=round(24/battery.interval_hours)
        split = (split // per_day) * per_day
    if not start < split < stop:
        raise ValueError("The holdout start must fall strictly inside the evaluation period.")
    validation_start = max(start, split-round(validation_days*24/battery.interval_hours))
    schedule=forecast_schedule(history,validation_start,split,battery)
    rows=[]
    for power in config.SWEEP_POWER_MW:
        for capacity in config.SWEEP_CAPACITY_MWH:
            candidate=replace(battery,power_mw=power,capacity_mwh=capacity)
            log(f"Validation: {power:g} MW / {capacity:g} MWh")
            _, metrics=run_strategies(history,validation_start,split,candidate,schedule=schedule)
            for row in metrics.to_dict("records"):
                rows.append({"Power MW":power,"Capacity MWh":capacity,**row})
            pd.DataFrame(rows).to_csv(folder/'configuration_validation.csv',index=False)
    table=pd.DataFrame(rows)
    eligible=table.loc[table.Strategy.eq('Forecast optimisation')]
    ranked=eligible.sort_values(['Net Margin AUD','Capacity MWh','Power MW'],ascending=[False,True,True])
    coarse_best=ranked.iloc[0]
    refined_rows=[]
    # Check the leading designs on a finer grid before consulting held-out data.
    for _,row in ranked.head(3).iterrows():
        candidate=replace(battery,power_mw=float(row['Power MW']),capacity_mwh=float(row['Capacity MWh']),
                          soc_step_mwh=battery.soc_step_mwh/2)
        log(f"Validation refinement: {candidate.power_mw:g} MW / {candidate.capacity_mwh:g} MWh")
        r=rolling_optimisation(history,validation_start,split,candidate,schedule=schedule)
        refined_rows.append({'Power MW':candidate.power_mw,'Capacity MWh':candidate.capacity_mwh,
                             'SOC Step MWh':candidate.soc_step_mwh,**indicators(r,candidate)})
    refined=pd.DataFrame(refined_rows)
    refined.to_csv(folder/'validation_precision.csv',index=False)
    best=refined.sort_values(['Net Margin AUD','Capacity MWh','Power MW'],ascending=[False,True,True]).iloc[0]
    chosen=replace(battery,power_mw=float(best['Power MW']),capacity_mwh=float(best['Capacity MWh']),
                   soc_step_mwh=battery.soc_step_mwh/2)
    log(f"Selected after validation refinement: {chosen.power_mw:g} MW / {chosen.capacity_mwh:g} MWh")
    candidate_text=(Path(__file__).parent/'config.py').read_text()
    for key,value in [('BATTERY_POWER_MW',chosen.power_mw),('BATTERY_CAPACITY_MWH',chosen.capacity_mwh),
                      ('SOC_STEP_MWH',chosen.soc_step_mwh)]:
        candidate_text=re.sub(rf'(?m)^{key} = .*$',f'{key} = {value}',candidate_text)
    (folder/'config_selected.py').write_text('# Selected on historical validation trading margin; excludes capital costs.\n'+candidate_text)
    holdout_schedule=forecast_schedule(history,split,stop,battery)
    held_rows=[]
    for name, candidate in [('Original configuration',replace(battery,soc_step_mwh=chosen.soc_step_mwh)),('Selected configuration',chosen)]:
        log(f"Holdout: {name}")
        _, m=run_strategies(history,split,stop,candidate,schedule=holdout_schedule)
        for row in m.to_dict('records'):
            held_rows.append({'Configuration':name,'Power MW':candidate.power_mw,'Capacity MWh':candidate.capacity_mwh,
                              'SOC Step MWh':candidate.soc_step_mwh,**row})
    holdout=pd.DataFrame(held_rows)
    holdout.to_csv(folder/'configuration_holdout.csv',index=False)
    period=f"{times.iloc[validation_start]:%d %b %Y}–{times.iloc[split-1]:%d %b %Y}"
    plot_sweep(table,folder,period)
    fig,ax=plt.subplots(figsize=(10,5.5),layout='constrained')
    for i,(name,color) in enumerate(COLORS.items()):
        subset=holdout.loc[holdout.Strategy.eq(name)]
        bars=ax.bar(np.arange(2)+(i-1)*.24,subset['Net Margin AUD']/1e6,width=.23,color=color,label=name)
        ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
    ax.set_xticks([0,1],[f'Original\n{battery.power_mw:g} MW / {battery.capacity_mwh:g} MWh',
                          f'Selected on validation\n{chosen.power_mw:g} MW / {chosen.capacity_mwh:g} MWh'])
    ax.set_ylabel('Net trading margin (AUD million)')
    ax.set_title(f"Later test: {times.iloc[split]:%d %b %Y}–{times.iloc[stop-1]:%d %b %Y}",loc='left',fontsize=16)
    ax.set_ylim(top=max(holdout['Net Margin AUD']/1e6)*1.25)
    ax.legend(frameon=False,loc='upper left')
    fig.supxlabel(f'All held-out strategies use a {chosen.soc_step_mwh:g} MWh SOC grid. Construction and fixed costs excluded.',fontsize=9)
    save_figure(fig,folder,'holdout_comparison')

    # One-at-a-time engineering assumptions; NOT parameters fitted to the holdout.
    variants=[('Original',battery),
              ('Efficiency 90% each way',replace(battery,charge_efficiency=.90,discharge_efficiency=.90)),
              ('Efficiency 98% each way',replace(battery,charge_efficiency=.98,discharge_efficiency=.98)),
              ('SOC window 20–80%',replace(battery,min_fraction=.20,max_fraction=.80)),
              ('SOC window 5–95%',replace(battery,min_fraction=.05,max_fraction=.95)),
              ('Cycling cost $10/MWh',replace(battery,cycle_cost_per_mwh=10)),
              ('Cycling cost $30/MWh',replace(battery,cycle_cost_per_mwh=30))]
    sensitivity=[]
    for name, candidate in variants:
        log(f"Sensitivity: {name}")
        result=rolling_optimisation(history,validation_start,split,candidate,schedule=schedule)
        sensitivity.append({'Scenario':name,**indicators(result,candidate)})
    sens=pd.DataFrame(sensitivity);sens.to_csv(folder/'sensitivity.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,5.4),layout='constrained')
    for ax,field,title,scale in zip(axes,['Net Margin AUD','EFC'],['Net margin (AUD million)','Equivalent full cycles'],[1e6,1]):
        ax.barh(sens.Scenario,sens[field]/scale,color='#007f86');ax.invert_yaxis()
        ax.set_title(title,loc='left',fontsize=12)
    axes[1].set_yticklabels([])
    fig.suptitle(f'Effect of changing one assumption\nValidation: {period}',fontsize=16,fontweight='bold')
    fig.supxlabel('Efficiency, SOC windows and cycling costs are scenarios; equipment prices and warranty trade-offs are not modelled.',fontsize=9)
    save_figure(fig,folder,'sensitivity')

    # Finer grids contain all coarser-grid feasible paths: oracle value cannot fall.
    refinements=[]
    for step in [1.0,0.5,0.25]:
        log(f"Precision check: {step:g} MWh grid, holdout oracle")
        candidate=replace(battery,soc_step_mwh=step)
        result=perfect_foresight(history.iloc[split:stop],candidate)
        refinements.append({'SOC Step MWh':step,**indicators(result,candidate)})
    refinement=pd.DataFrame(refinements);refinement.to_csv(folder/'grid_precision.csv',index=False)
    if np.any(np.diff(refinement['Net Margin AUD']) < -0.1):
        raise AssertionError('Refining nested SOC grids reduced the optimal margin.')
    return {'validation_start':str(times.iloc[validation_start]),'holdout_start':str(times.iloc[split]),
            'selected_battery':asdict(chosen),'validation':table,'holdout':holdout,
            'sensitivity':sens,'precision':refinement,'validation_refinement':refined,
            'coarse_winner_power_mw':float(coarse_best['Power MW']),
            'coarse_winner_capacity_mwh':float(coarse_best['Capacity MWh'])}


def markdown_table(df, columns):
    rows=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for row in df[columns].itertuples(index=False,name=None):
        rows.append('| '+' | '.join(f'{x:,.2f}' if isinstance(x,(float,np.floating)) else str(x) for x in row)+' |')
    return '\n'.join(rows)


def write_report(history,runs,metrics,battery,folder,metadata,sweep):
    first=next(iter(runs.values()))
    days=len(first)*battery.interval_hours/24
    forecast=metrics.loc[metrics.Strategy.eq('Forecast optimisation')].iloc[0]
    threshold=metrics.loc[metrics.Strategy.eq('Threshold (lagged)')].iloc[0]
    oracle=metrics.loc[metrics.Strategy.eq('Perfect foresight')].iloc[0]
    share=100*forecast['Net Margin AUD']/oracle['Net Margin AUD'] if oracle['Net Margin AUD'] else np.nan
    top=first.RRP.quantile(.99)
    event_rows=[]
    for name,data in runs.items():
        event_rows.append({'Strategy':name,'Cash flow at top 1% prices AUD':data.loc[data.RRP>=top,'Cash Flow'].sum(),
                           'Cash flow at negative prices AUD':data.loc[data.RRP<0,'Cash Flow'].sum()})
    pd.DataFrame(event_rows).to_csv(folder/'price_event_contributions.csv',index=False)
    monthly=pd.read_csv(folder/'monthly_margins.csv')
    loss_months=int((monthly['Forecast optimisation']<0).sum())
    lines=[
        '# Battery arbitrage analysis',
        f"Data: **{metadata['data_label']}**. {len(history):,} loaded intervals; {len(first):,} evaluated "
        f"five-minute intervals ({days:g} days). Evaluation: {metadata['evaluation_start']} to {metadata['evaluation_end_exclusive']} (exclusive).",
        '## Results',
        markdown_table(metrics,['Strategy','Net Margin AUD','Energy Bought MWh','Energy Sold MWh','EFC']),
        f"The causal forecast strategy captures **{share:.1f}%** of the perfect-information margin on the same "
        f"{battery.soc_step_mwh:g} MWh SOC grid. Its margin differs from the lagged threshold rule by "
        f"**AUD {forecast['Net Margin AUD']-threshold['Net Margin AUD']:,.0f}**. It has {loss_months} "
        f"negative-margin calendar {'month' if loss_months==1 else 'months'}.",
        '![Strategy comparison](strategy_comparison.png)',
        '## What the strategies know',
        '- **Threshold (lagged):** charge below $30/MWh, discharge above $150/MWh, using the previous completed interval’s price. '
        'The original code used the realised price of the interval being traded; that is an information advantage. '
        'Terminal reachability can override the rule near the end.',
        '- **Forecast optimisation:** a median of matching five-minute slots over the previous seven days; '
        'reforecast every hour, optimise a 24-hour horizon, execute the first hour. No current or future realised prices enter its decisions. '
        'The end of every planning horizon targets the initial SOC. A mathematical optimum for an inaccurate forecast can still lose money.',
        '- **Perfect foresight:** sees every actual price in the entire evaluation period and optimises it jointly. '
        'It is globally optimal on the stated SOC grid, not a proven optimum for continuous SOC. '
        'No forced daily SOC resets or hidden daily decomposition are used.',
        'All three strategies have identical grid power limits, efficiencies, SOC limits, SOC grid and final SOC. '
        'Charging and discharging cannot occur simultaneously. Positive EGrid means exported energy.',
        '![Dispatch example](dispatch_example.png)',
        '## What the battery assumptions mean',
        f"The supplied {battery.power_mw:g} MW / {battery.capacity_mwh:g} MWh battery has "
        f"**{battery.usable_mwh:g} MWh usable stored energy**. With discharge efficiency "
        f"{battery.discharge_efficiency:.0%}, that is {battery.usable_mwh*battery.discharge_efficiency:g} MWh "
        f"delivered to the grid, or {battery.usable_mwh*battery.discharge_efficiency/battery.power_mw:.3f} hours "
        'at rated power before accounting for SOC discretisation.',
        f"Round-trip efficiency is **{battery.charge_efficiency*battery.discharge_efficiency:.2%}**. "
        'The original cycle calculation divided energy bought by charging efficiency; the corrected value multiplies it.',
        'EFC = (grid imports × charging efficiency + grid exports ÷ discharge efficiency) / (2 × usable stored capacity). '
        'Cycling cost is the configured rate multiplied by half the internal two-way throughput. '
        'The base cycling-cost assumption is zero, reproducing the supplied model.',
    ]
    if sweep:
        held=sweep['holdout'].loc[sweep['holdout'].Strategy.eq('Forecast optimisation')]
        selected=sweep['selected_battery']
        precision=sweep['precision']
        coarse=precision.loc[precision['SOC Step MWh'].eq(.5),'Net Margin AUD'].iloc[0]
        fine=precision.loc[precision['SOC Step MWh'].eq(.25),'Net Margin AUD'].iloc[0]
        lines += ['## Configuration comparisons',
                  f"Nine power/capacity combinations were compared on {sweep['validation_start']} to "
                  f"{sweep['holdout_start']} (exclusive). At {battery.soc_step_mwh:g} MWh precision, "
                  f"{sweep['coarse_winner_power_mw']:g} MW / {sweep['coarse_winner_capacity_mwh']:g} MWh ranked first. "
                  'The leading three designs were then rerun at half the SOC step, using validation data only. '
                  f"This selected **{selected['power_mw']:g} MW / {selected['capacity_mwh']:g} MWh** before evaluating the holdout.",
                  '![Configuration comparison](configuration_comparison.png)',
                  markdown_table(sweep['validation_refinement'],['Power MW','Capacity MWh','SOC Step MWh','Net Margin AUD']),
                  'The ranking changes with numerical resolution. Treat the leading power ratings as a close comparison, '
                  'not a robust engineering design recommendation. The original and selected configurations are both evaluated '
                  f"on the later holdout at {selected['soc_step_mwh']:g} MWh precision.",
                  markdown_table(held,['Configuration','Power MW','Capacity MWh','Net Margin AUD','Net AUD per Usable MWh']),
                  '![Holdout comparison](holdout_comparison.png)',
                  '**This identifies the highest trading margin among the tested designs. It does not establish the most valuable investment.** '
                  'Larger batteries cost more, and the model assumes their trades do not change market prices. '
                  'The per-MWh comparison helps reveal diminishing utilisation but is not a substitute for NPV or a budget constraint.',
                  '![Sensitivity comparison](sensitivity.png)',
                  '## Numerical precision',
                  markdown_table(precision,['SOC Step MWh','Net Margin AUD']),
                  f"Refining the holdout oracle from 0.5 to 0.25 MWh changes margin by "
                  f"**{100*(fine/coarse-1):.2f}%**. This is an observed refinement effect, not a bound on the continuous optimum. "
                  'Finer grids take more memory and time. The original 0.5 MWh comparison remains consistent across strategies.']
    lines += ['## How to interpret these results',
              '- These are historical energy-trading margins, not project profit, current revenue forecasts or a bankable valuation. '
              'They exclude capital expenditure, fixed O&M, financing, tax, auxiliary loads, network charges, loss factors, outages, '
              'grid constraints, bid/dispatch mechanics, price impact and FCAS revenue. Efficiency is constant and degradation does not reduce capacity.',
              '- The uploaded period includes the June 2022 market suspension. Its observed prices are retained; '
              'intervention directions and compensation are not modelled. Results from this unusual period should not be extrapolated mechanically. '
              '[AEMO’s account](https://www.aemo.com.au/newsroom/news-updates/june-2022-series-of-events-and-reports).',
              '- The forecast is deliberately simple. Better validation should compare persistence, seasonal profiles and genuine archived forecasts '
              'using the same information cutoff, across seasons and later market regimes. Optimisation alone does not guarantee a better strategy.',
              '- For an advisory discussion, explain the assumptions that drive the result: forecast error, usable duration, '
              'terminal SOC, cycling cost, grid restrictions and what equipment would cost. Report both upside and weak periods.',
              '## Reproduce',
              'See README.md for commands and equations. The CSV outputs contain interval-level dispatch, monthly results, '
              'configuration comparisons and assumptions. No source price rows were interpolated, resampled or clipped. '
              'Calendar reporting uses interval-start dates so midnight interval endings are assigned to the correct month.']
    (folder/'INSIGHTS.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path(__file__).parent/'data')
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'results')
    parser.add_argument('--region',default=None)
    parser.add_argument('--start',help='Inclusive interval-start date, e.g. 2023-01-01')
    parser.add_argument('--end',help='Exclusive interval-start date, e.g. 2023-06-01')
    parser.add_argument('--demo',action='store_true',help='Explicitly generate synthetic prices')
    parser.add_argument('--sweep',action='store_true')
    parser.add_argument('--holdout-start',help='Chosen before reviewing held-out results')
    parser.add_argument('--validation-days',type=int,default=90)
    args=parser.parse_args()
    if args.validation_days < 1:
        parser.error('--validation-days must be positive')
    folder=args.output; folder.mkdir(parents=True,exist_ok=True)
    battery=Battery(); t0=time.perf_counter()
    if args.demo:
        history=synthetic_prices(); label='SYNTHETIC DEMONSTRATION'
        history.to_csv(folder/'synthetic_prices.csv',index=False)
    else:
        history=load_prices(args.data,battery.interval_hours,args.region)
        label=f"{history.REGION.iloc[0]} historical prices" if 'REGION' in history else 'Historical prices (region unspecified)'
        manifest=[{'File':p.name,'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'Bytes':p.stat().st_size} for p in sorted(args.data.glob('*.csv'))]
        pd.DataFrame(manifest).to_csv(folder/'source_manifest.csv',index=False)
    times=history.SETTLEMENTDATE-pd.Timedelta(hours=battery.interval_hours)
    warmup=round(config.FORECAST_LOOKBACK_DAYS*24/battery.interval_hours)
    start=max(warmup,int(times.searchsorted(pd.Timestamp(args.start)))) if args.start else warmup
    stop=int(times.searchsorted(pd.Timestamp(args.end))) if args.end else len(history)
    if not 1 <= start < stop <= len(history):
        parser.error('No evaluation data after the seven-day forecast warm-up and date filters.')
    metadata={'data_label':label,'battery':asdict(battery),'loaded_rows':len(history),'evaluated_rows':stop-start,
              'evaluation_start':str(times.iloc[start]),'evaluation_end_exclusive':str(history.SETTLEMENTDATE.iloc[stop-1]),
              'forecast_lookback_days':config.FORECAST_LOOKBACK_DAYS,
              'horizon_hours':config.FORECAST_HORIZON_HOURS,'replan_hours':config.REPLAN_HOURS,
              'buy_below':config.BUY_BELOW,'sell_above':config.SELL_ABOVE,
              'negative_price_share':float((history.RRP<0).mean()),'min_price':float(history.RRP.min()),'max_price':float(history.RRP.max())}
    metadata['software_versions']={name:importlib.metadata.version(name) for name in
                                  ['numpy','pandas','cvxpy','highspy','matplotlib','numba']}
    (folder/'assumptions.json').write_text(json.dumps(metadata,indent=2))
    log(f"Loaded {len(history):,} rows. Evaluating {stop-start:,} intervals: {metadata['evaluation_start']} to {metadata['evaluation_end_exclusive']}.")
    log('Building causal forecasts and evaluating all three strategies...')
    runs,metrics=run_strategies(history,start,stop,battery)
    metrics.to_csv(folder/'strategy_metrics.csv',index=False)
    for name,data in runs.items():
        slug=name.lower().replace(' ','_').replace('(','').replace(')','')
        data.to_csv(folder/f'{slug}_dispatch.csv.gz',index=False,compression='gzip')
    log(metrics[['Strategy','Net Margin AUD','EFC']].to_string(index=False))
    plot_results(runs,metrics,battery,folder,label)
    log(f'Baseline and charts completed in {time.perf_counter()-t0:.1f}s.')
    sweep=run_sweep(history,start,stop,battery,folder,args.holdout_start,args.validation_days) if args.sweep else None
    if sweep:
        metadata['sweep']={k:v for k,v in sweep.items() if not isinstance(v,pd.DataFrame)}
        (folder/'assumptions.json').write_text(json.dumps(metadata,indent=2))
    write_report(history,runs,metrics,battery,folder,metadata,sweep)
    write_html_report(folder)
    log(f'Finished in {time.perf_counter()-t0:.1f}s. Outputs: {folder.resolve()}')


if __name__=='__main__':
    main()
