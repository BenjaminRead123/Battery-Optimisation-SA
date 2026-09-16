"""Create a self-contained HTML report with embedded, exportable charts."""

import base64
import html
import json
from pathlib import Path

import pandas as pd


def write_html_report(folder):
    folder=Path(folder)
    metadata=json.loads((folder/'assumptions.json').read_text())
    metrics=pd.read_csv(folder/'strategy_metrics.csv')
    lookup=metrics.set_index('Strategy')
    forecast=lookup.loc['Forecast optimisation','Net Margin AUD']
    oracle=lookup.loc['Perfect foresight','Net Margin AUD']
    threshold=lookup.loc['Threshold (lagged)','Net Margin AUD']
    battery=metadata['battery']
    market='South Australia' if 'SA1' in metadata['data_label'] else metadata['data_label']
    usable=(battery['max_fraction']-battery['min_fraction'])*battery['capacity_mwh']
    delivered=usable*battery['discharge_efficiency']
    def money(value):
        return f'A${value/1e6:,.2f}m'
    def image(name,alt):
        encoded=base64.b64encode((folder/f'{name}.png').read_bytes()).decode()
        return f'<img src="data:image/png;base64,{encoded}" alt="{html.escape(alt)}">'
    def table(frame,columns):
        return frame[columns].to_html(index=False,border=0,float_format=lambda value:f'{value:,.2f}',classes='data-table')
    monthly=pd.read_csv(folder/'monthly_margins.csv')
    weak=monthly.sort_values('Forecast optimisation').iloc[0]
    sections=[f'''
    <header><p class="eyebrow">HISTORICAL ENERGY MODELLING</p>
    <h1>Battery arbitrage<br>{html.escape(market)}</h1>
    <p class="lead">Three strategies, the same battery, and the value of better information.</p>
    <p>{html.escape(metadata['evaluation_start'][:10])} to {html.escape(metadata['evaluation_end_exclusive'][:10])} (exclusive).
    {metadata['evaluated_rows']:,} five-minute intervals, after a seven-day forecast warm-up.</p>
    <div class="note">These are historical energy-trading margins. Construction, fixed operating costs,
    financing, grid constraints, price impact and FCAS are excluded. The baseline cycling-cost assumption is zero.</div>
    </header>
    <section><h2>What the original battery earned in the model</h2>
    <div class="cards">
      <div><span>Lagged threshold rule</span><strong>{money(threshold)}</strong></div>
      <div><span>Forecast optimisation</span><strong>{money(forecast)}</strong></div>
      <div><span>Perfect foresight</span><strong>{money(oracle)}</strong></div>
    </div>
    <p>The forecast strategy improves on the threshold rule by <b>{100*(forecast/threshold-1):.1f}%</b>,
    capturing <b>{100*forecast/oracle:.1f}%</b> of the full-period perfect-information margin on the same
    {battery['soc_step_mwh']:g} MWh SOC grid.</p>
    {image('strategy_comparison','Cumulative and monthly trading margins for all three strategies')}
    <p>The weakest forecast month is <b>{weak['Month']}</b>: A${weak['Forecast optimisation']:,.0f},
    compared with A${weak['Threshold (lagged)']:,.0f} for the threshold rule. A mathematically optimal
    schedule can perform poorly when the price forecast is wrong.</p>
    {table(metrics,['Strategy','Net Margin AUD','Energy Bought MWh','Energy Sold MWh','EFC'])}
    </section>
    <section><h2>What each strategy knows</h2>
    <table class="data-table"><thead><tr><th>Strategy</th><th>Information and decisions</th></tr></thead><tbody>
    <tr><td>Threshold</td><td>Uses the previous completed price. Charges below $30/MWh and discharges above $150/MWh.</td></tr>
    <tr><td>Forecast optimisation</td><td>Forecasts each five-minute price using the median matching slot over the previous seven days.
    Optimises 24 hours, executes one hour, and repeats.</td></tr>
    <tr><td>Perfect foresight</td><td>Knows every future price and optimises the complete evaluation period on the stated SOC grid.</td></tr>
    </tbody></table>
    <p>All strategies start and finish at the same stored energy. No simultaneous charging and discharging is allowed.
    The threshold rule has a final-inventory safeguard; the rolling optimiser targets initial SOC at the end of each planning horizon.</p>
    <p>The original loop used the price of the interval being traded as its own signal. The cleaned threshold strategy uses a one-interval lag
    so it does not see that interval’s outcome before deciding.</p>
    {image('dispatch_example','Prices and stored energy during a highly volatile three-day example')}
    </section>''']
    if (folder/'configuration_holdout.csv').exists() and 'sweep' in metadata:
        sweep=metadata['sweep']
        held=pd.read_csv(folder/'configuration_holdout.csv')
        fore=held.loc[held.Strategy.eq('Forecast optimisation')].set_index('Configuration')
        old=fore.loc['Original configuration','Net Margin AUD']
        new=fore.loc['Selected configuration','Net Margin AUD']
        chosen=sweep['selected_battery']
        fine=pd.read_csv(folder/'validation_precision.csv')
        precision=pd.read_csv(folder/'grid_precision.csv')
        coarse_value=precision.loc[precision['SOC Step MWh'].eq(.5),'Net Margin AUD'].iloc[0]
        fine_value=precision.loc[precision['SOC Step MWh'].eq(.25),'Net Margin AUD'].iloc[0]
        sensitivity=pd.read_csv(folder/'sensitivity.csv')
        reduced=sensitivity.set_index('Scenario')
        cycle_reduction=100*(1-reduced.loc['Cycling cost $10/MWh','EFC']/reduced.loc['Original','EFC'])
        sections.append(f'''
        <section><h2>Changing the battery configuration</h2>
        <p>Nine designs were compared using validation data from {sweep['validation_start'][:10]} to
        {sweep['holdout_start'][:10]} (exclusive). Only those earlier prices were used to select a design.</p>
        {image('configuration_comparison','Validation margins for nine power and capacity combinations')}
        <div class="note"><b>The power ranking is sensitive to numerical precision.</b>
        {sweep['coarse_winner_power_mw']:g} MW / {sweep['coarse_winner_capacity_mwh']:g} MWh leads on the coarse grid.
        Refining the three leading designs to {chosen['soc_step_mwh']:g} MWh steps selects
        {chosen['power_mw']:g} MW / {chosen['capacity_mwh']:g} MWh. This narrow ranking is not a robust equipment recommendation.</div>
        {table(fine,['Power MW','Capacity MWh','SOC Step MWh','Net Margin AUD'])}
        <p>On the later test period, forecast optimisation earns <b>{money(new)}</b> with the selected design versus
        <b>{money(old)}</b> with the original, a <b>{100*(new/old-1):.1f}%</b> difference. Both use the same finer SOC grid.</p>
        {image('holdout_comparison','Original and selected configurations on the later test period')}
        <p><b>A higher trading margin does not establish the best investment.</b> Equipment prices, lifetime,
        financing, network costs and a budget constraint are needed. Power and capacity affect different costs.
        The model also assumes that changing battery size leaves the market prices unchanged.</p>
        </section>
        <section><h2>Efficiency, operating range and cycling costs</h2>
        {image('sensitivity','Effect of changing efficiency, SOC limits or throughput cost one at a time')}
        <p>The base forecast strategy cycles {lookup.loc['Forecast optimisation','EFC']/(metadata['evaluated_rows']*battery['interval_hours']/24):.2f}
        times per day on average. In validation, adding an assumed $10 per MWh of discharge-equivalent internal throughput
        reduces equivalent full cycles by {cycle_reduction:.1f}%. This changes the dispatch decisions; it is not just a deduction
        applied after running the zero-cost strategy.</p>
        <p>A wider SOC window and higher efficiency may improve trading outcomes, but neither is a free specification change.
        Equipment cost, warranties and degradation must also be considered.</p>
        </section>
        <section><h2>How precise is “perfect”?</h2>
        <p>The dynamic programme is globally optimal for the chosen finite set of stored-energy states. It optimises the entire
        selected period with no daily resets. It is not a proof of the continuous-SOC optimum.</p>
        {table(precision,['SOC Step MWh','Net Margin AUD'])}
        <p>Refining the holdout oracle from 0.5 to 0.25 MWh raises margin by {100*(fine_value/coarse_value-1):.2f}%.
        This is an observed sensitivity, not a bound on the remaining numerical error. A separate CVXPY formulation is included
        for continuous-SOC optimisation on shorter periods.</p>
        </section>''')
    sections.append(f'''
    <section><h2>Corrections and assumptions to explain</h2>
    <ul><li><b>Stored energy:</b> {battery['capacity_mwh']:g} MWh nameplate with a
    {battery['min_fraction']*battery['capacity_mwh']:g}–{battery['max_fraction']*battery['capacity_mwh']:g} MWh window leaves {usable:g} MWh usable.
    At {battery['discharge_efficiency']:.0%} discharge efficiency, {delivered:g} MWh reaches the grid:
    {delivered/battery['power_mw']:.3f} hours at {battery['power_mw']:g} MW before discretisation.</li>
    <li><b>Round-trip efficiency:</b> {battery['charge_efficiency']:g} × {battery['discharge_efficiency']:g}
    = {battery['charge_efficiency']*battery['discharge_efficiency']:.2%}.</li>
    <li><b>Cycle accounting:</b> battery-side charging is grid imports multiplied by charging efficiency.
    EFC is total internal charge/discharge throughput divided by twice usable capacity.</li>
    <li><b>Physical controls:</b> power and SOC bounds, exclusive charge/discharge, and equal starting/ending inventory.</li>
    <li><b>Data checks:</b> sorted timestamps, region checks, duplicate checks and exact frequency checks.
    No price clipping, interpolation or resampling. Calendar reporting uses interval-start dates.</li></ul>
    <p>The 2021–2023 prices include the June 2022 market suspension. Intervention instructions and compensation are not
    represented. These historical outcomes should not be read as present-day revenue forecasts.
    <a href="https://www.aemo.com.au/newsroom/news-updates/june-2022-series-of-events-and-reports">AEMO’s account of June 2022</a>.</p>
    <p>The runnable project contains the source data, Python modules, independent checks, interval dispatch,
    editable configuration files, CSV summaries and SVG charts. See README.md and results/INSIGHTS.md.</p>
    <p class="small">Model checks cover energy conservation, SOC/power limits, causality, terminal inventory,
    exhaustive small-case optimality and an independent CVXPY comparison.</p>
    </section>''')
    css='''body{margin:0;background:#f3f5f7;color:#152334;font-family:Arial,sans-serif;line-height:1.65}
    main{max-width:1120px;margin:36px auto;background:white;padding:56px 64px;box-sizing:border-box}
    header{border-bottom:2px solid #007f86;padding-bottom:28px}h1{font-size:48px;line-height:1.1;letter-spacing:-1.4px;margin:14px 0 20px}
    h2{font-size:25px;line-height:1.3;margin-top:0}.eyebrow{color:#007f86;letter-spacing:2px;font-size:12px;font-weight:bold}
    .lead{font-size:21px;color:#475569}.note{background:#eef6f6;border-left:4px solid #007f86;padding:16px 20px;margin:24px 0}
    section{padding:36px 0;border-bottom:1px solid #e2e8f0}img{width:100%;height:auto;margin:16px 0}
    .cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.cards div{background:#f3f6f8;padding:22px}
    .cards span{display:block;font-size:14px;color:#526075}.cards strong{display:block;font-size:31px;margin-top:8px}
    .data-table{border-collapse:collapse;width:100%;font-size:13px;margin:22px 0}.data-table th,.data-table td{padding:11px 10px;text-align:right;border-bottom:1px solid #dce4ec}
    .data-table th:first-child,.data-table td:first-child{text-align:left}.data-table th{background:#f3f6f8;font-weight:bold}
    a{color:#007f86}.small{font-size:13px;color:#64748b}li{margin-bottom:8px}
    @media(max-width:700px){main{padding:25px 18px;margin:0}h1{font-size:34px}.cards{grid-template-columns:1fr}.data-table{font-size:11px;display:block;overflow-x:auto}}
    @media print{main{margin:0;padding:0}section{break-inside:avoid}.cards div{border:1px solid #ddd}body{background:white}h1{font-size:36px}}
    '''
    output='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Battery arbitrage analysis</title><style>'+css+'</style></head><body><main>'+''.join(sections)+'</main></body></html>'
    (folder/'Battery_Analysis.html').write_text(output,encoding='utf-8')


if __name__=='__main__':
    write_html_report(Path(__file__).parent/'results')
