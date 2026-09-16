# Selected on historical validation trading margin; excludes capital costs.
"""Edit these assumptions, then run ``python main.py --sweep``.

Power and cash flows are measured at the grid connection. SOC is stored energy.
The defaults reproduce the supplied screenshot. Dollar amounts are AUD.
"""

BATTERY_POWER_MW = 50.0
BATTERY_CAPACITY_MWH = 400.0
CHARGE_EFFICIENCY = 0.94
DISCHARGE_EFFICIENCY = 0.94

# Derive energy limits from capacity so changing capacity stays consistent.
INITIAL_SOC_MWH = 0.50 * BATTERY_CAPACITY_MWH
MIN_SOC_MWH = 0.10 * BATTERY_CAPACITY_MWH
MAX_SOC_MWH = 0.90 * BATTERY_CAPACITY_MWH
INTERVAL_HOURS = 5 / 60

BUY_BELOW = 30.0
SELL_ABOVE = 150.0
FORECAST_LOOKBACK_DAYS = 7
FORECAST_HORIZON_HOURS = 24
REPLAN_HOURS = 1  # Solve hourly; execute 12 five-minute decisions before replanning.

# A sensitivity assumption, NOT a quoted equipment cost. Half of total internal
# charge/discharge throughput is charged at this rate. Zero matches the original.
CYCLE_COST_PER_MWH = 0.0

# Each mixed-integer solve must finish optimally to the stated tolerance.
SOLVER_TIME_LIMIT_SECONDS = 60
SOLVER_RELATIVE_GAP = 1e-6

# Dynamic programming finds the global optimum on this stored-energy grid.
# All reported strategies use the same grid; smaller steps cost more memory/time.
SOC_STEP_MWH = 0.25

# Bounded design comparison. Rankings exclude construction and fixed costs.
SWEEP_POWER_MW = [50.0, 100.0, 150.0]
SWEEP_CAPACITY_MWH = [100.0, 200.0, 400.0]
VALIDATION_FRACTION = 0.50
