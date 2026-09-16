"""Exact finite-state optimisation with O(intervals * SOC states) work.

The action is the next SOC. It implies either charge, discharge, or idle, never
both. Sliding maxima exploit the linear cash flow within each operating mode.
Numba compiles the loops; this is dynamic programming, not machine learning.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def solve_grid(prices, n_states, initial_state, final_state, step, max_charge_steps,
               max_discharge_steps, eta_c, eta_d, cycle_cost):
    n = len(prices)
    # Only policies are retained, avoiding a full float64 value-function matrix.
    policy = np.empty((n, n_states), dtype=np.uint16)
    future = np.full(n_states, -np.inf)
    future[final_state] = 0.0
    next_values = np.empty(n_states)
    charge_values, discharge_values = np.empty(n_states), np.empty(n_states)
    charge_next, discharge_next = np.empty(n_states, dtype=np.int64), np.empty(n_states, dtype=np.int64)
    queue = np.empty(n_states, dtype=np.int64)
    values = np.empty(n_states)

    for t in range(n - 1, -1, -1):
        # Charging cash flow for movement i -> j is -(p/eta_c + wear/2)*(j-i)*step.
        charge_slope = (prices[t] / eta_c + cycle_cost / 2) * step
        for j in range(n_states):
            values[j] = future[j] - charge_slope * j
        head, tail = 0, 0
        for i in range(n_states - 1, -1, -1):
            while head < tail and queue[head] > i + max_charge_steps:
                head += 1
            while head < tail and values[queue[tail - 1]] <= values[i]:
                tail -= 1
            queue[tail] = i
            tail += 1
            best = queue[head]
            charge_values[i] = values[best] + charge_slope * i
            charge_next[i] = best

        # Discharging i -> j earns (p*eta_d - wear/2)*(i-j)*step.
        discharge_slope = (prices[t] * eta_d - cycle_cost / 2) * step
        for j in range(n_states):
            values[j] = future[j] - discharge_slope * j
        head, tail = 0, 0
        for i in range(n_states):
            while head < tail and queue[head] < i - max_discharge_steps:
                head += 1
            while head < tail and values[queue[tail - 1]] <= values[i]:
                tail -= 1
            queue[tail] = i
            tail += 1
            best = queue[head]
            discharge_values[i] = values[best] + discharge_slope * i
            discharge_next[i] = best

        for i in range(n_states):
            # Prefer idle on numerical ties to avoid gratuitous zero-value cycles.
            value, chosen = future[i], i
            if charge_values[i] > value + 1e-8:
                value, chosen = charge_values[i], charge_next[i]
            if discharge_values[i] > value + 1e-8:
                value, chosen = discharge_values[i], discharge_next[i]
            next_values[i] = value
            policy[t, i] = chosen
        future, next_values = next_values, future

    optimum = future[initial_state]
    charge, discharge = np.zeros(n), np.zeros(n)
    state = initial_state
    for t in range(n):
        next_state = int(policy[t, state])
        movement = (next_state - state) * step
        if movement > 0:
            charge[t] = movement / eta_c
        else:
            discharge[t] = -movement * eta_d
        state = next_state
    return charge, discharge, optimum


class GridOptimizer:
    def __init__(self, battery):
        self.battery = battery

    def solve(self, prices, initial, final):
        b = self.battery
        prices = np.asarray(prices, dtype=np.float64)
        if prices.ndim != 1 or not len(prices) or not np.isfinite(prices).all():
            raise ValueError("Need a nonempty finite price vector.")
        def index(value):
            raw = (value - b.min_mwh) / b.soc_step_mwh
            if not np.isclose(raw, round(raw), atol=1e-5, rtol=0):
                raise ValueError("SOC is not on the selected energy grid.")
            result = int(round(raw))
            if not 0 <= result <= round(b.usable_mwh / b.soc_step_mwh):
                raise ValueError("SOC is outside operating limits.")
            return result
        n_states = int(round(b.usable_mwh / b.soc_step_mwh)) + 1
        if n_states > 65535:
            raise ValueError("Too many SOC states; use a larger step.")
        max_c = int(np.floor(b.charge_efficiency * b.max_grid_mwh / b.soc_step_mwh + 1e-9))
        max_d = int(np.floor(b.max_grid_mwh / b.discharge_efficiency / b.soc_step_mwh + 1e-9))
        c, d, optimum = solve_grid(prices, n_states, index(initial), index(final),
                                   b.soc_step_mwh, max_c, max_d,
                                   b.charge_efficiency, b.discharge_efficiency, b.cycle_cost_per_mwh)
        if not np.isfinite(optimum):
            raise ValueError("Terminal SOC is unreachable in this horizon on this SOC grid.")
        actual = prices @ (d - c) - b.cycle_cost_per_mwh * (
            b.charge_efficiency * c.sum() + d.sum() / b.discharge_efficiency) / 2
        if not np.isclose(actual, optimum, atol=0.02, rtol=1e-8):
            raise AssertionError("Policy reconstruction differs from the dynamic-programming optimum.")
        return c, d
