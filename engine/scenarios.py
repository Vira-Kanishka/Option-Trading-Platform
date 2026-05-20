"""
Scenario stress-test engine.

Computes strategy P&L across a Cartesian grid of:
    - underlying price shocks (±5%, ±10%, ±20% by default)
    - volatility shocks (absolute or relative)
    - optional time-decay (evaluate at t = T - dt rather than t = 0)

Returns a structured DataFrame for tabular display and a 2D matrix for heatmaps.

Author: Kanishk Devgan
Project: Amoghopāya
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .strategies import Strategy


DEFAULT_SPOT_SHOCKS = np.array([-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20])
DEFAULT_VOL_SHOCKS = np.array([-0.30, -0.15, 0.0, 0.15, 0.30])  # relative


@dataclass
class ScenarioResult:
    """Output of a scenario run."""
    pnl_grid: pd.DataFrame      # rows=vol shock, cols=spot shock, values=P&L
    spot_shocks: np.ndarray
    vol_shocks: np.ndarray
    base_S: float
    base_sigma: float
    days_forward: int


def run_scenarios(
    strat: Strategy,
    S: float,
    sigma: float,
    r: float = 0.05,
    q: float = 0.0,
    spot_shocks: np.ndarray = DEFAULT_SPOT_SHOCKS,
    vol_shocks: np.ndarray = DEFAULT_VOL_SHOCKS,
    vol_shock_kind: str = "relative",  # "relative" or "absolute"
    days_forward: int = 0,
    exercise_style: str = "european",
) -> ScenarioResult:
    """Compute mark-to-market P&L across a (spot, vol) grid.

    P&L convention: positive = gain vs entry_premium.
    Time-forward decay is applied by shrinking each option leg's expiry_T by
    days_forward/365 before re-marking. If any leg expires within the window,
    the leg's payoff collapses to intrinsic.

    exercise_style: "european" (closed-form BSM, fast) or "american" (binomial
    tree per cell). American stress grids are materially slower because each cell
    re-prices every leg on a tree; acceptable for the small default grid.
    """
    # Clone legs with shortened time to expiry
    from copy import deepcopy
    strat_fwd = deepcopy(strat)
    dt = days_forward / 365.0
    for leg in strat_fwd.legs:
        if leg.expiry_T is not None:
            # Replace via object.__setattr__ since Leg is frozen
            object.__setattr__(leg, "expiry_T", max(leg.expiry_T - dt, 0.0))

    rows = []
    for dv in vol_shocks:
        if vol_shock_kind == "relative":
            sigma_shocked = max(sigma * (1.0 + dv), 1e-6)
        else:  # absolute
            sigma_shocked = max(sigma + dv, 1e-6)
        row = []
        for ds in spot_shocks:
            S_shocked = S * (1.0 + ds)
            pnl = strat_fwd.pnl(S_shocked, r, q, sigma_shocked, exercise_style)
            row.append(pnl)
        rows.append(row)

    df = pd.DataFrame(
        rows,
        index=[f"{int(v*100):+d}%" for v in vol_shocks],
        columns=[f"{int(s*100):+d}%" for s in spot_shocks],
    )
    df.index.name = "vol shock"
    df.columns.name = "spot shock"

    return ScenarioResult(
        pnl_grid=df,
        spot_shocks=spot_shocks,
        vol_shocks=vol_shocks,
        base_S=S,
        base_sigma=sigma,
        days_forward=days_forward,
    )
