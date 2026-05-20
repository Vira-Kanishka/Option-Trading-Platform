"""
American option pricing via the Cox-Ross-Rubinstein (CRR) binomial tree.

The core BSM engine prices European options in closed form. US single-stock and
ETF options (including SPY) are American-style: they may be exercised at any time
up to expiry. Early exercise is only optimal in narrow cases:

    - Deep in-the-money PUTS when rates are high (capturing time value of the strike)
    - CALLS just before a large ex-dividend date (capturing the dividend)

For short-dated, near-the-money, defined-risk structures the European
approximation is excellent. This module exists so the platform can quantify the
early-exercise premium when it matters, rather than silently assuming it away.

Method
------
Cox-Ross-Rubinstein binomial tree with `steps` time steps. At each node we take
the greater of the continuation value (discounted expected value) and the
intrinsic value (immediate exercise). The European price from the same tree is
computed alongside so the early-exercise premium is available directly.

Convergence is O(1/steps); 200-500 steps gives cent-level accuracy for typical
equity options. We expose `american_price` (the value), `early_exercise_premium`
(American minus European tree price), and `binomial_european_price` (for
validating tree convergence against the closed-form BSM).

Author: Kanishk Devgan
Project: Amoghopāya
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

OptionType = Literal["call", "put"]

_MIN_T = 1e-8


def _crr_params(T: float, r: float, q: float, sigma: float, steps: int):
    """Return (dt, u, d, p, disc) for a CRR tree."""
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    # Risk-neutral up-probability with continuous dividend yield q
    a = math.exp((r - q) * dt)
    p = (a - d) / (u - d)
    disc = math.exp(-r * dt)
    return dt, u, d, p, disc


def _binomial(S: float, K: float, T: float, r: float, q: float, sigma: float,
              option_type: OptionType, steps: int, american: bool) -> float:
    """Core CRR backward induction. If american, allow early exercise at each node."""
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    dt, u, d, p, disc = _crr_params(T, r, q, sigma, steps)

    # Clamp probability into [0, 1] for numerical safety at extreme params
    p = min(1.0, max(0.0, p))

    # Terminal asset prices: S * u^j * d^(steps-j) for j in 0..steps
    j = np.arange(steps + 1)
    asset = S * (u ** j) * (d ** (steps - j))

    # Terminal payoffs
    if option_type == "call":
        values = np.maximum(asset - K, 0.0)
    else:
        values = np.maximum(K - asset, 0.0)

    # Backward induction
    for step in range(steps - 1, -1, -1):
        values = disc * (p * values[1:step + 2] + (1.0 - p) * values[0:step + 1])
        if american:
            j = np.arange(step + 1)
            asset = S * (u ** j) * (d ** (step - j))
            if option_type == "call":
                intrinsic = np.maximum(asset - K, 0.0)
            else:
                intrinsic = np.maximum(K - asset, 0.0)
            values = np.maximum(values, intrinsic)

    return float(values[0])


def american_price(S: float, K: float, T: float, r: float, q: float, sigma: float,
                   option_type: OptionType, steps: int = 300) -> float:
    """American option price via CRR binomial tree.

    `steps` controls accuracy/speed. 300 is a good default for equity options.
    """
    return _binomial(S, K, T, r, q, sigma, option_type, steps, american=True)


def binomial_european_price(S: float, K: float, T: float, r: float, q: float,
                            sigma: float, option_type: OptionType,
                            steps: int = 300) -> float:
    """European price from the SAME tree. Used to (a) compute the early-exercise
    premium consistently and (b) validate tree convergence against closed-form BSM.
    """
    return _binomial(S, K, T, r, q, sigma, option_type, steps, american=False)


def early_exercise_premium(S: float, K: float, T: float, r: float, q: float,
                           sigma: float, option_type: OptionType,
                           steps: int = 300) -> float:
    """American minus European, computed on the same tree for consistency.

    This is the value attributable purely to the early-exercise right. It is
    >= 0 up to discretisation error. Near zero for most short-dated near-the-money
    options; meaningfully positive for deep-ITM puts at high rates and for calls
    facing a large dividend yield.
    """
    am = _binomial(S, K, T, r, q, sigma, option_type, steps, american=True)
    eu = _binomial(S, K, T, r, q, sigma, option_type, steps, american=False)
    return am - eu


def should_use_american(S: float, K: float, T: float, r: float, q: float,
                        sigma: float, option_type: OptionType,
                        threshold: float = 0.005, steps: int = 200) -> bool:
    """Heuristic: is the early-exercise premium large enough to bother with the
    American price, rather than using the faster closed-form European BSM?

    Returns True when the premium exceeds `threshold` * S (default 0.5% of spot).
    Lets the platform price European by default and only switch to the tree for
    the legs where it actually matters.
    """
    prem = early_exercise_premium(S, K, T, r, q, sigma, option_type, steps)
    return prem > threshold * S


# ---------------------------------------------------------------------------
# American Greeks
# ---------------------------------------------------------------------------
# No closed form: the early-exercise boundary makes the value function
# non-smooth, so these are numerical.
#
# Delta/Gamma come straight off the lattice (the step-2 recombining nodes give
# three asset prices and three values around spot, so a finite difference on the
# nodes themselves gives both at no extra cost). Same idea as in 't Hout
# (arXiv:2401.13361) for PDE grids, done here on the binomial tree. Cleaner than
# bumping spot and re-pricing.
#
# Vega/Theta/Rho have no lattice shortcut, so they come from bumps: central for
# Vega and Rho, forward for Theta.

def _binomial_with_early_nodes(S, K, T, r, q, sigma, option_type, steps, american):
    """Run the tree but also return the option values and asset prices at the
    first two time steps (step 1 and step 2 from the root), used for lattice
    Delta/Gamma extraction.

    Returns (root_value, nodes) where nodes is a dict with the step-2 lattice:
        asset_2 = [S*d^2, S, S*u^2]  (the three step-2 nodes for a recombining tree)
        value_2 = corresponding option values
    For a recombining CRR tree, after two steps the middle node returns to S
    (since u*d = 1), giving a natural centred stencil around the spot.
    """
    if T <= 0 or steps < 2:
        # Degenerate: fall back to plain pricing, no lattice Greeks available
        v = _binomial(S, K, T, r, q, sigma, option_type, max(steps, 1), american)
        return v, None

    dt, u, d, p, disc = _crr_params(T, r, q, sigma, steps)
    p = min(1.0, max(0.0, p))

    j = np.arange(steps + 1)
    asset = S * (u ** j) * (d ** (steps - j))
    if option_type == "call":
        values = np.maximum(asset - K, 0.0)
    else:
        values = np.maximum(K - asset, 0.0)

    captured = {}
    for step in range(steps - 1, -1, -1):
        values = disc * (p * values[1:step + 2] + (1.0 - p) * values[0:step + 1])
        if american:
            jj = np.arange(step + 1)
            a_step = S * (u ** jj) * (d ** (step - jj))
            if option_type == "call":
                intrinsic = np.maximum(a_step - K, 0.0)
            else:
                intrinsic = np.maximum(K - a_step, 0.0)
            values = np.maximum(values, intrinsic)
        if step == 2:
            jj = np.arange(3)
            a2 = S * (u ** jj) * (d ** (2 - jj))  # [S d^2, S, S u^2]
            captured["asset_2"] = a2
            captured["value_2"] = values.copy()
        if step == 1:
            jj = np.arange(2)
            a1 = S * (u ** jj) * (d ** (1 - jj))  # [S d, S u]
            captured["asset_1"] = a1
            captured["value_1"] = values.copy()

    return float(values[0]), (captured if captured else None)


def american_greeks(S: float, K: float, T: float, r: float, q: float, sigma: float,
                    option_type: OptionType, steps: int = 400):
    """Greeks for an American option, returned as a dict matching bsm.Greeks fields.

    Delta, Gamma : extracted from the lattice (step-2 recombining nodes), giving a
                   centred stencil around spot at no extra pricing cost. Smooth and
                   accurate; not corrected further.
    Vega, Theta, Rho : computed by re-pricing bumps, then control-variate
                   corrected against the exact closed-form European Greek to cancel
                   the tree's sawtooth discretisation error (see body). Brings their
                   accuracy close to delta/gamma rather than the ~1% a raw bump gives.

    Returns a dict: {price, delta, gamma, vega, theta, rho}.
    """
    price, nodes = _binomial_with_early_nodes(S, K, T, r, q, sigma, option_type, steps, american=True)

    # --- Delta and Gamma from the step-2 lattice (centred around S) ---
    if nodes and "asset_2" in nodes:
        a2 = nodes["asset_2"]      # [S d^2, S, S u^2]
        v2 = nodes["value_2"]
        s_dn, s_mid, s_up = a2[0], a2[1], a2[2]
        v_dn, v_mid, v_up = v2[0], v2[1], v2[2]
        delta = (v_up - v_dn) / (s_up - s_dn)
        # Non-uniform second difference for gamma
        gamma = 2.0 * (
            (v_up - v_mid) / (s_up - s_mid) - (v_mid - v_dn) / (s_mid - s_dn)
        ) / (s_up - s_dn)
    else:
        # Fallback: bump spot directly
        h = max(1e-4 * S, 1e-6)
        v_up = american_price(S + h, K, T, r, q, sigma, option_type, steps)
        v_dn = american_price(S - h, K, T, r, q, sigma, option_type, steps)
        v_mid = price
        delta = (v_up - v_dn) / (2 * h)
        gamma = (v_up - 2 * v_mid + v_dn) / (h * h)

    # Vega/Theta/Rho via bumps, control-variate corrected and step-averaged.
    # Bump Greeks on a binomial tree inherit "sawtooth" error: the early-exercise
    # boundary jumps between nodes as the tree refines, so a single-resolution bump
    # is noisy (~1%). Two fixes together:
    #
    #   1. Control variate: compute each Greek by the same bump on the European
    #      tree, measure that tree's error vs the exact closed-form BSM Greek, and
    #      subtract it from the American estimate (shared discretisation error
    #      largely cancels).
    #   2. Step averaging: repeat over a few adjacent step counts and average,
    #      cancelling the residual sawtooth (its phase shifts with step count).
    #
    # Delta and Gamma aren't corrected; they come off the lattice and are already
    # smooth.

    from .bsm import bs_greeks as _bs_greeks

    eu_exact = _bs_greeks(S, K, T, r, q, sigma, option_type)

    def _t(s, k, t, rr, qq, v, am, n):
        return _binomial(s, k, t, rr, qq, v, option_type, n, am)

    # Average over a small spread of step counts to smooth the sawtooth.
    step_set = [steps, steps + 1, steps + 2, steps + 3]

    h_sig, h_r = 1e-3, 1e-4
    h_t = min(1e-3, T / 10) if T > 0 else 1e-3

    vega_acc = theta_acc = rho_acc = 0.0
    for n in step_set:
        # Vega
        am_v = (_t(S, K, T, r, q, sigma + h_sig, True, n)
                - _t(S, K, T, r, q, max(sigma - h_sig, 1e-6), True, n)) / (2 * h_sig)
        eu_v = (_t(S, K, T, r, q, sigma + h_sig, False, n)
                - _t(S, K, T, r, q, max(sigma - h_sig, 1e-6), False, n)) / (2 * h_sig)
        vega_acc += am_v - (eu_v - eu_exact.vega)

        # Theta (theta = dV/dt = -dV/dT, per year)
        am_p = _t(S, K, T, r, q, sigma, True, n)
        eu_p = _t(S, K, T, r, q, sigma, False, n)
        am_th = (_t(S, K, max(T - h_t, 1e-8), r, q, sigma, True, n) - am_p) / h_t
        eu_th = (_t(S, K, max(T - h_t, 1e-8), r, q, sigma, False, n) - eu_p) / h_t
        theta_acc += am_th - (eu_th - eu_exact.theta)

        # Rho
        am_r = (_t(S, K, T, r + h_r, q, sigma, True, n)
                - _t(S, K, T, r - h_r, q, sigma, True, n)) / (2 * h_r)
        eu_r = (_t(S, K, T, r + h_r, q, sigma, False, n)
                - _t(S, K, T, r - h_r, q, sigma, False, n)) / (2 * h_r)
        rho_acc += am_r - (eu_r - eu_exact.rho)

    m = len(step_set)
    vega, theta, rho = vega_acc / m, theta_acc / m, rho_acc / m

    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }
