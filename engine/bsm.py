"""
Black-Scholes-Merton pricing and Greeks.

Closed-form European option pricing with continuous dividend yield q.
All Greeks derived analytically; no finite differences.

Conventions
-----------
S  : spot price
K  : strike
T  : time to expiry in years (ACT/365)
r  : continuously-compounded risk-free rate
q  : continuously-compounded dividend yield (0 for most equity indices intraday)
sigma : annualised volatility (decimal, e.g. 0.20 for 20%)

Greeks returned "per underlying unit move" unless noted:
    delta : dP/dS
    gamma : d^2 P / dS^2
    vega  : dP/dsigma   (per 1.0 change in sigma, i.e. 100 vol points)
    theta : dP/dt       (calendar; per year, divide by 365 for per-day)
    rho   : dP/dr       (per 1.0 change in r)

We don't scale vega/theta to "per 1 vol point" or "per day" inside this module.
Scaling is a presentation concern; the engine stores raw derivatives.

Author: Kanishk Devgan
Project: Amoghopāya
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

OptionType = Literal["call", "put"]

# Numerical floors: keep formulas stable at zero vol / zero time without NaN
_MIN_T = 1e-8
_MIN_SIGMA = 1e-8


# ---------------------------------------------------------------------------
# Core d1, d2
# ---------------------------------------------------------------------------

def _d1_d2(S: float, K: float, T: float, r: float, q: float, sigma: float) -> tuple[float, float]:
    """Return (d1, d2). Guards against degenerate inputs."""
    T = max(T, _MIN_T)
    sigma = max(sigma, _MIN_SIGMA)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2


# ---------------------------------------------------------------------------
# Price
# ---------------------------------------------------------------------------

def bs_price(S: float, K: float, T: float, r: float, q: float, sigma: float,
             option_type: OptionType) -> float:
    """Black-Scholes-Merton price of a European call or put."""
    if T <= 0:
        # At expiry, price collapses to intrinsic
        if option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)

    if option_type == "call":
        return S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    return K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Greeks:
    """Greeks for a single European option, per one contract on one underlying unit.

    vega is per 1.0 change in sigma (so vega * 0.01 gives P&L for a 1 vol-point move).
    theta is per year (so theta / 365 gives per-calendar-day decay).
    """
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def bs_greeks(S: float, K: float, T: float, r: float, q: float, sigma: float,
              option_type: OptionType) -> Greeks:
    """Closed-form Greeks. Uses the q-adjusted Black-Scholes formulas."""
    if T <= 0:
        # At expiry: delta is 0/1 step, all other Greeks zero
        intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        if option_type == "call":
            delta = 1.0 if S > K else 0.0
        else:
            delta = -1.0 if S < K else 0.0
        return Greeks(price=intrinsic, delta=delta, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)

    T = max(T, _MIN_T)
    sigma = max(sigma, _MIN_SIGMA)
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    sqrtT = math.sqrt(T)
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    pdf_d1 = norm.pdf(d1)

    # Price
    if option_type == "call":
        price = S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
        delta = disc_q * norm.cdf(d1)
        theta = (
            -S * disc_q * pdf_d1 * sigma / (2.0 * sqrtT)
            - r * K * disc_r * norm.cdf(d2)
            + q * S * disc_q * norm.cdf(d1)
        )
        rho = K * T * disc_r * norm.cdf(d2)
    else:
        price = K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)
        delta = -disc_q * norm.cdf(-d1)
        theta = (
            -S * disc_q * pdf_d1 * sigma / (2.0 * sqrtT)
            + r * K * disc_r * norm.cdf(-d2)
            - q * S * disc_q * norm.cdf(-d1)
        )
        rho = -K * T * disc_r * norm.cdf(-d2)

    gamma = disc_q * pdf_d1 / (S * sigma * sqrtT)
    vega = S * disc_q * pdf_d1 * sqrtT

    return Greeks(price=price, delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


# ---------------------------------------------------------------------------
# Implied volatility
# ---------------------------------------------------------------------------

def implied_vol(price: float, S: float, K: float, T: float, r: float, q: float,
                option_type: OptionType,
                lo: float = 1e-4, hi: float = 5.0) -> float:
    """Invert BSM for sigma given a market price using Brent's method.

    Raises ValueError if the target price is outside the no-arbitrage bounds.
    Returns sigma in [lo, hi].
    """
    # Arbitrage bounds
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    if option_type == "call":
        lower_bound = max(S * disc_q - K * disc_r, 0.0)
        upper_bound = S * disc_q
    else:
        lower_bound = max(K * disc_r - S * disc_q, 0.0)
        upper_bound = K * disc_r

    if price < lower_bound - 1e-10 or price > upper_bound + 1e-10:
        raise ValueError(
            f"Price {price:.4f} outside no-arbitrage bounds "
            f"[{lower_bound:.4f}, {upper_bound:.4f}]"
        )

    def objective(sigma: float) -> float:
        return bs_price(S, K, T, r, q, sigma, option_type) - price

    # If even at lo vol we're above the target, IV is numerically ~0
    if objective(lo) > 0:
        return lo
    if objective(hi) < 0:
        return hi

    return brentq(objective, lo, hi, xtol=1e-8, rtol=1e-8, maxiter=100)


def historical_vol(prices: np.ndarray, window: int = 30,
                   annualisation: int = 252) -> float:
    """Annualised historical (realised) volatility from a price series.

    Computes the sample standard deviation of log returns over the last
    `window` observations, then annualises by sqrt(annualisation).

    Parameters
    ----------
    prices : np.ndarray
        1-D array of contiguous closing prices, most recent last.
        Must contain at least `window + 1` observations.
    window : int, default 30
        Number of returns to use (so window+1 prices).
    annualisation : int, default 252
        Number of trading periods per year. Use 252 for daily US equities.

    Returns
    -------
    float : annualised volatility as a decimal (e.g. 0.185 = 18.5%)

    Notes
    -----
    Presented in the UI alongside implied volatility, not in place of it.
    HV is a reference input for assessing whether current IV is rich or cheap,
    but BSM pricing and Greeks in the engine are always driven by IV.
    """
    prices = np.asarray(prices, dtype=float)
    if prices.ndim != 1:
        raise ValueError("prices must be 1-D")
    if len(prices) < window + 1:
        raise ValueError(f"Need at least {window + 1} prices, got {len(prices)}")
    if np.any(prices <= 0):
        raise ValueError("All prices must be positive")

    # Log returns over last `window` observations
    recent = prices[-(window + 1):]
    log_returns = np.diff(np.log(recent))
    return float(np.std(log_returns, ddof=1) * np.sqrt(annualisation))
