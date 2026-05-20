"""
Strategy composition.

A Strategy is a collection of Legs. Each Leg is a signed quantity of either
an option (with strike and type) or the underlying itself.

Aggregation rules
-----------------
For any observable X that is additive across positions (price, delta, gamma,
vega, theta, rho, payoff-at-expiry), the strategy-level value is

    X_strategy = sum_i  side_i * qty_i * X_leg_i

where side_i in {+1 long, -1 short} and qty_i is the contract count.
Underlying legs contribute delta = side * qty, and zero for all other Greeks.

We express sizes in "contracts" but abstract the contract multiplier out to
the caller; all Greeks/payoffs returned here are per underlying unit.
For US equity options (100 shares per contract), multiply by 100 at the
presentation layer. This keeps the engine unit-clean.

Author: Kanishk Devgan
Project: Amoghopāya
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .bsm import Greeks, bs_greeks, bs_price
from .american import american_greeks, american_price

ExerciseStyle = Literal["european", "american"]

Side = Literal[1, -1]  # +1 long, -1 short
LegKind = Literal["call", "put", "underlying"]


# ---------------------------------------------------------------------------
# Leg
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Leg:
    """A single position in an option or in the underlying.

    For underlying legs, `strike` and `expiry_T` are ignored.

    Per-leg implied volatility
    --------------------------
    `iv` optionally overrides the strategy-level sigma for THIS leg only. Real
    option markets exhibit a volatility surface: implied vol varies by strike and
    expiry. A single flat sigma misprices off-ATM strikes; an OTM put on an
    equity index trades at a materially higher IV than the ATM. When `iv` is set,
    the leg prices and computes Greeks at its own vol; when `iv` is None, the leg
    falls back to the sigma passed in by the caller (the legacy flat-vol behaviour,
    fully backward compatible).
    """
    kind: LegKind
    side: Side
    qty: float = 1.0
    strike: float | None = None
    expiry_T: float | None = None  # time to expiry in years
    iv: float | None = None        # per-leg implied vol; None -> use caller's sigma

    def __post_init__(self) -> None:
        if self.kind in ("call", "put"):
            if self.strike is None or self.expiry_T is None:
                raise ValueError(f"Option leg needs strike and expiry_T: {self}")
            if self.strike <= 0:
                raise ValueError(f"Strike must be positive: {self}")
        if self.side not in (1, -1):
            raise ValueError(f"Side must be +1 or -1: {self}")
        if self.qty <= 0:
            raise ValueError(f"qty must be positive; use side to flip sign: {self}")
        if self.iv is not None and self.iv <= 0:
            raise ValueError(f"Per-leg iv must be positive if set: {self}")

    def _vol(self, sigma: float) -> float:
        """Resolve the volatility to use: per-leg iv if set, else caller's sigma."""
        return self.iv if self.iv is not None else sigma

    # ---- per-leg observables ----

    def payoff_at_expiry(self, S_T: np.ndarray | float) -> np.ndarray | float:
        """Terminal payoff per unit at underlying price S_T.

        For options: intrinsic at expiry.
        For underlying: S_T - cost_basis is handled at strategy level;
        here we return S_T itself (the leg's terminal value), and the
        strategy's net_premium absorbs the cost basis.
        """
        if self.kind == "call":
            intrinsic = np.maximum(S_T - self.strike, 0.0)
        elif self.kind == "put":
            intrinsic = np.maximum(self.strike - S_T, 0.0)
        else:  # underlying
            intrinsic = np.asarray(S_T, dtype=float) if isinstance(S_T, np.ndarray) else float(S_T)
        return self.side * self.qty * intrinsic

    def mark_price(self, S: float, r: float, q: float, sigma: float,
                   exercise_style: ExerciseStyle = "european") -> float:
        """Signed mark-to-market price. Positive = asset; negative = liability.

        Uses this leg's own implied vol if `iv` is set, else the caller's sigma.
        With exercise_style="american", prices on a CRR binomial tree; otherwise
        closed-form European BSM.
        """
        if self.kind == "underlying":
            return self.side * self.qty * S
        vol = self._vol(sigma)
        if exercise_style == "american":
            p = american_price(S, self.strike, self.expiry_T, r, q, vol, self.kind)
        else:
            p = bs_price(S, self.strike, self.expiry_T, r, q, vol, self.kind)
        return self.side * self.qty * p

    def greeks(self, S: float, r: float, q: float, sigma: float,
               exercise_style: ExerciseStyle = "european") -> Greeks:
        """Signed Greeks contribution (price, delta, gamma, vega, theta, rho).

        Uses this leg's own implied vol if `iv` is set, else the caller's sigma.
        With exercise_style="american", Greeks come from the binomial lattice
        (Delta/Gamma) and tree bumps (Vega/Theta/Rho); otherwise closed-form BSM.
        """
        if self.kind == "underlying":
            # Underlying: delta=1 per unit, gamma=vega=theta=rho=0
            sign = self.side * self.qty
            return Greeks(price=sign * S, delta=sign, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)
        vol = self._vol(sigma)
        sign = self.side * self.qty
        if exercise_style == "american":
            ag = american_greeks(S, self.strike, self.expiry_T, r, q, vol, self.kind)
            return Greeks(
                price=sign * ag["price"],
                delta=sign * ag["delta"],
                gamma=sign * ag["gamma"],
                vega=sign * ag["vega"],
                theta=sign * ag["theta"],
                rho=sign * ag["rho"],
            )
        g = bs_greeks(S, self.strike, self.expiry_T, r, q, vol, self.kind)
        return Greeks(
            price=sign * g.price,
            delta=sign * g.delta,
            gamma=sign * g.gamma,
            vega=sign * g.vega,
            theta=sign * g.theta,
            rho=sign * g.rho,
        )


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

@dataclass
class Strategy:
    """A named collection of legs.

    `entry_premium` (optional override) is the net debit/credit the user paid
    to enter. If None, we compute it from mark prices at construction time
    when calling `set_entry_premium_from_marks`. By convention:
        + debit paid (money out)  ->  entry_premium > 0
        - credit received         ->  entry_premium < 0
    """
    name: str
    legs: list[Leg]
    entry_premium: float = 0.0
    meta: dict = field(default_factory=dict)

    # ---- construction helpers ----

    def set_entry_premium_from_marks(self, S: float, r: float, q: float, sigma: float,
                                     exercise_style: ExerciseStyle = "european") -> None:
        """Set entry_premium to the net mark price at given inputs."""
        self.entry_premium = self.mark_value(S, r, q, sigma, exercise_style)

    # ---- aggregation ----

    def payoff_at_expiry(self, S_T: np.ndarray) -> np.ndarray:
        """Net P&L at expiry across a grid of terminal prices.

        Returns: P&L per unit-underlying (multiply by contract multiplier at UI).
        Payoff at expiry is identical for European and American (both collapse to
        intrinsic at T), so this method needs no exercise_style argument.
        """
        S_T = np.asarray(S_T, dtype=float)
        total = np.zeros_like(S_T)
        for leg in self.legs:
            total = total + leg.payoff_at_expiry(S_T)
        return total - self.entry_premium

    def mark_value(self, S: float, r: float, q: float, sigma: float,
                   exercise_style: ExerciseStyle = "european") -> float:
        """Net mark-to-market value of the basket (premium + stock)."""
        return sum(leg.mark_price(S, r, q, sigma, exercise_style) for leg in self.legs)

    def pnl(self, S: float, r: float, q: float, sigma: float,
            exercise_style: ExerciseStyle = "european") -> float:
        """Current P&L vs entry premium."""
        return self.mark_value(S, r, q, sigma, exercise_style) - self.entry_premium

    def aggregate_greeks(self, S: float, r: float, q: float, sigma: float,
                         exercise_style: ExerciseStyle = "european") -> Greeks:
        """Leg-additive Greeks."""
        totals = {"price": 0.0, "delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
        for leg in self.legs:
            g = leg.greeks(S, r, q, sigma, exercise_style)
            totals["price"] += g.price
            totals["delta"] += g.delta
            totals["gamma"] += g.gamma
            totals["vega"] += g.vega
            totals["theta"] += g.theta
            totals["rho"] += g.rho
        return Greeks(**totals)

    # ---- payoff diagnostics ----

    def breakevens(self, S_ref: float, span: float = 0.6, n: int = 20001) -> list[float]:
        """Numerical breakevens: S_T where payoff_at_expiry crosses zero.

        span: fraction of S_ref above/below to scan. 0.6 -> [0.4*S_ref, 1.6*S_ref].
        """
        lo = max(1e-6, S_ref * (1 - span))
        hi = S_ref * (1 + span)
        grid = np.linspace(lo, hi, n)
        pnl = self.payoff_at_expiry(grid)
        sign = np.sign(pnl)
        # Sign changes (ignore exact zeros to avoid double-counting)
        sign[sign == 0] = 1
        crossings = np.where(np.diff(sign) != 0)[0]
        bes = []
        for idx in crossings:
            # Linear interpolation between grid[idx] and grid[idx+1]
            x0, x1 = grid[idx], grid[idx + 1]
            y0, y1 = pnl[idx], pnl[idx + 1]
            if y1 == y0:
                continue
            be = x0 - y0 * (x1 - x0) / (y1 - y0)
            bes.append(be)
        return bes

    def max_profit_loss(self, S_ref: float, span: float = 2.0, n: int = 20001) -> tuple[float, float]:
        """Observed (max_profit, max_loss) over a wide grid.

        Uses a wide span (default 200%) so unbounded strategies are flagged.
        Returned values are numerical; callers should overlay theoretical
        bounds (e.g. "unlimited" for naked short calls).
        """
        lo = max(1e-6, S_ref * (1 - min(span, 0.99)))
        hi = S_ref * (1 + span)
        grid = np.linspace(lo, hi, n)
        pnl = self.payoff_at_expiry(grid)
        return float(np.max(pnl)), float(np.min(pnl))


# ---------------------------------------------------------------------------
# Strategy factories (the 20-strategy catalogue)
# ---------------------------------------------------------------------------

def covered_call(S0: float, K_call: float, T: float, qty: float = 1.0) -> Strategy:
    """Long 1 share underlying + short 1 call at K_call."""
    return Strategy(
        name="Covered Call",
        legs=[
            Leg("underlying", side=1, qty=qty),
            Leg("call", side=-1, qty=qty, strike=K_call, expiry_T=T),
        ],
        meta={"risk_class": "low", "view": "neutral-to-mildly-bullish"},
    )


def protective_put(S0: float, K_put: float, T: float, qty: float = 1.0) -> Strategy:
    """Long 1 share underlying + long 1 put at K_put (insurance)."""
    return Strategy(
        name="Protective Put",
        legs=[
            Leg("underlying", side=1, qty=qty),
            Leg("put", side=1, qty=qty, strike=K_put, expiry_T=T),
        ],
        meta={"risk_class": "low", "view": "bullish-with-hedge"},
    )


def bull_call_spread(K_lo: float, K_hi: float, T: float, qty: float = 1.0) -> Strategy:
    """Long call at K_lo + short call at K_hi (K_lo < K_hi). Debit."""
    if not K_lo < K_hi:
        raise ValueError("bull_call_spread requires K_lo < K_hi")
    return Strategy(
        name="Bull Call Spread",
        legs=[
            Leg("call", side=1, qty=qty, strike=K_lo, expiry_T=T),
            Leg("call", side=-1, qty=qty, strike=K_hi, expiry_T=T),
        ],
        meta={"risk_class": "medium", "view": "moderately-bullish"},
    )


def bear_put_spread(K_hi: float, K_lo: float, T: float, qty: float = 1.0) -> Strategy:
    """Long put at K_hi + short put at K_lo (K_lo < K_hi). Debit."""
    if not K_lo < K_hi:
        raise ValueError("bear_put_spread requires K_lo < K_hi")
    return Strategy(
        name="Bear Put Spread",
        legs=[
            Leg("put", side=1, qty=qty, strike=K_hi, expiry_T=T),
            Leg("put", side=-1, qty=qty, strike=K_lo, expiry_T=T),
        ],
        meta={"risk_class": "medium", "view": "moderately-bearish"},
    )


def bull_put_spread(K_hi: float, K_lo: float, T: float, qty: float = 1.0) -> Strategy:
    """Short put at K_hi + long put at K_lo. Credit (K_lo < K_hi)."""
    if not K_lo < K_hi:
        raise ValueError("bull_put_spread requires K_lo < K_hi")
    return Strategy(
        name="Bull Put Spread",
        legs=[
            Leg("put", side=-1, qty=qty, strike=K_hi, expiry_T=T),
            Leg("put", side=1, qty=qty, strike=K_lo, expiry_T=T),
        ],
        meta={"risk_class": "medium", "view": "moderately-bullish (credit)"},
    )


def bear_call_spread(K_lo: float, K_hi: float, T: float, qty: float = 1.0) -> Strategy:
    """Short call at K_lo + long call at K_hi. Credit (K_lo < K_hi)."""
    if not K_lo < K_hi:
        raise ValueError("bear_call_spread requires K_lo < K_hi")
    return Strategy(
        name="Bear Call Spread",
        legs=[
            Leg("call", side=-1, qty=qty, strike=K_lo, expiry_T=T),
            Leg("call", side=1, qty=qty, strike=K_hi, expiry_T=T),
        ],
        meta={"risk_class": "medium", "view": "moderately-bearish (credit)"},
    )


def straddle(K: float, T: float, qty: float = 1.0) -> Strategy:
    """Long call + long put at same strike K. Long vol."""
    return Strategy(
        name="Long Straddle",
        legs=[
            Leg("call", side=1, qty=qty, strike=K, expiry_T=T),
            Leg("put", side=1, qty=qty, strike=K, expiry_T=T),
        ],
        meta={"risk_class": "high", "view": "direction-agnostic, long-vol"},
    )


def strangle(K_put: float, K_call: float, T: float, qty: float = 1.0) -> Strategy:
    """Long OTM put + long OTM call (K_put < K_call). Long vol, cheaper than straddle."""
    if not K_put < K_call:
        raise ValueError("strangle requires K_put < K_call")
    return Strategy(
        name="Long Strangle",
        legs=[
            Leg("put", side=1, qty=qty, strike=K_put, expiry_T=T),
            Leg("call", side=1, qty=qty, strike=K_call, expiry_T=T),
        ],
        meta={"risk_class": "high", "view": "direction-agnostic, long-vol"},
    )


def butterfly(K_lo: float, K_mid: float, K_hi: float, T: float,
              option: Literal["call", "put"] = "call", qty: float = 1.0) -> Strategy:
    """Long butterfly: +1 K_lo, -2 K_mid, +1 K_hi. Requires K_mid = (K_lo+K_hi)/2 for symmetry."""
    if not (K_lo < K_mid < K_hi):
        raise ValueError("butterfly requires K_lo < K_mid < K_hi")
    return Strategy(
        name=f"Long {option.capitalize()} Butterfly",
        legs=[
            Leg(option, side=1, qty=qty, strike=K_lo, expiry_T=T),
            Leg(option, side=-1, qty=2 * qty, strike=K_mid, expiry_T=T),
            Leg(option, side=1, qty=qty, strike=K_hi, expiry_T=T),
        ],
        meta={"risk_class": "medium", "view": "pin at K_mid, short-vol"},
    )


def collar(S0: float, K_put: float, K_call: float, T: float, qty: float = 1.0) -> Strategy:
    """Long 1 share + long put at K_put + short call at K_call.
    Caps upside, floors downside. Often near-zero-cost.
    """
    if not K_put < K_call:
        raise ValueError("collar requires K_put < K_call")
    return Strategy(
        name="Collar",
        legs=[
            Leg("underlying", side=1, qty=qty),
            Leg("put", side=1, qty=qty, strike=K_put, expiry_T=T),
            Leg("call", side=-1, qty=qty, strike=K_call, expiry_T=T),
        ],
        meta={"risk_class": "low", "view": "bounded bullish"},
    )


def long_call(K: float, T: float, qty: float = 1.0) -> Strategy:
    """Single long call. Unlimited upside, defined loss = premium paid."""
    return Strategy(
        name="Long Call",
        legs=[
            Leg("call", side=1, qty=qty, strike=K, expiry_T=T),
        ],
        meta={"risk_class": "medium", "view": "strongly bullish"},
    )


def long_put(K: float, T: float, qty: float = 1.0) -> Strategy:
    """Single long put. Large downside gains, defined loss = premium paid."""
    return Strategy(
        name="Long Put",
        legs=[
            Leg("put", side=1, qty=qty, strike=K, expiry_T=T),
        ],
        meta={"risk_class": "medium", "view": "strongly bearish"},
    )


def cash_secured_put(K: float, T: float, qty: float = 1.0) -> Strategy:
    """Short put backed by cash to buy stock at K if assigned.

    Economically equivalent to covered call via put-call parity.
    Max loss = K - premium (i.e., underlying goes to zero after assignment).
    Requires cash collateral of K per share.
    """
    return Strategy(
        name="Cash-Secured Put",
        legs=[
            Leg("put", side=-1, qty=qty, strike=K, expiry_T=T),
        ],
        meta={"risk_class": "low", "view": "neutral-to-bullish, income"},
    )


def put_butterfly(K_lo: float, K_mid: float, K_hi: float, T: float,
                  qty: float = 1.0) -> Strategy:
    """Long put butterfly: +1 K_hi, -2 K_mid, +1 K_lo.

    Identical payoff to a call butterfly at the same strikes (put-call parity on
    each leg cancels out). Included for catalog completeness; in practice choose
    whichever side has better liquidity.
    """
    if not (K_lo < K_mid < K_hi):
        raise ValueError("put_butterfly requires K_lo < K_mid < K_hi")
    return Strategy(
        name="Long Put Butterfly",
        legs=[
            Leg("put", side=1, qty=qty, strike=K_hi, expiry_T=T),
            Leg("put", side=-1, qty=2 * qty, strike=K_mid, expiry_T=T),
            Leg("put", side=1, qty=qty, strike=K_lo, expiry_T=T),
        ],
        meta={"risk_class": "medium", "view": "pin at K_mid, short-vol"},
    )


def iron_butterfly(K_lo: float, K_mid: float, K_hi: float, T: float,
                   qty: float = 1.0) -> Strategy:
    """Short iron butterfly: short straddle at K_mid + long wings at K_lo/K_hi.

    Legs: +1 put @ K_lo, -1 put @ K_mid, -1 call @ K_mid, +1 call @ K_hi.
    Net credit received. Max profit = credit (at K_mid at expiry).
    Max loss = (K_mid - K_lo) - credit  (equal on both sides since wings
    are equidistant).

    Short-vol, defined-risk equivalent of a short straddle.
    """
    if not (K_lo < K_mid < K_hi):
        raise ValueError("iron_butterfly requires K_lo < K_mid < K_hi")
    return Strategy(
        name="Iron Butterfly",
        legs=[
            Leg("put", side=1, qty=qty, strike=K_lo, expiry_T=T),
            Leg("put", side=-1, qty=qty, strike=K_mid, expiry_T=T),
            Leg("call", side=-1, qty=qty, strike=K_mid, expiry_T=T),
            Leg("call", side=1, qty=qty, strike=K_hi, expiry_T=T),
        ],
        meta={"risk_class": "medium", "view": "pin at K_mid, short-vol, credit"},
    )


def iron_condor(K_put_long: float, K_put_short: float,
                K_call_short: float, K_call_long: float,
                T: float, qty: float = 1.0) -> Strategy:
    """Short iron condor: short OTM put spread + short OTM call spread.

    Legs: +1 put @ K_put_long, -1 put @ K_put_short,
          -1 call @ K_call_short, +1 call @ K_call_long.
    Requires K_put_long < K_put_short < K_call_short < K_call_long.

    Net credit received. Max profit = credit (if underlying stays between the
    short strikes at expiry). Max loss = max(wing width) - credit.

    The canonical retail income strategy, bounded risk on both sides.
    """
    if not (K_put_long < K_put_short < K_call_short < K_call_long):
        raise ValueError(
            "iron_condor requires K_put_long < K_put_short < K_call_short < K_call_long"
        )
    return Strategy(
        name="Iron Condor",
        legs=[
            Leg("put", side=1, qty=qty, strike=K_put_long, expiry_T=T),
            Leg("put", side=-1, qty=qty, strike=K_put_short, expiry_T=T),
            Leg("call", side=-1, qty=qty, strike=K_call_short, expiry_T=T),
            Leg("call", side=1, qty=qty, strike=K_call_long, expiry_T=T),
        ],
        meta={"risk_class": "medium", "view": "range-bound, short-vol, credit"},
    )


def short_call_butterfly(K_lo: float, K_mid: float, K_hi: float, T: float,
                         qty: float = 1.0) -> Strategy:
    """Short call butterfly: -1 K_lo, +2 K_mid, -1 K_hi.

    Exact mirror of long call butterfly. Net credit received.
    Max profit = credit (when underlying moves far from K_mid in either direction).
    Max loss at K_mid = (K_mid - K_lo) - credit  (assuming symmetric wings).
    Long vol, valley-shaped payoff.
    """
    if not (K_lo < K_mid < K_hi):
        raise ValueError("short_call_butterfly requires K_lo < K_mid < K_hi")
    return Strategy(
        name="Short Call Butterfly",
        legs=[
            Leg("call", side=-1, qty=qty, strike=K_lo, expiry_T=T),
            Leg("call", side=1, qty=2 * qty, strike=K_mid, expiry_T=T),
            Leg("call", side=-1, qty=qty, strike=K_hi, expiry_T=T),
        ],
        meta={"risk_class": "medium", "view": "breakout from K_mid, long-vol, credit"},
    )


def short_put_butterfly(K_lo: float, K_mid: float, K_hi: float, T: float,
                        qty: float = 1.0) -> Strategy:
    """Short put butterfly: -1 K_hi, +2 K_mid, -1 K_lo.

    Payoff identical to short call butterfly at the same strikes; put-construction
    chosen when the put side has better bid-ask liquidity.
    """
    if not (K_lo < K_mid < K_hi):
        raise ValueError("short_put_butterfly requires K_lo < K_mid < K_hi")
    return Strategy(
        name="Short Put Butterfly",
        legs=[
            Leg("put", side=-1, qty=qty, strike=K_hi, expiry_T=T),
            Leg("put", side=1, qty=2 * qty, strike=K_mid, expiry_T=T),
            Leg("put", side=-1, qty=qty, strike=K_lo, expiry_T=T),
        ],
        meta={"risk_class": "medium", "view": "breakout from K_mid, long-vol, credit"},
    )


def reverse_iron_butterfly(K_lo: float, K_mid: float, K_hi: float, T: float,
                           qty: float = 1.0) -> Strategy:
    """Reverse (long) iron butterfly: long straddle at K_mid + short wings at K_lo/K_hi.

    Legs: -1 put @ K_lo, +1 put @ K_mid, +1 call @ K_mid, -1 call @ K_hi.
    Net debit paid. Max profit at the wings; max loss at K_mid = net debit.
    Long-vol, valley-shaped, defined-risk equivalent of a long straddle.
    """
    if not (K_lo < K_mid < K_hi):
        raise ValueError("reverse_iron_butterfly requires K_lo < K_mid < K_hi")
    return Strategy(
        name="Reverse Iron Butterfly",
        legs=[
            Leg("put", side=-1, qty=qty, strike=K_lo, expiry_T=T),
            Leg("put", side=1, qty=qty, strike=K_mid, expiry_T=T),
            Leg("call", side=1, qty=qty, strike=K_mid, expiry_T=T),
            Leg("call", side=-1, qty=qty, strike=K_hi, expiry_T=T),
        ],
        meta={"risk_class": "medium", "view": "breakout from K_mid, long-vol, debit"},
    )


def broken_wing_butterfly(K_lo: float, K_mid: float, K_hi: float, T: float,
                          option: Literal["call", "put"] = "call",
                          qty: float = 1.0) -> Strategy:
    """Broken-wing (asymmetric) butterfly: +1 / -2 / +1 with unequal wing widths.

    Structurally identical to the standard butterfly (same leg ratios), but the
    wings K_mid - K_lo and K_hi - K_mid are not equal. The asymmetry introduces
    directional skew while keeping the short-vol tent characteristic at K_mid.

    Defined risk: max loss is the narrower wing minus net premium.

    Common construction: skew one wing wider to reduce the debit (or turn it into
    a small credit) in exchange for more loss potential on the narrower side.

    Parameters:
        K_lo, K_mid, K_hi : strikes (must be strictly increasing)
        option : "call" or "put" (payoff identical; choose by liquidity)
    """
    if not (K_lo < K_mid < K_hi):
        raise ValueError("broken_wing_butterfly requires K_lo < K_mid < K_hi")
    lower_wing = K_mid - K_lo
    upper_wing = K_hi - K_mid
    if abs(lower_wing - upper_wing) < 1e-6:
        # Symmetric: still legal, but the product point is asymmetry, so warn the caller
        # via meta (not an exception; some callers may still want this)
        symmetry_note = " (symmetric configuration; consider butterfly() instead)"
    else:
        symmetry_note = ""

    if option == "call":
        legs = [
            Leg("call", side=1, qty=qty, strike=K_lo, expiry_T=T),
            Leg("call", side=-1, qty=2 * qty, strike=K_mid, expiry_T=T),
            Leg("call", side=1, qty=qty, strike=K_hi, expiry_T=T),
        ]
    else:
        legs = [
            Leg("put", side=1, qty=qty, strike=K_hi, expiry_T=T),
            Leg("put", side=-1, qty=2 * qty, strike=K_mid, expiry_T=T),
            Leg("put", side=1, qty=qty, strike=K_lo, expiry_T=T),
        ]

    # Directional bias based on which wing is wider
    if upper_wing > lower_wing:
        view = "pin at K_mid with bullish skew (wider upper wing), short-vol"
    elif lower_wing > upper_wing:
        view = "pin at K_mid with bearish skew (wider lower wing), short-vol"
    else:
        view = "pin at K_mid, short-vol"

    return Strategy(
        name=f"Broken-Wing {option.capitalize()} Butterfly",
        legs=legs,
        meta={"risk_class": "medium", "view": view + symmetry_note},
    )


# Convenience registry, used by the recommender and the UI
STRATEGY_CATALOG = {
    "covered_call": covered_call,
    "protective_put": protective_put,
    "bull_call_spread": bull_call_spread,
    "bear_put_spread": bear_put_spread,
    "bull_put_spread": bull_put_spread,
    "bear_call_spread": bear_call_spread,
    "straddle": straddle,
    "strangle": strangle,
    "butterfly": butterfly,
    "collar": collar,
    # Wave 1 additions
    "long_call": long_call,
    "long_put": long_put,
    "cash_secured_put": cash_secured_put,
    "put_butterfly": put_butterfly,
    "iron_butterfly": iron_butterfly,
    "iron_condor": iron_condor,
    # Wave 2 additions (butterfly family completeness)
    "short_call_butterfly": short_call_butterfly,
    "short_put_butterfly": short_put_butterfly,
    "reverse_iron_butterfly": reverse_iron_butterfly,
    "broken_wing_butterfly": broken_wing_butterfly,
}
