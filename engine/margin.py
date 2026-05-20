"""
Simplified Reg-T initial margin.

Implements the retail margin rules for US equity options as governed by
FINRA Rule 4210 and CBOE. These are INITIAL margin requirements at order entry,
not variation / maintenance.

We keep it intentionally simplified: the goal is product risk transparency,
not clearing-house precision. Edge cases (dividends, early exercise risk,
haircuts, portfolio margin) are out of scope.

Rules implemented
-----------------
Long options (debit):           premium only
Naked short call:               max(20% * S - OTM_amount, 10% * S) + premium   [CBOE Rule 10.3]
Naked short put:                max(20% * S - OTM_amount, 10% * K) + premium
Covered call (long stock):      0 additional (stock collateralises)
Protective put:                 0 additional (cost of put is the cost)
Debit vertical spreads:         net debit paid (max loss)
Credit vertical spreads:        width of strikes - net credit received (max loss)
Long straddle/strangle:         sum of debits
Long butterfly (debit):         net debit paid
Collar:                         0 additional (stock collateralises; premiums net)
Short straddle/strangle:        greater of the two naked margins + other leg's premium

All dollar figures are per-share; UI multiplies by contract multiplier (×100).

Author: Kanishk Devgan
Project: Amoghopāya
"""

from __future__ import annotations

from .bsm import bs_price
from .strategies import Leg, Strategy


def _otm_amount_call(S: float, K: float) -> float:
    return max(K - S, 0.0)


def _otm_amount_put(S: float, K: float) -> float:
    return max(S - K, 0.0)


def _leg_mark(leg: Leg, S: float, r: float, q: float, sigma: float) -> float:
    """Absolute mark price per unit (unsigned, qty-weighted)."""
    if leg.kind == "underlying":
        return leg.qty * S
    return leg.qty * bs_price(S, leg.strike, leg.expiry_T, r, q, sigma, leg.kind)


def naked_short_call_margin(S: float, K: float, premium: float) -> float:
    """FINRA/CBOE naked short call initial margin, per share."""
    m1 = 0.20 * S - _otm_amount_call(S, K) + premium
    m2 = 0.10 * S + premium
    return max(m1, m2)


def naked_short_put_margin(S: float, K: float, premium: float) -> float:
    """FINRA/CBOE naked short put initial margin, per share."""
    m1 = 0.20 * S - _otm_amount_put(S, K) + premium
    m2 = 0.10 * K + premium
    return max(m1, m2)


def strategy_margin(strat: Strategy, S: float, r: float, q: float, sigma: float) -> dict:
    """Return a dict with margin breakdown.

    Keys:
        initial_margin : per-share dollar amount the broker reserves
        notes          : human-readable explanation
        bucket         : one of {long_debit, covered, vertical_debit, vertical_credit,
                                 naked_short, long_vol, short_vol, collar, complex}
    """
    name = strat.name.lower()
    legs = strat.legs

    # Covered structures (underlying present)
    has_underlying = any(l.kind == "underlying" for l in legs)
    option_legs = [l for l in legs if l.kind != "underlying"]

    # --- Covered call ---
    if "covered call" in name and has_underlying:
        return {
            "initial_margin": 0.0,
            "bucket": "covered",
            "notes": "Stock collateralises the short call; no additional margin.",
        }

    # --- Protective put ---
    if "protective put" in name and has_underlying:
        put_leg = next(l for l in option_legs if l.kind == "put" and l.side == 1)
        premium = _leg_mark(put_leg, S, r, q, sigma)
        return {
            "initial_margin": premium,
            "bucket": "long_debit",
            "notes": "Long put is fully paid; cost = initial margin.",
        }

    # --- Collar ---
    if "collar" in name and has_underlying:
        put_leg = next(l for l in option_legs if l.kind == "put" and l.side == 1)
        call_leg = next(l for l in option_legs if l.kind == "call" and l.side == -1)
        net_debit = _leg_mark(put_leg, S, r, q, sigma) - _leg_mark(call_leg, S, r, q, sigma)
        return {
            "initial_margin": max(net_debit, 0.0),
            "bucket": "collar",
            "notes": "Short call covered by stock; long put paid in full. Net premium shown.",
        }

    # --- Vertical spreads (two legs, same type) ---
    if len(option_legs) == 2 and not has_underlying:
        a, b = option_legs
        if a.kind == b.kind:
            p_a = _leg_mark(a, S, r, q, sigma)
            p_b = _leg_mark(b, S, r, q, sigma)
            net = a.side * p_a + b.side * p_b   # signed mark
            strikes = sorted([a.strike, b.strike])
            width = strikes[1] - strikes[0]
            if net > 0:  # debit spread
                return {
                    "initial_margin": net,
                    "bucket": "vertical_debit",
                    "notes": f"Debit vertical: max loss = net debit ({net:.2f}).",
                }
            else:  # credit spread
                max_loss = width - abs(net)
                return {
                    "initial_margin": max_loss,
                    "bucket": "vertical_credit",
                    "notes": f"Credit vertical: max loss = width ({width:.2f}) - credit ({abs(net):.2f}).",
                }

    # --- Iron butterfly / iron condor (4 legs, mixed calls+puts, net credit) ---
    # Structure: long put wing, short put, short call, long call wing
    # Max loss = max(put_spread_width, call_spread_width) - net credit
    if len(option_legs) == 4 and not has_underlying:
        calls = [l for l in option_legs if l.kind == "call"]
        puts = [l for l in option_legs if l.kind == "put"]
        if len(calls) == 2 and len(puts) == 2:
            long_call_leg = next((l for l in calls if l.side == 1), None)
            short_call_leg = next((l for l in calls if l.side == -1), None)
            long_put_leg = next((l for l in puts if l.side == 1), None)
            short_put_leg = next((l for l in puts if l.side == -1), None)
            if all([long_call_leg, short_call_leg, long_put_leg, short_put_leg]):
                call_width = long_call_leg.strike - short_call_leg.strike
                put_width = short_put_leg.strike - long_put_leg.strike
                # Net signed premium (positive = debit, negative = credit)
                net = sum(l.side * _leg_mark(l, S, r, q, sigma) / l.qty for l in option_legs)
                if net < 0:  # net credit (iron butterfly or iron condor)
                    max_wing = max(call_width, put_width)
                    max_loss = max_wing - abs(net)
                    bucket_name = ("iron_butterfly"
                                   if short_call_leg.strike == short_put_leg.strike
                                   else "iron_condor")
                    return {
                        "initial_margin": max(max_loss, 0.0),
                        "bucket": bucket_name,
                        "notes": f"{bucket_name.replace('_', ' ').title()}: "
                                 f"max loss = max wing width ({max_wing:.2f}) "
                                 f"− credit ({abs(net):.2f}).",
                    }
                else:  # net debit (reverse iron butterfly, long the structure)
                    # Max loss = net debit paid (at K_mid the structure is worthless)
                    return {
                        "initial_margin": net,
                        "bucket": "reverse_iron_butterfly",
                        "notes": f"Reverse iron butterfly: max loss = net debit paid ({net:.2f}).",
                    }

    # --- 3-leg butterfly-shape structures (±1, ∓2, ±1 in one option type) ---
    # Handles:  long call/put butterfly (debit),
    #           short call/put butterfly (credit),
    #           broken-wing butterfly (either direction, possibly credit)
    if len(option_legs) == 3 and not has_underlying:
        kinds = {l.kind for l in option_legs}
        if len(kinds) == 1:  # all same option type
            # Sort by strike
            sorted_legs = sorted(option_legs, key=lambda l: l.strike)
            wing_lo, middle, wing_hi = sorted_legs
            # Detect butterfly leg signature: outer legs same sign, middle opposite,
            # and |middle.qty| = 2 * |wing.qty|
            if (wing_lo.side == wing_hi.side
                    and middle.side == -wing_lo.side
                    and abs(middle.qty - 2 * wing_lo.qty) < 1e-9
                    and abs(wing_hi.qty - wing_lo.qty) < 1e-9):
                lower_wing = middle.strike - wing_lo.strike
                upper_wing = wing_hi.strike - middle.strike
                narrower = min(lower_wing, upper_wing)
                net = strat.mark_value(S, r, q, sigma)  # positive debit / negative credit
                if wing_lo.side == 1:  # long butterfly variants (outer legs long)
                    # Max loss = net debit (worthless at expiry away from K_mid)
                    # For broken-wing with credit: max loss = narrower wing − credit
                    if net > 0:
                        return {
                            "initial_margin": net,
                            "bucket": "long_butterfly",
                            "notes": f"Long butterfly: max loss = net debit ({net:.2f}).",
                        }
                    else:
                        max_loss = narrower - abs(net)
                        return {
                            "initial_margin": max(max_loss, 0.0),
                            "bucket": "broken_wing_credit",
                            "notes": f"Broken-wing (credit): max loss = narrower wing ({narrower:.2f}) − credit ({abs(net):.2f}).",
                        }
                else:  # short butterfly variants (outer legs short)
                    # Max loss at K_mid = wing width − credit received
                    # (wings are long so they cap the loss)
                    if net < 0:  # credit, the expected case
                        max_loss = narrower - abs(net)
                        return {
                            "initial_margin": max(max_loss, 0.0),
                            "bucket": "short_butterfly",
                            "notes": f"Short butterfly: max loss at K_mid = narrower wing ({narrower:.2f}) − credit ({abs(net):.2f}).",
                        }
                    else:
                        return {
                            "initial_margin": narrower + net,  # conservative
                            "bucket": "short_butterfly_debit",
                            "notes": "Short butterfly with debit (unusual); treating as narrower-wing + debit.",
                        }

    # Butterfly (debit): fallback if the 3-leg pattern wasn't matched above
    if "butterfly" in name:
        p = strat.mark_value(S, r, q, sigma)  # should be small positive for long butterfly
        return {
            "initial_margin": max(p, 0.0),
            "bucket": "long_debit",
            "notes": "Long butterfly: max loss = net debit paid.",
        }

    # --- Straddle / strangle (long) ---
    if ("straddle" in name or "strangle" in name) and all(l.side == 1 for l in option_legs):
        p = strat.mark_value(S, r, q, sigma)
        return {
            "initial_margin": max(p, 0.0),
            "bucket": "long_vol",
            "notes": "Long premium structure: max loss = total debit paid.",
        }

    # --- Cash-secured put (single short put, cash-backed) ---
    if "cash-secured put" in name:
        leg = option_legs[0]
        p = _leg_mark(leg, S, r, q, sigma)
        # Cash collateral = strike minus premium received
        return {
            "initial_margin": leg.strike - p,
            "bucket": "cash_secured",
            "notes": f"Cash-secured: collateral = K ({leg.strike:.2f}) − credit ({p:.2f}).",
        }

    # --- Naked shorts (single-leg) ---
    if len(option_legs) == 1 and not has_underlying:
        leg = option_legs[0]
        p = _leg_mark(leg, S, r, q, sigma)
        if leg.side == 1:
            return {
                "initial_margin": p,
                "bucket": "long_debit",
                "notes": "Long option: premium is the max loss.",
            }
        # Short single option
        if leg.kind == "call":
            m = naked_short_call_margin(S, leg.strike, p)
        else:
            m = naked_short_put_margin(S, leg.strike, p)
        return {
            "initial_margin": m,
            "bucket": "naked_short",
            "notes": "Reg-T naked short: max(20%·S − OTM, 10%·S or K) + premium.",
        }

    # --- Fallback: conservative, treat unknown complex as naked on worst leg ---
    return {
        "initial_margin": strat.mark_value(S, r, q, sigma),
        "bucket": "complex",
        "notes": "Complex structure: shown as net mark; broker may apply stricter rules.",
    }
