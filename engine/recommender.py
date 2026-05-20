"""
Strategy recommender.

Takes a user's market view (directional + volatility) and ranks strategies from
the catalogue. Two rankers are provided:

  recommend_eu  (current) - ranks by the expected utility of each strategy's
      payoff distribution under the user's view, with risk aversion derived from
      the Capacity-for-Loss score. No tunable weights: direction and volatility
      come from a view-implied lognormal, risk from the curvature of a CRRA
      utility function.

  recommend  (legacy) - the original weighted-sum scorer on three fit-axes
      (direction, vol, risk). Kept for backward compatibility and reference; its
      weights are hand-picked, which is the limitation recommend_eu removes.

Author: Kanishk Devgan
Project: Amoghopāya
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DirectionalView = Literal["bullish", "bearish", "neutral"]
VolView = Literal["up", "down", "neutral"]
RiskAppetite = Literal["low", "medium", "high"]


# -----------------------------------------------------------------------------
# Strategy profiles
# -----------------------------------------------------------------------------
# Each profile encodes the *qualitative* Greek signature of the strategy
# (near the money, in the typical construction).
#
# direction: -1 bearish, 0 neutral, +1 bullish
# vega_sign: -1 short-vol, 0 neutral, +1 long-vol
# theta_sign: -1 bleeder, 0 neutral, +1 decay-collector
# risk_class: low / medium / high
# max_loss: "defined" / "undefined" (naked shorts)
# -----------------------------------------------------------------------------

STRATEGY_PROFILES: dict[str, dict] = {
    "covered_call": {
        "label": "Covered Call",
        "direction": 0,       # delta ~ 0.5 but capped; neutral-to-mildly-bullish
        "vega_sign": -1,
        "theta_sign": +1,
        "risk_class": "low",
        "max_loss": "defined (stock downside to zero)",
        "one_liner": "Earn premium on stock you own, capping upside at K.",
    },
    "protective_put": {
        "label": "Protective Put",
        "direction": +1,
        "vega_sign": +1,
        "theta_sign": -1,
        "risk_class": "low",
        "max_loss": "defined (S₀ − K + premium)",
        "one_liner": "Own stock with a floor; downside capped at the put strike.",
    },
    "bull_call_spread": {
        "label": "Bull Call Spread",
        "direction": +1,
        "vega_sign": 0,
        "theta_sign": 0,
        "risk_class": "medium",
        "max_loss": "defined (net debit)",
        "one_liner": "Moderately bullish; defined risk; cheaper than a naked call.",
    },
    "bear_put_spread": {
        "label": "Bear Put Spread",
        "direction": -1,
        "vega_sign": 0,
        "theta_sign": 0,
        "risk_class": "medium",
        "max_loss": "defined (net debit)",
        "one_liner": "Moderately bearish; defined risk; cheaper than a naked put.",
    },
    "bull_put_spread": {
        "label": "Bull Put Spread (credit)",
        "direction": +1,
        "vega_sign": -1,
        "theta_sign": +1,
        "risk_class": "medium",
        "max_loss": "defined (width − credit)",
        "one_liner": "Collect premium expecting price to stay above the short put.",
    },
    "bear_call_spread": {
        "label": "Bear Call Spread (credit)",
        "direction": -1,
        "vega_sign": -1,
        "theta_sign": +1,
        "risk_class": "medium",
        "max_loss": "defined (width − credit)",
        "one_liner": "Collect premium expecting price to stay below the short call.",
    },
    "straddle": {
        "label": "Long Straddle",
        "direction": 0,
        "vega_sign": +1,
        "theta_sign": -1,
        "risk_class": "high",
        "max_loss": "defined (total debit)",
        "one_liner": "Direction-agnostic; profits from a large move either way.",
    },
    "strangle": {
        "label": "Long Strangle",
        "direction": 0,
        "vega_sign": +1,
        "theta_sign": -1,
        "risk_class": "high",
        "max_loss": "defined (total debit)",
        "one_liner": "Cheaper vol play than a straddle; needs a bigger move to pay off.",
    },
    "butterfly": {
        "label": "Long Butterfly",
        "direction": 0,
        "vega_sign": -1,
        "theta_sign": +1,
        "risk_class": "medium",
        "max_loss": "defined (net debit)",
        "one_liner": "Pin-risk play: max payoff if underlying expires at K_mid.",
    },
    "collar": {
        "label": "Collar",
        "direction": +1,
        "vega_sign": 0,
        "theta_sign": 0,
        "risk_class": "low",
        "max_loss": "defined (S₀ − K_put − net credit)",
        "one_liner": "Bounded bullish: give up upside above K_call for downside protection.",
    },
    "long_call": {
        "label": "Long Call",
        "direction": +1,
        "vega_sign": +1,
        "theta_sign": -1,
        "risk_class": "medium",
        "max_loss": "defined (premium paid)",
        "one_liner": "Direct leveraged bullish exposure; unlimited upside, loss capped at the debit.",
    },
    "long_put": {
        "label": "Long Put",
        "direction": -1,
        "vega_sign": +1,
        "theta_sign": -1,
        "risk_class": "medium",
        "max_loss": "defined (premium paid)",
        "one_liner": "Direct leveraged bearish exposure; profits grow as underlying falls, loss capped at the debit.",
    },
    "cash_secured_put": {
        "label": "Cash-Secured Put",
        "direction": +1,
        "vega_sign": -1,
        "theta_sign": +1,
        "risk_class": "low",
        "max_loss": "defined (K − premium)",
        "one_liner": "Collect premium agreeing to buy stock at K if assigned; mirror of covered call.",
    },
    "put_butterfly": {
        "label": "Long Put Butterfly",
        "direction": 0,
        "vega_sign": -1,
        "theta_sign": +1,
        "risk_class": "medium",
        "max_loss": "defined (net debit)",
        "one_liner": "Same payoff as call butterfly; chosen when put-side strikes have better liquidity.",
    },
    "iron_butterfly": {
        "label": "Iron Butterfly",
        "direction": 0,
        "vega_sign": -1,
        "theta_sign": +1,
        "risk_class": "medium",
        "max_loss": "defined (wing width − credit)",
        "one_liner": "Short-vol credit structure; max profit at K_mid, bounded losses on both sides.",
    },
    "iron_condor": {
        "label": "Iron Condor",
        "direction": 0,
        "vega_sign": -1,
        "theta_sign": +1,
        "risk_class": "medium",
        "max_loss": "defined (max wing width − credit)",
        "one_liner": "The canonical retail income play: profit if the underlying stays range-bound.",
    },
    "short_call_butterfly": {
        "label": "Short Call Butterfly",
        "direction": 0,
        "vega_sign": +1,
        "theta_sign": -1,
        "risk_class": "medium",
        "max_loss": "defined (wing width − credit)",
        "one_liner": "Reversed butterfly collecting premium; profits on a breakout away from K_mid.",
    },
    "short_put_butterfly": {
        "label": "Short Put Butterfly",
        "direction": 0,
        "vega_sign": +1,
        "theta_sign": -1,
        "risk_class": "medium",
        "max_loss": "defined (wing width − credit)",
        "one_liner": "Put-construction of the short butterfly; same payoff, choose by liquidity.",
    },
    "reverse_iron_butterfly": {
        "label": "Reverse Iron Butterfly",
        "direction": 0,
        "vega_sign": +1,
        "theta_sign": -1,
        "risk_class": "medium",
        "max_loss": "defined (net debit)",
        "one_liner": "Long-vol, defined-risk version of a straddle; profits on a breakout either way.",
    },
    "broken_wing_butterfly": {
        "label": "Broken-Wing Butterfly",
        "direction": 0,
        "vega_sign": -1,
        "theta_sign": +1,
        "risk_class": "medium",
        "max_loss": "defined (narrower wing − net premium adjustment)",
        "one_liner": "Butterfly with asymmetric wings; skews directionally while keeping the pin character.",
    },
}


# -----------------------------------------------------------------------------
# Scoring
# -----------------------------------------------------------------------------

def _dir_to_int(v: DirectionalView) -> int:
    return {"bullish": +1, "bearish": -1, "neutral": 0}[v]


def _vol_to_int(v: VolView) -> int:
    return {"up": +1, "down": -1, "neutral": 0}[v]


def _risk_rank(r: RiskAppetite) -> int:
    return {"low": 1, "medium": 2, "high": 3}[r]


@dataclass
class Recommendation:
    key: str
    label: str
    score: float
    rationale: str
    profile: dict


def recommend(
    directional: DirectionalView,
    vol: VolView,
    risk: RiskAppetite,
    top_n: int = 3,
    weights: tuple[float, float, float] = (1.0, 0.7, 0.8),
) -> list[Recommendation]:
    """Rank strategies for a given view.

    weights = (direction_weight, vol_weight, risk_weight).
    """
    dir_int = _dir_to_int(directional)
    vol_int = _vol_to_int(vol)
    user_risk = _risk_rank(risk)

    w_dir, w_vol, w_risk = weights
    results: list[Recommendation] = []

    for key, p in STRATEGY_PROFILES.items():
        # Directional fit: 1 if signs agree; 0.5 if one is neutral; 0 if opposite
        if p["direction"] == dir_int:
            dir_score = 1.0
        elif p["direction"] == 0 or dir_int == 0:
            dir_score = 0.5
        else:
            dir_score = 0.0

        # Vol fit
        if p["vega_sign"] == vol_int:
            vol_score = 1.0
        elif p["vega_sign"] == 0 or vol_int == 0:
            vol_score = 0.5
        else:
            vol_score = 0.0

        # Risk fit: penalise strategies riskier than user appetite
        strat_risk = _risk_rank(p["risk_class"])
        if strat_risk <= user_risk:
            risk_score = 1.0
        else:
            # Each step beyond the user's appetite drops 0.4
            risk_score = max(0.0, 1.0 - 0.4 * (strat_risk - user_risk))

        score = w_dir * dir_score + w_vol * vol_score + w_risk * risk_score

        # Rationale
        reasons = []
        if dir_score >= 1.0:
            reasons.append(f"direction matches ({directional})")
        elif dir_score >= 0.5:
            reasons.append(f"direction-neutral exposure, compatible with {directional} view")
        else:
            reasons.append(f"direction mismatch (strategy leans {'bullish' if p['direction']>0 else 'bearish'})")

        if vol_score >= 1.0:
            reasons.append(f"vega sign matches vol view ({vol})")
        elif vol_score >= 0.5:
            reasons.append("vol-neutral")
        else:
            reasons.append("vega opposed to vol view")

        if strat_risk > user_risk:
            reasons.append(f"above stated risk appetite ({p['risk_class']} vs {risk})")

        rationale = "; ".join(reasons) + f". {p['one_liner']}"

        results.append(Recommendation(
            key=key,
            label=p["label"],
            score=round(score, 3),
            rationale=rationale,
            profile=p,
        ))

    # Primary sort: score descending.
    # Tiebreaker 1: prefer strategies whose risk class matches user appetite
    # exactly (risk_match=True beats risk_match=False), so we surface the
    # strategy best calibrated to the user's stated risk rather than always
    # the safest one.
    # Tiebreaker 2: alphabetical label for deterministic ordering.
    user_risk_str = risk
    results.sort(
        key=lambda r: (
            -r.score,
            0 if r.profile["risk_class"] == user_risk_str else 1,
            r.label,
        )
    )
    return results[:top_n]


# =============================================================================
# Expected-utility recommender (no weights)
# =============================================================================
# The weighted-sum recommender above blends three fit-scores with hand-picked
# weights, which encode a preference with no principled basis. This ranker drops
# them: it ranks strategies by the expected utility of their payoff distribution
# under the user's view, with risk aversion taken from the Capacity-for-Loss
# profile.
#
# No direction/vol/risk weights. Direction and volatility are captured by the
# view-implied distribution; risk by the curvature of the utility function. Two
# assumptions remain, both explicit (not hidden coefficients):
#
#   1. View -> distribution. Terminal price modelled as lognormal with:
#        drift  : bullish/bearish -> +/- 0.5 * sigma * sqrt(T); neutral -> 0
#                 (a moderate half-sigma tilt, self-scaling to the underlying's vol)
#        spread : vol up/down -> sigma * 1.3 / sigma * 0.7; neutral -> sigma
#                 (vol view sets the realised-vol spread the payoff is evaluated
#                  against; strategies are still priced at market sigma)
#
#   2. Utility. CRRA U(W) = W^(1-g)/(1-g), with g from the Capacity-for-Loss
#      score (0-10): high CFL -> low g (risk-tolerant), low CFL -> high g
#      (downside penalised). Reuses the suitability evidence from onboarding.
#
# Magnitudes are moderate defaults, exposed as parameters so the belief inputs
# can be adjusted.

import math as _math

import numpy as _np

from . import strategies as _strat


# View magnitude defaults (documented belief assumptions, adjustable)
_DRIFT_SIGMA_FRACTION = 0.5     # directional tilt = +/- 0.5 * sigma * sqrt(T)
_VOL_UP_MULT = 1.3              # vol-up view: realised spread = sigma * 1.3
_VOL_DOWN_MULT = 0.7           # vol-down view: realised spread = sigma * 0.7


def cfl_to_risk_aversion(cfl_score: float) -> float:
    """Map a Capacity-for-Loss score (0-10) to a CRRA risk-aversion coefficient g.

    High CFL (can absorb loss) -> low g (near risk-neutral, g ~ 1).
    Low CFL (cannot absorb loss) -> high g (strongly downside-averse, g ~ 6).

    The linear map g = 6 - 0.5 * cfl_score yields g in [1, 6] for cfl in [0, 10].
    g = 1 is treated as the log-utility limit (handled in the utility function).
    """
    g = 6.0 - 0.5 * float(cfl_score)
    return max(1.0, min(6.0, g))


def _crra_utility(wealth: _np.ndarray, gamma: float) -> _np.ndarray:
    """CRRA utility over (positive) terminal wealth.

    U(W) = W^(1-g)/(1-g) for g != 1; U(W) = ln(W) for g == 1.
    Wealth is floored at a small positive epsilon so the utility stays finite
    even when a strategy's payoff drives the account toward zero.
    """
    eps = 1e-6
    w = _np.maximum(wealth, eps)
    if abs(gamma - 1.0) < 1e-9:
        return _np.log(w)
    return _np.power(w, 1.0 - gamma) / (1.0 - gamma)


def _view_distribution(S: float, sigma: float, T: float,
                       directional: DirectionalView, vol: VolView,
                       n: int = 4001):
    """Return (S_T grid, probability weights) for the view-implied lognormal.

    Terminal price S_T = S * exp((mu - 0.5 sig_v^2) T + sig_v sqrt(T) Z), Z ~ N(0,1),
    where mu encodes the directional tilt and sig_v the view-adjusted spread.
    Returned as a discretised grid with normalised probability masses for
    numerical expectation.
    """
    T = max(T, 1e-6)
    sigma = max(sigma, 1e-6)

    # Directional drift: +/- 0.5 * sigma * sqrt(T) as an annualised drift mu so that
    # mu * T contributes the half-sigma move over the horizon.
    tilt = _DRIFT_SIGMA_FRACTION * sigma / _math.sqrt(T)
    if directional == "bullish":
        mu = tilt
    elif directional == "bearish":
        mu = -tilt
    else:
        mu = 0.0

    # View-adjusted volatility spread
    if vol == "up":
        sig_v = sigma * _VOL_UP_MULT
    elif vol == "down":
        sig_v = sigma * _VOL_DOWN_MULT
    else:
        sig_v = sigma

    # Discretise Z over +/- 5 sd
    z = _np.linspace(-5.0, 5.0, n)
    pdf = _np.exp(-0.5 * z * z)
    pdf /= pdf.sum()
    S_T = S * _np.exp((mu - 0.5 * sig_v * sig_v) * T + sig_v * _math.sqrt(T) * z)
    return S_T, pdf


# Canonical strategy instances at standard moneyness, for payoff evaluation.
# Strikes are placed relative to spot S using round, liquid offsets.
def _canonical_strategy(key: str, S: float, T: float):
    """Build a representative instance of strategy `key` at standard moneyness.

    Returns a Strategy, or None if the key is not constructible here.
    """
    f = _strat
    atm = S
    lo5, hi5 = S * 0.95, S * 1.05
    lo10, hi10 = S * 0.90, S * 1.10
    builders = {
        "covered_call": lambda: f.covered_call(S, hi5, T),
        "protective_put": lambda: f.protective_put(S, lo5, T),
        "bull_call_spread": lambda: f.bull_call_spread(atm, hi5, T),
        "bear_put_spread": lambda: f.bear_put_spread(atm, lo5, T),
        "bull_put_spread": lambda: f.bull_put_spread(atm, lo5, T),
        "bear_call_spread": lambda: f.bear_call_spread(atm, hi5, T),
        "straddle": lambda: f.straddle(atm, T),
        "strangle": lambda: f.strangle(lo5, hi5, T),
        "butterfly": lambda: f.butterfly(lo5, atm, hi5, T),
        "collar": lambda: f.collar(S, lo5, hi5, T),
        "long_call": lambda: f.long_call(atm, T),
        "long_put": lambda: f.long_put(atm, T),
        "cash_secured_put": lambda: f.cash_secured_put(lo5, T),
        "put_butterfly": lambda: f.put_butterfly(lo5, atm, hi5, T),
        "iron_butterfly": lambda: f.iron_butterfly(lo5, atm, hi5, T),
        "iron_condor": lambda: f.iron_condor(lo10, lo5, hi5, hi10, T),
        "short_call_butterfly": lambda: f.short_call_butterfly(lo5, atm, hi5, T),
        "short_put_butterfly": lambda: f.short_put_butterfly(lo5, atm, hi5, T),
        "reverse_iron_butterfly": lambda: f.reverse_iron_butterfly(lo5, atm, hi5, T),
        "broken_wing_butterfly": lambda: f.broken_wing_butterfly(lo5, atm, hi10, T),
    }
    builder = builders.get(key)
    if builder is None:
        return None
    try:
        return builder()
    except Exception:
        return None


@dataclass
class EURecommendation:
    key: str
    label: str
    expected_utility: float
    certainty_equivalent: float
    expected_pnl: float
    prob_profit: float
    rationale: str
    profile: dict


def recommend_eu(
    directional: DirectionalView,
    vol: VolView,
    cfl_score: float,
    S: float = 100.0,
    sigma: float = 0.20,
    T: float = 30.0 / 365.0,
    r: float = 0.05,
    q: float = 0.0,
    top_n: int = 3,
    account_wealth: float | None = None,
) -> list[EURecommendation]:
    """Rank strategies by expected utility under the user's view (no weights).

    directional, vol : the user's qualitative market view.
    cfl_score        : Capacity-for-Loss score (0-10) from the suitability module;
                       maps to CRRA risk aversion.
    S, sigma, T, r, q: market inputs. Strategies are priced at market sigma; the
                       payoff is evaluated against the view-implied distribution.
    account_wealth   : wealth base for CRRA utility. Defaults to a multiple of S
                       so per-contract P&L is a sensible fraction of the account.

    Returns the top_n strategies by expected utility, each with its certainty
    equivalent, expected P&L, and probability of profit under the view.
    """
    gamma = cfl_to_risk_aversion(cfl_score)
    S_T, probs = _view_distribution(S, sigma, T, directional, vol)

    # Wealth base: model a per-contract trade as a modest fraction of the account.
    # Default account = 100 * S (i.e. one contract notional ~ 1% of account), so
    # option P&L moves utility meaningfully but cannot bankrupt the account in the
    # typical defined-risk case.
    W0 = account_wealth if account_wealth is not None else 100.0 * S

    results: list[EURecommendation] = []
    for key, prof in STRATEGY_PROFILES.items():
        strat = _canonical_strategy(key, S, T)
        if strat is None:
            continue
        # Entry premium at market sigma (what you actually pay/receive)
        strat.set_entry_premium_from_marks(S, r, q, sigma)
        # Payoff per contract across the terminal grid (x100 contract multiplier)
        pnl_per_share = strat.payoff_at_expiry(S_T)
        pnl_contract = pnl_per_share * 100.0

        # Terminal wealth and expected utility under the view distribution
        wealth = W0 + pnl_contract
        util = _crra_utility(wealth, gamma)
        eu = float(_np.sum(probs * util))

        # Certainty equivalent: invert CRRA so EU is expressed back in wealth units
        if abs(gamma - 1.0) < 1e-9:
            ce = float(_math.exp(eu))
        else:
            ce = float(((1.0 - gamma) * eu) ** (1.0 / (1.0 - gamma)))
        ce_gain = ce - W0  # certainty-equivalent gain over doing nothing

        expected_pnl = float(_np.sum(probs * pnl_contract))
        prob_profit = float(_np.sum(probs * (pnl_contract > 0)))

        rationale = (
            f"Expected utility under a {directional}/{vol}-vol view with risk "
            f"aversion g={gamma:.1f} (from CFL). Certainty-equivalent gain "
            f"{ce_gain:+.0f}; expected P&L {expected_pnl:+.0f}; "
            f"probability of profit {prob_profit*100:.0f}%. {prof.get('one_liner','')}"
        )

        results.append(EURecommendation(
            key=key,
            label=prof["label"],
            expected_utility=eu,
            certainty_equivalent=ce,
            expected_pnl=expected_pnl,
            prob_profit=prob_profit,
            rationale=rationale,
            profile=prof,
        ))

    # Rank by expected utility (higher is better). Deterministic tiebreak by label.
    results.sort(key=lambda x: (-x.expected_utility, x.label))
    return results[:top_n]
