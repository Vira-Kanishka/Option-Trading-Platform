"""
Suitability gate and risk flags.

The module exposes two suitability rubrics:

**V2 (evidence-based, current)**: mirrors the React app's user-facing rubric.
    Two independent scores drive the approval level:
        experience_score in [0, 10], from trade count, years active,
                                     formal credentials.
        capacity_score   in [0, 10], from liquid capital, allocation
                                     percentage, and a Capacity-for-Loss (CFL)
                                     scenario test.
    A level is granted only when both scores meet the level threshold. This
    prevents a wealthy beginner (high capital, no experience) reaching L3
    and vice versa.

    Levels collapse to 1-3 because the v1 catalogue is all defined-risk.
    A fourth tier could be added with naked shorts or ratio spreads.

**V1 (legacy, FINRA-style)**: older four-tier self-declared rubric, retained
    for reference and backward compatibility with pre-v2 tests. Not used by
    the React app.

The two rubrics use different ``InvestorProfile`` classes (``InvestorProfile``
for V1, ``InvestorProfileV2`` for V2) to avoid silent behavioural changes for
callers of the legacy API.

Additionally this module exposes ``risk_flags`` which returns per-strategy
risk warnings used in the Review screen.

Author: Kanishk Devgan
Project: Amoghopāya
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .strategies import Strategy


# =============================================================================
# V2 rubric (evidence-based two-factor; current React app rubric)
# =============================================================================

TradesLastYear = Literal["none", "1_10", "11_50", "50_plus"]
YearsActive = Literal["under_1", "1_3", "3_10", "over_10"]
Credential = Literal["series_7", "finance_role", "derivatives_course",
                     "cfa_l2", "frm", "cqf", "msc", "phd"]
ImpactIf50pctLoss = Literal["unaffected", "stressful", "significant", "devastating"]


@dataclass
class InvestorProfileV2:
    """Evidence-based investor profile, matching the React app schema."""
    # Experience inputs
    trades_last_year: TradesLastYear = "none"
    years_active: YearsActive = "under_1"
    credentials: list[Credential] = field(default_factory=list)
    # Capacity inputs
    liquid_capital: float = 50_000.0     # USD
    allocation_pct: float = 10.0         # % of liquid_capital allocated
    impact_if_50pct_loss: ImpactIf50pctLoss = "stressful"


def compute_experience_score(p: InvestorProfileV2) -> int:
    """Experience score ∈ [0, 10].

    Weights:
        trades_last_year: 0-4
        years_active: 0-3
        credentials: +1 each, capped at 3
    """
    trades_pts = {"none": 0, "1_10": 1, "11_50": 3, "50_plus": 4}.get(p.trades_last_year, 0)
    years_pts = {"under_1": 0, "1_3": 1, "3_10": 2, "over_10": 3}.get(p.years_active, 0)
    cred_pts = min(3, len(p.credentials or []))
    return min(10, trades_pts + years_pts + cred_pts)


def compute_capacity_score(p: InvestorProfileV2) -> int:
    """Capacity score ∈ [0, 10].

    Weights:
        capital_pts: 0-6 (log-step)
        alloc_pts: 0-2 (sweet spot 5-30%)
        impact_pts: +2 / +1 / -2 / -5 depending on scenario response
    Final result is floored at 0 and capped at 10.
    """
    lc = p.liquid_capital
    if lc >= 1_000_000: capital_pts = 6
    elif lc >= 500_000: capital_pts = 5
    elif lc >= 100_000: capital_pts = 4
    elif lc >= 50_000:  capital_pts = 3
    elif lc >= 10_000:  capital_pts = 2
    else:               capital_pts = 0

    ap = p.allocation_pct
    if ap > 50:    alloc_pts = 0  # reckless
    elif ap > 30:  alloc_pts = 1
    elif ap >= 5:  alloc_pts = 2
    elif ap >= 1:  alloc_pts = 1
    else:          alloc_pts = 0

    impact_pts = {
        "unaffected": 2,
        "stressful": 1,
        "significant": -2,   # hard downgrade
        "devastating": -5,   # floor
    }.get(p.impact_if_50pct_loss, 0)

    return max(0, min(10, capital_pts + alloc_pts + impact_pts))


def approval_level_v2(p: InvestorProfileV2) -> int:
    """Two-factor approval level 1-3.

    Level 3 requires experience >= 6 AND capacity >= 5.
    Level 2 requires experience >= 3 AND capacity >= 3.
    Otherwise Level 1.

    Both gates must be met; a high score on only one axis is insufficient.
    """
    exp = compute_experience_score(p)
    cap = compute_capacity_score(p)
    if exp >= 6 and cap >= 5:
        return 3
    if exp >= 3 and cap >= 3:
        return 2
    return 1


# Lot-count caps per level. Scales with both experience and capacity gates.
LOT_CAPS: dict[int, int] = {1: 1, 2: 3, 3: 25}


@dataclass
class UpgradePath:
    next_level: int
    gaps: list[str]


def upgrade_path(p: InvestorProfileV2):
    """Return what the user needs to improve to upgrade to the next level.

    Returns ``None`` if already at the top level (3).
    """
    exp = compute_experience_score(p)
    cap = compute_capacity_score(p)
    level = approval_level_v2(p)
    if level == 3:
        return None
    thresholds = {1: (3, 3), 2: (6, 5)}
    exp_need, cap_need = thresholds[level]
    gaps: list[str] = []
    if exp < exp_need:
        gaps.append(f"experience score +{exp_need - exp} (currently {exp}/10)")
    if cap < cap_need:
        gaps.append(f"CFL score +{cap_need - cap} (currently {cap}/10)")
    return UpgradePath(next_level=level + 1, gaps=gaps)


# =============================================================================
# Level -> allowed strategy keys (shared between V1 and V2)
# =============================================================================
# The v1 catalogue is all defined-risk. Level 3 unlocks the full 20 strategies.
# If naked shorts or ratio spreads were added, a Level 4 would be appropriate.
# =============================================================================

_LEVEL_1_KEYS = {
    "covered_call",
    "protective_put",
    "cash_secured_put",
}

_LEVEL_2_KEYS = _LEVEL_1_KEYS | {
    "long_call",
    "long_put",
    "straddle",
    "strangle",
}

_LEVEL_3_KEYS = _LEVEL_2_KEYS | {
    "bull_call_spread", "bear_put_spread",
    "bull_put_spread", "bear_call_spread",
    "butterfly", "put_butterfly",
    "short_call_butterfly", "short_put_butterfly",
    "iron_butterfly", "reverse_iron_butterfly",
    "broken_wing_butterfly",
    "iron_condor",
    "collar",
}

APPROVAL_LEVELS: dict[int, set[str]] = {
    1: _LEVEL_1_KEYS,
    2: _LEVEL_2_KEYS,
    3: _LEVEL_3_KEYS,
    4: _LEVEL_3_KEYS,  # legacy: same as L3 (v1 catalog is defined-risk only)
}


def allowed_strategy_keys_v2(p: InvestorProfileV2) -> set[str]:
    """Allowed strategies for a V2 profile."""
    return APPROVAL_LEVELS[approval_level_v2(p)]


# =============================================================================
# V1 rubric (legacy FINRA-style; kept for backward compatibility)
# =============================================================================
# Pre-v2 tests and any consumer holding a reference to these symbols continue
# to work unchanged. New code should use the V2 API.
# =============================================================================

Experience = Literal["none", "some", "experienced", "professional"]
Objective = Literal["income", "growth", "hedging", "speculation"]
LossTolerance = Literal["low", "medium", "high"]


@dataclass
class InvestorProfile:
    """Legacy self-declared investor profile. See ``InvestorProfileV2`` for current."""
    experience: Experience
    objective: Objective
    loss_tolerance: LossTolerance
    liquid_net_worth: float
    annual_income: float
    age: int


def assign_approval_level(profile: InvestorProfile) -> int:
    """Legacy approval assignment. Levels 1-4."""
    base = {"none": 1, "some": 2, "experienced": 3, "professional": 4}[profile.experience]
    if profile.loss_tolerance == "low" and base > 2:
        base = 2
    if profile.liquid_net_worth < 10_000 and base > 2:
        base = 2
    if profile.annual_income < 25_000 and base > 1:
        base = max(1, base - 1)
    if profile.age >= 75 and base > 2:
        base = 2
    if profile.objective == "income" and profile.experience == "none":
        base = min(base, 1)
    return max(1, min(4, base))


def allowed_strategy_keys(profile: InvestorProfile) -> set[str]:
    """Legacy API: allowed strategies for a V1 profile."""
    return APPROVAL_LEVELS[assign_approval_level(profile)]


# =============================================================================
# Risk flags (shared, strategy-level; used by Review screen)
# =============================================================================

def risk_flags(strat: Strategy, S: float, r: float, q: float, sigma: float) -> list[dict]:
    """Return a list of risk-flag dicts to display in the Review screen.

    Each flag has: severity ("info"/"warn"/"danger"), code, message.
    """
    flags: list[dict] = []
    option_legs = [l for l in strat.legs if l.kind != "underlying"]
    has_underlying = any(l.kind == "underlying" for l in strat.legs)

    net_short_calls = (
        sum(-l.side * l.qty for l in option_legs if l.kind == "call" and l.side == -1)
        - sum(l.side * l.qty for l in option_legs if l.kind == "call" and l.side == +1)
    )
    if net_short_calls > 0 and not has_underlying:
        flags.append({
            "severity": "danger",
            "code": "UNLIMITED_UPSIDE_LOSS",
            "message": "Net short calls without stock coverage: theoretically unlimited loss if underlying rallies.",
        })

    net_short_puts = (
        sum(-l.side * l.qty for l in option_legs if l.kind == "put" and l.side == -1)
        - sum(l.side * l.qty for l in option_legs if l.kind == "put" and l.side == +1)
    )
    if net_short_puts > 0:
        flags.append({
            "severity": "warn",
            "code": "LARGE_DOWNSIDE_LOSS",
            "message": "Net short puts: loss grows as underlying falls, bounded only at zero.",
        })

    short_legs_close_to_expiry = [
        l for l in option_legs
        if l.side == -1 and l.expiry_T is not None and l.expiry_T <= 30 / 365
    ]
    if short_legs_close_to_expiry:
        flags.append({
            "severity": "warn",
            "code": "ASSIGNMENT_RISK",
            "message": "Short leg within 30 days of expiry: rising assignment/exercise probability, especially if ITM.",
        })

    greeks = strat.aggregate_greeks(S, r, q, sigma)
    if greeks.theta < 0 and abs(greeks.theta) > 0.005 * S:
        flags.append({
            "severity": "info",
            "code": "THETA_DECAY",
            "message": f"Strategy bleeds approx ${abs(greeks.theta)/365:.2f} per share per day at current vol.",
        })

    if abs(greeks.vega) > 0.05 * S:
        direction = "long vol" if greeks.vega > 0 else "short vol"
        flags.append({
            "severity": "info",
            "code": "VEGA_EXPOSURE",
            "message": f"Large vega exposure ({direction}); a 1-vol-point move moves P&L by approx ${abs(greeks.vega)/100:.2f}/share.",
        })

    expiries = {round(l.expiry_T, 6) for l in option_legs if l.expiry_T is not None}
    if len(expiries) == 1:
        flags.append({
            "severity": "info",
            "code": "SINGLE_EXPIRY",
            "message": "All legs share one expiry: no calendar diversification.",
        })

    return flags
