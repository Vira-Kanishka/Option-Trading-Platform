"""
Amoghopāya options engine.

Public API:
    bsm          : BSM pricing + Greeks + implied vol (closed-form, European)
    american     : American option pricing via CRR binomial tree + early-exercise premium
    strategies   : Leg (with optional per-leg IV), Strategy, and factories for the catalogue
    scenarios    : stress-test grid
    margin       : Reg-T initial margin
    recommender  : view → top-N strategies
    suitability  : investor profile → allowed strategies + risk flags
"""

from . import american, bsm, margin, recommender, scenarios, strategies, suitability

__all__ = ["bsm", "american", "strategies", "scenarios", "margin", "recommender", "suitability"]
__version__ = "0.2.0"
