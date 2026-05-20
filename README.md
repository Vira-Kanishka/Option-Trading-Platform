# Amoghopāya : Option Trading Platform

**A strategy-first options trading platform.** Built for the Risk Management challenge task, MSc Financial Technology, University of Birmingham.

![tests](https://github.com/USERNAME/amoghopaya/actions/workflows/tests.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

Amoghopāya covers the full options trade lifecycle across twenty defined-risk strategies: strategy discovery, suitability gating, trade construction, risk review, position management, and a strategy-level volatility-and-Greeks analysis lab. It pairs a React prototype (the user-facing app) with a tested Python quantitative engine.

> **Live demo:** _add your CodeSandbox / Vercel link here_
>
> **Note:** This is an educational prototype. It is paper-trading only, places no real orders, and does not constitute financial advice.

---

## What it does

- **Twenty defined-risk strategies** : covered call, protective put, cash-secured put, the four verticals, long straddle/strangle, the butterfly family (long/short, call/put, iron, reverse iron, broken-wing), iron condor, collar, and the single-leg directionals.
- **Evidence-based suitability** : a two-factor model combining an *experience score* (trade count, years active, credentials) and a *Capacity-for-Loss score* (liquid capital, allocation, a 50%-loss scenario test). Both gates must be met for each approval level; a wealthy beginner cannot reach Level 3 on capital alone.
- **European and American exercise** : choose the exercise style up front and it threads through pricing, Greeks, P&L, and stress tests. European options use closed-form Black-Scholes-Merton with continuous dividend yield; American options use a Cox-Ross-Rubinstein binomial tree.
- **Analytic and lattice Greeks** : all five Greeks (delta, gamma, vega, theta, rho). For European options they are closed-form and validated against finite differences in CI. For American options, Delta and Gamma are extracted from the binomial lattice; Vega, Theta, and Rho come from bumps with a control-variate-plus-step-averaging correction that cancels the binomial sawtooth, bringing them to better than 0.5% against an independent high-resolution reference (validated in the test suite).
- **Scenario stress testing** : P&L across a grid of spot and volatility shocks, with time decay.
- **Reg-T initial margin** : implemented across fourteen strategy buckets following FINRA Rule 4210 / CBOE conventions.
- **Expected-utility recommender (no weights)** : ranks strategies by the expected utility of their payoff distribution under the user's view, with risk aversion derived from the Capacity-for-Loss profile. There are no tunable direction/vol/risk weights: direction and volatility are captured by a view-implied lognormal, risk by the curvature of a CRRA utility function. Every recommendation reports its certainty-equivalent gain, expected P&L, and probability of profit.

## Repository layout

```
amoghopaya/
├── engine/                 # Python quantitative engine
│   ├── bsm.py              # Black-Scholes-Merton pricing, analytic Greeks, implied vol
│   ├── american.py         # CRR binomial American pricing + early-exercise premium
│   ├── strategies.py       # Leg (with optional per-leg IV), Strategy, 20 factories
│   ├── scenarios.py        # spot/vol stress grid
│   ├── margin.py           # Reg-T initial margin
│   ├── recommender.py      # expected-utility strategy ranking (no weights)
│   └── suitability.py      # two-factor approval rubric + risk flags
├── tests/
│   └── test_engine.py      # 125 tests
├── web/
│   └── amoghopaya_app.jsx  # React prototype (single-file component)
├── docs/                   # design notes
├── conftest.py
├── requirements.txt
└── .github/workflows/      # CI
```

## Quick start — the engine

```bash
git clone https://github.com/USERNAME/amoghopaya.git
cd amoghopaya
pip install -r requirements.txt
pytest -q
```

You should see `125 passed`.

### Example usage

```python
from engine import bsm, strategies, american

# Price a single option and inspect its Greeks
price = bsm.bs_price(S=450, K=450, T=30/365, r=0.05, q=0.013, sigma=0.20, option_type="call")
greeks = bsm.bs_greeks(S=450, K=450, T=30/365, r=0.05, q=0.013, sigma=0.20, option_type="call")

# Build a bull call spread and aggregate its Greeks
spread = strategies.bull_call_spread(K_lo=445, K_hi=460, T=30/365)
spread.set_entry_premium_from_marks(S=450, r=0.05, q=0.013, sigma=0.20)
net_greeks = spread.aggregate_greeks(S=450, r=0.05, q=0.013, sigma=0.20)

# Quantify the early-exercise premium for an American put
prem = american.early_exercise_premium(S=400, K=450, T=1.0, r=0.06, q=0.0, sigma=0.25, option_type="put")
```

### Per-leg implied volatility

Each leg can carry its own implied vol to reflect the volatility surface, rather than a single flat sigma:

```python
from engine.strategies import Leg, Strategy

# A vertical priced with a skew: the lower strike trades at a richer vol
spread = Strategy("skewed_call_spread", [
    Leg("call", side=1, qty=1, strike=445, expiry_T=0.5, iv=0.24),
    Leg("call", side=-1, qty=1, strike=460, expiry_T=0.5, iv=0.20),
])
value = spread.mark_value(S=450, r=0.05, q=0.013, sigma=0.22)  # sigma is the fallback
```

## The web app

`web/amoghopaya_app.jsx` is a single-file React component. To run it locally you need a Vite + React + Tailwind scaffold; the live demo above is the easiest way to see it. Dependencies: React 18, Recharts, lucide-react, Tailwind.

## Modelling notes and limitations

This is a teaching prototype, and several choices are deliberate simplifications:

- **European or American exercise.** The platform supports both styles, selected up front and threaded through pricing, Greeks, P&L, and stress tests. European options use closed-form BSM; American options use a CRR binomial tree. American Greeks are extracted from the lattice (Delta, Gamma) and small bumps (Vega, Theta, Rho), following the lattice-extraction idea of in 't Hout (arXiv:2401.13361). For the short-dated, near-the-money defined-risk catalogue the two styles are nearly identical; they diverge for cases like deep-ITM long-dated puts, where the early-exercise premium is material.
- **Hand-picked policy parameters.** The suitability scoring brackets, risk-flag thresholds, and stress-shock magnitudes are design choices, not data-calibrated constants. The recommender ranks by expected utility but it rests on two explicit, documented belief assumptions (the view-to-distribution drift/spread magnitudes and the CRRA utility form with CFL-derived risk aversion). The pricing/Greeks/margin/parity layer underneath is derived or regulation-specified. Production use would require calibration and regulatory review.
- **Flat or per-leg vol, not a full surface model.** The engine supports per-leg implied vol but does not implement a stochastic or local volatility model (Heston, Dupire, SABR). For listed-strike pricing, per-leg IV from a vendor surface would be an improvement.

## Author

Kanishk Devgan 

## License

MIT — see [LICENSE](LICENSE).
