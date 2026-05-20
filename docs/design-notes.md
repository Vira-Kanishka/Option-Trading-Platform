# Design notes

Short notes on the non-obvious design decisions behind Amoghopāya.

## Why a two-factor suitability model

Most retail brokers gate options access on self-declared experience ("beginner / intermediate / expert"). That is gameable — anyone can tick "expert." Amoghopāya scores two independent axes from evidence:

- **Experience** — trade count over the last year, years active, and formal credentials. Each contributes points; credentials are capped so a stack of certificates cannot substitute for activity.
- **Capacity for Loss** — liquid capital, the fraction of it being allocated, and a scenario question: what happens if you lose 50% of what you put in? This follows the FCA's Capacity-for-Loss framing, which asks not just whether a loss is tolerable in attitude but whether it is survivable in fact.

Both gates must clear for each level. This is intentionally stricter than a single composite score: a wealthy beginner (high capacity, zero experience) is held at Level 1, and an experienced trader who says a 50% loss would be devastating is pulled down regardless of track record.

The level then caps both *which* strategies unlock and the *position size* (`LOT_CAPS`).

## Why closed-form pricing

The user drags sliders and watches payoff curves and a Greek surface redraw in real time. Closed-form Black-Scholes-Merton gives microsecond pricing and machine-precision analytic Greeks, which keeps the UI responsive. Monte Carlo would be noisy and slow; binomial trees are reserved for the American early-exercise question, where they are actually needed.

## European and American exercise

The platform supports both exercise styles. The style is chosen up front and threads through every downstream calculation: pricing, Greeks, P&L, and the stress grid all recompute under the selected style.

European options price in closed-form BSM. American options price on a Cox-Ross-Rubinstein binomial tree with an early-exercise check at every node. American options have no clean closed-form Greeks because the early-exercise boundary makes the value function non-smooth there, so Delta and Gamma are extracted directly from the binomial lattice (the recombining step-2 nodes give a centred stencil around spot at no extra cost), and Vega, Theta, and Rho come from small re-pricings. This lattice-extraction approach follows the idea in in 't Hout (arXiv:2401.13361), which extracts the same Greeks from a PDE grid; the binomial analogue is simpler to implement and adequate for a defined-risk catalogue.

For the platform's short-dated, near-the-money structures the two styles are nearly identical (the early-exercise premium is negligible), so the default is European for speed. The styles diverge for cases like deep-ITM long-dated puts, where the early-exercise premium is material and the American path matters.

### Accuracy of the American Greeks

American Greeks have no closed form, so they are approximated. Delta and Gamma come straight from the binomial lattice (the recombining step-2 nodes) and are smooth and accurate. Vega, Theta, and Rho are computed by re-pricing bumps, which on a binomial tree suffer "sawtooth" noise: the early-exercise boundary jumps between discrete nodes as the tree refines, so a single-resolution bump is only accurate to roughly 1%. Two standard corrections are applied together to fix this: a control variate (compute each Greek by the same bump on the European tree, measure that tree's error against the exact closed-form BSM Greek, and subtract it — the shared discretisation error cancels) and step averaging (average over four adjacent step counts, whose sawtooth phases differ and so cancel). After both corrections, Vega and Theta agree with a high-resolution independent finite-difference reference to better than 0.5% across in-, at-, and out-of-the-money cases, and a non-dividend American call (which must equal European by theorem) matches closed-form BSM to within 0.02% on every Greek. This is validated in the test suite.

## Why analytic Greeks, validated numerically

All Greeks are exact partial derivatives of the BSM formula — no finite differencing in the pricing path, so no numerical noise. To guard against formula regressions, the test suite re-derives every Greek by central finite difference and checks it against the analytic value on every CI run. This automates a manual verification that originally caught six sign/scaling errors in the UI explainers.

## Why an expected-utility recommender (and no weights)

A recommender that suggests financial products to retail users must be explainable — to the user and to a compliance function. An earlier version scored each strategy on three axes (direction, volatility, risk) and blended them with hand-picked weights. Those weights had no principled basis: they encoded a preference, not a measurable fact, and no amount of UI (sliders, dials) removes that — it only relocates the arbitrariness.

The current recommender removes the weights entirely. It ranks strategies by the expected utility of their payoff distribution under the user's view:

- Direction and volatility are captured by a **view-implied lognormal** terminal-price distribution: a bullish/bearish view sets a drift of ±0.5·σ·√T, and an up/down vol view sets the realised-vol spread to σ×1.3 or σ×0.7. Strategies are still *priced* at market implied vol; the view only shapes the distribution their payoff is *evaluated against*.
- Risk is captured by the curvature of a **CRRA utility function**, with the risk-aversion coefficient derived from the user's Capacity-for-Loss score — reusing the suitability evidence already collected in onboarding. High CFL → low risk aversion; low CFL → strong downside aversion.

There are no direction/vol/risk weights to tune. The two remaining inputs (the drift/spread magnitudes and the CRRA form) are explicit, documented belief assumptions rather than hidden coefficients, and the magnitudes are adjustable. This is the honest distinction: you cannot build a recommender with zero assumptions, because recommendation is a value judgement — but you can move the judgement from arbitrary blending weights to a stated utility function and a stated view, which is defensible. Each recommendation reports its certainty-equivalent gain, expected P&L, and probability of profit under the view.

Full calibration would still be possible — preference learning from expert-labelled rankings, outcome-based backtesting, or eliciting the risk-aversion parameter more precisely — but those need data the prototype does not have.

## Why defined-risk only

The catalogue excludes naked short options. Every strategy has a bounded maximum loss. This is a deliberate scope choice for a retail-facing educational tool and keeps the margin model tractable. A higher approval tier with naked or ratio structures would be the natural extension, with a correspondingly stricter suitability gate.
