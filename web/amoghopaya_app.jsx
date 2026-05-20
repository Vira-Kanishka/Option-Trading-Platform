import React, { useState, useMemo, useEffect } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Area, AreaChart } from "recharts";

// =============================================================================
// BSM engine (JS port of engine/bsm.py — kept compact and self-contained)
// =============================================================================

// Abramowitz & Stegun 7.1.26 approximation for erf
function erf(x) {
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x);
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
  const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const t = 1.0 / (1.0 + p * x);
  const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  return sign * y;
}
const N = x => 0.5 * (1 + erf(x / Math.SQRT2));
const phi = x => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);

function d1d2(S, K, T, r, q, sigma) {
  T = Math.max(T, 1e-8); sigma = Math.max(sigma, 1e-8);
  const sqrtT = Math.sqrt(T);
  const d1 = (Math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
  return [d1, d1 - sigma * sqrtT];
}

function bsPrice(S, K, T, r, q, sigma, type) {
  if (T <= 0) return type === "call" ? Math.max(S - K, 0) : Math.max(K - S, 0);
  const [d1, d2] = d1d2(S, K, T, r, q, sigma);
  const dR = Math.exp(-r * T), dQ = Math.exp(-q * T);
  return type === "call"
    ? S * dQ * N(d1) - K * dR * N(d2)
    : K * dR * N(-d2) - S * dQ * N(-d1);
}

function bsGreeks(S, K, T, r, q, sigma, type) {
  if (T <= 0) {
    const intrinsic = type === "call" ? Math.max(S - K, 0) : Math.max(K - S, 0);
    const delta = type === "call" ? (S > K ? 1 : 0) : (S < K ? -1 : 0);
    return { price: intrinsic, delta, gamma: 0, vega: 0, theta: 0, rho: 0 };
  }
  T = Math.max(T, 1e-8); sigma = Math.max(sigma, 1e-8);
  const [d1, d2] = d1d2(S, K, T, r, q, sigma);
  const sqrtT = Math.sqrt(T), dR = Math.exp(-r * T), dQ = Math.exp(-q * T), pd1 = phi(d1);
  let price, delta, theta, rho;
  if (type === "call") {
    price = S * dQ * N(d1) - K * dR * N(d2);
    delta = dQ * N(d1);
    theta = -S * dQ * pd1 * sigma / (2 * sqrtT) - r * K * dR * N(d2) + q * S * dQ * N(d1);
    rho = K * T * dR * N(d2);
  } else {
    price = K * dR * N(-d2) - S * dQ * N(-d1);
    delta = -dQ * N(-d1);
    theta = -S * dQ * pd1 * sigma / (2 * sqrtT) + r * K * dR * N(-d2) - q * S * dQ * N(-d1);
    rho = -K * T * dR * N(-d2);
  }
  const gamma = dQ * pd1 / (S * sigma * sqrtT);
  const vega = S * dQ * pd1 * sqrtT;
  return { price, delta, gamma, vega, theta, rho };
}

// ===========================================================================
// American option pricing — CRR binomial tree (mirrors engine/american.py)
// ===========================================================================
// US single-stock/ETF options are American-style. Early exercise is only optimal
// in narrow cases (deep-ITM puts at high rates; calls before a large dividend),
// but the platform supports a full American path so prices, Greeks, and stress
// tests reflect the chosen exercise style.

function binomialPrice(S, K, T, r, q, sigma, type, steps, american) {
  if (T <= 0) return type === "call" ? Math.max(S - K, 0) : Math.max(K - S, 0);
  sigma = Math.max(sigma, 1e-8);
  const dt = T / steps;
  const u = Math.exp(sigma * Math.sqrt(dt));
  const d = 1 / u;
  const a = Math.exp((r - q) * dt);
  let p = (a - d) / (u - d);
  p = Math.min(1, Math.max(0, p));
  const disc = Math.exp(-r * dt);
  const values = new Array(steps + 1);
  for (let j = 0; j <= steps; j++) {
    const asset = S * Math.pow(u, j) * Math.pow(d, steps - j);
    values[j] = type === "call" ? Math.max(asset - K, 0) : Math.max(K - asset, 0);
  }
  for (let step = steps - 1; step >= 0; step--) {
    for (let j = 0; j <= step; j++) {
      const v = disc * (p * values[j + 1] + (1 - p) * values[j]);
      if (american) {
        const asset = S * Math.pow(u, j) * Math.pow(d, step - j);
        const intrinsic = type === "call" ? Math.max(asset - K, 0) : Math.max(K - asset, 0);
        values[j] = Math.max(v, intrinsic);
      } else {
        values[j] = v;
      }
    }
  }
  return values[0];
}

function americanPrice(S, K, T, r, q, sigma, type, steps = 300) {
  if (T <= 0) return type === "call" ? Math.max(S - K, 0) : Math.max(K - S, 0);
  sigma = Math.max(sigma, 1e-8);
  const dt = T / steps;
  const u = Math.exp(sigma * Math.sqrt(dt));
  const d = 1 / u;
  const a = Math.exp((r - q) * dt);
  let p = (a - d) / (u - d);
  p = Math.min(1, Math.max(0, p));
  const disc = Math.exp(-r * dt);

  // Terminal payoffs
  const values = new Array(steps + 1);
  for (let j = 0; j <= steps; j++) {
    const asset = S * Math.pow(u, j) * Math.pow(d, steps - j);
    values[j] = type === "call" ? Math.max(asset - K, 0) : Math.max(K - asset, 0);
  }
  // Backward induction with early-exercise check
  for (let step = steps - 1; step >= 0; step--) {
    for (let j = 0; j <= step; j++) {
      let v = disc * (p * values[j + 1] + (1 - p) * values[j]);
      const asset = S * Math.pow(u, j) * Math.pow(d, step - j);
      const intrinsic = type === "call" ? Math.max(asset - K, 0) : Math.max(K - asset, 0);
      values[j] = Math.max(v, intrinsic);
    }
  }
  return values[0];
}

// American Greeks: Delta/Gamma from the lattice (step-2 recombining nodes),
// Vega/Theta/Rho from small bumps. Mirrors engine/american.american_greeks.
function americanGreeks(S, K, T, r, q, sigma, type, steps = 300) {
  if (T <= 0) {
    const intrinsic = type === "call" ? Math.max(S - K, 0) : Math.max(K - S, 0);
    const delta = type === "call" ? (S > K ? 1 : 0) : (S < K ? -1 : 0);
    return { price: intrinsic, delta, gamma: 0, vega: 0, theta: 0, rho: 0 };
  }
  sigma = Math.max(sigma, 1e-8);
  const dt = T / steps;
  const u = Math.exp(sigma * Math.sqrt(dt));
  const d = 1 / u;
  const a = Math.exp((r - q) * dt);
  let p = (a - d) / (u - d);
  p = Math.min(1, Math.max(0, p));
  const disc = Math.exp(-r * dt);

  const values = new Array(steps + 1);
  for (let j = 0; j <= steps; j++) {
    const asset = S * Math.pow(u, j) * Math.pow(d, steps - j);
    values[j] = type === "call" ? Math.max(asset - K, 0) : Math.max(K - asset, 0);
  }
  let cap2 = null;
  for (let step = steps - 1; step >= 0; step--) {
    for (let j = 0; j <= step; j++) {
      const v = disc * (p * values[j + 1] + (1 - p) * values[j]);
      const asset = S * Math.pow(u, j) * Math.pow(d, step - j);
      const intrinsic = type === "call" ? Math.max(asset - K, 0) : Math.max(K - asset, 0);
      values[j] = Math.max(v, intrinsic);
    }
    if (step === 2) {
      cap2 = {
        sDn: S * d * d, sMid: S, sUp: S * u * u,
        vDn: values[0], vMid: values[1], vUp: values[2],
      };
    }
  }
  const price = values[0];

  let delta, gamma;
  if (cap2) {
    delta = (cap2.vUp - cap2.vDn) / (cap2.sUp - cap2.sDn);
    gamma = 2 * (
      (cap2.vUp - cap2.vMid) / (cap2.sUp - cap2.sMid)
      - (cap2.vMid - cap2.vDn) / (cap2.sMid - cap2.sDn)
    ) / (cap2.sUp - cap2.sDn);
  } else {
    const h = Math.max(1e-4 * S, 1e-6);
    const vu = americanPrice(S + h, K, T, r, q, sigma, type, steps);
    const vd = americanPrice(S - h, K, T, r, q, sigma, type, steps);
    delta = (vu - vd) / (2 * h);
    gamma = (vu - 2 * price + vd) / (h * h);
  }

  // Vega/Theta/Rho via bumps, control-variate corrected and step-averaged.
  // (Mirrors engine/american.american_greeks.) Cancels the binomial sawtooth by
  // subtracting the European tree's bump error vs exact BSM, averaged over a few
  // adjacent step counts. Delta/Gamma above come from the lattice and are exact.
  const euExact = bsGreeks(S, K, T, r, q, sigma, type);
  const tree = (s, k, t, rr, qq, v, am, n) => binomialPrice(s, k, t, rr, qq, v, type, n, am);
  const stepSet = [steps, steps + 1, steps + 2, steps + 3];
  const hSig = 1e-3, hR = 1e-4;
  const hT = T > 0 ? Math.min(1e-3, T / 10) : 1e-3;
  let vegaAcc = 0, thetaAcc = 0, rhoAcc = 0;
  for (const n of stepSet) {
    const amV = (tree(S, K, T, r, q, sigma + hSig, true, n)
               - tree(S, K, T, r, q, Math.max(sigma - hSig, 1e-6), true, n)) / (2 * hSig);
    const euV = (tree(S, K, T, r, q, sigma + hSig, false, n)
               - tree(S, K, T, r, q, Math.max(sigma - hSig, 1e-6), false, n)) / (2 * hSig);
    vegaAcc += amV - (euV - euExact.vega);

    const amP = tree(S, K, T, r, q, sigma, true, n);
    const euP = tree(S, K, T, r, q, sigma, false, n);
    const amTh = (tree(S, K, Math.max(T - hT, 1e-8), r, q, sigma, true, n) - amP) / hT;
    const euTh = (tree(S, K, Math.max(T - hT, 1e-8), r, q, sigma, false, n) - euP) / hT;
    thetaAcc += amTh - (euTh - euExact.theta);

    const amR = (tree(S, K, T, r + hR, q, sigma, true, n)
               - tree(S, K, T, r - hR, q, sigma, true, n)) / (2 * hR);
    const euR = (tree(S, K, T, r + hR, q, sigma, false, n)
               - tree(S, K, T, r - hR, q, sigma, false, n)) / (2 * hR);
    rhoAcc += amR - (euR - euExact.rho);
  }
  const m = stepSet.length;
  const vega = vegaAcc / m, theta = thetaAcc / m, rho = rhoAcc / m;

  return { price, delta, gamma, vega, theta, rho };
}

// Unified dispatch: price/greeks by exercise style. Keeps call sites clean.
function optionPrice(S, K, T, r, q, sigma, type, style = "european") {
  return style === "american"
    ? americanPrice(S, K, T, r, q, sigma, type)
    : bsPrice(S, K, T, r, q, sigma, type);
}
function optionGreeks(S, K, T, r, q, sigma, type, style = "european") {
  return style === "american"
    ? americanGreeks(S, K, T, r, q, sigma, type)
    : bsGreeks(S, K, T, r, q, sigma, type);
}

// Historical (realised) volatility from a price series.
// Uses sample std of log returns; annualises by sqrt(252).
// Presented in the UI alongside IV as a reference for whether vol is rich/cheap.
function historicalVol(prices, window = 30) {
  if (prices.length < window + 1) return null;
  const recent = prices.slice(-(window + 1));
  const logReturns = [];
  for (let i = 1; i < recent.length; i++) {
    logReturns.push(Math.log(recent[i] / recent[i - 1]));
  }
  const mean = logReturns.reduce((s, x) => s + x, 0) / logReturns.length;
  const variance = logReturns.reduce((s, x) => s + (x - mean) ** 2, 0) / (logReturns.length - 1);
  return Math.sqrt(variance) * Math.sqrt(252);
}

// Mulberry32 — tiny deterministic PRNG. Used to generate a plausible
// synthetic price history for HV display when no real market data is present.
function seededRandom(seed) {
  let a = seed >>> 0;
  return function() {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Generate a synthetic GBM path seeded from ticker+sigma, for HV display.
// The generated HV will be close to but distinct from the supplied IV
// (demonstrating the IV/HV distinction without pretending to have real data).
function syntheticPriceHistory(ticker, currentSpot, sigmaSeed, nDays = 90) {
  const seed = ticker.split("").reduce((s, c) => s + c.charCodeAt(0), 0) * 1000
              + Math.round(sigmaSeed * 10000);
  const rng = seededRandom(seed);
  const targetHV = sigmaSeed * (0.85 + 0.30 * rng());  // HV slightly different from IV
  const daily = targetHV / Math.sqrt(252);
  const prices = [];
  let p = currentSpot / Math.exp(rng() * 0.02);  // small offset from today
  for (let i = 0; i < nDays; i++) {
    // Box-Muller for normal sample
    const u1 = Math.max(rng(), 1e-10), u2 = rng();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    p = p * Math.exp(-0.5 * daily * daily + daily * z);
    prices.push(p);
  }
  return prices;
}

// =============================================================================
// Strategy definitions (JS mirror of engine/strategies.py)
// =============================================================================

const STRAT_DEFS = {
  covered_call: {
    label: "Covered Call",
    tagline: "Income on held stock",
    risk: "low", approvalLevel: 1,
    direction: "neutral-to-bullish", vol: "short",
    describe: (S, T, p) => `+1 underlying; −1 call @ ${p.K_call}`,
    defaultParams: S => ({ K_call: Math.round(S * 1.05) }),
    legs: (S, T, p) => [
      { kind: "underlying", side: 1, qty: 1 },
      { kind: "call", side: -1, qty: 1, K: p.K_call, T },
    ],
    maxProfit: "Capped at K_call − S₀ + premium",
    maxLoss: "Stock down to zero, less premium received",
  },
  protective_put: {
    label: "Protective Put",
    tagline: "Bullish, with downside floor",
    risk: "low", approvalLevel: 1,
    direction: "bullish", vol: "long",
    describe: (S, T, p) => `+1 underlying; +1 put @ ${p.K_put}`,
    defaultParams: S => ({ K_put: Math.round(S * 0.95) }),
    legs: (S, T, p) => [
      { kind: "underlying", side: 1, qty: 1 },
      { kind: "put", side: 1, qty: 1, K: p.K_put, T },
    ],
    maxProfit: "Unbounded to the upside",
    maxLoss: "S₀ − K_put + premium",
  },
  bull_call_spread: {
    label: "Bull Call Spread",
    tagline: "Defined-risk bullish (debit)",
    risk: "medium", approvalLevel: 3,
    direction: "bullish", vol: "neutral",
    describe: (S, T, p) => `+1 call @ ${p.K_lo}; −1 call @ ${p.K_hi}`,
    defaultParams: S => ({ K_lo: Math.round(S * 0.98), K_hi: Math.round(S * 1.05) }),
    legs: (S, T, p) => [
      { kind: "call", side: 1, qty: 1, K: p.K_lo, T },
      { kind: "call", side: -1, qty: 1, K: p.K_hi, T },
    ],
    maxProfit: "Width − net debit",
    maxLoss: "Net debit paid",
  },
  bear_put_spread: {
    label: "Bear Put Spread",
    tagline: "Defined-risk bearish (debit)",
    risk: "medium", approvalLevel: 3,
    direction: "bearish", vol: "neutral",
    describe: (S, T, p) => `+1 put @ ${p.K_hi}; −1 put @ ${p.K_lo}`,
    defaultParams: S => ({ K_hi: Math.round(S * 1.02), K_lo: Math.round(S * 0.95) }),
    legs: (S, T, p) => [
      { kind: "put", side: 1, qty: 1, K: p.K_hi, T },
      { kind: "put", side: -1, qty: 1, K: p.K_lo, T },
    ],
    maxProfit: "Width − net debit",
    maxLoss: "Net debit paid",
  },
  bull_put_spread: {
    label: "Bull Put Spread",
    tagline: "Collect premium above K_hi",
    risk: "medium", approvalLevel: 3,
    direction: "bullish", vol: "short",
    describe: (S, T, p) => `−1 put @ ${p.K_hi}; +1 put @ ${p.K_lo}`,
    defaultParams: S => ({ K_hi: Math.round(S * 0.98), K_lo: Math.round(S * 0.92) }),
    legs: (S, T, p) => [
      { kind: "put", side: -1, qty: 1, K: p.K_hi, T },
      { kind: "put", side: 1, qty: 1, K: p.K_lo, T },
    ],
    maxProfit: "Net credit received",
    maxLoss: "Width − net credit",
  },
  bear_call_spread: {
    label: "Bear Call Spread",
    tagline: "Collect premium below K_lo",
    risk: "medium", approvalLevel: 3,
    direction: "bearish", vol: "short",
    describe: (S, T, p) => `−1 call @ ${p.K_lo}; +1 call @ ${p.K_hi}`,
    defaultParams: S => ({ K_lo: Math.round(S * 1.02), K_hi: Math.round(S * 1.08) }),
    legs: (S, T, p) => [
      { kind: "call", side: -1, qty: 1, K: p.K_lo, T },
      { kind: "call", side: 1, qty: 1, K: p.K_hi, T },
    ],
    maxProfit: "Net credit received",
    maxLoss: "Width − net credit",
  },
  straddle: {
    label: "Long Straddle",
    tagline: "Direction-agnostic, long vol",
    risk: "high", approvalLevel: 2,
    direction: "neutral", vol: "long",
    describe: (S, T, p) => `+1 call @ ${p.K}; +1 put @ ${p.K}`,
    defaultParams: S => ({ K: Math.round(S) }),
    legs: (S, T, p) => [
      { kind: "call", side: 1, qty: 1, K: p.K, T },
      { kind: "put", side: 1, qty: 1, K: p.K, T },
    ],
    maxProfit: "Unbounded on large move",
    maxLoss: "Total debit paid",
  },
  strangle: {
    label: "Long Strangle",
    tagline: "Cheaper vol play than straddle",
    risk: "high", approvalLevel: 2,
    direction: "neutral", vol: "long",
    describe: (S, T, p) => `+1 put @ ${p.K_put}; +1 call @ ${p.K_call}`,
    defaultParams: S => ({ K_put: Math.round(S * 0.95), K_call: Math.round(S * 1.05) }),
    legs: (S, T, p) => [
      { kind: "put", side: 1, qty: 1, K: p.K_put, T },
      { kind: "call", side: 1, qty: 1, K: p.K_call, T },
    ],
    maxProfit: "Unbounded on large move",
    maxLoss: "Total debit paid",
  },
  butterfly: {
    label: "Long Butterfly",
    tagline: "Pin at K_mid, short vol",
    risk: "medium", approvalLevel: 3,
    direction: "neutral", vol: "short",
    describe: (S, T, p) => `+1 @ ${p.K_lo}; −2 @ ${p.K_mid}; +1 @ ${p.K_hi}`,
    defaultParams: S => ({ K_lo: Math.round(S * 0.95), K_mid: Math.round(S), K_hi: Math.round(S * 1.05) }),
    legs: (S, T, p) => [
      { kind: "call", side: 1, qty: 1, K: p.K_lo, T },
      { kind: "call", side: -1, qty: 2, K: p.K_mid, T },
      { kind: "call", side: 1, qty: 1, K: p.K_hi, T },
    ],
    maxProfit: "Width/2 − net debit",
    maxLoss: "Net debit paid",
  },
  collar: {
    label: "Collar",
    tagline: "Bounded bullish",
    risk: "low", approvalLevel: 3,
    direction: "bullish", vol: "neutral",
    describe: (S, T, p) => `+1 underlying; +1 put @ ${p.K_put}; −1 call @ ${p.K_call}`,
    defaultParams: S => ({ K_put: Math.round(S * 0.95), K_call: Math.round(S * 1.05) }),
    legs: (S, T, p) => [
      { kind: "underlying", side: 1, qty: 1 },
      { kind: "put", side: 1, qty: 1, K: p.K_put, T },
      { kind: "call", side: -1, qty: 1, K: p.K_call, T },
    ],
    maxProfit: "K_call − S₀ + net credit",
    maxLoss: "S₀ − K_put + net debit",
  },
  // ---- Wave 1 additions (academic completeness, single-expiry, defined-risk) ----
  long_call: {
    label: "Long Call",
    tagline: "Leveraged bullish, unlimited upside",
    risk: "medium", approvalLevel: 2,
    direction: "bullish", vol: "long",
    describe: (S, T, p) => `+1 call @ ${p.K}`,
    defaultParams: S => ({ K: Math.round(S) }),
    legs: (S, T, p) => [
      { kind: "call", side: 1, qty: 1, K: p.K, T },
    ],
    maxProfit: "Unbounded as underlying rises",
    maxLoss: "Premium paid (debit)",
  },
  long_put: {
    label: "Long Put",
    tagline: "Leveraged bearish, large downside gains",
    risk: "medium", approvalLevel: 2,
    direction: "bearish", vol: "long",
    describe: (S, T, p) => `+1 put @ ${p.K}`,
    defaultParams: S => ({ K: Math.round(S) }),
    legs: (S, T, p) => [
      { kind: "put", side: 1, qty: 1, K: p.K, T },
    ],
    maxProfit: "K − premium (underlying to zero)",
    maxLoss: "Premium paid (debit)",
  },
  cash_secured_put: {
    label: "Cash-Secured Put",
    tagline: "Income, cash-backed",
    risk: "low", approvalLevel: 1,
    direction: "bullish", vol: "short",
    describe: (S, T, p) => `−1 put @ ${p.K}  (cash-backed)`,
    defaultParams: S => ({ K: Math.round(S * 0.95) }),
    legs: (S, T, p) => [
      { kind: "put", side: -1, qty: 1, K: p.K, T },
    ],
    maxProfit: "Premium received (credit)",
    maxLoss: "K − premium (if assigned and stock to zero)",
  },
  put_butterfly: {
    label: "Long Put Butterfly",
    tagline: "Pin at K_mid (put construction)",
    risk: "medium", approvalLevel: 3,
    direction: "neutral", vol: "short",
    describe: (S, T, p) => `+1 put @ ${p.K_hi}; −2 puts @ ${p.K_mid}; +1 put @ ${p.K_lo}`,
    defaultParams: S => ({ K_lo: Math.round(S * 0.95), K_mid: Math.round(S), K_hi: Math.round(S * 1.05) }),
    legs: (S, T, p) => [
      { kind: "put", side: 1, qty: 1, K: p.K_hi, T },
      { kind: "put", side: -1, qty: 2, K: p.K_mid, T },
      { kind: "put", side: 1, qty: 1, K: p.K_lo, T },
    ],
    maxProfit: "Width/2 − net debit",
    maxLoss: "Net debit paid",
  },
  iron_butterfly: {
    label: "Iron Butterfly",
    tagline: "Short-vol credit, pin at K_mid",
    risk: "medium", approvalLevel: 3,
    direction: "neutral", vol: "short",
    describe: (S, T, p) => `+1 put @ ${p.K_lo}; −1 put @ ${p.K_mid}; −1 call @ ${p.K_mid}; +1 call @ ${p.K_hi}`,
    defaultParams: S => ({ K_lo: Math.round(S * 0.95), K_mid: Math.round(S), K_hi: Math.round(S * 1.05) }),
    legs: (S, T, p) => [
      { kind: "put", side: 1, qty: 1, K: p.K_lo, T },
      { kind: "put", side: -1, qty: 1, K: p.K_mid, T },
      { kind: "call", side: -1, qty: 1, K: p.K_mid, T },
      { kind: "call", side: 1, qty: 1, K: p.K_hi, T },
    ],
    maxProfit: "Net credit received",
    maxLoss: "Wing width − net credit",
  },
  iron_condor: {
    label: "Iron Condor",
    tagline: "Range-bound income, short vol",
    risk: "medium", approvalLevel: 3,
    direction: "neutral", vol: "short",
    describe: (S, T, p) => `+1 put @ ${p.K_put_long}; −1 put @ ${p.K_put_short}; −1 call @ ${p.K_call_short}; +1 call @ ${p.K_call_long}`,
    defaultParams: S => ({
      K_put_long: Math.round(S * 0.92),
      K_put_short: Math.round(S * 0.97),
      K_call_short: Math.round(S * 1.03),
      K_call_long: Math.round(S * 1.08),
    }),
    legs: (S, T, p) => [
      { kind: "put", side: 1, qty: 1, K: p.K_put_long, T },
      { kind: "put", side: -1, qty: 1, K: p.K_put_short, T },
      { kind: "call", side: -1, qty: 1, K: p.K_call_short, T },
      { kind: "call", side: 1, qty: 1, K: p.K_call_long, T },
    ],
    maxProfit: "Net credit received",
    maxLoss: "max(put width, call width) − net credit",
  },
  // ---- Wave 2 additions (butterfly family completeness) ----
  short_call_butterfly: {
    label: "Short Call Butterfly",
    tagline: "Breakout from K_mid, long vol, credit",
    risk: "medium", approvalLevel: 3,
    direction: "neutral", vol: "long",
    describe: (S, T, p) => `−1 call @ ${p.K_lo}; +2 calls @ ${p.K_mid}; −1 call @ ${p.K_hi}`,
    defaultParams: S => ({ K_lo: Math.round(S * 0.95), K_mid: Math.round(S), K_hi: Math.round(S * 1.05) }),
    legs: (S, T, p) => [
      { kind: "call", side: -1, qty: 1, K: p.K_lo, T },
      { kind: "call", side: 1, qty: 2, K: p.K_mid, T },
      { kind: "call", side: -1, qty: 1, K: p.K_hi, T },
    ],
    maxProfit: "Net credit received",
    maxLoss: "Wing width − net credit (at K_mid)",
  },
  short_put_butterfly: {
    label: "Short Put Butterfly",
    tagline: "Breakout from K_mid (put construction)",
    risk: "medium", approvalLevel: 3,
    direction: "neutral", vol: "long",
    describe: (S, T, p) => `−1 put @ ${p.K_hi}; +2 puts @ ${p.K_mid}; −1 put @ ${p.K_lo}`,
    defaultParams: S => ({ K_lo: Math.round(S * 0.95), K_mid: Math.round(S), K_hi: Math.round(S * 1.05) }),
    legs: (S, T, p) => [
      { kind: "put", side: -1, qty: 1, K: p.K_hi, T },
      { kind: "put", side: 1, qty: 2, K: p.K_mid, T },
      { kind: "put", side: -1, qty: 1, K: p.K_lo, T },
    ],
    maxProfit: "Net credit received",
    maxLoss: "Wing width − net credit (at K_mid)",
  },
  reverse_iron_butterfly: {
    label: "Reverse Iron Butterfly",
    tagline: "Long-vol breakout, defined-risk debit",
    risk: "medium", approvalLevel: 3,
    direction: "neutral", vol: "long",
    describe: (S, T, p) => `−1 put @ ${p.K_lo}; +1 put @ ${p.K_mid}; +1 call @ ${p.K_mid}; −1 call @ ${p.K_hi}`,
    defaultParams: S => ({ K_lo: Math.round(S * 0.95), K_mid: Math.round(S), K_hi: Math.round(S * 1.05) }),
    legs: (S, T, p) => [
      { kind: "put", side: -1, qty: 1, K: p.K_lo, T },
      { kind: "put", side: 1, qty: 1, K: p.K_mid, T },
      { kind: "call", side: 1, qty: 1, K: p.K_mid, T },
      { kind: "call", side: -1, qty: 1, K: p.K_hi, T },
    ],
    maxProfit: "Wing width − net debit",
    maxLoss: "Net debit paid (at K_mid)",
  },
  broken_wing_butterfly: {
    label: "Broken-Wing Butterfly",
    tagline: "Asymmetric wings, directional skew",
    risk: "medium", approvalLevel: 3,
    direction: "neutral", vol: "short",
    describe: (S, T, p) => `+1 call @ ${p.K_lo}; −2 calls @ ${p.K_mid}; +1 call @ ${p.K_hi}`,
    // Default: bullish skew, wider upper wing
    defaultParams: S => ({ K_lo: Math.round(S * 0.97), K_mid: Math.round(S), K_hi: Math.round(S * 1.10) }),
    legs: (S, T, p) => [
      { kind: "call", side: 1, qty: 1, K: p.K_lo, T },
      { kind: "call", side: -1, qty: 2, K: p.K_mid, T },
      { kind: "call", side: 1, qty: 1, K: p.K_hi, T },
    ],
    maxProfit: "Width of narrower wing − net debit (at K_mid)",
    maxLoss: "Max(|net debit|, wider-wing − narrower-wing − credit); bounded.",
  },
};

// =============================================================================
// Strategy explainers — pedagogical content for "Learn more" modal
// =============================================================================
// Each explainer carries: whenToUse (market view + reasoning), greekSignature
// (plain-English Greek behaviour), commonMistake (most common retail error),
// and payoffShape (SVG path for a small iconic payoff sketch).
// Content is deliberately terse: ~2 sentences per field so the modal fits a
// single viewport without scrolling.

const STRAT_EXPLAINERS = {
  covered_call: {
    whenToUse: "Expresses a neutral-to-mildly-bullish view on a held underlying. The short call monetises time value against capped upside; the position is equivalent to a short put at the same strike by put-call parity.",
    greekSignature: "Net delta is positive but less than the underlying alone; gamma and vega are negative (short call dominates); theta is positive. Greek magnitudes peak when the short call sits near the money and decay as spot moves below the strike.",
    commonMistake: "Risk-reward is identical to a cash-secured put at the same strike. Preference between the two is driven by existing stock position, dividend capture, and tax lot considerations rather than payoff.",
    payoffShape: "hockey_down",
  },
  protective_put: {
    whenToUse: "Long underlying combined with a long OTM put to floor the downside at the put strike. The structure converts unbounded linear exposure into capped downside at the cost of the put premium.",
    greekSignature: "Net delta is positive but less than unity (the put carries negative delta). Gamma and vega are positive, concentrated near the put strike. Theta is negative; the hedge carries carry cost.",
    commonMistake: "Equivalent to a long call at the put strike plus cash, by put-call parity. Rolling the put forward rather than letting it expire is the standard approach to maintain continuous protection.",
    payoffShape: "hockey_up",
  },
  bull_call_spread: {
    whenToUse: "Debit structure expressing a bounded bullish view between two strikes. Lower cost than an outright long call in exchange for a capped payoff above the short strike.",
    greekSignature: "Net delta is positive, peaking between the strikes. Gamma flips sign across the short strike: positive below, negative above. Vega is small; its sign tracks which leg is closer to the money (slightly positive when spot sits below the midpoint, negative when above).",
    commonMistake: "Spread width, leg deltas, and time to expiry jointly determine the risk-reward ratio. At 30-60 DTE, a one-standard-deviation-wide spread is a common construction balancing probability of max profit against premium paid.",
    payoffShape: "stairs_up",
  },
  bear_put_spread: {
    whenToUse: "Debit structure expressing a bounded bearish view. Mirror of the bull call spread with long puts replacing long calls.",
    greekSignature: "Net delta negative, bounded in magnitude. Gamma flips sign across the short strike. Vega sign tracks which leg sits closer to the money: slightly positive when spot is above the midpoint, negative when below.",
    commonMistake: "For symmetric strikes, payoff is identical to a bull put credit spread at the same strikes by put-call parity. Choice between the two reduces to a financing and margin question.",
    payoffShape: "stairs_down",
  },
  bull_put_spread: {
    whenToUse: "Credit structure expressing a bullish-to-neutral view. Maximum profit is the credit received; maximum loss is the strike width minus the credit.",
    greekSignature: "Positive delta, small in magnitude. Negative vega and positive theta; the short put dominates the combined vega. Gamma is negative between strikes.",
    commonMistake: "Max loss / max profit ratio is (width − credit) / credit. At typical short-leg deltas around 0.30, the ratio is unfavourable on a per-trade basis; the structure is premised on frequency of success rather than per-trade expectancy.",
    payoffShape: "stairs_up",
  },
  bear_call_spread: {
    whenToUse: "Credit structure expressing a bearish-to-neutral view. Mirror of the bull put spread.",
    greekSignature: "Negative delta, small. Negative vega, positive theta. Negative gamma between strikes.",
    commonMistake: "Short call assignment risk rises sharply near ex-dividend dates for dividend-paying underlyings. Holders of deep-ITM short calls face early exercise if the extrinsic value falls below the dividend amount.",
    payoffShape: "stairs_down",
  },
  straddle: {
    whenToUse: "Long call and long put at the same strike. Expresses a long-volatility, direction-agnostic view. Profit requires the underlying to move beyond the combined premium paid in either direction.",
    greekSignature: "Delta near zero at the strike (exactly zero only when r = q; the risk-free carry tilts it positive) and accumulates directionally as spot drifts. Gamma is large and positive, peaked at the strike. Vega is large and positive. Theta is the dominant cost: among the most theta-negative structures per unit premium.",
    commonMistake: "Implied volatility is typically elevated ahead of scheduled events. Post-event IV collapse (vol crush) often exceeds the realised move, producing negative P&L even when the directional thesis is correct.",
    payoffShape: "v_valley",
  },
  strangle: {
    whenToUse: "Long OTM call and long OTM put. Cheaper analogue of the straddle with strikes displaced from the money; requires larger realised move to break even.",
    greekSignature: "Delta near zero between strikes. Gamma is positive and peaked at each strike. Vega positive but smaller per premium unit than a straddle. Theta negative; decays faster as spot approaches expiry.",
    commonMistake: "Break-even distance from spot equals the total premium paid. Strikes selected at approximately ±1 standard deviation (≈16 delta) balance cost against probability of profit; deeper OTM construction cheapens the premium but pushes break-evens beyond historically realised ranges.",
    payoffShape: "v_wide",
  },
  butterfly: {
    whenToUse: "Three-strike debit structure expressing a pin thesis at the middle strike. Long wings at K_lo and K_hi, short two contracts at K_mid. Maximum payoff occurs if the underlying expires at K_mid.",
    greekSignature: "Delta near zero at K_mid. Gamma is negative and concentrated at K_mid (short gamma near the pin). Vega is negative. Theta is positive and accelerates in the final two weeks to expiry.",
    commonMistake: "Structure is economically equivalent whether constructed with calls or puts at symmetric strikes. At asymmetric (broken-wing) strikes, the two constructions differ by (1 − e^(−rT))·(−K_lo + 2K_mid − K_hi), which is non-zero for r > 0.",
    payoffShape: "tent",
  },
  collar: {
    whenToUse: "Long underlying combined with a long OTM put (floor) and short OTM call (ceiling). Typical structure for long-term stock holdings where outright liquidation is undesirable.",
    greekSignature: "Net delta is positive and bounded, approaching zero at the wings. Vega and theta are small but non-zero, typically tilted toward the short call's sign (vega negative, theta positive) because the call contributes larger magnitude than the equidistant put under equity skew.",
    commonMistake: "Zero-cost collar construction requires selecting a short call strike whose premium equals the long put premium. On symmetric skew this produces asymmetric strike widths around spot: the call strike is typically closer to the money than the put strike.",
    payoffShape: "step_bounded",
  },
  long_call: {
    whenToUse: "Expresses a bullish directional view with bounded downside (the premium paid). Leverage relative to the underlying is a function of strike moneyness and time to expiry.",
    greekSignature: "Delta ranges from zero (deep OTM) to one (deep ITM), passing through 0.5 at approximately ATM; slightly above 0.5 when r > q because of the forward tilt. Gamma is positive and peaked near the strike. Vega is positive and largest for ATM long-dated calls. Theta is negative.",
    commonMistake: "Expected profit at horizon depends on realised drift, realised volatility, and IV evolution. For ATM options, theta magnitude scales as 1/√(T − t), so daily decay accelerates as expiry approaches; the final two weeks are the most adverse period for unprofitable positions.",
    payoffShape: "hockey_up",
  },
  long_put: {
    whenToUse: "Expresses a bearish directional view with bounded downside. Used as a portfolio tail hedge against correlated equity exposure.",
    greekSignature: "Delta from zero to negative one. Gamma positive, peaked near the strike. Vega positive. Theta negative.",
    commonMistake: "When used as a hedge, the notional delta of the put position should match the dollar delta of the underlying exposure, not the share count. This requires scaling by both the put's delta and its contract multiplier.",
    payoffShape: "hockey_down",
  },
  cash_secured_put: {
    whenToUse: "Short put backed by cash sufficient to purchase the underlying at the strike. Combines premium collection with willingness to acquire the underlying at strike − premium.",
    greekSignature: "Positive delta (short put). Negative vega, positive theta. Greeks are most pronounced near the short strike.",
    commonMistake: "Identical P&L and Greek profile to a covered call at the same strike, by put-call parity. Practical difference lies in whether the underlying is already held: cash-secured put if not, covered call if already long stock.",
    payoffShape: "step_capped",
  },
  put_butterfly: {
    whenToUse: "Put-constructed analogue of the long call butterfly. Same strikes, same payoff at expiry for symmetric construction.",
    greekSignature: "Identical Greek signature to call butterfly at symmetric strikes. Delta near zero at K_mid; negative gamma, negative vega, positive theta.",
    commonMistake: "Choice between put and call construction reduces to execution: select the side with tighter bid-ask spreads at the chosen strikes. Typically immaterial on liquid index ETFs; material on individual names with skewed liquidity.",
    payoffShape: "tent",
  },
  iron_butterfly: {
    whenToUse: "Four-leg credit structure: short ATM straddle bracketed by long OTM wings. Capped both sides; maximum profit at K_mid equal to net credit received.",
    greekSignature: "Delta near zero at K_mid. Large negative gamma at K_mid (short gamma dominates). Negative vega. Positive theta. Max loss is wing width minus credit.",
    commonMistake: "Delta-neutral at entry but accumulates directional exposure as spot drifts. Gamma scalping or strike adjustment is typical management during the life of the trade on volatile underlyings.",
    payoffShape: "tent_capped",
  },
  iron_condor: {
    whenToUse: "Four-leg credit structure with a flat profit zone between the two short strikes. Short volatility, range-bound thesis.",
    greekSignature: "Delta near zero when centred. Negative gamma inside the short strikes, positive gamma outside the long strikes. Negative vega, positive theta. Max loss is max(put-width, call-width) minus net credit.",
    commonMistake: "Credit-to-max-loss ratio depends on strike selection, not just width. Short-strike deltas around 0.15-0.20 typically produce credits in the range of 1/3 to 1/4 of width, the standard retail construction.",
    payoffShape: "plateau",
  },
  short_call_butterfly: {
    whenToUse: "Credit structure with P&L inverted from the long butterfly. Profits on breakouts away from K_mid in either direction; maximum loss at K_mid.",
    greekSignature: "Delta near zero at K_mid. Positive gamma (valley shape at expiry). Positive vega. Negative theta. Sign-inverted across all Greeks from the long butterfly.",
    commonMistake: "P&L at every underlying price equals the negative of the long butterfly at the same strikes. Margin requirement is the narrower wing minus the credit received.",
    payoffShape: "valley_capped",
  },
  short_put_butterfly: {
    whenToUse: "Put-constructed analogue of the short call butterfly. Same payoff and Greek signature at symmetric strikes.",
    greekSignature: "Identical to short call butterfly at symmetric strikes.",
    commonMistake: "Construction choice is a liquidity question, not a payoff question. Execution preference typically goes to the side with tighter spreads on the short (middle) strike, since that leg carries two contracts.",
    payoffShape: "valley_capped",
  },
  reverse_iron_butterfly: {
    whenToUse: "Debit structure with valley-shaped payoff. Long straddle at K_mid flanked by short wings at K_lo and K_hi. Capped upside, capped downside, long volatility.",
    greekSignature: "Delta near zero at K_mid. Positive gamma, positive vega, negative theta: defined-risk long-volatility profile. Max loss is the net debit paid, realised at K_mid.",
    commonMistake: "Breakeven at each wing equals K_lo + debit and K_hi − debit. For IV-elevated environments, the combined structure becomes expensive and the required realised move to reach max profit exceeds historical ranges.",
    payoffShape: "valley_capped",
  },
  broken_wing_butterfly: {
    whenToUse: "Three-strike structure with asymmetric wings: (K_mid − K_lo) ≠ (K_hi − K_mid). Expresses a pin thesis at K_mid with a directional tilt.",
    greekSignature: "Delta at K_mid is non-zero and carries the opposite sign to the wider wing: a wider upper wing produces bearish delta, a wider lower wing produces bullish delta. Gamma negative at K_mid. Vega negative. Theta positive. Can be structured as a credit if the short middle strike generates sufficient premium.",
    commonMistake: "Call and put constructions differ at expiry by (1 − e^(−rT))·(−K_lo + 2K_mid − K_hi) for r > 0. This reflects the time value of money on the strike sum and is typically cents per share for retail-scale positions.",
    payoffShape: "tent_skewed",
  },
};

// =============================================================================
// Strategy engine — payoff, Greeks, margin, risk flags
// =============================================================================

function legMark(leg, S, r, q, sigma, style = "european") {
  if (leg.kind === "underlying") return leg.side * leg.qty * S;
  return leg.side * leg.qty * optionPrice(S, leg.K, leg.T, r, q, sigma, leg.kind, style);
}

function legGreeks(leg, S, r, q, sigma, style = "european") {
  const sign = leg.side * leg.qty;
  if (leg.kind === "underlying") {
    return { price: sign * S, delta: sign, gamma: 0, vega: 0, theta: 0, rho: 0 };
  }
  const g = optionGreeks(S, leg.K, leg.T, r, q, sigma, leg.kind, style);
  return {
    price: sign * g.price, delta: sign * g.delta, gamma: sign * g.gamma,
    vega: sign * g.vega, theta: sign * g.theta, rho: sign * g.rho,
  };
}

function legPayoff(leg, ST) {
  if (leg.kind === "underlying") return leg.side * leg.qty * ST;
  const intrinsic = leg.kind === "call" ? Math.max(ST - leg.K, 0) : Math.max(leg.K - ST, 0);
  return leg.side * leg.qty * intrinsic;
}

function strategyMark(legs, S, r, q, sigma, style = "european") {
  return legs.reduce((s, l) => s + legMark(l, S, r, q, sigma, style), 0);
}

function strategyGreeks(legs, S, r, q, sigma, style = "european") {
  const tot = { price: 0, delta: 0, gamma: 0, vega: 0, theta: 0, rho: 0 };
  for (const leg of legs) {
    const g = legGreeks(leg, S, r, q, sigma, style);
    for (const k of Object.keys(tot)) tot[k] += g[k];
  }
  return tot;
}

// Early-exercise premium of the whole strategy: the absolute value the American
// right adds over European, summed over option legs. Returns the total premium
// (always >= 0) and a per-leg breakdown. Used to make the Euro/Amer toggle's
// effect legible even when it is small (it is near zero for most short-dated,
// low-dividend structures, and material for deep-ITM puts).
function earlyExercisePremium(legs, S, r, q, sigma) {
  let total = 0;
  for (const leg of legs) {
    if (leg.kind === "underlying") continue;
    const eu = bsPrice(S, leg.K, leg.T, r, q, sigma, leg.kind);
    const am = americanPrice(S, leg.K, leg.T, r, q, sigma, leg.kind);
    total += Math.abs(am - eu) * leg.qty;
  }
  return total;
}

function strategyPayoff(legs, ST, entryPremium) {
  return legs.reduce((s, l) => s + legPayoff(l, ST), 0) - entryPremium;
}

function payoffCurve(legs, S, entryPremium, span = 0.4, n = 121) {
  const lo = S * (1 - span), hi = S * (1 + span);
  const out = [];
  for (let i = 0; i < n; i++) {
    const st = lo + (hi - lo) * i / (n - 1);
    out.push({ S: st, pnl: strategyPayoff(legs, st, entryPremium) });
  }
  return out;
}

function findBreakevens(curve) {
  const bes = [];
  for (let i = 1; i < curve.length; i++) {
    const a = curve[i - 1], b = curve[i];
    if (a.pnl === 0) bes.push(a.S);
    else if ((a.pnl > 0 && b.pnl < 0) || (a.pnl < 0 && b.pnl > 0)) {
      bes.push(a.S - a.pnl * (b.S - a.S) / (b.pnl - a.pnl));
    }
  }
  return bes;
}

function computeMargin(legs, S, r, q, sigma, entryPremium, stratKey) {
  const optionLegs = legs.filter(l => l.kind !== "underlying");
  const hasUnderlying = legs.some(l => l.kind === "underlying");
  if (stratKey === "covered_call") return { amount: 0, note: "Stock collateralises the short call." };
  if (stratKey === "protective_put") {
    const put = optionLegs.find(l => l.kind === "put" && l.side === 1);
    return { amount: legMark(put, S, r, q, sigma), note: "Long put cost = initial margin." };
  }
  if (stratKey === "collar") {
    const put = optionLegs.find(l => l.kind === "put" && l.side === 1);
    const call = optionLegs.find(l => l.kind === "call" && l.side === -1);
    return { amount: Math.max(legMark(put, S, r, q, sigma) + legMark(call, S, r, q, sigma), 0),
             note: "Stock covers short call; net premium shown." };
  }
  // Cash-secured put: collateral = K − premium
  if (stratKey === "cash_secured_put") {
    const put = optionLegs.find(l => l.kind === "put" && l.side === -1);
    const p = Math.abs(legMark(put, S, r, q, sigma));  // abs because short leg sign flips it
    return { amount: put.K - p, note: `Cash collateral = K (${put.K.toFixed(2)}) − credit (${p.toFixed(2)}).` };
  }
  // Iron butterfly / iron condor / reverse iron butterfly: 4-leg mixed call+put structures
  if ((stratKey === "iron_butterfly" || stratKey === "iron_condor" || stratKey === "reverse_iron_butterfly") && optionLegs.length === 4) {
    const calls = optionLegs.filter(l => l.kind === "call");
    const puts = optionLegs.filter(l => l.kind === "put");
    const longCall = calls.find(l => l.side === 1);
    const shortCall = calls.find(l => l.side === -1);
    const longPut = puts.find(l => l.side === 1);
    const shortPut = puts.find(l => l.side === -1);
    const net = legs.reduce((s, l) => s + legMark(l, S, r, q, sigma), 0);
    if (stratKey === "reverse_iron_butterfly") {
      // Net debit: max loss = debit paid
      return {
        amount: Math.max(net, 0),
        note: `Reverse iron butterfly: max loss = net debit (${net.toFixed(2)}).`,
      };
    }
    // Iron butterfly or iron condor (net credit)
    const callWidth = longCall.K - shortCall.K;
    const putWidth = shortPut.K - longPut.K;
    const maxWing = Math.max(callWidth, putWidth);
    const credit = Math.abs(net);
    const label = stratKey === "iron_butterfly" ? "Iron butterfly" : "Iron condor";
    return {
      amount: Math.max(maxWing - credit, 0),
      note: `${label}: max wing ${maxWing.toFixed(2)} − credit ${credit.toFixed(2)}.`,
    };
  }
  // 3-leg butterfly-pattern: long/short call/put butterfly, broken-wing butterfly
  // Pattern: all same kind, outer legs same sign, middle leg opposite sign with 2x qty
  if (optionLegs.length === 3 && !hasUnderlying) {
    const kinds = new Set(optionLegs.map(l => l.kind));
    if (kinds.size === 1) {
      const sorted = [...optionLegs].sort((a, b) => a.K - b.K);
      const [wingLo, middle, wingHi] = sorted;
      const isButterflyShape =
        wingLo.side === wingHi.side &&
        middle.side === -wingLo.side &&
        Math.abs(middle.qty - 2 * wingLo.qty) < 1e-9 &&
        Math.abs(wingHi.qty - wingLo.qty) < 1e-9;
      if (isButterflyShape) {
        const lowerWing = middle.K - wingLo.K;
        const upperWing = wingHi.K - middle.K;
        const narrower = Math.min(lowerWing, upperWing);
        const net = strategyMark(legs, S, r, q, sigma);
        if (wingLo.side === 1) {
          // Long butterfly (outer legs long)
          if (net > 0) {
            return { amount: net, note: `Long butterfly: max loss = net debit (${net.toFixed(2)}).` };
          }
          const maxLoss = narrower - Math.abs(net);
          return {
            amount: Math.max(maxLoss, 0),
            note: `Broken-wing (credit): max loss = narrower wing (${narrower.toFixed(2)}) − credit (${Math.abs(net).toFixed(2)}).`,
          };
        } else {
          // Short butterfly (outer legs short)
          if (net < 0) {
            const maxLoss = narrower - Math.abs(net);
            return {
              amount: Math.max(maxLoss, 0),
              note: `Short butterfly: max loss at K_mid = narrower wing (${narrower.toFixed(2)}) − credit (${Math.abs(net).toFixed(2)}).`,
            };
          }
          return {
            amount: narrower + net,
            note: "Short butterfly with debit (unusual); treating as narrower-wing + debit.",
          };
        }
      }
    }
  }
  if (optionLegs.length === 2 && !hasUnderlying && optionLegs[0].kind === optionLegs[1].kind) {
    const [a, b] = optionLegs;
    const net = legMark(a, S, r, q, sigma) + legMark(b, S, r, q, sigma);
    const width = Math.abs(a.K - b.K);
    if (net > 0) return { amount: net, note: "Debit vertical: max loss = net debit." };
    return { amount: width - Math.abs(net), note: `Credit vertical: width ${width.toFixed(2)} − credit ${Math.abs(net).toFixed(2)}.` };
  }
  // Single-leg short put (not cash-secured): naked, not in our catalog but handled defensively
  if (optionLegs.length === 1 && !hasUnderlying) {
    const leg = optionLegs[0];
    const p = Math.abs(legMark(leg, S, r, q, sigma));
    if (leg.side === 1) return { amount: p, note: "Long option: premium is the max loss." };
    // naked short (safety fallback; shouldn't be reachable for the supported catalog)
    const otm = leg.kind === "call" ? Math.max(leg.K - S, 0) : Math.max(S - leg.K, 0);
    const regT = Math.max(0.20 * S - otm, 0.10 * (leg.kind === "put" ? leg.K : S)) + p;
    return { amount: regT, note: "Reg-T naked short: max(20%·S − OTM, 10%·S or K) + premium." };
  }
  const p = strategyMark(legs, S, r, q, sigma);
  return { amount: Math.max(p, 0), note: "Long-premium structure: max loss = total debit." };
}

function computeRiskFlags(legs, S, greeks, stratKey) {
  const flags = [];
  const optionLegs = legs.filter(l => l.kind !== "underlying");
  const hasUnderlying = legs.some(l => l.kind === "underlying");
  const shortCalls = optionLegs.filter(l => l.kind === "call" && l.side === -1).reduce((s, l) => s + l.qty, 0);
  const longCalls = optionLegs.filter(l => l.kind === "call" && l.side === 1).reduce((s, l) => s + l.qty, 0);
  const shortPuts = optionLegs.filter(l => l.kind === "put" && l.side === -1).reduce((s, l) => s + l.qty, 0);
  const longPuts = optionLegs.filter(l => l.kind === "put" && l.side === 1).reduce((s, l) => s + l.qty, 0);

  if (shortCalls > longCalls && !hasUnderlying) {
    flags.push({ severity: "danger", code: "UNLIMITED_UPSIDE_LOSS",
      message: "Net short calls without stock coverage: theoretically unlimited loss on a rally." });
  }
  if (shortPuts > longPuts) {
    flags.push({ severity: "warn", code: "LARGE_DOWNSIDE_LOSS",
      message: "Net short puts: loss grows as underlying falls, capped only at zero." });
  }
  if (optionLegs.some(l => l.side === -1 && l.T <= 30 / 365)) {
    flags.push({ severity: "warn", code: "ASSIGNMENT_RISK",
      message: "Short leg within 30 days of expiry: rising assignment risk if it goes ITM." });
  }
  if (greeks.theta < 0 && Math.abs(greeks.theta) > 0.005 * S) {
    flags.push({ severity: "info", code: "THETA_DECAY",
      message: `Strategy bleeds ≈$${(Math.abs(greeks.theta) / 365).toFixed(2)} per share per day at current vol.` });
  }
  if (Math.abs(greeks.vega) > 0.05 * S) {
    flags.push({ severity: "info", code: "VEGA_EXPOSURE",
      message: `${greeks.vega > 0 ? "Long" : "Short"} vega: a 1-vol-point move shifts P&L by ≈$${(Math.abs(greeks.vega) / 100).toFixed(2)}/share.` });
  }
  return flags;
}

// =============================================================================
// Recommender — expected-utility ranking, no weights. Mirrors engine/recommender.py
// =============================================================================
// Ranks strategies by the expected utility of their payoff distribution under
// the user's view, with risk aversion derived from the Capacity-for-Loss score.
// There are NO direction/vol/risk weights: direction and vol are captured by the
// view-implied distribution; risk by the curvature of the CRRA utility function.
//
// Two explicit, documented belief assumptions (adjustable, not hidden weights):
//   drift  : bullish/bearish -> +/- 0.5 * sigma * sqrt(T); neutral -> 0
//   spread : vol up/down -> sigma * 1.3 / sigma * 0.7; neutral -> sigma

const DRIFT_SIGMA_FRACTION = 0.5;
const VOL_UP_MULT = 1.3;
const VOL_DOWN_MULT = 0.7;

// Map Capacity-for-Loss score (0-10) to CRRA risk aversion g in [1, 6].
// High CFL -> low g (risk-tolerant); low CFL -> high g (downside-averse).
function cflToRiskAversion(cflScore) {
  return Math.max(1, Math.min(6, 6 - 0.5 * cflScore));
}

// CRRA utility over terminal wealth (array). U(W)=W^(1-g)/(1-g), or ln(W) at g=1.
function crraUtility(wealth, gamma) {
  const eps = 1e-6;
  if (Math.abs(gamma - 1) < 1e-9) return wealth.map(w => Math.log(Math.max(w, eps)));
  return wealth.map(w => Math.pow(Math.max(w, eps), 1 - gamma) / (1 - gamma));
}

// View-implied lognormal terminal-price grid + probability weights.
function viewDistribution(S, sigma, T, directional, vol, n = 2001) {
  T = Math.max(T, 1e-6); sigma = Math.max(sigma, 1e-6);
  const tilt = DRIFT_SIGMA_FRACTION * sigma / Math.sqrt(T);
  const mu = directional === "bullish" ? tilt : directional === "bearish" ? -tilt : 0;
  const sigV = vol === "up" ? sigma * VOL_UP_MULT : vol === "down" ? sigma * VOL_DOWN_MULT : sigma;

  const ST = new Array(n), pdf = new Array(n);
  let psum = 0;
  for (let i = 0; i < n; i++) {
    const z = -5 + 10 * i / (n - 1);
    const p = Math.exp(-0.5 * z * z);
    pdf[i] = p; psum += p;
    ST[i] = S * Math.exp((mu - 0.5 * sigV * sigV) * T + sigV * Math.sqrt(T) * z);
  }
  for (let i = 0; i < n; i++) pdf[i] /= psum;
  return { ST, pdf };
}

// Rank strategies by expected utility. cflScore from the suitability profile.
function recommendEU(directional, vol, cflScore, market) {
  const { S, sigma, T, r, q } = market;
  const gamma = cflToRiskAversion(cflScore);
  const { ST, pdf } = viewDistribution(S, sigma, T, directional, vol);
  const W0 = 100 * S;  // account base: one contract notional ~ 1% of account

  const scored = Object.entries(STRAT_DEFS).map(([key, def]) => {
    const params = def.defaultParams(S);
    const legs = def.legs(S, T, params);
    const entryPremium = strategyMark(legs, S, r, q, sigma, market.exerciseStyle || "european");

    // Payoff per contract (x100) and expected utility under the view
    let eu = 0, expPnl = 0, pProfit = 0;
    const util = [];
    for (let i = 0; i < ST.length; i++) {
      const pnlContract = (strategyPayoff(legs, ST[i], entryPremium)) * 100;
      const w = Math.max(W0 + pnlContract, 1e-6);
      const u = Math.abs(gamma - 1) < 1e-9 ? Math.log(w) : Math.pow(w, 1 - gamma) / (1 - gamma);
      eu += pdf[i] * u;
      expPnl += pdf[i] * pnlContract;
      if (pnlContract > 0) pProfit += pdf[i];
    }
    // Certainty-equivalent gain over doing nothing
    let ce;
    if (Math.abs(gamma - 1) < 1e-9) ce = Math.exp(eu);
    else ce = Math.pow((1 - gamma) * eu, 1 / (1 - gamma));
    const ceGain = ce - W0;

    return { key, def, eu, ceGain, expPnl, pProfit };
  });

  scored.sort((a, b) => (b.eu !== a.eu ? b.eu - a.eu : a.def.label.localeCompare(b.def.label)));
  return scored.slice(0, 3);
}

// =============================================================================
// Presentation helpers
// =============================================================================

const fmtMoney = v => (v >= 0 ? "" : "−") + "$" + Math.abs(v).toFixed(2);
const fmtSigned = v => (v >= 0 ? "+" : "−") + "$" + Math.abs(v).toFixed(2);
const fmtPct = v => (v >= 0 ? "+" : "") + (v * 100).toFixed(1) + "%";
const fmtNum = (v, dp = 3) => v.toFixed(dp);

// =============================================================================
// UI primitives
// =============================================================================

function Tab({ active, onClick, children }) {
  return (
    <button onClick={onClick}
      style={{
        fontFamily: "Instrument Sans, ui-sans-serif, sans-serif",
        borderBottom: active ? "2px solid #2E4A7B" : "2px solid transparent",
        color: active ? "#2E4A7B" : "#78716c",
        fontWeight: active ? 600 : 400,
      }}
      className="px-4 py-2 text-xs tracking-widest uppercase transition-colors hover:opacity-80">
      {children}
    </button>
  );
}

function StatBlock({ label, value, sublabel, positive, negative }) {
  const color = positive ? "text-emerald-700" : negative ? "text-rose-700" : "text-stone-900";
  return (
    <div className="px-4 py-3 border border-stone-300 bg-white">
      <div className="text-[10px] tracking-[0.15em] uppercase text-stone-500 mb-1"
           style={{ fontFamily: "Instrument Sans" }}>{label}</div>
      <div className={`text-2xl ${color}`}
           style={{ fontFamily: "JetBrains Mono, ui-monospace, monospace", fontVariantNumeric: "tabular-nums" }}>
        {value}
      </div>
      {sublabel && <div className="text-[11px] text-stone-500 mt-0.5">{sublabel}</div>}
    </div>
  );
}

function Divider() { return <div className="border-t border-dotted border-stone-300 my-6" />; }

function RiskPill({ severity, code, message }) {
  const colors = {
    danger: "bg-rose-50 border-rose-300 text-rose-900",
    warn:   "bg-amber-50 border-amber-300 text-amber-900",
    info:   "bg-stone-50 border-stone-300 text-stone-700",
  };
  return (
    <div className={`border px-3 py-2 ${colors[severity]} flex gap-2 items-start text-sm`}>
      <span className="text-[10px] tracking-widest uppercase font-semibold mt-0.5 whitespace-nowrap">
        {severity}
      </span>
      <span className="flex-1">
        <span className="font-mono text-xs mr-2">{code}</span>
        {message}
      </span>
    </div>
  );
}

// =============================================================================
// Application shell
// =============================================================================

// ============================================================================
// Suitability: evidence-based scoring rubric
// ============================================================================
//
// Two independent scores drive the approval level:
//   experienceScore in [0, 10], from trade count, years active, credentials
//   capacityScore   in [0, 10], from at-risk capital and Capacity for Loss
//
// A level is granted only when both scores meet the level's threshold.
// This prevents a high-net-worth beginner (high capacity, no experience) from
// reaching Level 3, and vice versa.
//
// Note: real brokers use a fourth tier gating naked short options. This build
// excludes naked shorts from the v1 catalog by design, so the rubric collapses
// to three tiers. A fourth tier could be added with unlimited-loss strategies.
// ============================================================================

const INITIAL_PROFILE = {
  // Experience inputs
  tradesLastYear: "none",       // "none" | "1_10" | "11_50" | "50_plus"
  yearsActive: "under_1",       // "under_1" | "1_3" | "3_10" | "over_10"
  credentials: [],              // array of: "series_7" | "finance_role" | "derivatives_course"
  // Capacity inputs
  liquidCapital: 50000,         // USD, liquid capital not needed for 3-year essentials
  allocationPct: 10,            // % of liquidCapital allocated to options trading
  impactIf50pctLoss: "stressful", // "unaffected" | "stressful" | "significant" | "devastating"
};

function computeExperienceScore(p) {
  const tradesPts = { none: 0, "1_10": 1, "11_50": 3, "50_plus": 4 }[p.tradesLastYear] ?? 0;
  const yearsPts  = { under_1: 0, "1_3": 1, "3_10": 2, over_10: 3 }[p.yearsActive] ?? 0;
  // Each credential worth 1 point, capped at 3
  const credPts = Math.min(3, (p.credentials || []).length);
  return Math.min(10, tradesPts + yearsPts + credPts);
}

function computeCapacityScore(p) {
  // Capital component: log-scaled so $10k isn't zero but $1M isn't infinite
  // $10k → 2pts, $50k → 3pts, $100k → 4pts, $500k → 5pts, $1M+ → 6pts
  const capitalPts =
    p.liquidCapital >= 1_000_000 ? 6 :
    p.liquidCapital >= 500_000 ? 5 :
    p.liquidCapital >= 100_000 ? 4 :
    p.liquidCapital >= 50_000 ? 3 :
    p.liquidCapital >= 10_000 ? 2 : 0;

  // Allocation: moderate is best; too little = no skin in game, too much = overexposed
  // Sweet spot 5–20%; penalties either side
  const allocPts =
    p.allocationPct > 50 ? 0 :    // reckless over-allocation
    p.allocationPct > 30 ? 1 :
    p.allocationPct >= 5 ? 2 :
    p.allocationPct >= 1 ? 1 : 0;

  // Capacity for Loss: heavily weights this qualitative check
  const impactPts = {
    unaffected: 2,
    stressful: 1,
    significant: -2,   // hard downgrade: user says they can't actually afford to lose
    devastating: -5,   // force floor
  }[p.impactIf50pctLoss] ?? 0;

  return Math.max(0, Math.min(10, capitalPts + allocPts + impactPts));
}

function approvalLevel(p) {
  const exp = computeExperienceScore(p);
  const cap = computeCapacityScore(p);
  // Both scores must meet threshold
  if (exp >= 6 && cap >= 5) return 3;
  if (exp >= 3 && cap >= 3) return 2;
  return 1;
}

// Lot-count caps per level — scale with both experience and capacity
const LOT_CAPS = { 1: 1, 2: 3, 3: 25 };

// What's the next threshold the user needs to cross to upgrade?
function upgradePath(p) {
  const exp = computeExperienceScore(p);
  const cap = computeCapacityScore(p);
  const level = approvalLevel(p);
  if (level === 3) return null;
  const nextThresholds = { 1: [3, 3], 2: [6, 5] };
  const [expNeed, capNeed] = nextThresholds[level];
  const gaps = [];
  if (exp < expNeed) gaps.push(`experience score +${expNeed - exp} (currently ${exp}/10)`);
  if (cap < capNeed) gaps.push(`CFL score +${capNeed - cap} (currently ${cap}/10)`);
  return { nextLevel: level + 1, gaps };
}

export default function Amoghopaya() {
  const [stage, setStage] = useState("onboard"); // onboard | app
  const [profile, setProfile] = useState(INITIAL_PROFILE);
  const [tab, setTab] = useState("gallery");  // gallery | builder | review | lab | recommender | dashboard
  const [selectedStrat, setSelectedStrat] = useState(null);
  const [stratParams, setStratParams] = useState({});
  // Order sizing and type (brief requires "quantity, and order type concept")
  const [lotSize, setLotSize] = useState(1);
  const [orderType, setOrderType] = useState("market");  // "market" | "limit"
  const [limitPrice, setLimitPrice] = useState(0);
  const [positions, setPositions] = useState([]);

  const level = approvalLevel(profile);

  // Global market context (can be tweaked in the lab)
  const [market, setMarket] = useState({
    underlying: "SPY", S: 450, r: 0.05, q: 0.015, sigma: 0.18, T: 45 / 365,
    exerciseStyle: "european",  // "european" | "american"
  });

  if (stage === "onboard") {
    return <Onboard profile={profile} setProfile={setProfile} level={level} onDone={() => setStage("app")} />;
  }

  return (
    <div className="min-h-screen bg-[#FAF9F6]" style={{ fontFamily: "Instrument Sans, system-ui, sans-serif" }}>
      {/* Header */}
      <header className="border-b border-stone-300 bg-white">
        <div className="max-w-7xl mx-auto px-8 py-5 flex items-center justify-between">
          <div className="flex flex-col">
            <h1 className="text-3xl tracking-tight text-[#2E4A7B] leading-none"
                style={{ fontFamily: "Playfair Display, Georgia, serif", fontWeight: 700 }}>
              Amoghopāya
            </h1>
            <span className="text-xs tracking-wide text-stone-500 mt-0.5"
                  style={{ fontFamily: "Playfair Display, Georgia, serif" }}>
              Option Trading Platform
            </span>
          </div>
          <div className="flex items-center gap-4 text-xs tracking-wider uppercase text-stone-600">
            <span className="flex items-center gap-1.5">
              <span className="text-stone-400">Approval</span>
              <span className="text-[#2E4A7B] font-bold text-sm">L{level}</span>
            </span>
            <span className="text-stone-300">│</span>
            <span className="font-mono normal-case">
              <span className="text-stone-400 text-[10px] mr-1">SPOT</span>
              <span className="text-stone-800">{market.underlying} ${market.S.toFixed(2)}</span>
            </span>
            <span className="text-stone-300">│</span>
            <span className="flex items-center gap-1.5">
              <span className="text-stone-400 text-[10px]">STYLE</span>
              <span className="flex border border-stone-300">
                {["european", "american"].map(st => {
                  const sel = market.exerciseStyle === st;
                  return (
                    <button key={st}
                      onClick={() => setMarket({ ...market, exerciseStyle: st })}
                      style={sel
                        ? { backgroundColor: "#2E4A7B", color: "#FFFFFF" }
                        : { backgroundColor: "#FFFFFF", color: "#78716c" }}
                      className="px-2 py-0.5 text-[10px] uppercase tracking-wider transition-colors">
                      {st === "european" ? "Euro" : "Amer"}
                    </button>
                  );
                })}
              </span>
            </span>
            <span className="text-stone-300">│</span>
            <button onClick={() => setStage("onboard")}
                    className="hover:text-[#2E4A7B] border border-stone-300 px-3 py-1 hover:border-[#2E4A7B]">
              Profile
            </button>
          </div>
        </div>
        {/* Tabs */}
        <nav className="max-w-7xl mx-auto px-8 flex gap-1 border-t border-stone-200">
          <Tab active={tab === "gallery"} onClick={() => setTab("gallery")}>Strategies</Tab>
          <Tab active={tab === "builder"} onClick={() => setTab("builder")}>Builder</Tab>
          <Tab active={tab === "review"} onClick={() => setTab("review")}>Review</Tab>
          <Tab active={tab === "lab"} onClick={() => setTab("lab")}>Vol & Greeks Lab</Tab>
          <Tab active={tab === "recommender"} onClick={() => setTab("recommender")}>Recommender</Tab>
          <Tab active={tab === "dashboard"} onClick={() => setTab("dashboard")}>Positions</Tab>
        </nav>
      </header>

      <main className="max-w-7xl mx-auto px-8 py-8">
        {tab === "gallery" && (
          <Gallery level={level} onPick={key => {
            setSelectedStrat(key);
            setStratParams(STRAT_DEFS[key].defaultParams(market.S));
            setTab("builder");
          }} />
        )}
        {tab === "builder" && (
          <Builder market={market} setMarket={setMarket}
                   stratKey={selectedStrat} stratParams={stratParams} setStratParams={setStratParams}
                   lotSize={lotSize} setLotSize={setLotSize}
                   lotCap={LOT_CAPS[level]}
                   onReview={() => setTab("review")}
                   onBack={() => setTab("gallery")} />
        )}
        {tab === "review" && (
          <Review market={market} stratKey={selectedStrat} stratParams={stratParams}
                  lotSize={lotSize}
                  orderType={orderType} setOrderType={setOrderType}
                  limitPrice={limitPrice} setLimitPrice={setLimitPrice}
                  onConfirm={(position) => {
                    setPositions([...positions, {
                      ...position,
                      id: Date.now(),
                      lots: lotSize,
                      closedAt: null,
                      closingValue: null,
                      realizedPnl: null,
                    }]);
                    setTab("dashboard");
                  }}
                  onBack={() => setTab("builder")} />
        )}
        {tab === "lab" && (
          <Lab
            market={market}
            stratKey={selectedStrat}
            stratParams={stratParams}
            onGoToLibrary={() => setTab("gallery")} />
        )}
        {tab === "recommender" && (
          <Recommender market={market} profile={profile} onPick={key => {
            setSelectedStrat(key);
            setStratParams(STRAT_DEFS[key].defaultParams(market.S));
            setTab("builder");
          }} />
        )}
        {tab === "dashboard" && <Dashboard positions={positions} setPositions={setPositions} market={market} />}
      </main>

      <footer className="border-t border-stone-300 bg-white mt-16">
        <div className="max-w-7xl mx-auto px-8 py-4 text-xs text-stone-500 flex justify-between">
          <span>Amoghopāya Option Trading Platform · Kanishk Devgan</span>
          <span className="font-mono">Paper trading</span>
        </div>
      </footer>
    </div>
  );
}

// =============================================================================
// Onboarding
// =============================================================================

function Onboard({ profile, setProfile, level, onDone }) {
  const expScore = computeExperienceScore(profile);
  const capScore = computeCapacityScore(profile);
  const upgrade = upgradePath(profile);
  const lotCap = LOT_CAPS[level];
  const [showIntroDetail, setShowIntroDetail] = useState(false);

  const levelDescriptions = {
    1: "Income-oriented basics: Covered Call, Protective Put, Cash-Secured Put.",
    2: "Level 1 + outright long options (Long Call, Long Put, Long Straddle, Long Strangle).",
    3: "Level 2 + all vertical spreads, butterflies, iron condor, collar. All 20 strategies unlocked; highest position-size cap.",
  };

  const toggleCredential = (c) => {
    const set = new Set(profile.credentials || []);
    if (set.has(c)) set.delete(c); else set.add(c);
    setProfile({ ...profile, credentials: Array.from(set) });
  };

  return (
    <div className="min-h-screen bg-[#FAF9F6] px-6 py-10"
         style={{ fontFamily: "Instrument Sans, system-ui, sans-serif" }}>
      <div className="max-w-3xl w-full mx-auto">
        <div className="mb-8 text-center">
          <h1 className="text-6xl tracking-tight text-[#2E4A7B] leading-none"
              style={{ fontFamily: "Playfair Display, Georgia, serif", fontWeight: 700 }}>
            Amoghopāya
          </h1>
          <div className="text-xl tracking-wide text-stone-500 mt-1 mb-2"
               style={{ fontFamily: "Playfair Display, Georgia, serif" }}>
            Option Trading Platform
          </div>
          <div className="mt-4 text-stone-700 max-w-lg mx-auto text-sm">
            Before trading, complete a two-factor approval assessment:
            <div className="mt-1">
              trading experience &amp; capacity for loss (CFL).{" "}
              <button onClick={() => setShowIntroDetail(v => !v)}
                      className="underline text-[#2E4A7B] hover:opacity-70">
                {showIntroDetail ? "▴" : "▾"}
              </button>
            </div>
            {showIntroDetail && (
              <div className="mt-3 text-stone-600 text-xs leading-relaxed border-l-2 border-[#2E4A7B] pl-3 py-1 text-left">
                Options approval is a two-factor determination, modelled on the appropriateness and suitability principles in MiFID II and the FCA's Capacity-for-Loss guidance. Experience is scored from trading history and formal credentials; Capacity for Loss (CFL) is scored from liquid surplus and loss-absorption profile. Levels 1–3 gate the strategy catalogue and position sizing, and each level imposes an independent minimum on <em>both</em> factors, so neither experience nor capital alone is sufficient to advance.
              </div>
            )}
          </div>
        </div>

        {/* ========== EXPERIENCE SECTION ========== */}
        <div className="bg-white border border-stone-300 p-8 mb-6">
          <div className="flex items-baseline justify-between mb-5 pb-3 border-b border-stone-200">
            <div>
              <div className="text-[10px] tracking-[0.15em] uppercase text-stone-500">Part 1</div>
              <h2 className="text-2xl text-[#2E4A7B]"
                  style={{ fontFamily: "Playfair Display, Georgia, serif" }}>Options experience</h2>
            </div>
            <div className="text-right">
              <div className="text-[10px] uppercase tracking-wider text-stone-500">Score</div>
              <div className="text-2xl font-bold text-[#2E4A7B]">{expScore}<span className="text-sm text-stone-500">/10</span></div>
            </div>
          </div>
          <div className="space-y-5">
            <Field label="Options trades placed in the last 12 months">
              <RadioGroup value={profile.tradesLastYear} onChange={v => setProfile({ ...profile, tradesLastYear: v })}
                options={[
                  { v: "none", l: "None" }, { v: "1_10", l: "1–10" },
                  { v: "11_50", l: "11–50" }, { v: "50_plus", l: "50+" },
                ]} />
            </Field>
            <Field label="Years actively trading options">
              <RadioGroup value={profile.yearsActive} onChange={v => setProfile({ ...profile, yearsActive: v })}
                options={[
                  { v: "under_1", l: "< 1 year" }, { v: "1_3", l: "1–3 years" },
                  { v: "3_10", l: "3–10 years" }, { v: "over_10", l: "10+ years" },
                ]} />
            </Field>
            <div>
              <div className="text-[10px] tracking-[0.15em] uppercase text-stone-500 mb-2">
                Credentials (tick all that apply)
              </div>
              <div className="space-y-2">
                {[
                  { v: "series_7", l: "Hold or have held a FINRA Series 7, 55, 63, or 66 license" },
                  { v: "finance_role", l: "Work or have worked in a trading-related role in financial services" },
                  { v: "derivatives_course", l: "Completed a formal derivatives qualification (CFA Level II, FRM, CQF, MSc, PhD)" },
                ].map(({ v, l }) => {
                  const checked = (profile.credentials || []).includes(v);
                  return (
                    <label key={v} className="flex items-start gap-3 cursor-pointer text-sm">
                      <input type="checkbox" checked={checked}
                             onChange={() => toggleCredential(v)}
                             className="mt-0.5 w-4 h-4 accent-[#2E4A7B]" />
                      <span className={checked ? "text-stone-900" : "text-stone-600"}>{l}</span>
                    </label>
                  );
                })}
              </div>
            </div>
          </div>
        </div>

        {/* ========== CAPACITY SECTION ========== */}
        <div className="bg-white border border-stone-300 p-8 mb-6">
          <div className="flex items-baseline justify-between mb-5 pb-3 border-b border-stone-200">
            <div>
              <div className="text-[10px] tracking-[0.15em] uppercase text-stone-500">Part 2</div>
              <h2 className="text-2xl text-[#2E4A7B]"
                  style={{ fontFamily: "Playfair Display, Georgia, serif" }}>Capacity for Loss</h2>
            </div>
            <div className="text-right">
              <div className="text-[10px] uppercase tracking-wider text-stone-500">Score</div>
              <div className="text-2xl font-bold text-[#2E4A7B]">{capScore}<span className="text-sm text-stone-500">/10</span></div>
            </div>
          </div>
          <div className="space-y-5">
            <div className="grid grid-cols-2 gap-4">
              <Field label="Liquid capital, surplus to 3-year essentials (USD)">
                <NumberInput value={profile.liquidCapital} step={5000}
                             onChange={v => setProfile({ ...profile, liquidCapital: v })} />
                <div className="text-[10px] text-stone-500 mt-1">
                  Capital you genuinely don't need for housing, education, emergencies over the next 3 years.
                </div>
              </Field>
              <Field label="% of this capital allocated to options">
                <NumberInput value={profile.allocationPct} step={5} min={0} max={100}
                             onChange={v => setProfile({ ...profile, allocationPct: Math.max(0, Math.min(100, v)) })} />
                <div className="text-[10px] text-stone-500 mt-1">
                  At-risk: <span className="font-mono text-stone-700">
                    ${Math.round((profile.liquidCapital || 0) * (profile.allocationPct || 0) / 100).toLocaleString()}
                  </span>
                </div>
              </Field>
            </div>
            <Field label="If your options allocation dropped 50% in one month, your life would be:">
              <div className="grid grid-cols-2 gap-2">
                {[
                  { v: "unaffected", l: "Unaffected", d: "Loss is within routine variation." },
                  { v: "stressful", l: "Stressful but manageable", d: "Uncomfortable; no lifestyle change." },
                  { v: "significant", l: "Significantly impacted", d: "Would force budget or plan changes." },
                  { v: "devastating", l: "Financially devastating", d: "Would jeopardise essentials." },
                ].map(({ v, l, d }) => {
                  const sel = profile.impactIf50pctLoss === v;
                  return (
                    <button key={v} onClick={() => setProfile({ ...profile, impactIf50pctLoss: v })}
                      style={sel
                        ? { borderColor: "#2E4A7B", borderWidth: 2, backgroundColor: "#EEF2F8" }
                        : { borderColor: "#d6d3d1", borderWidth: 1 }}
                      className="text-left px-3 py-2 transition-colors hover:opacity-90">
                      <div style={{ color: sel ? "#2E4A7B" : "#1c1917", fontWeight: sel ? 600 : 400 }}
                           className="text-sm">{l}</div>
                      <div className="text-[10px] text-stone-500 mt-0.5">{d}</div>
                    </button>
                  );
                })}
              </div>
            </Field>
          </div>
        </div>

        {/* ========== RESULT ========== */}
        <div className="bg-white border border-stone-300 p-8">
          <div className="flex items-start justify-between mb-5 pb-4 border-b border-stone-200">
            <div>
              <div className="text-[10px] tracking-[0.15em] uppercase text-stone-500">Assigned approval level</div>
              <div className="flex items-baseline gap-3 mt-1">
                <div className="text-5xl text-[#2E4A7B] font-bold"
                     style={{ fontFamily: "Playfair Display, Georgia, serif" }}>{level}</div>
                <div className="text-sm uppercase tracking-wider text-stone-500">of 3</div>
              </div>
              <div className="text-sm text-stone-700 mt-2 leading-relaxed">{levelDescriptions[level]}</div>
            </div>
            <div className="text-right">
              <div className="text-[10px] uppercase tracking-wider text-stone-500">Position size cap</div>
              <div className="text-2xl font-bold text-stone-900 mt-1">{lotCap}<span className="text-sm text-stone-500"> lot{lotCap === 1 ? "" : "s"}</span></div>
              <div className="text-[10px] text-stone-500 mt-1">max per order</div>
            </div>
          </div>
          <div className="mb-5">
            <div className="uppercase tracking-wider text-[10px] text-stone-500 mb-2">Why this level</div>
            <div className="border border-stone-200">
              <div className="grid grid-cols-12 gap-0 text-[10px] uppercase tracking-wider text-stone-500 bg-stone-50 border-b border-stone-200">
                <div className="col-span-1 px-3 py-2">Level</div>
                <div className="col-span-4 px-3 py-2">Experience</div>
                <div className="col-span-4 px-3 py-2">CFL</div>
                <div className="col-span-3 px-3 py-2 text-right">Status</div>
              </div>
              {[
                { lvl: 1, expNeed: 0, capNeed: 0 },
                { lvl: 2, expNeed: 3, capNeed: 3 },
                { lvl: 3, expNeed: 6, capNeed: 5 },
              ].map(({ lvl, expNeed, capNeed }) => {
                const expOk = expScore >= expNeed;
                const capOk = capScore >= capNeed;
                const qualified = expOk && capOk;
                const isCurrent = lvl === level;
                return (
                  <div key={lvl}
                    style={{
                      backgroundColor: isCurrent ? "#EEF2F8" : "#FFFFFF",
                      borderLeft: isCurrent ? "3px solid #2E4A7B" : "3px solid transparent",
                    }}
                    className="grid grid-cols-12 gap-0 text-xs border-b border-stone-100 last:border-b-0">
                    <div className="col-span-1 px-3 py-2 font-bold" style={{ color: isCurrent ? "#2E4A7B" : "#44403c" }}>
                      {lvl}
                    </div>
                    <div className="col-span-4 px-3 py-2 text-stone-700">
                      {expNeed === 0 ? (
                        <span className="text-stone-500">no minimum</span>
                      ) : (
                        <>
                          need ≥ {expNeed}
                          <span className="ml-2" style={{ color: expOk ? "#047857" : "#b91c1c" }}>
                            (you: {expScore} {expOk ? "✓" : "✗"})
                          </span>
                        </>
                      )}
                    </div>
                    <div className="col-span-4 px-3 py-2 text-stone-700">
                      {capNeed === 0 ? (
                        <span className="text-stone-500">no minimum</span>
                      ) : (
                        <>
                          need ≥ {capNeed}
                          <span className="ml-2" style={{ color: capOk ? "#047857" : "#b91c1c" }}>
                            (you: {capScore} {capOk ? "✓" : "✗"})
                          </span>
                        </>
                      )}
                    </div>
                    <div className="col-span-3 px-3 py-2 text-right text-[11px]">
                      {isCurrent ? (
                        <span className="font-semibold" style={{ color: "#2E4A7B" }}>You are here</span>
                      ) : qualified ? (
                        <span style={{ color: "#047857" }}>Qualified</span>
                      ) : (
                        <span className="text-stone-400">—</span>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
          {upgrade && (
            <div className="mb-5 text-xs text-stone-700 leading-relaxed bg-amber-50 border-l-2 border-amber-400 pl-3 py-2">
              <div className="uppercase tracking-wider text-[10px] text-amber-700 mb-1">To reach Level {upgrade.nextLevel}</div>
              {upgrade.gaps.map((g, i) => <div key={i}>• {g}</div>)}
            </div>
          )}
          <button onClick={onDone}
            style={{ backgroundColor: "#2E4A7B", color: "#FFFFFF", fontWeight: 600 }}
            className="w-full px-6 py-3 text-sm tracking-wider uppercase hover:opacity-90">
            Enter platform →
          </button>
          <div className="mt-4 text-[10px] text-stone-400 leading-relaxed text-center">
            Self-reported inputs are used only to compute your approval level for this session and are not stored or transmitted.
            This is an educational paper-trading prototype: the assessment is illustrative and is not a regulated suitability determination or financial advice.
          </div>
        </div>
      </div>
    </div>
  );
}

function Field({ label, children }) {
  return (
    <div>
      <div className="text-[10px] tracking-[0.15em] uppercase text-stone-500 mb-2">{label}</div>
      {children}
    </div>
  );
}

function RadioGroup({ value, onChange, options }) {
  return (
    <div className="flex gap-0 border border-stone-300">
      {options.map(o => {
        const selected = value === o.v;
        return (
          <button key={o.v} onClick={() => onChange(o.v)}
            style={selected
              ? { backgroundColor: "#2E4A7B", color: "#FFFFFF", fontWeight: 600 }
              : { backgroundColor: "#FFFFFF", color: "#44403c" }}
            className="flex-1 px-4 py-2 text-sm border-r border-stone-300 last:border-r-0 transition-colors hover:opacity-90">
            {o.l}
          </button>
        );
      })}
    </div>
  );
}

function NumberInput({ value, onChange, step = 1, min = 0, max = 9999999 }) {
  return (
    <input type="number" value={value} step={step} min={min} max={max}
      onChange={e => onChange(+e.target.value)}
      className="w-full border border-stone-300 px-3 py-2 font-mono text-sm bg-white focus:outline-none focus:border-[#2E4A7B]" />
  );
}

// =============================================================================
// Payoff shape sketches — iconic SVG previews for the explainer modal
// =============================================================================
// Each shape is a small 120×60 SVG path that renders the strategy's expiry
// payoff silhouette. These are pedagogical icons, not to-scale plots, just
// enough to convey the shape. x: 0 (OTM left) to 120 (OTM right). y: 60 (max
// loss) to 0 (max profit).

const PAYOFF_PATHS = {
  hockey_up:     "M 0,50 L 50,50 L 110,10",
  hockey_down:   "M 10,10 L 70,50 L 120,50",
  stairs_up:     "M 0,50 L 40,50 L 70,20 L 120,20",
  stairs_down:   "M 0,20 L 50,20 L 80,50 L 120,50",
  v_valley:      "M 0,15 L 60,55 L 120,15",
  v_wide:        "M 0,15 L 40,50 L 80,50 L 120,15",
  tent:          "M 0,50 L 50,50 L 60,15 L 70,50 L 120,50",
  tent_capped:   "M 0,40 L 40,40 L 60,15 L 80,40 L 120,40",
  tent_skewed:   "M 0,45 L 40,45 L 60,15 L 90,40 L 120,40",
  valley_capped: "M 0,25 L 40,25 L 60,50 L 80,25 L 120,25",
  plateau:       "M 0,50 L 30,50 L 50,20 L 70,20 L 90,50 L 120,50",
  step_capped:   "M 0,50 L 55,50 L 60,15 L 120,15",
  step_bounded:  "M 0,50 L 30,50 L 60,30 L 90,15 L 120,15",
};

function PayoffSketch({ shape, height = 60, width = 120 }) {
  const path = PAYOFF_PATHS[shape] || PAYOFF_PATHS.hockey_up;
  return (
    <svg viewBox={`0 0 ${width} ${height}`} width={width} height={height}
         style={{ display: "block" }}>
      <line x1="0" y1={height / 2} x2={width} y2={height / 2}
            stroke="#D6D3D1" strokeWidth="0.5" strokeDasharray="2,2" />
      <path d={path} fill="none" stroke="#2E4A7B" strokeWidth="2"
            strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

// =============================================================================
// Explain modal — "Learn more" per-strategy pedagogical panel
// =============================================================================

function ExplainModal({ stratKey, onClose }) {
  if (!stratKey) return null;
  const def = STRAT_DEFS[stratKey];
  const ex = STRAT_EXPLAINERS[stratKey];
  if (!def || !ex) return null;

  return (
    <div
      onClick={onClose}
      style={{
        position: "fixed", inset: 0, backgroundColor: "rgba(28, 25, 23, 0.6)",
        zIndex: 50, display: "flex", alignItems: "center", justifyContent: "center", padding: "24px",
      }}>
      <div onClick={e => e.stopPropagation()}
           style={{
             backgroundColor: "#FFFFFF", maxWidth: "640px", width: "100%",
             maxHeight: "90vh", overflowY: "auto", border: "1px solid #D6D3D1",
             fontFamily: "Instrument Sans, system-ui, sans-serif",
           }}>
        {/* Header */}
        <div style={{ padding: "24px 28px 16px 28px", borderBottom: "1px solid #E7E5E4" }}>
          <div className="flex items-start justify-between">
            <div>
              <div className="text-[10px] tracking-[0.15em] uppercase text-stone-500 mb-1">Strategy reference</div>
              <h2 style={{ fontFamily: "Playfair Display, Georgia, serif", fontSize: "28px", fontWeight: 700, color: "#2E4A7B" }}>
                {def.label}
              </h2>
              <div className="text-sm text-stone-600 mt-1">{def.tagline}</div>
            </div>
            <button onClick={onClose}
                    style={{ fontSize: "22px", color: "#78716c", padding: "4px 12px", lineHeight: 1 }}
                    className="hover:opacity-70">✕</button>
          </div>
          <div className="flex gap-2 mt-4">
            <span className="text-[10px] tracking-wider uppercase px-2 py-0.5 border border-stone-300 text-stone-600">{def.direction}</span>
            <span className="text-[10px] tracking-wider uppercase px-2 py-0.5 border border-stone-300 text-stone-600">
              {def.vol === "long" ? "+vega" : def.vol === "short" ? "−vega" : "~vega"}
            </span>
            <span className={`text-[10px] tracking-wider uppercase px-2 py-0.5 border ${
              def.risk === "low" ? "border-emerald-300 text-emerald-700 bg-emerald-50" :
              def.risk === "medium" ? "border-amber-300 text-amber-700 bg-amber-50" :
              "border-rose-300 text-rose-700 bg-rose-50"
            }`}>{def.risk} risk</span>
            <span className="text-[10px] tracking-wider uppercase px-2 py-0.5 border border-stone-300 text-stone-600">
              Level {def.approvalLevel}
            </span>
          </div>
        </div>

        {/* Payoff sketch */}
        <div style={{ padding: "20px 28px", borderBottom: "1px solid #E7E5E4", backgroundColor: "#FAFAF9" }}>
          <div className="text-[10px] tracking-[0.15em] uppercase text-stone-500 mb-3">Payoff at expiry</div>
          <div className="flex items-center gap-4">
            <PayoffSketch shape={ex.payoffShape} />
            <div className="text-xs text-stone-600">
              <div className="flex gap-4">
                <div>
                  <div className="uppercase tracking-wider text-[9px] text-stone-400">Max profit</div>
                  <div className="text-stone-700 mt-0.5">{def.maxProfit}</div>
                </div>
              </div>
              <div className="mt-2">
                <div className="uppercase tracking-wider text-[9px] text-stone-400">Max loss</div>
                <div className="text-stone-700 mt-0.5">{def.maxLoss}</div>
              </div>
            </div>
          </div>
        </div>

        {/* Content sections */}
        <div style={{ padding: "24px 28px" }}>
          <div className="mb-6">
            <div className="text-[10px] tracking-[0.15em] uppercase text-stone-500 mb-2">Thesis</div>
            <div className="text-sm text-stone-800 leading-relaxed">{ex.whenToUse}</div>
          </div>
          <div className="mb-6">
            <div className="text-[10px] tracking-[0.15em] uppercase text-stone-500 mb-2">Risk profile</div>
            <div className="text-sm text-stone-800 leading-relaxed">{ex.greekSignature}</div>
          </div>
          <div className="mb-2 p-3"
               style={{ borderLeft: "2px solid #C99B33", backgroundColor: "#FAF7EF" }}>
            <div className="text-[10px] tracking-[0.15em] uppercase text-[#8B6E1F] mb-1">Structural note</div>
            <div className="text-sm text-stone-800 leading-relaxed">{ex.commonMistake}</div>
          </div>
        </div>

        {/* Footer */}
        <div style={{ padding: "16px 28px", borderTop: "1px solid #E7E5E4", backgroundColor: "#FAFAF9" }}>
          <button onClick={onClose}
            style={{ backgroundColor: "#2E4A7B", color: "#FFFFFF" }}
            className="px-5 py-2 text-xs tracking-wider uppercase hover:opacity-90">
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

// =============================================================================
// Gallery
// =============================================================================

function Gallery({ level, onPick }) {
  const [explainKey, setExplainKey] = useState(null);
  return (
    <div>
      <div className="mb-6 flex items-baseline justify-between">
        <div>
          <h2 className="text-2xl text-stone-900" style={{ fontFamily: "Playfair Display, Georgia, serif" }}>
            Strategy Library
          </h2>
          <div className="text-sm text-stone-500 mt-1">
            Twenty defined-risk and income strategies, filtered to your approval level.
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        {Object.entries(STRAT_DEFS).map(([key, def]) => {
          const locked = def.approvalLevel > level;
          return (
            <div key={key}
              className={`text-left border transition-all ${
                locked ? "border-stone-200 bg-stone-50 opacity-60"
                       : "border-stone-300 bg-white hover:border-[#2E4A7B] hover:shadow-sm"
              }`}>
              <button onClick={() => !locked && onPick(key)} disabled={locked}
                className="w-full text-left p-5 disabled:cursor-not-allowed">
                <div className="flex items-start justify-between mb-3">
                  <div className={`text-[10px] tracking-[0.15em] uppercase px-2 py-0.5 border ${
                    def.risk === "low" ? "border-emerald-300 text-emerald-700 bg-emerald-50" :
                    def.risk === "medium" ? "border-amber-300 text-amber-700 bg-amber-50" :
                    "border-rose-300 text-rose-700 bg-rose-50"
                  }`}>{def.risk}</div>
                  {locked && <span className="text-[10px] text-stone-400 uppercase tracking-wider">L{def.approvalLevel}</span>}
                </div>
                <div className="text-lg font-semibold text-stone-900 mb-1">{def.label}</div>
                <div className="text-xs text-stone-500">{def.tagline}</div>
                <div className="mt-4 pt-3 border-t border-dotted border-stone-300 flex gap-3 text-[10px] uppercase tracking-wider text-stone-500">
                  <span>{def.direction}</span>
                  <span className="text-stone-300">·</span>
                  <span>{def.vol === "long" ? "+vega" : def.vol === "short" ? "−vega" : "~vega"}</span>
                </div>
              </button>
              <button onClick={() => setExplainKey(key)}
                className="w-full text-[10px] tracking-[0.15em] uppercase text-stone-500 hover:text-[#2E4A7B] border-t border-stone-200 py-2 transition-colors">
                Learn more
              </button>
            </div>
          );
        })}
      </div>

      <ExplainModal stratKey={explainKey} onClose={() => setExplainKey(null)} />
    </div>
  );
}

// =============================================================================
// Builder
// =============================================================================

function Builder({ market, setMarket, stratKey, stratParams, setStratParams, lotSize, setLotSize, lotCap, onReview, onBack }) {
  if (!stratKey) {
    return (
      <div className="text-center py-20">
        <div className="text-stone-500 mb-4">No strategy selected.</div>
        <button onClick={onBack} className="text-[#2E4A7B] underline">Go to Strategy Library</button>
      </div>
    );
  }
  const def = STRAT_DEFS[stratKey];
  const style = market.exerciseStyle || "european";
  const legs = def.legs(market.S, market.T, stratParams);
  const entryPremium = strategyMark(legs, market.S, market.r, market.q, market.sigma, style);
  const eePremium = earlyExercisePremium(legs, market.S, market.r, market.q, market.sigma);
  const curve = payoffCurve(legs, market.S, entryPremium, 0.3, 81);
  const breakevens = findBreakevens(curve);
  const greeks = strategyGreeks(legs, market.S, market.r, market.q, market.sigma, style);
  const minPnl = Math.min(...curve.map(d => d.pnl));
  const maxPnl = Math.max(...curve.map(d => d.pnl));

  // Strategy-level Greeks sampled across ±30% spot range.
  // Each Greek is normalised to its own max-abs across the grid so all four
  // curves land in [-1, +1] regardless of the strategy's intrinsic scale.
  // Raw values are carried alongside for tooltip display.
  const stratGreekCurve = useMemo(() => {
    const lo = market.S * 0.7, hi = market.S * 1.3;
    const raw = [];
    for (let i = 0; i < 61; i++) {
      const s = lo + (hi - lo) * i / 60;
      const g = strategyGreeks(legs, s, market.r, market.q, market.sigma, style);
      raw.push({ S: s, delta: g.delta, gamma: g.gamma, vega: g.vega, theta: g.theta });
    }
    const maxAbs = {
      delta: Math.max(...raw.map(d => Math.abs(d.delta)), 1e-6),
      gamma: Math.max(...raw.map(d => Math.abs(d.gamma)), 1e-6),
      vega:  Math.max(...raw.map(d => Math.abs(d.vega)),  1e-6),
      theta: Math.max(...raw.map(d => Math.abs(d.theta)), 1e-6),
    };
    return raw.map(d => ({
      S: d.S,
      delta: d.delta / maxAbs.delta,
      gamma: d.gamma / maxAbs.gamma,
      vega:  d.vega  / maxAbs.vega,
      theta: d.theta / maxAbs.theta,
      rawDelta: d.delta, rawGamma: d.gamma, rawVega: d.vega, rawTheta: d.theta,
    }));
  }, [legs, market.S, market.r, market.q, market.sigma, style]);

  // Historical volatility for reference (synthetic demo data, deterministic per ticker+IV)
  const hvPrices = useMemo(
    () => syntheticPriceHistory(market.underlying, market.S, market.sigma, 90),
    [market.underlying, market.S, market.sigma]
  );
  const hv30 = historicalVol(hvPrices, 30);

  const paramKeys = Object.keys(stratParams);

  return (
    <div>
      <div className="flex items-start justify-between mb-6">
        <div>
          <div className="text-xs tracking-widest uppercase text-stone-500 mb-1">Build strategy</div>
          <h2 className="text-3xl" style={{ fontFamily: "Playfair Display, Georgia, serif" }}>{def.label}</h2>
          <div className="text-sm text-stone-500 mt-1">{def.tagline}</div>
        </div>
        <div className="flex gap-2">
          <button onClick={onBack} className="px-4 py-2 border border-stone-300 text-sm hover:bg-stone-50">← Library</button>
          <button onClick={onReview}
            style={{ backgroundColor: "#2E4A7B", color: "#FFFFFF", fontWeight: 600 }}
            className="px-6 py-2 text-sm tracking-wider uppercase hover:opacity-90">
            Review →
          </button>
        </div>
      </div>

      {/* Exercise-style banner: makes the Euro/Amer toggle's effect legible */}
      <div className="mb-6 border border-stone-300 bg-white px-5 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-[10px] tracking-[0.15em] uppercase text-stone-500">Pricing under</span>
          <span className="font-semibold text-[#2E4A7B]">
            {style === "american" ? "American exercise (binomial tree)" : "European exercise (closed-form BSM)"}
          </span>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <span className="text-stone-500">Early-exercise premium:</span>
          <span className="font-mono text-stone-800">${eePremium.toFixed(4)}</span>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-6">
        {/* LEFT: parameters */}
        <div className="col-span-1 space-y-6">
          <div className="bg-white border border-stone-300 p-5">
            <div className="text-[10px] tracking-[0.15em] uppercase text-stone-500 mb-3">Underlying</div>
            <div className="grid grid-cols-2 gap-3">
              <Field label="Symbol">
                <input value={market.underlying}
                  onChange={e => setMarket({ ...market, underlying: e.target.value.toUpperCase() })}
                  className="w-full border border-stone-300 px-3 py-2 font-mono text-sm bg-white uppercase" />
              </Field>
              <Field label="Spot (S)">
                <NumberInput value={market.S} onChange={v => setMarket({ ...market, S: v })} step={1} />
              </Field>
              <Field label="IV (σ)">
                <NumberInput value={+market.sigma.toFixed(3)} step={0.01}
                  onChange={v => setMarket({ ...market, sigma: v })} />
              </Field>
              <Field label="Days to expiry">
                <NumberInput value={Math.round(market.T * 365)} step={1}
                  onChange={v => setMarket({ ...market, T: v / 365 })} />
              </Field>
              <Field label="Rate r">
                <NumberInput value={+market.r.toFixed(3)} step={0.005}
                  onChange={v => setMarket({ ...market, r: v })} />
              </Field>
              <Field label="Div yield q">
                <NumberInput value={+market.q.toFixed(3)} step={0.005}
                  onChange={v => setMarket({ ...market, q: v })} />
              </Field>
              <Field label={`Quantity (lots, max ${lotCap})`}>
                <NumberInput value={lotSize} step={1} min={1} max={lotCap}
                  onChange={v => setLotSize(Math.max(1, Math.min(lotCap, Math.round(v))))} />
              </Field>
              <div className="flex items-end text-[10px] text-stone-500 leading-tight">
                1 lot = 100 shares (US std).<br/>
                Total cost × {lotSize * 100}.<br/>
                <span className="text-[#2E4A7B]">Level cap: {lotCap} lot{lotCap === 1 ? "" : "s"}</span>
              </div>
            </div>
            {/* IV vs HV reference comparison — brief requires HV/IV distinction */}
            {hv30 !== null && (
              <div className="mt-3 pt-3 border-t border-dotted border-stone-300 text-xs flex items-center justify-between">
                <span className="text-stone-500">
                  Implied vol: <span className="font-mono text-stone-900">{(market.sigma * 100).toFixed(1)}%</span>
                  <span className="mx-2 text-stone-300">·</span>
                  Historical (30d): <span className="font-mono text-stone-900">{(hv30 * 100).toFixed(1)}%</span>
                </span>
                <span className={`text-[10px] uppercase tracking-wider font-semibold ${
                  market.sigma > hv30 * 1.1 ? "text-amber-700"
                  : market.sigma < hv30 * 0.9 ? "text-emerald-700"
                  : "text-stone-500"
                }`}>
                  {market.sigma > hv30 * 1.1 ? "IV rich vs realised"
                   : market.sigma < hv30 * 0.9 ? "IV cheap vs realised"
                   : "IV ≈ realised"}
                </span>
              </div>
            )}
          </div>

          <div className="bg-white border border-stone-300 p-5">
            <div className="text-[10px] tracking-[0.15em] uppercase text-stone-500 mb-3">Strategy strikes</div>
            <div className="space-y-3">
              {paramKeys.map(k => (
                <Field key={k} label={k.replace(/_/g, " ")}>
                  <NumberInput value={stratParams[k]} step={1}
                    onChange={v => setStratParams({ ...stratParams, [k]: v })} />
                </Field>
              ))}
            </div>
          </div>

          <div className="bg-white border border-stone-300 p-5">
            <div className="text-[10px] tracking-[0.15em] uppercase text-stone-500 mb-3">Legs</div>
            <div className="space-y-2 text-sm font-mono">
              {legs.map((l, i) => (
                <div key={i} className="flex justify-between items-center py-1 border-b border-dotted border-stone-200 last:border-0">
                  <span>
                    <span className={`px-2 py-0.5 text-[10px] mr-2 ${
                      l.side === 1 ? "bg-emerald-50 text-emerald-800 border border-emerald-300"
                                   : "bg-rose-50 text-rose-800 border border-rose-300"
                    }`}>{l.side === 1 ? "+" : "−"}{l.qty}</span>
                    {l.kind === "underlying" ? "stock" : `${l.kind.toUpperCase()} @${l.K}`}
                  </span>
                  <span className="text-stone-500 text-xs">
                    {l.kind !== "underlying" && `$${bsPrice(market.S, l.K, l.T, market.r, market.q, market.sigma, l.kind).toFixed(2)}`}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* RIGHT: chart + stats */}
        <div className="col-span-2 space-y-4">
          <div className="bg-white border border-stone-300 p-5">
            <div className="text-[10px] tracking-[0.15em] uppercase text-stone-500 mb-3">Payoff at expiry</div>
            <ResponsiveContainer width="100%" height={260}>
              <AreaChart data={curve} margin={{ top: 10, right: 30, left: 10, bottom: 30 }}>
                <CartesianGrid stroke="#E7E5E4" strokeDasharray="2 2" />
                <XAxis dataKey="S" type="number" domain={["auto", "auto"]}
                  tickFormatter={v => v.toFixed(0)} stroke="#78716C" fontSize={11}
                  tick={{ fontFamily: "JetBrains Mono" }}
                  label={{ value: "Underlying price at expiry (S_T)", position: "insideBottom", offset: -14,
                           fill: "#57534E", fontSize: 11, fontFamily: "Instrument Sans, system-ui, sans-serif" }} />
                <YAxis tickFormatter={v => fmtSigned(v)} stroke="#78716C" fontSize={11}
                  tick={{ fontFamily: "JetBrains Mono" }} width={76}
                  label={{ value: "P&L per share ($)", angle: -90, position: "insideLeft", offset: 12,
                           style: { textAnchor: "middle", fill: "#57534E", fontSize: 11,
                                    fontFamily: "Instrument Sans, system-ui, sans-serif" } }} />
                <Tooltip
                  formatter={(v) => fmtSigned(v)}
                  labelFormatter={v => `S = $${v.toFixed(2)}`}
                  contentStyle={{ fontFamily: "JetBrains Mono", fontSize: 12, border: "1px solid #D6D3D1" }} />
                <ReferenceLine y={0} stroke="#44403C" strokeWidth={1} />
                <ReferenceLine x={market.S} stroke="#2E4A7B" strokeDasharray="3 3" strokeWidth={1} />
                {breakevens.map((be, i) => (
                  <ReferenceLine key={i} x={be} stroke="#A16207" strokeDasharray="2 2" strokeWidth={1} />
                ))}
                <Area type="monotone" dataKey="pnl" stroke="#2E4A7B" fill="#2E4A7B" fillOpacity={0.08} strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
            {/* Reference-line legend: cleaner than on-chart labels */}
            <div className="flex gap-5 mt-2 text-[11px] text-stone-600 pl-4">
              <span className="flex items-center gap-2">
                <span style={{ display: "inline-block", width: "18px", height: 0, borderTop: "1.5px dashed #2E4A7B" }}/>
                <span>Current spot: <span className="font-mono text-[#2E4A7B]">${market.S.toFixed(2)}</span></span>
              </span>
              {breakevens.length > 0 && (
                <span className="flex items-center gap-2">
                  <span style={{ display: "inline-block", width: "18px", height: 0, borderTop: "1.5px dashed #A16207" }}/>
                  <span>Break-even{breakevens.length > 1 ? "s" : ""}: <span className="font-mono text-[#A16207]">
                    {breakevens.map(be => `$${be.toFixed(2)}`).join(" · ")}
                  </span></span>
                </span>
              )}
            </div>
          </div>

          <div className="grid grid-cols-4 gap-3">
            <StatBlock label="Net premium"
              value={fmtSigned(entryPremium)}
              sublabel={`per share · ${entryPremium >= 0 ? "debit (cost)" : "credit (received)"}`}
              negative={entryPremium > 0} positive={entryPremium < 0} />
            <StatBlock label="Max profit"
              value={isFinite(maxPnl) ? fmtSigned(maxPnl) : "∞"}
              sublabel="per share · over ±30% grid" positive />
            <StatBlock label="Max loss"
              value={isFinite(minPnl) ? fmtSigned(minPnl) : "∞"}
              sublabel="per share · over ±30% grid" negative />
            <StatBlock label="Breakevens"
              value={breakevens.length === 0 ? "—" : breakevens.map(be => be.toFixed(2)).join(" / ")}
              sublabel={breakevens.length === 1 ? "single BE" : breakevens.length === 2 ? "two BEs" : ""} />
          </div>

          {/* Dollar totals for the actual sized position */}
          <div className="bg-stone-50 border border-stone-300 p-4">
            <div className="text-[10px] tracking-[0.15em] uppercase text-stone-500 mb-2">
              Total for {lotSize} {lotSize === 1 ? "lot" : "lots"} · 1 lot = 100 shares
            </div>
            <div className="grid grid-cols-3 gap-4 text-sm font-mono" style={{ fontVariantNumeric: "tabular-nums" }}>
              <div>
                <div className="text-[10px] uppercase tracking-wider text-stone-500">Total premium</div>
                <div className={`text-lg ${entryPremium > 0 ? "text-rose-700" : "text-emerald-700"}`}>
                  {fmtSigned(entryPremium * lotSize * 100)}
                </div>
              </div>
              <div>
                <div className="text-[10px] uppercase tracking-wider text-stone-500">Max profit ($)</div>
                <div className="text-lg text-emerald-700">{fmtSigned(maxPnl * lotSize * 100)}</div>
              </div>
              <div>
                <div className="text-[10px] uppercase tracking-wider text-stone-500">Max loss ($)</div>
                <div className="text-lg text-rose-700">{fmtSigned(minPnl * lotSize * 100)}</div>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-5 gap-3">
            <StatBlock label="Δ Delta" value={fmtNum(greeks.delta, 3)} sublabel="per share" />
            <StatBlock label="Γ Gamma" value={fmtNum(greeks.gamma, 4)} sublabel="per share" />
            <StatBlock label="ν Vega" value={fmtNum(greeks.vega / 100, 3)} sublabel="per 1 vol pt" />
            <StatBlock label="Θ Theta" value={fmtNum(greeks.theta / 365, 3)} sublabel="per day" />
            <StatBlock label="ρ Rho" value={fmtNum(greeks.rho / 100, 3)} sublabel="per 1bp" />
          </div>

          {/* Strategy-level Greek sensitivity across the spot range */}
          <div className="bg-white border border-stone-300 p-5">
            <div className="text-[10px] tracking-[0.15em] uppercase text-stone-500 mb-3">
              Greeks across spot range
            </div>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={stratGreekCurve} margin={{ top: 10, right: 20, left: 0, bottom: 28 }}>
                <CartesianGrid stroke="#E7E5E4" strokeDasharray="2 2" />
                <XAxis dataKey="S" type="number" domain={["auto", "auto"]}
                  tickFormatter={v => v.toFixed(0)} stroke="#78716C" fontSize={11}
                  tick={{ fontFamily: "JetBrains Mono" }}
                  label={{ value: "Underlying spot (S)", position: "insideBottom", offset: -14,
                           fill: "#57534E", fontSize: 11, fontFamily: "Instrument Sans, system-ui, sans-serif" }} />
                <YAxis stroke="#78716C" fontSize={11} width={76}
                  tick={{ fontFamily: "JetBrains Mono" }} domain={[-1.1, 1.1]}
                  label={{ value: "Normalised Greek value", angle: -90, position: "insideLeft", offset: 12,
                           style: { textAnchor: "middle", fill: "#57534E", fontSize: 11,
                                    fontFamily: "Instrument Sans, system-ui, sans-serif" } }} />
                <Tooltip
                  contentStyle={{ fontFamily: "JetBrains Mono", fontSize: 11, border: "1px solid #D6D3D1" }}
                  labelFormatter={v => `S = $${v.toFixed(2)}`}
                  formatter={(value, name, props) => {
                    // Display raw Greek values in conventional units.
                    // Engine stores vega per 1.0 sigma and theta per year; convert
                    // to per-1-vol-pt and per-day for user-facing display.
                    const display = {
                      delta: props.payload.rawDelta,
                      gamma: props.payload.rawGamma,
                      vega:  props.payload.rawVega / 100,
                      theta: props.payload.rawTheta / 365,
                    }[name];
                    const suffix = { vega: " /vol pt", theta: " /day" }[name] || "";
                    return [display !== undefined
                      ? display.toFixed(name === "gamma" ? 5 : 4) + suffix
                      : value.toFixed(4), name];
                  }} />
                <ReferenceLine x={market.S} stroke="#2E4A7B" strokeDasharray="3 3" strokeWidth={1} />
                <ReferenceLine y={0} stroke="#44403C" strokeWidth={0.8} />
                <Line type="monotone" name="delta" dataKey="delta" stroke="#2E4A7B" strokeWidth={2} dot={false} />
                <Line type="monotone" name="gamma" dataKey="gamma" stroke="#059669" strokeWidth={1.5} dot={false} />
                <Line type="monotone" name="vega" dataKey="vega" stroke="#A16207" strokeWidth={1.5} dot={false} />
                <Line type="monotone" name="theta" dataKey="theta" stroke="#BE123C" strokeWidth={1.5} dot={false} />
              </LineChart>
            </ResponsiveContainer>
            <div className="flex gap-5 mt-2 text-[10px] text-stone-500 justify-center">
              <span><span className="inline-block w-3 h-0.5 bg-[#2E4A7B] mr-1 align-middle"/>Delta</span>
              <span><span className="inline-block w-3 h-0.5 bg-emerald-700 mr-1 align-middle"/>Gamma</span>
              <span><span className="inline-block w-3 h-0.5 bg-amber-700 mr-1 align-middle"/>Vega</span>
              <span><span className="inline-block w-3 h-0.5 bg-rose-700 mr-1 align-middle"/>Theta</span>
            </div>
            <div className="text-[10px] text-stone-400 text-center mt-1 italic">
              Each curve normalised to its own ±1 range across the grid. Hover to see actual values in conventional units (vega per vol pt, theta per day).
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// =============================================================================
// Review
// =============================================================================

function Review({ market, stratKey, stratParams, lotSize, orderType, setOrderType, limitPrice, setLimitPrice, onConfirm, onBack }) {
  if (!stratKey) return <div className="text-stone-500 py-20 text-center">Select a strategy first.</div>;
  const def = STRAT_DEFS[stratKey];
  const style = market.exerciseStyle || "european";
  const legs = def.legs(market.S, market.T, stratParams);
  const entryPremium = strategyMark(legs, market.S, market.r, market.q, market.sigma, style);
  const greeks = strategyGreeks(legs, market.S, market.r, market.q, market.sigma, style);
  const margin = computeMargin(legs, market.S, market.r, market.q, market.sigma, entryPremium, stratKey);
  const flags = computeRiskFlags(legs, market.S, greeks, stratKey);

  // Payoff curve and bounds: same engine as the Builder so the numbers match
  const curve = payoffCurve(legs, market.S, entryPremium, 0.3, 81);
  const breakevens = findBreakevens(curve);
  const minPnl = Math.min(...curve.map(d => d.pnl));
  const maxPnl = Math.max(...curve.map(d => d.pnl));

  // Scenario grid: 15 days forward, ±5/10/20% spot x ±15/30% vol shocks
  const spotShocks = [-0.20, -0.10, -0.05, 0, 0.05, 0.10, 0.20];
  const volShocks = [-0.30, -0.15, 0, 0.15, 0.30];
  const daysFwd = 15;
  const scenarioGrid = volShocks.map(dv => {
    const sigShocked = Math.max(market.sigma * (1 + dv), 1e-6);
    return spotShocks.map(ds => {
      const Sshocked = market.S * (1 + ds);
      const Tshocked = Math.max(market.T - daysFwd / 365, 0);
      const shockedLegs = legs.map(l => l.kind === "underlying" ? l : { ...l, T: Tshocked });
      return strategyMark(shockedLegs, Sshocked, market.r, market.q, sigShocked, style) - entryPremium;
    });
  });
  // Index of the centre cell (current scenario: 0% spot, 0% vol)
  const centreSpotIdx = spotShocks.indexOf(0);
  const centreVolIdx = volShocks.indexOf(0);

  const hasBlocker = flags.some(f => f.severity === "danger");
  const expiryDays = Math.round(market.T * 365);

  return (
    <div>
      {/* ========== HEADER ========== */}
      <div className="flex items-start justify-between mb-6">
        <div>
          <div className="text-xs tracking-widest uppercase text-stone-500 mb-1">Review order</div>
          <h2 className="text-3xl" style={{ fontFamily: "Playfair Display, Georgia, serif" }}>{def.label}</h2>
          <div className="text-sm text-stone-600 mt-1 font-mono">
            {market.underlying} · {def.describe(market.S, market.T, stratParams)} · {expiryDays}d to expiry
          </div>
        </div>
        <div className="flex gap-2">
          <button onClick={onBack} className="px-4 py-2 border border-stone-300 text-sm hover:bg-stone-50">← Edit</button>
          <button
            onClick={() => {
              const expiresOn = new Date(Date.now() + market.T * 365 * 86400000).toISOString().slice(0, 10);
              onConfirm({
                stratKey, underlying: market.underlying, S0: market.S, params: stratParams,
                T: market.T, sigma: market.sigma, entryPremium,
                openedAt: new Date().toISOString().slice(0, 10),
                expiresOn,
                orderType, limitPrice: orderType === "limit" ? limitPrice : null,
              });
            }}
            disabled={hasBlocker}
            style={hasBlocker
              ? { backgroundColor: "#E7E5E4", color: "#A8A29E", cursor: "not-allowed" }
              : { backgroundColor: "#2E4A7B", color: "#FFFFFF", fontWeight: 600 }}
            className="px-6 py-2 text-sm tracking-wider uppercase hover:opacity-90">
            Confirm paper trade →
          </button>
        </div>
      </div>

      {/* ========== ROW 1 — PRIMARY METRICS (4 big cards) ========== */}
      <div className="grid grid-cols-4 gap-3 mb-4">
        <div className="bg-white border border-stone-300 p-5">
          <div className="text-[10px] tracking-[0.15em] uppercase text-stone-500 mb-2">Net premium</div>
          <div className="text-3xl font-mono"
               style={{ color: entryPremium >= 0 ? "#9F1239" : "#065F46" }}>
            {fmtSigned(entryPremium)}
          </div>
          <div className="text-xs text-stone-500 mt-1">
            per share · {entryPremium >= 0 ? "debit" : "credit"}
          </div>
        </div>

        <div className="bg-white border border-stone-300 p-5">
          <div className="text-[10px] tracking-[0.15em] uppercase text-stone-500 mb-2">Max profit</div>
          <div className="text-3xl font-mono text-emerald-700">
            {isFinite(maxPnl) ? fmtSigned(maxPnl) : "∞"}
          </div>
          <div className="text-xs text-stone-500 mt-1">per share · capped by structure</div>
        </div>

        <div className="bg-white border border-stone-300 p-5">
          <div className="text-[10px] tracking-[0.15em] uppercase text-stone-500 mb-2">Max loss</div>
          <div className="text-3xl font-mono text-rose-700">
            {isFinite(minPnl) ? fmtSigned(minPnl) : "−∞"}
          </div>
          <div className="text-xs text-stone-500 mt-1">per share · defined risk</div>
        </div>

        <div className="bg-white border border-stone-300 p-5">
          <div className="text-[10px] tracking-[0.15em] uppercase text-stone-500 mb-2">
            Break-even{breakevens.length > 1 ? "s" : ""}
          </div>
          <div className="text-2xl font-mono text-stone-900">
            {breakevens.length === 0
              ? "—"
              : breakevens.map(be => `$${be.toFixed(2)}`).join(" / ")}
          </div>
          <div className="text-xs text-stone-500 mt-1">
            at expiry ·{" "}
            {breakevens.length === 0
              ? "no break-even within grid"
              : breakevens.length === 1 ? "single point" : `${breakevens.length} points`}
          </div>
        </div>
      </div>

      {/* ========== ROW 2 — TOTALS (2 cards scaled by position size) ========== */}
      <div className="grid grid-cols-2 gap-3 mb-6">
        <div className="bg-stone-50 border border-stone-300 p-4">
          <div className="flex items-baseline justify-between">
            <div>
              <div className="text-[10px] tracking-[0.15em] uppercase text-stone-500 mb-1">
                Total position {entryPremium >= 0 ? "cost" : "credit received"}
              </div>
              <div className="text-2xl font-mono"
                   style={{ color: entryPremium >= 0 ? "#9F1239" : "#065F46" }}>
                {fmtSigned(entryPremium * lotSize * 100)}
              </div>
            </div>
            <div className="text-xs text-stone-500 text-right">
              {lotSize} {lotSize === 1 ? "lot" : "lots"} × 100 shares<br />
              = {lotSize * 100} share equivalent
            </div>
          </div>
        </div>
        <div className="bg-stone-50 border border-stone-300 p-4">
          <div className="flex items-baseline justify-between">
            <div>
              <div className="text-[10px] tracking-[0.15em] uppercase text-stone-500 mb-1">
                Initial margin required
              </div>
              <div className="text-2xl font-mono text-stone-900">
                ${(margin.amount * lotSize * 100).toFixed(2)}
              </div>
            </div>
            <div className="text-xs text-stone-500 text-right max-w-[200px]">
              {margin.note}
            </div>
          </div>
        </div>
      </div>

      {/* ========== RISK FLAGS ========== */}
      {flags.length > 0 && (
        <div className="mb-6">
          <div className="text-[10px] tracking-[0.15em] uppercase text-stone-500 mb-2">Risk review</div>
          <div className="space-y-2">
            {flags.map((f, i) => <RiskPill key={i} {...f} />)}
          </div>
          {hasBlocker && (
            <div className="mt-2 text-xs text-rose-700">
              One or more danger-level flags must be resolved before confirming.
            </div>
          )}
        </div>
      )}

      <div className="grid grid-cols-3 gap-6">
        <div className="col-span-2 space-y-4">
          {/* Payoff chart */}
          <div className="bg-white border border-stone-300 p-5">
            <div className="text-[10px] tracking-[0.15em] uppercase text-stone-500 mb-3">Payoff at expiry</div>
            <ResponsiveContainer width="100%" height={240}>
              <AreaChart data={curve} margin={{ top: 10, right: 30, left: 10, bottom: 30 }}>
                <CartesianGrid stroke="#E7E5E4" strokeDasharray="2 2" />
                <XAxis dataKey="S" type="number" domain={["auto", "auto"]}
                  tickFormatter={v => v.toFixed(0)} stroke="#78716C" fontSize={11}
                  tick={{ fontFamily: "JetBrains Mono" }}
                  label={{ value: "Underlying price at expiry (S_T)", position: "insideBottom", offset: -14,
                           fill: "#57534E", fontSize: 11, fontFamily: "Instrument Sans, system-ui, sans-serif" }} />
                <YAxis tickFormatter={v => fmtSigned(v)} stroke="#78716C" fontSize={11}
                  tick={{ fontFamily: "JetBrains Mono" }} width={76}
                  label={{ value: "P&L per share ($)", angle: -90, position: "insideLeft", offset: 12,
                           style: { textAnchor: "middle", fill: "#57534E", fontSize: 11,
                                    fontFamily: "Instrument Sans, system-ui, sans-serif" } }} />
                <Tooltip
                  formatter={(v) => fmtSigned(v)}
                  labelFormatter={v => `S = $${v.toFixed(2)}`}
                  contentStyle={{ fontFamily: "JetBrains Mono", fontSize: 12, border: "1px solid #D6D3D1" }} />
                <ReferenceLine y={0} stroke="#44403C" strokeWidth={1} />
                <ReferenceLine x={market.S} stroke="#2E4A7B" strokeDasharray="3 3" strokeWidth={1} />
                {breakevens.map((be, i) => (
                  <ReferenceLine key={i} x={be} stroke="#A16207" strokeDasharray="2 2" strokeWidth={1} />
                ))}
                <Area type="monotone" dataKey="pnl" stroke="#2E4A7B" fill="#2E4A7B" fillOpacity={0.08} strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
            <div className="flex gap-5 mt-2 text-[11px] text-stone-600 pl-4">
              <span className="flex items-center gap-2">
                <span style={{ display: "inline-block", width: "18px", height: 0, borderTop: "1.5px dashed #2E4A7B" }}/>
                <span>Current spot: <span className="font-mono text-[#2E4A7B]">${market.S.toFixed(2)}</span></span>
              </span>
              {breakevens.length > 0 && (
                <span className="flex items-center gap-2">
                  <span style={{ display: "inline-block", width: "18px", height: 0, borderTop: "1.5px dashed #A16207" }}/>
                  <span>Break-even{breakevens.length > 1 ? "s" : ""}: <span className="font-mono text-[#A16207]">
                    {breakevens.map(be => `$${be.toFixed(2)}`).join(" · ")}
                  </span></span>
                </span>
              )}
            </div>
          </div>

          {/* Scenario grid with centre cell marked */}
          <div className="bg-white border border-stone-300 p-5">
            <div className="flex items-baseline justify-between mb-3">
              <div className="text-[10px] tracking-[0.15em] uppercase text-stone-500">
                Scenario grid: P&L per share · {daysFwd} days forward
              </div>
              <div className="text-xs text-stone-500">spot shock × vol shock</div>
            </div>
            <table className="w-full text-xs" style={{ fontFamily: "JetBrains Mono" }}>
              <thead>
                <tr className="border-b border-stone-300">
                  <th className="py-2 text-left text-stone-500 font-normal w-20">vol ↓ / spot →</th>
                  {spotShocks.map((ds, j) => (
                    <th key={ds}
                        className={`py-2 text-right font-normal ${j === centreSpotIdx ? "text-[#2E4A7B] font-semibold" : "text-stone-500"}`}>
                      {(ds >= 0 ? "+" : "") + (ds * 100).toFixed(0) + "%"}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(() => {
                  const maxAbs = Math.max(...scenarioGrid.flat().map(Math.abs), 1);
                  return volShocks.map((dv, i) => (
                    <tr key={dv} className="border-b border-stone-100 last:border-0">
                      <td className={`py-2 ${i === centreVolIdx ? "text-[#2E4A7B] font-semibold" : "text-stone-500"}`}>
                        {(dv >= 0 ? "+" : "") + (dv * 100).toFixed(0) + "%"}
                      </td>
                      {scenarioGrid[i].map((v, j) => {
                        const isCentre = i === centreVolIdx && j === centreSpotIdx;
                        const intensity = Math.min(Math.abs(v) / maxAbs, 1);
                        const bg = isCentre
                          ? "#EEF2F8"
                          : Math.abs(v) < 0.01
                            ? "transparent"
                            : v > 0
                              ? `rgba(5, 150, 105, ${0.08 + intensity * 0.3})`
                              : `rgba(190, 18, 60, ${0.08 + intensity * 0.3})`;
                        const textColor = isCentre
                          ? "#2E4A7B"
                          : Math.abs(v) < 0.01
                            ? "#57534E"
                            : v > 0 ? "#065F46" : "#9F1239";
                        return (
                          <td key={j}
                              className="py-2 px-2 text-right tabular-nums"
                              style={{
                                backgroundColor: bg,
                                color: textColor,
                                fontWeight: isCentre ? 700 : 400,
                                outline: isCentre ? "1.5px solid #2E4A7B" : "none",
                                outlineOffset: isCentre ? "-1.5px" : "0",
                              }}>
                            {(v >= 0 ? "+" : "−") + "$" + Math.abs(v).toFixed(2)}
                          </td>
                        );
                      })}
                    </tr>
                  ));
                })()}
              </tbody>
            </table>
            <div className="text-[10px] text-stone-500 mt-2 italic">
              Highlighted cell: current scenario (no spot shock, no vol shock, 15 days forward from today).
            </div>
          </div>
        </div>

        <div className="space-y-4">
          {/* Order type */}
          <div className="bg-white border border-stone-300 p-5">
            <div className="text-[10px] tracking-[0.15em] uppercase text-stone-500 mb-3">Order type</div>
            <div className="flex gap-0 border border-stone-300 mb-3">
              <button onClick={() => setOrderType("market")}
                style={orderType === "market"
                  ? { backgroundColor: "#2E4A7B", color: "#FFFFFF", fontWeight: 600 }
                  : { backgroundColor: "#FFFFFF", color: "#44403c" }}
                className="flex-1 px-3 py-1.5 text-xs uppercase tracking-wider border-r border-stone-300 hover:opacity-90">
                Market
              </button>
              <button onClick={() => setOrderType("limit")}
                style={orderType === "limit"
                  ? { backgroundColor: "#2E4A7B", color: "#FFFFFF", fontWeight: 600 }
                  : { backgroundColor: "#FFFFFF", color: "#44403c" }}
                className="flex-1 px-3 py-1.5 text-xs uppercase tracking-wider hover:opacity-90">
                Limit
              </button>
            </div>
            {orderType === "limit" ? (
              <div>
                <div className="text-[10px] uppercase tracking-wider text-stone-500 mb-1">Limit price ($/share)</div>
                <NumberInput value={limitPrice || +entryPremium.toFixed(2)} step={0.01}
                  onChange={v => setLimitPrice(v)} />
                <div className="text-[10px] text-stone-500 mt-1">
                  Order fills only at or better than this price.
                </div>
              </div>
            ) : (
              <div className="text-xs text-stone-500">
                Fills at the next available market price. Simpler but no price protection.
              </div>
            )}
          </div>

          {/* Dollar sensitivities — kept for the immediate-risk view around current spot */}
          <div className="bg-white border border-stone-300 p-5">
            <div className="text-[10px] tracking-[0.15em] uppercase text-stone-500 mb-3">
              Dollar P&L sensitivities<br/>
              <span className="text-stone-400 normal-case">per contract (×100)</span>
            </div>
            <div className="space-y-2 text-sm">
              <Sens label="+1% underlying" value={greeks.delta * market.S * 0.01 * 100 + 0.5 * greeks.gamma * (market.S * 0.01) ** 2 * 100} />
              <Sens label="−1% underlying" value={-greeks.delta * market.S * 0.01 * 100 + 0.5 * greeks.gamma * (market.S * 0.01) ** 2 * 100} />
              <Sens label="+1 vol point" value={greeks.vega * 0.01 * 100} />
              <Sens label="+1 day pass" value={greeks.theta / 365 * 100} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function Sens({ label, value }) {
  const pos = value >= 0;
  return (
    <div className="border border-stone-200 px-3 py-2">
      <div className="text-[10px] uppercase tracking-wider text-stone-500">{label}</div>
      <div className={`text-lg font-mono ${pos ? "text-emerald-700" : "text-rose-700"}`}>
        {fmtSigned(value)}
      </div>
    </div>
  );
}

// =============================================================================
// Vol & Greeks Lab (stretch feature #2)
// Strategy-driven stress-testing sandbox: takes the user's selected strategy
// from the Builder and lets them vary market conditions via sliders.
// Strikes and structure are read-only; only market parameters change.
// =============================================================================

function Lab({ market, stratKey, stratParams, onGoToLibrary }) {
  // Hooks are called unconditionally here (rules of hooks)
  // State for market-parameter sliders. Defaults are the Builder's current values
  // when a strategy exists; otherwise benign fallbacks (these values are unused
  // in the empty-state render).
  const fallbackSpot = 100;
  const [S, setS] = useState(market?.S ?? fallbackSpot);
  const [sigma, setSigma] = useState(market?.sigma ?? 0.25);
  const [daysToExpiry, setDaysToExpiry] = useState(Math.round((market?.T ?? 30/365) * 365));
  const [r, setR] = useState(market?.r ?? 0.05);
  const [q, setQ] = useState(market?.q ?? 0.015);

  const T = Math.max(daysToExpiry / 365, 1e-6);
  const def = stratKey ? STRAT_DEFS[stratKey] : null;

  // Derive legs from the Builder's anchor spot; strikes stay fixed as user drags spot.
  // Use useMemo so legsAtSpot gets a stable reference when inputs don't change,
  // making downstream useMemo blocks actually memoise.
  const legsAtSpot = useMemo(() => {
    if (!def || !market) return [];
    const legs = def.legs(market.S, T, stratParams);
    return legs.map(l => l.kind === "underlying" ? l : { ...l, T });
  }, [def, market, stratParams, T]);

  const entryPremium = useMemo(() => {
    if (!def || !market) return 0;
    const builderLegs = def.legs(market.S, market.T, stratParams);
    return strategyMark(builderLegs, market.S, market.r, market.q, market.sigma, market.exerciseStyle || "european");
  }, [def, market, stratParams]);

  const curve = useMemo(
    () => legsAtSpot.length ? payoffCurve(legsAtSpot, S, entryPremium, 0.3, 81) : [],
    [legsAtSpot, S, entryPremium]
  );

  const greekCurve = useMemo(() => {
    if (!legsAtSpot.length) return [];
    const style = market?.exerciseStyle || "european";
    const lo = S * 0.7, hi = S * 1.3;
    const raw = [];
    for (let i = 0; i < 61; i++) {
      const s = lo + (hi - lo) * i / 60;
      const g = strategyGreeks(legsAtSpot, s, r, q, sigma, style);
      raw.push({ S: s, delta: g.delta, gamma: g.gamma, vega: g.vega, theta: g.theta });
    }
    const maxAbs = {
      delta: Math.max(...raw.map(d => Math.abs(d.delta)), 1e-6),
      gamma: Math.max(...raw.map(d => Math.abs(d.gamma)), 1e-6),
      vega:  Math.max(...raw.map(d => Math.abs(d.vega)),  1e-6),
      theta: Math.max(...raw.map(d => Math.abs(d.theta)), 1e-6),
    };
    return raw.map(d => ({
      S: d.S,
      delta: d.delta / maxAbs.delta,
      gamma: d.gamma / maxAbs.gamma,
      vega:  d.vega  / maxAbs.vega,
      theta: d.theta / maxAbs.theta,
      rawDelta: d.delta, rawGamma: d.gamma, rawVega: d.vega, rawTheta: d.theta,
    }));
  }, [legsAtSpot, r, q, sigma, S, market]);

  // ---- Now that all hooks are called, we can conditionally return ----
  if (!stratKey) {
    return (
      <div>
        <div className="mb-6">
          <div className="text-xs tracking-widest uppercase text-stone-500 mb-1">Interactive</div>
          <h2 className="text-3xl" style={{ fontFamily: "Playfair Display, Georgia, serif" }}>Volatility & Greeks Lab</h2>
          <div className="text-sm text-stone-500 mt-1">
            Stress-test your selected strategy under different market conditions.
          </div>
        </div>
        <div className="bg-white border border-stone-300 p-16 text-center">
          <div className="text-stone-600 mb-4 text-lg" style={{ fontFamily: "Playfair Display, Georgia, serif" }}>
            No strategy selected
          </div>
          <div className="text-sm text-stone-500 mb-6 max-w-md mx-auto leading-relaxed">
            Pick a strategy from the Library first. The Lab will then let you vary spot, volatility,
            time to expiry, and carry parameters to see how the position responds.
          </div>
          <button onClick={onGoToLibrary}
            style={{ backgroundColor: "#2E4A7B", color: "#FFFFFF", fontWeight: 600 }}
            className="px-6 py-3 text-sm tracking-wider uppercase hover:opacity-90">
            Go to Library →
          </button>
        </div>
      </div>
    );
  }

  // ---- Strategy selected: compute instantaneous quantities ----
  const labStyle = market?.exerciseStyle || "european";
  const currentMark = strategyMark(legsAtSpot, S, r, q, sigma, labStyle);
  const currentPnl = currentMark - entryPremium;
  const greeks = strategyGreeks(legsAtSpot, S, r, q, sigma, labStyle);
  const labEEPremium = earlyExercisePremium(legsAtSpot, S, r, q, sigma);
  const breakevens = findBreakevens(curve);

  // Compare with tolerance to avoid float-comparison artefacts from slider drags
  const near = (a, b, eps = 1e-6) => Math.abs(a - b) < eps;
  const builderDays = Math.round(market.T * 365);
  const isModified =
    !near(S, market.S) ||
    !near(sigma, market.sigma) ||
    daysToExpiry !== builderDays ||
    !near(r, market.r) ||
    !near(q, market.q);

  const resetToBuilder = () => {
    setS(market.S);
    setSigma(market.sigma);
    setDaysToExpiry(builderDays);
    setR(market.r);
    setQ(market.q);
  };

  return (
    <div>
      <div className="flex items-start justify-between mb-6">
        <div>
          <div className="text-xs tracking-widest uppercase text-stone-500 mb-1">Interactive</div>
          <h2 className="text-3xl" style={{ fontFamily: "Playfair Display, Georgia, serif" }}>
            Volatility & Greeks Lab
          </h2>
          <div className="text-sm text-stone-600 mt-1">
            <span className="font-semibold">{def.label}</span> on {market.underlying} ·{" "}
            <span className="font-mono">{def.describe(market.S, market.T, stratParams)}</span>
          </div>
          <div className="text-xs text-stone-500 mt-1">
            Strikes and structure are fixed. Move sliders to stress-test market conditions.
          </div>
        </div>
        {isModified && (
          <button onClick={resetToBuilder}
            className="px-4 py-2 border border-[#2E4A7B] text-[#2E4A7B] text-xs tracking-wider uppercase hover:bg-stone-50">
            Reset to Builder values
          </button>
        )}
      </div>

      {/* Exercise-style banner: explains why curves do or don't move on toggle */}
      <div className="mb-6 border border-stone-300 bg-white px-5 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-[10px] tracking-[0.15em] uppercase text-stone-500">Pricing under</span>
          <span className="font-semibold text-[#2E4A7B]">
            {labStyle === "american" ? "American exercise (binomial tree)" : "European exercise (closed-form BSM)"}
          </span>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <span className="text-stone-500">Early-exercise premium:</span>
          <span className="font-mono text-stone-800">${labEEPremium.toFixed(4)}</span>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-6">
        {/* LEFT: sliders + live Greeks */}
        <div className="col-span-1 space-y-4">
          <div className="bg-white border border-stone-300 p-5 space-y-5">
            <div className="text-[10px] tracking-[0.15em] uppercase text-stone-500 mb-1">Market parameters</div>
            <Slider label="Spot (S)" value={S} min={market.S * 0.5} max={market.S * 1.5} step={0.5}
              onChange={setS} fmt={v => "$" + v.toFixed(2)} />
            <Slider label="Volatility (σ)" value={sigma} min={0.05} max={1.0} step={0.01}
              onChange={setSigma} fmt={v => (v * 100).toFixed(1) + "%"} />
            <Slider label="Days to expiry" value={daysToExpiry} min={1} max={builderDays} step={1}
              onChange={setDaysToExpiry} fmt={v => v.toFixed(0) + "d"} />
            <Slider label="Rate (r)" value={r} min={-0.02} max={0.10} step={0.0025}
              onChange={setR} fmt={v => (v * 100).toFixed(2) + "%"} />
            <Slider label="Div yield (q)" value={q} min={0} max={0.08} step={0.0025}
              onChange={setQ} fmt={v => (v * 100).toFixed(2) + "%"} />
          </div>

          <div className="bg-white border border-stone-300 p-5">
            <div className="text-[10px] tracking-[0.15em] uppercase text-stone-500 mb-3">
              Live strategy Greeks
            </div>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between py-1.5 border-b border-stone-100">
                <span className="text-stone-500">Δ Delta</span>
                <span className="font-mono tabular-nums">{greeks.delta.toFixed(4)}</span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-stone-100">
                <span className="text-stone-500">Γ Gamma</span>
                <span className="font-mono tabular-nums">{greeks.gamma.toFixed(5)}</span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-stone-100">
                <span className="text-stone-500">ν Vega</span>
                <span className="font-mono tabular-nums">{(greeks.vega / 100).toFixed(4)}
                  <span className="text-[10px] text-stone-400 ml-1">/vol pt</span>
                </span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-stone-100">
                <span className="text-stone-500">Θ Theta</span>
                <span className="font-mono tabular-nums">{(greeks.theta / 365).toFixed(4)}
                  <span className="text-[10px] text-stone-400 ml-1">/day</span>
                </span>
              </div>
              <div className="flex justify-between py-1.5">
                <span className="text-stone-500">ρ Rho</span>
                <span className="font-mono tabular-nums">{(greeks.rho / 100).toFixed(4)}
                  <span className="text-[10px] text-stone-400 ml-1">/1bp</span>
                </span>
              </div>
            </div>
          </div>

          <div className="bg-stone-50 border border-stone-300 p-4">
            <div className="text-[10px] tracking-[0.15em] uppercase text-stone-500 mb-2">Current position P&L</div>
            <div className="text-2xl font-mono"
                 style={{ color: currentPnl >= 0 ? "#065F46" : "#9F1239" }}>
              {fmtSigned(currentPnl)}
            </div>
            <div className="text-xs text-stone-500 mt-1">
              per share · vs. entry premium of <span className="font-mono">{fmtSigned(entryPremium)}</span>
            </div>
          </div>
        </div>

        {/* RIGHT: charts */}
        <div className="col-span-2 space-y-4">
          {/* Payoff curve at current slider settings */}
          <div className="bg-white border border-stone-300 p-5">
            <div className="text-[10px] tracking-[0.15em] uppercase text-stone-500 mb-3">
              Payoff at expiry · {daysToExpiry} day{daysToExpiry === 1 ? "" : "s"} from now
            </div>
            <ResponsiveContainer width="100%" height={240}>
              <AreaChart data={curve} margin={{ top: 10, right: 30, left: 10, bottom: 30 }}>
                <CartesianGrid stroke="#E7E5E4" strokeDasharray="2 2" />
                <XAxis dataKey="S" type="number" domain={["auto", "auto"]}
                  tickFormatter={v => v.toFixed(0)} stroke="#78716C" fontSize={11}
                  tick={{ fontFamily: "JetBrains Mono" }}
                  label={{ value: "Underlying price at expiry (S_T)", position: "insideBottom", offset: -14,
                           fill: "#57534E", fontSize: 11, fontFamily: "Instrument Sans, system-ui, sans-serif" }} />
                <YAxis tickFormatter={v => fmtSigned(v)} stroke="#78716C" fontSize={11}
                  tick={{ fontFamily: "JetBrains Mono" }} width={76}
                  label={{ value: "P&L per share ($)", angle: -90, position: "insideLeft", offset: 12,
                           style: { textAnchor: "middle", fill: "#57534E", fontSize: 11,
                                    fontFamily: "Instrument Sans, system-ui, sans-serif" } }} />
                <Tooltip formatter={(v) => fmtSigned(v)}
                  labelFormatter={v => `S = $${v.toFixed(2)}`}
                  contentStyle={{ fontFamily: "JetBrains Mono", fontSize: 12, border: "1px solid #D6D3D1" }} />
                <ReferenceLine y={0} stroke="#44403C" strokeWidth={1} />
                <ReferenceLine x={S} stroke="#2E4A7B" strokeDasharray="3 3" strokeWidth={1} />
                {breakevens.map((be, i) => (
                  <ReferenceLine key={i} x={be} stroke="#A16207" strokeDasharray="2 2" strokeWidth={1} />
                ))}
                <Area type="monotone" dataKey="pnl" stroke="#2E4A7B" fill="#2E4A7B" fillOpacity={0.08} strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
            <div className="flex gap-5 mt-2 text-[11px] text-stone-600 pl-4">
              <span className="flex items-center gap-2">
                <span style={{ display: "inline-block", width: "18px", height: 0, borderTop: "1.5px dashed #2E4A7B" }}/>
                <span>Current spot: <span className="font-mono text-[#2E4A7B]">${S.toFixed(2)}</span></span>
              </span>
              {breakevens.length > 0 && (
                <span className="flex items-center gap-2">
                  <span style={{ display: "inline-block", width: "18px", height: 0, borderTop: "1.5px dashed #A16207" }}/>
                  <span>Break-even{breakevens.length > 1 ? "s" : ""}: <span className="font-mono text-[#A16207]">
                    {breakevens.map(be => `$${be.toFixed(2)}`).join(" · ")}
                  </span></span>
                </span>
              )}
            </div>
          </div>

          {/* Greeks across spot range */}
          <div className="bg-white border border-stone-300 p-5">
            <div className="text-[10px] tracking-[0.15em] uppercase text-stone-500 mb-3">
              Strategy Greeks across spot range
            </div>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={greekCurve} margin={{ top: 10, right: 20, left: 0, bottom: 28 }}>
                <CartesianGrid stroke="#E7E5E4" strokeDasharray="2 2" />
                <XAxis dataKey="S" type="number" domain={["auto", "auto"]}
                  tickFormatter={v => v.toFixed(0)} stroke="#78716C" fontSize={11}
                  tick={{ fontFamily: "JetBrains Mono" }}
                  label={{ value: "Underlying spot (S)", position: "insideBottom", offset: -14,
                           fill: "#57534E", fontSize: 11, fontFamily: "Instrument Sans, system-ui, sans-serif" }} />
                <YAxis stroke="#78716C" fontSize={11} width={76}
                  tick={{ fontFamily: "JetBrains Mono" }} domain={[-1.1, 1.1]}
                  label={{ value: "Normalised Greek value", angle: -90, position: "insideLeft", offset: 12,
                           style: { textAnchor: "middle", fill: "#57534E", fontSize: 11,
                                    fontFamily: "Instrument Sans, system-ui, sans-serif" } }} />
                <Tooltip
                  contentStyle={{ fontFamily: "JetBrains Mono", fontSize: 11, border: "1px solid #D6D3D1" }}
                  labelFormatter={v => `S = $${v.toFixed(2)}`}
                  formatter={(value, name, props) => {
                    const display = {
                      delta: props.payload.rawDelta,
                      gamma: props.payload.rawGamma,
                      vega:  props.payload.rawVega / 100,
                      theta: props.payload.rawTheta / 365,
                    }[name];
                    const suffix = { vega: " /vol pt", theta: " /day" }[name] || "";
                    return [display !== undefined
                      ? display.toFixed(name === "gamma" ? 5 : 4) + suffix
                      : value.toFixed(4), name];
                  }} />
                <ReferenceLine x={S} stroke="#2E4A7B" strokeDasharray="3 3" strokeWidth={1} />
                <ReferenceLine y={0} stroke="#44403C" strokeWidth={0.8} />
                <Line type="monotone" name="delta" dataKey="delta" stroke="#2E4A7B" strokeWidth={2} dot={false} />
                <Line type="monotone" name="gamma" dataKey="gamma" stroke="#059669" strokeWidth={1.5} dot={false} />
                <Line type="monotone" name="vega" dataKey="vega" stroke="#A16207" strokeWidth={1.5} dot={false} />
                <Line type="monotone" name="theta" dataKey="theta" stroke="#BE123C" strokeWidth={1.5} dot={false} />
              </LineChart>
            </ResponsiveContainer>
            <div className="flex gap-5 mt-2 text-[10px] text-stone-500 justify-center">
              <span><span className="inline-block w-3 h-0.5 bg-[#2E4A7B] mr-1 align-middle"/>Delta</span>
              <span><span className="inline-block w-3 h-0.5 bg-emerald-700 mr-1 align-middle"/>Gamma</span>
              <span><span className="inline-block w-3 h-0.5 bg-amber-700 mr-1 align-middle"/>Vega</span>
              <span><span className="inline-block w-3 h-0.5 bg-rose-700 mr-1 align-middle"/>Theta</span>
            </div>
            <div className="text-[10px] text-stone-400 text-center mt-1 italic">
              Each curve normalised to its own ±1 range across the grid. Hover to see actual values in conventional units.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function Slider({ label, value, min, max, step, onChange, fmt }) {
  return (
    <div>
      <div className="flex items-baseline justify-between mb-1">
        <span className="text-[10px] tracking-[0.15em] uppercase text-stone-500">{label}</span>
        <span className="text-sm font-mono text-[#2E4A7B] tabular-nums">{fmt(value)}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(+e.target.value)}
        className="w-full accent-[#2E4A7B]" />
    </div>
  );
}

// =============================================================================
// Recommender (stretch feature #1)
// =============================================================================

function Recommender({ onPick, market, profile }) {
  const [directional, setDirectional] = useState("bullish");
  const [vol, setVol] = useState("neutral");

  const cflScore = computeCapacityScore(profile);
  const gamma = cflToRiskAversion(cflScore);
  const recs = recommendEU(directional, vol, cflScore, market);

  const riskPosture = gamma >= 4.5 ? "strongly risk-averse"
                    : gamma >= 3 ? "risk-averse"
                    : gamma >= 2 ? "moderate"
                    : "risk-tolerant";

  return (
    <div>
      <div className="mb-6">
        <div className="text-xs tracking-widest uppercase text-stone-500 mb-1">Guided</div>
        <h2 className="text-3xl" style={{ fontFamily: "Playfair Display, Georgia, serif" }}>Strategy Recommender</h2>
        <div className="text-sm text-stone-500 mt-1">
          Describe your market view. Strategies are ranked by expected utility under that view, using the risk
          aversion implied by your Capacity-for-Loss profile.
        </div>
      </div>

      <div className="grid grid-cols-3 gap-6">
        <div className="bg-white border border-stone-300 p-6 space-y-5">
          <Field label="Directional view">
            <RadioGroup value={directional} onChange={setDirectional}
              options={[{ v: "bullish", l: "Bullish" }, { v: "neutral", l: "Neutral" }, { v: "bearish", l: "Bearish" }]} />
          </Field>
          <Field label="Volatility view">
            <RadioGroup value={vol} onChange={setVol}
              options={[{ v: "up", l: "Up" }, { v: "neutral", l: "Neutral" }, { v: "down", l: "Down" }]} />
          </Field>
          <div className="pt-3 border-t border-stone-200">
            <div className="text-[10px] tracking-[0.15em] uppercase text-stone-500 mb-1">Risk aversion (from CFL)</div>
            <div className="flex items-baseline gap-2">
              <span className="font-mono text-lg text-[#2E4A7B]">γ = {gamma.toFixed(1)}</span>
              <span className="text-xs text-stone-500">{riskPosture}</span>
            </div>
            <div className="text-[10px] text-stone-500 mt-1 leading-relaxed">
              Derived from your onboarding Capacity-for-Loss score ({cflScore}/10), not set here. Update it via Profile.
            </div>
          </div>
          <div className="pt-3 border-t border-stone-200 text-[10px] text-stone-500 leading-relaxed">
            Ranking by expected CRRA utility under a view-implied lognormal
            (drift ±0.5σ√T, vol spread ×1.3/×0.7). Auditable, no ML.
          </div>
        </div>

        <div className="col-span-2 space-y-3">
          {recs.map((r, i) => (
            <div key={r.key} className="bg-white border border-stone-300 p-5">
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-baseline gap-3">
                  <span className="text-xs tracking-widest uppercase text-stone-400">#{i + 1}</span>
                  <div>
                    <div className="text-xl font-semibold">{r.def.label}</div>
                    <div className="text-xs text-stone-500">{r.def.tagline}</div>
                  </div>
                </div>
                <button onClick={() => onPick(r.key)}
                  className="px-3 py-1 text-xs uppercase tracking-wider border border-[#2E4A7B] text-[#2E4A7B] hover:bg-[#2E4A7B] hover:text-white">
                  Build →
                </button>
              </div>
              <div className="grid grid-cols-3 gap-3 border-t border-dotted border-stone-300 pt-3">
                <div>
                  <div className="text-[10px] uppercase tracking-wider text-stone-500">Certainty-equiv. gain</div>
                  <div className={`font-mono text-sm ${r.ceGain >= 0 ? "text-emerald-700" : "text-rose-700"}`}>
                    {r.ceGain >= 0 ? "+" : "−"}${Math.abs(Math.round(r.ceGain)).toLocaleString()}
                  </div>
                </div>
                <div>
                  <div className="text-[10px] uppercase tracking-wider text-stone-500">Expected P&L</div>
                  <div className={`font-mono text-sm ${r.expPnl >= 0 ? "text-emerald-700" : "text-rose-700"}`}>
                    {r.expPnl >= 0 ? "+" : "−"}${Math.abs(Math.round(r.expPnl)).toLocaleString()}
                  </div>
                </div>
                <div>
                  <div className="text-[10px] uppercase tracking-wider text-stone-500">Prob. of profit</div>
                  <div className="font-mono text-sm text-stone-800">{(r.pProfit * 100).toFixed(0)}%</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// =============================================================================
// Dashboard: portfolio view with net Greeks
// =============================================================================

function Dashboard({ positions, setPositions, market }) {
  const today = new Date();
  const [confirmingClose, setConfirmingClose] = useState(null);  // position id pending close confirmation

  // Split into open and closed
  const openPositions = positions.filter(p => !p.closedAt);
  const closedPositions = positions.filter(p => p.closedAt);

  // Enrich open positions with live mark / Greeks
  const enrichedOpen = openPositions.map(pos => {
    const def = STRAT_DEFS[pos.stratKey];
    const T = Math.max(pos.T - 0, 0);
    const legs = def.legs(pos.S0, T, pos.params);
    const currentLegs = legs.map(l => l.kind === "underlying" ? l : { ...l, T });
    const currentValue = strategyMark(currentLegs, market.S, market.r, market.q, market.sigma, market.exerciseStyle || "european");
    const pnl = currentValue - pos.entryPremium;
    const greeks = strategyGreeks(currentLegs, market.S, market.r, market.q, market.sigma, market.exerciseStyle || "european");
    const daysToExpiry = pos.expiresOn
      ? Math.max(0, Math.round((new Date(pos.expiresOn) - today) / 86400000))
      : Math.round(pos.T * 365);
    const lots = pos.lots || 1;
    return { ...pos, def, currentValue, pnl, greeks, daysToExpiry, lots };
  });

  // Enrich closed positions with their frozen close value (no live Greeks)
  const enrichedClosed = closedPositions.map(pos => {
    const def = STRAT_DEFS[pos.stratKey];
    return {
      ...pos,
      def,
      lots: pos.lots || 1,
      realizedPnl: pos.realizedPnl ?? (pos.closingValue - pos.entryPremium),
    };
  });

  // Portfolio-level metrics only on open positions
  const portfolioGreeks = enrichedOpen.reduce(
    (acc, p) => ({
      delta: acc.delta + p.greeks.delta * p.lots,
      gamma: acc.gamma + p.greeks.gamma * p.lots,
      vega: acc.vega + p.greeks.vega * p.lots,
      theta: acc.theta + p.greeks.theta * p.lots,
    }),
    { delta: 0, gamma: 0, vega: 0, theta: 0 }
  );
  const openUnrealizedPnl = enrichedOpen.reduce((s, p) => s + p.pnl * p.lots, 0);
  const realizedPnl = enrichedClosed.reduce((s, p) => s + p.realizedPnl * p.lots, 0);

  // Concentration flags only use open positions
  const concentrationFlags = [];
  if (enrichedOpen.length >= 2) {
    const underlyingCounts = enrichedOpen.reduce((acc, p) => {
      acc[p.underlying] = (acc[p.underlying] || 0) + 1;
      return acc;
    }, {});
    Object.entries(underlyingCounts).forEach(([sym, count]) => {
      const pct = count / enrichedOpen.length;
      if (pct > 0.5) {
        concentrationFlags.push({
          severity: "warn",
          code: "UNDERLYING_CONCENTRATION",
          message: `${(pct * 100).toFixed(0)}% of open positions (${count}/${enrichedOpen.length}) are on ${sym}. A single-name event would affect the whole book.`,
        });
      }
    });
    const expiryWeeks = enrichedOpen.reduce((acc, p) => {
      if (!p.expiresOn) return acc;
      const wk = p.expiresOn.slice(0, 7) + "-wk" + Math.ceil(parseInt(p.expiresOn.slice(8, 10)) / 7);
      acc[wk] = (acc[wk] || 0) + 1;
      return acc;
    }, {});
    Object.entries(expiryWeeks).forEach(([wk, count]) => {
      const pct = count / enrichedOpen.length;
      if (pct > 0.5 && enrichedOpen.length >= 2) {
        concentrationFlags.push({
          severity: "warn",
          code: "EXPIRY_CONCENTRATION",
          message: `${(pct * 100).toFixed(0)}% of positions share one expiry window. Consider laddering expiries to reduce event risk.`,
        });
      }
    });
    const nearExpiry = enrichedOpen.filter(p => p.daysToExpiry <= 7 && p.daysToExpiry > 0);
    if (nearExpiry.length > 0) {
      concentrationFlags.push({
        severity: "info",
        code: "NEAR_EXPIRY",
        message: `${nearExpiry.length} position${nearExpiry.length > 1 ? "s" : ""} expiring within 7 days: gamma and assignment risk peak here.`,
      });
    }
  }

  const confirmClosePosition = (p) => {
    const closingDate = new Date().toISOString().slice(0, 10);
    setPositions(positions.map(pos =>
      pos.id === p.id
        ? { ...pos, closedAt: closingDate, closingValue: p.currentValue, realizedPnl: p.pnl }
        : pos
    ));
    setConfirmingClose(null);
  };

  return (
    <div>
      <div className="mb-6">
        <div className="text-xs tracking-widest uppercase text-stone-500 mb-1">Portfolio</div>
        <h2 className="text-3xl" style={{ fontFamily: "Playfair Display, Georgia, serif" }}>Positions Dashboard</h2>
      </div>

      {/* Top stats row: P&L always visible; Greeks hidden when no open positions (they would all be zero) */}
      <div className={`grid gap-3 mb-6 ${enrichedOpen.length > 0 ? "grid-cols-6" : "grid-cols-2"}`}>
        <StatBlock label="Unrealised P&L" value={fmtSigned(openUnrealizedPnl * 100)}
                   sublabel={`${enrichedOpen.length} open ${enrichedOpen.length === 1 ? "position" : "positions"}`}
                   positive={openUnrealizedPnl > 0} negative={openUnrealizedPnl < 0} />
        <StatBlock label="Realised P&L" value={fmtSigned(realizedPnl * 100)}
                   sublabel={`${enrichedClosed.length} closed ${enrichedClosed.length === 1 ? "position" : "positions"}`}
                   positive={realizedPnl > 0} negative={realizedPnl < 0} />
        {enrichedOpen.length > 0 && (
          <>
            <StatBlock label="Net Δ" value={portfolioGreeks.delta.toFixed(3)} sublabel="per share × lots" />
            <StatBlock label="Net Γ" value={portfolioGreeks.gamma.toFixed(4)} sublabel="per share × lots" />
            <StatBlock label="Net ν" value={(portfolioGreeks.vega / 100).toFixed(3)} sublabel="per vol pt" />
            <StatBlock label="Net Θ" value={(portfolioGreeks.theta / 365).toFixed(3)} sublabel="per day" />
          </>
        )}
      </div>

      {concentrationFlags.length > 0 && (
        <div className="mb-6">
          <div className="text-[10px] tracking-[0.15em] uppercase text-stone-500 mb-2">Portfolio concentration review</div>
          <div className="space-y-2">
            {concentrationFlags.map((f, i) => <RiskPill key={i} {...f} />)}
          </div>
        </div>
      )}

      {/* ========== OPEN POSITIONS ========== */}
      <div className="mb-8">
        <div className="flex items-baseline justify-between mb-3">
          <div className="text-[10px] tracking-[0.15em] uppercase text-stone-500">
            Open positions ({enrichedOpen.length})
          </div>
        </div>
        {enrichedOpen.length === 0 ? (
          <div className="bg-white border border-stone-300 p-12 text-center text-stone-500">
            {enrichedClosed.length === 0
              ? "No open positions. Pick a strategy from the Library to get started."
              : `No open positions. ${enrichedClosed.length} closed ${enrichedClosed.length === 1 ? "trade" : "trades"} below.`}
          </div>
        ) : (
          <div className="bg-white border border-stone-300">
            <table className="w-full text-sm">
              <thead className="bg-stone-50 border-b border-stone-300">
                <tr className="text-left text-[10px] tracking-[0.15em] uppercase text-stone-500">
                  <th className="px-4 py-3">Strategy</th>
                  <th className="px-4 py-3">Symbol</th>
                  <th className="px-4 py-3 text-right">Lots</th>
                  <th className="px-4 py-3">Opened</th>
                  <th className="px-4 py-3 text-right">DTE</th>
                  <th className="px-4 py-3 text-right">Entry ($)</th>
                  <th className="px-4 py-3 text-right">Mark ($)</th>
                  <th className="px-4 py-3 text-right">P&L ($)</th>
                  <th className="px-4 py-3 text-right">Δ</th>
                  <th className="px-4 py-3 text-right">ν</th>
                  <th className="px-4 py-3 text-right">Θ/day</th>
                  <th className="px-4 py-3"></th>
                </tr>
              </thead>
              <tbody style={{ fontFamily: "JetBrains Mono", fontSize: 13 }}>
                {enrichedOpen.map(p => (
                  <React.Fragment key={p.id}>
                    <tr className="border-b border-stone-100 last:border-0">
                      <td className="px-4 py-3">{p.def.label}</td>
                      <td className="px-4 py-3">{p.underlying}</td>
                      <td className="px-4 py-3 text-right">{p.lots}</td>
                      <td className="px-4 py-3 text-stone-500">{p.openedAt}</td>
                      <td className={`px-4 py-3 text-right ${p.daysToExpiry <= 7 ? "text-amber-700 font-semibold" : "text-stone-700"}`}>
                        {p.daysToExpiry}d
                      </td>
                      <td className="px-4 py-3 text-right">{fmtSigned(p.entryPremium * p.lots * 100)}</td>
                      <td className="px-4 py-3 text-right">{fmtSigned(p.currentValue * p.lots * 100)}</td>
                      <td className={`px-4 py-3 text-right font-semibold ${p.pnl >= 0 ? "text-emerald-700" : "text-rose-700"}`}>
                        {fmtSigned(p.pnl * p.lots * 100)}
                      </td>
                      <td className="px-4 py-3 text-right">{p.greeks.delta.toFixed(3)}</td>
                      <td className="px-4 py-3 text-right">{(p.greeks.vega / 100).toFixed(3)}</td>
                      <td className="px-4 py-3 text-right">{(p.greeks.theta / 365).toFixed(3)}</td>
                      <td className="px-4 py-3 text-right">
                        {confirmingClose === p.id ? null : (
                          <button onClick={() => setConfirmingClose(p.id)}
                            className="text-[10px] uppercase tracking-wider text-stone-500 hover:text-[#2E4A7B] border border-stone-300 px-2 py-1 hover:border-[#2E4A7B]">
                            Close
                          </button>
                        )}
                      </td>
                    </tr>
                    {confirmingClose === p.id && (
                      <tr className="border-b border-stone-100">
                        <td colSpan={12} className="px-4 py-3"
                            style={{ backgroundColor: "#FAF7EF", borderLeft: "3px solid #C99B33" }}>
                          <div className="flex items-center justify-between gap-4">
                            <div className="text-sm text-stone-800">
                              <span className="font-semibold">Close {p.def.label} on {p.underlying}?</span>{" "}
                              Mark-to-market at <span className="font-mono">${p.currentValue.toFixed(2)}</span>/share.
                              Realised P&L:{" "}
                              <span className="font-mono font-semibold" style={{ color: p.pnl >= 0 ? "#047857" : "#b91c1c" }}>
                                {fmtSigned(p.pnl * p.lots * 100)}
                              </span>
                              {" "}({p.lots} lot{p.lots === 1 ? "" : "s"} × 100 shares)
                            </div>
                            <div className="flex gap-2 shrink-0">
                              <button onClick={() => setConfirmingClose(null)}
                                className="text-xs uppercase tracking-wider text-stone-600 border border-stone-300 px-3 py-1.5 hover:bg-white">
                                Cancel
                              </button>
                              <button onClick={() => confirmClosePosition(p)}
                                style={{ backgroundColor: "#2E4A7B", color: "#FFFFFF", fontWeight: 600 }}
                                className="text-xs uppercase tracking-wider px-3 py-1.5 hover:opacity-90">
                                Confirm close
                              </button>
                            </div>
                          </div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* ========== CLOSED POSITIONS ========== */}
      {enrichedClosed.length > 0 && (() => {
        // Enrich each closed position with hold days
        const withHold = enrichedClosed.map(p => {
          const openedDate = new Date(p.openedAt);
          const closedDate = new Date(p.closedAt);
          const holdDays = Math.max(0, Math.round((closedDate - openedDate) / 86400000));
          return { ...p, holdDays };
        });

        // Summary stats (only meaningful at 3+ trades)
        const pnlDollars = withHold.map(p => p.realizedPnl * p.lots * 100);
        const wins = pnlDollars.filter(x => x > 0);
        const losses = pnlDollars.filter(x => x < 0);
        const winRate = pnlDollars.length > 0 ? (wins.length / pnlDollars.length) * 100 : 0;
        const avgWin = wins.length > 0 ? wins.reduce((s, x) => s + x, 0) / wins.length : 0;
        const avgLoss = losses.length > 0 ? losses.reduce((s, x) => s + x, 0) / losses.length : 0;
        const largestWin = wins.length > 0 ? Math.max(...wins) : 0;
        const largestLoss = losses.length > 0 ? Math.min(...losses) : 0;
        const showStats = withHold.length >= 3;

        return (
          <div>
            <div className="flex items-baseline justify-between mb-3">
              <div className="text-[10px] tracking-[0.15em] uppercase text-stone-500">
                Closed positions ({withHold.length})
              </div>
              <div className="text-xs text-stone-500">realised track record</div>
            </div>

            {/* Summary stats strip — only when 3+ trades so sample size is meaningful */}
            {showStats && (
              <div className="grid grid-cols-5 gap-3 mb-3">
                <StatBlock label="Win rate" value={winRate.toFixed(0) + "%"}
                           sublabel={`${wins.length}W / ${losses.length}L`} />
                <StatBlock label="Avg winner" value={fmtSigned(avgWin)}
                           sublabel="per closed trade" positive={avgWin > 0} />
                <StatBlock label="Avg loser" value={fmtSigned(avgLoss)}
                           sublabel="per closed trade" negative={avgLoss < 0} />
                <StatBlock label="Largest win" value={fmtSigned(largestWin)}
                           sublabel="single trade" positive={largestWin > 0} />
                <StatBlock label="Largest loss" value={fmtSigned(largestLoss)}
                           sublabel="single trade" negative={largestLoss < 0} />
              </div>
            )}

            <div className="bg-white border border-stone-300">
              <table className="w-full text-sm">
                <thead className="bg-stone-50 border-b border-stone-300">
                  <tr className="text-left text-[10px] tracking-[0.15em] uppercase text-stone-500">
                    <th className="px-4 py-3">Strategy</th>
                    <th className="px-4 py-3">Symbol</th>
                    <th className="px-4 py-3 text-right">Lots</th>
                    <th className="px-4 py-3">Opened</th>
                    <th className="px-4 py-3">Closed</th>
                    <th className="px-4 py-3 text-right">Hold</th>
                    <th className="px-4 py-3 text-right">Entry ($)</th>
                    <th className="px-4 py-3 text-right">Close mark ($)</th>
                    <th className="px-4 py-3 text-right">Realised P&L ($)</th>
                  </tr>
                </thead>
                <tbody style={{ fontFamily: "JetBrains Mono", fontSize: 13 }}>
                  {withHold.map(p => {
                    // Colour Entry by debit/credit: debit (+) is rose, credit (−) is emerald
                    const entryColor = p.entryPremium > 0 ? "text-rose-700"
                                     : p.entryPremium < 0 ? "text-emerald-700"
                                     : "text-stone-700";
                    // Colour Realised P&L: positive = emerald, negative = rose, exactly zero = neutral stone
                    const pnlDollar = p.realizedPnl * p.lots * 100;
                    const pnlColor = Math.abs(pnlDollar) < 0.005 ? "text-stone-500"
                                   : pnlDollar > 0 ? "text-emerald-700"
                                   : "text-rose-700";
                    return (
                      <tr key={p.id} className="border-b border-stone-100 last:border-0">
                        <td className="px-4 py-3">{p.def.label}</td>
                        <td className="px-4 py-3">{p.underlying}</td>
                        <td className="px-4 py-3 text-right">{p.lots}</td>
                        <td className="px-4 py-3 text-stone-500">{p.openedAt}</td>
                        <td className="px-4 py-3 text-stone-500">{p.closedAt}</td>
                        <td className="px-4 py-3 text-right text-stone-500">{p.holdDays}d</td>
                        <td className={`px-4 py-3 text-right ${entryColor}`}>
                          {fmtSigned(p.entryPremium * p.lots * 100)}
                          <span className="text-[10px] text-stone-400 ml-1">
                            {p.entryPremium > 0 ? "dr" : p.entryPremium < 0 ? "cr" : ""}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-right">{fmtSigned(p.closingValue * p.lots * 100)}</td>
                        <td className={`px-4 py-3 text-right font-semibold ${pnlColor}`}>
                          {fmtSigned(pnlDollar)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        );
      })()}
    </div>
  );
}
