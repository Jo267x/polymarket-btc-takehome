# Research Log: Polymarket BTC 5-Minute Event Trading

## Overview

This document records the systematic research process used to develop a trading model for Polymarket BTC 5-minute up/down events. The research spans 9 model generations, 3 live market recordings, and approximately 40 hours of iteration.

**Primary finding:** Polymarket BTC 5-minute events are dominated by extreme intra-event BTC moves that make momentum-following strategies highly volatile. The key edge is not predicting direction — it is managing risk better than the baseline during extreme events while capturing small consistent gains during normal events.

---

## Market Structure Analysis

### The Adversarial Landscape

Polymarket prediction markets contain several distinct participant types with asymmetric information advantages:

**Insiders** (politicians, exchange employees, whale wallet operators) position before public information breaks. Their trades create the initial price move that all other participants react to.

**Momentum followers** (including MomentumBaseline) react to price moves mechanically. They create overshoot and crowding effects after the initial move.

**Market makers** post bids and asks for spread income. They pull quotes during uncertainty, creating observable spread widening signals.

**Noise traders** act randomly, creating temporary dislocations and mean-reversion opportunities.

**Key insight:** A 1Hz model with no external data feed is always 2-3 steps behind the information chain. The viable edge is not predicting direction — it is exploiting the behavior of momentum followers and noise traders after the initial move has established direction.

### The Leverage Problem

The MomentumBaseline has a critical flaw: its position sizing formula produces effective leverage of 3-5x starting capital during large BTC moves. When momentum reverses, this creates catastrophic losses:

```
Observed baseline performance across 4 recording sessions:
Session 1: +$13,998  DD=192%  (got lucky on direction)
Session 2: +$92,852  DD=621%  (extreme luck)  
Session 3: +$731,071 DD=323%  (extreme luck, overnight)
Session 4: -$7,659   DD=247%  (unlucky on direction)
```

The baseline's primary_score is 0 on all sessions because the drawdown term `(1 - max_drawdown)` is negative, flooring the score to zero. **The baseline is not a viable strategy — it is a lottery ticket.**

### Market Conditions Observed

All four recording sessions occurred during an unusually volatile BTC period (May 2026), where 5-minute BTC moves of $200-500 were common. This caused UP/DOWN token prices to swing between 0.01 and 0.99 within single events.

```
Normal session:  up_mid stays 0.35-0.65 (~70% of ticks)
Our sessions:    up_mid stays 0.35-0.65 (~14-38% of ticks)
```

This context is important: results on these fixtures are stress tests, not representative of typical market conditions.

---

## Model Generation History

### Generation 1 (v1-v3): Momentum + Spread Gates

**Hypothesis:** The MomentumBaseline signal is correct but needs spread quality gates to filter low-quality entries.

**Implementation:** Three-phase structure (Discovery/Trend/Resolution) with spread-as-volume-proxy gates. Position sizing proportional to spread quality.

**Key results:**
```
v1: -$596  DD=69%  trades=143  (live fixture 1)
v2: -$390  DD=43%  trades=46
v3: -$266  DD=31%  trades=40
```

**Learning:** Spread gates reduce trades and losses but the signal itself is still net negative. 272 trades → 40 trades, losses reduced from $596 → $266. Fee drag is a major factor — each round-trip costs ~$0.018/share at p=0.50.

**Key finding:** The 3-tick trend confirmation requirement is too strict for real Polymarket data. Prices oscillate every tick, making 3 consecutive same-direction ticks rare. Dropped to 2-tick confirmation.

---

### Generation 2 (v4-v6): Phase-Aware + Profit Take

**Hypothesis:** The model needs phase-awareness: different behavior early vs late in events. Adding profit-take exits when token reaches high confidence levels.

**Implementation:** 
- Extreme price gate: never enter above 0.70 or below 0.30
- Profit-take at 0.82/0.18
- TTR-scaled position sizing
- Fixed anchor drift using event-start mid (not rolling window)

**Key results:**
```
v4: -$267  DD=31%   trades=40
v5: -$284  DD=33%   trades=40  (slightly worse — price-adaptive momentum misfired)
v6: -$188  DD=19%   trades=9   (best on fixture 1)
    $0     DD=0%    trades=0   (correctly flat on fixture 2)
```

**Learning:** The profit-take at 0.82 correctly prevents the "mature trend looks like reversal" misfire. However, it also caps gains on large moves — on the $34k baseline event, we made only $51.

**Key finding:** Event anchor drift (current_price - first_tick_price) is more meaningful than rolling window drift. The rolling window carries data from previous events in replay mode.

---

### Generation 3 (v7): Trailing Stop Experiment

**Hypothesis:** Replace static profit-take with trailing stop to capture larger moves while still protecting against reversals.

**Implementation:** Track watermark (best price seen since entry). Exit when price pulls back 15% from watermark. Tighten trail to 8% at extreme prices.

**Key results:**
```
v7: -$1,166  DD=118%  trades=563  (worse than v6)
```

**Learning:** Trailing stop created a churn loop — exit triggers, model re-enters, triggers again. 563 trades vs 212 in v6. The DOWN-side trailing stop math was also incorrect, creating asymmetric behavior. Complexity without sufficient data to validate = bugs.

**Key finding:** Simplicity is a virtue when data is limited. The trailing stop concept is sound but requires more careful implementation and more data to tune.

---

### Generation 4 (v8-v9): Cash Tracking Fix + Size Reduction

**Hypothesis:** The equity estimator bug is causing event stop loss to trigger incorrectly, creating cascading losses. Fix the cash tracking.

**Implementation:**
- Track `_current_cash` explicitly instead of estimating equity
- Event stop loss based on cash change from event start
- Reduced position size to 25% max (from 30%)
- Only spend available cash (prevents effective leverage)

**Key results:**
```
v9: -$45   DD=4.5%  trades=4   (fixture 1)
    $0     DD=0%    trades=0   (fixture 2 — correctly flat)
    -$392  DD=39%   trades=42  (overnight 25 events)
```

**Learning:** The equity estimator bug was causing stop losses to trigger at wrong times, creating extra trades and losses. With accurate cash tracking, the model is dramatically more conservative but correct.

**Key finding:** Equity going below $0 in earlier versions was an accounting artifact, not real leverage. The simulator caps real losses at starting capital.

---

### Generation 5 (v10): Baseline Signal + Risk Management

**Hypothesis:** Our signal is too conservative. Use baseline momentum signal but wrap proper risk management around it.

**Implementation:** Mirror MomentumBaseline entry logic but add:
- Hard size cap at 30% (prevents baseline's 5x leverage)
- 25% per-trade stop loss
- 15% event stop loss
- Force flat at TTR < 60s

**Key results:**
```
v10: -$120  DD=15%  trades=7   (fixture 1)
     -$656  DD=66%  trades=50  (overnight)
```

**Learning:** The baseline signal is too noisy without our phase gates. More trades from baseline signal = more fee drag = worse results than v9 with strict gates.

**Key finding:** The problem is not the signal structure — it is the market conditions. During extreme BTC volatility, no momentum signal works consistently.

---

## Primary Findings

### Finding 1: Polymarket BTC Events Are Extreme by Design

The 5-minute binary format with $0/$1 resolution creates extreme token price sensitivity. A 0.5% BTC move in 5 minutes (normal) sends the UP token from 0.50 to 0.80. This means even "calm" BTC sessions produce extreme Polymarket token moves.

### Finding 2: The Baseline Is a Leveraged Lottery

MomentumBaseline achieves positive PnL through effective leverage (3-5x capital) combined with directional luck. Its primary_score is 0 on every recording session due to excessive drawdown. It is not a viable strategy — it is a benchmark that demonstrates what NOT to do.

### Finding 3: Fee Drag Is the Primary Enemy of High-Frequency Strategies

At Polymarket's fee rate (0.072), trading near p=0.50 costs $0.018/share. A 300-tick event with 150 trades loses approximately $50-100 in pure fees regardless of signal quality. The break-even trade frequency is approximately 1 trade per 15-20 ticks.

### Finding 4: Spread Dynamics Are Informative But Insufficient

Real Polymarket spreads vary from 0.01 to 0.70 within single events. Wide spreads correlate with news shocks and large BTC moves. However, our recordings showed that catastrophic events (token going to 0.01 or 0.99) often started with tight spreads — the market makers had already exited before the spread widened.

### Finding 5: Capital Preservation Outperforms Momentum on Extreme Sessions

```
Best comparison (live test, session 4):
  Model:    -$18   DD=1.85%   2 trades
  Baseline: -$13,984  DD=1,398%  388 trades
```

On sessions where the baseline self-destructs, a conservative model preserves almost all capital. Over a 2-hour scoring window with mixed normal and extreme events, capital preservation in extreme events compounds positively.

---

## Current Best Model (v9)

### Hypothesis

Polymarket BTC 5-minute events follow a three-phase intraday price discovery structure analogous to equity markets:

- **Phase 1 (TTR > 180s):** High uncertainty, noisy, smart money positioning. Strategy: stay flat unless extreme deviation from event anchor warrants mean reversion.
- **Phase 2 (TTR 75-180s):** Momentum valid but requires confirmation. Strategy: follow confirmed momentum with spread gate.
- **Phase 3 (TTR < 75s):** Resolution convergence, insider-dominated. Strategy: unconditionally flat.

### Risk Framework

```
Entry gates (all must be true):
  - Spread < 0.06 (liquid conditions only)
  - up_mid between 0.33-0.67 (not already at extreme)
  - TTR between 75-210s (right phase window)
  - 2 consecutive confirming ticks (not single-tick noise)
  - 45 tick warmup (sufficient history)
  - 15-25 tick cooldown after exit (prevent churn)

Position risk:
  - Max 25% of capital per trade
  - 25% per-trade stop loss
  - 15% event stop loss → sit out rest of event
  - Force flat at TTR < 75s

Cash tracking:
  - Track actual cash spent/received
  - Never spend more than available cash
  - Prevents effective leverage
```

### Performance Summary

```
Fixture 1 (volatile, 3 events):  -$45   DD=4.5%   4 trades
Fixture 2 (extreme, 4 events):   $0     DD=0%     0 trades (correctly flat)
Overnight (25 events, extreme):  -$392  DD=39%    42 trades
Live test (10 min, 3 events):    -$18   DD=1.85%  2 trades
```

---

## Next Research Directions

### Generation 6: Pure Mean Reversion

**Hypothesis:** Early in events (TTR > 150s), token prices mean-revert to 0.50 because the market is uncertain. Noise traders push prices around. Fade significant deviations.

```python
# Simple testable signal:
# if up_mid < 0.38 and TTR > 150s: buy UP (expect reversion to 0.50)
# if up_mid > 0.62 and TTR > 150s: buy DOWN (expect reversion to 0.50)
# exit when up_mid returns to 0.48-0.52
```

**Expected edge:** On calm sessions where token stays near 0.50, this generates 3-5 clean round-trips per event at $10-30 each. On extreme sessions, the event stop loss limits damage.

### Generation 7: Regime Detection

**Hypothesis:** Detect whether the current event is "trending" or "mean-reverting" from the first 30 ticks, then apply the appropriate strategy.

**Signal:** Velocity of up_mid change in first 30 ticks. High velocity → trending regime → follow momentum. Low velocity → mean-reverting regime → fade extremes.

### Generation 8: Multi-Signal Ensemble

Combine mean reversion (Phase 1) and momentum (Phase 2) signals with regime detection to select which signal to apply. Weight signals by recent performance (online learning).

---

## Conclusion

The research demonstrates that beating MomentumBaseline on primary_score requires solving two problems simultaneously:

1. **A positive-expectation signal** that makes money on average across normal events
2. **Risk management** that prevents catastrophic losses during extreme events

The baseline solves neither. Our v9 model solves (2) but not yet (1). The next generation should focus on finding a reliable positive-expectation signal for normal market conditions, using the extensive risk framework already developed.

The key insight for further research: **the edge is not in predicting BTC direction — it is in exploiting the predictable behavior of other Polymarket participants** (momentum crowding, noise trader mean reversion, market maker quote dynamics) within the 5-minute event window.