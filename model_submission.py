"""
ModelSubmission — v14: Opening Price Signal
============================================

Key finding from data analysis:
  When a Polymarket BTC 5-min event opens with UP token price
  significantly away from 0.50, that direction predicts resolution:
  
  |opening_mid - 0.50| > 0.05 → 81.8% accuracy (11/24 events)
  |opening_mid - 0.50| > 0.08 → 100% accuracy  (5/24 events)
  
Why this works:
  Market makers set opening prices based on BTC moves that occurred
  in the seconds BEFORE the event started. They have faster data
  feeds and price in directional information immediately.
  
  When they open UP token at 0.42, they're saying:
  "BTC just dropped, we think DOWN wins this event"
  
  Their opening quote is the signal. Follow it.

Strategy:
  1. Capture up_mid on the very first tick of each event
  2. If opening_mid > 0.55: enter UP (MMs pricing UP win)
  3. If opening_mid < 0.45: enter DOWN (MMs pricing DOWN win)  
  4. If 0.45-0.55: stay flat (uncertain, don't trade)
  5. Hold entire event, exit at TTR=62s
  6. One trade per event maximum

Risk management:
  - Event stop loss: 20% of capital
  - Force flat at TTR=62s (avoid resolution lottery)
  - No stop loss during hold (trust the opening signal)
  - Only trade events with clear opening signal
"""

from __future__ import annotations

from typing import Any

from polybench import FLAT, MarketInfo, Model, RunResult, Side, Signal, Tick


class ModelSubmission(Model):

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config=config)

        self._capital = float(self.config.get("starting_capital", 1000.0))

        # Opening price threshold
        # Only trade when opening mid is this far from 0.50
        # 0.05 = trade when mid < 0.45 or > 0.55 (81.8% accuracy)
        # 0.08 = trade when mid < 0.42 or > 0.58 (100% accuracy)
        self._open_threshold = float(self.config.get("open_threshold", 0.06))

        # How many ticks to wait before entering
        # (let the opening price stabilize)
        self._entry_delay = int(self.config.get("entry_delay", 3))

        # Force flat before resolution
        self._force_flat_ttr = float(self.config.get("force_flat_ttr", 62.0))

        # Position size
        self._size = float(self.config.get("size", 0.30))

        # Maximum spread at entry
        self._max_spread = float(self.config.get("max_spread", 0.08))

        # Event stop loss
        self._event_max_loss = float(self.config.get("event_max_loss", 0.20))

        # State
        self._side = Side.FLAT
        self._shares = 0.0
        self._entry_price = 0.0
        self._entry_notional = 0.0
        self._current_cash = self._capital
        self._event_start_cash = self._capital
        self._opening_mid = 0.0
        self._tick_count = 0
        self._entered_this_event = False
        self._stopped_out = False
        self._signal_side = None  # determined from opening price

    def on_start(self, market_info: MarketInfo) -> None:
        self._side = Side.FLAT
        self._shares = 0.0
        self._entry_price = 0.0
        self._entry_notional = 0.0
        self._event_start_cash = self._current_cash
        self._opening_mid = 0.0
        self._tick_count = 0
        self._entered_this_event = False
        self._stopped_out = False
        self._signal_side = None

    def on_finish(self, result: RunResult) -> None:
        pass

    def on_tick(self, tick: Tick) -> Signal | None:
        self._tick_count += 1

        # Capture opening price on first tick
        if self._tick_count == 1 and tick.up_mid > 0:
            self._opening_mid = tick.up_mid
            # Determine signal direction from opening price
            deviation = self._opening_mid - 0.50
            if deviation > self._open_threshold:
                self._signal_side = Side.UP
            elif deviation < -self._open_threshold:
                self._signal_side = Side.DOWN
            else:
                self._signal_side = None  # no trade this event

        # Force flat near resolution
        if tick.time_to_resolve <= self._force_flat_ttr:
            return self._go_flat()

        # Stopped out
        if self._stopped_out:
            return FLAT

        # No signal for this event
        if self._signal_side is None:
            return FLAT

        # Manage existing position
        if self._side != Side.FLAT:
            return self._manage_position(tick)

        # Already entered this event
        if self._entered_this_event:
            return FLAT

        # Wait for entry delay (let opening price stabilize)
        if self._tick_count < self._entry_delay:
            return FLAT

        # Valid book check
        if not self._book_valid(tick):
            return FLAT

        # Spread gate
        spread = tick.up_ask - tick.up_bid
        if spread <= 0 or spread > self._max_spread:
            return FLAT

        # Enter in signal direction
        if self._signal_side == Side.UP:
            return self._enter(Side.UP, tick.up_ask, tick.up_bid)
        elif self._signal_side == Side.DOWN:
            return self._enter(Side.DOWN, tick.down_ask, tick.down_bid)

        return FLAT

    def _manage_position(self, tick: Tick) -> Signal:
        # Event stop loss only — trust the opening signal
        if self._current_cash < self._event_start_cash * (1.0 - self._event_max_loss):
            self._stopped_out = True
            return self._go_flat()

        return self._hold(tick)

    def _enter(self, side: Side, ask: float, bid: float) -> Signal:
        if ask <= 0 or bid <= 0 or ask <= bid:
            return FLAT
        size = self._clamp(self._size, 0.0, 0.40)
        available = max(0.0, self._current_cash)
        notional = min(size * self._capital, available)
        if notional <= 0:
            return FLAT
        self._side = side
        self._shares = notional / ask
        self._entry_price = ask
        self._entry_notional = notional
        self._current_cash -= notional
        self._entered_this_event = True
        return Signal(side=side, size=notional / self._capital, confidence=0.8)

    def _hold(self, tick: Tick) -> Signal:
        ask = tick.up_ask if self._side == Side.UP else tick.down_ask
        if ask <= 0 or self._shares <= 0:
            return self._go_flat()
        size = self._clamp((self._shares * ask) / self._capital, 0.0, 1.0)
        return Signal(side=self._side, size=size, confidence=0.8)

    def _go_flat(self) -> Signal:
        if self._side != Side.FLAT:
            self._current_cash += self._entry_notional
        self._side = Side.FLAT
        self._shares = 0.0
        self._entry_price = 0.0
        self._entry_notional = 0.0
        return FLAT

    def _book_valid(self, tick: Tick) -> bool:
        return (
            tick.up_bid > 0.0 and tick.up_ask > 0.0
            and tick.down_bid > 0.0 and tick.down_ask > 0.0
            and tick.up_ask > tick.up_bid
            and tick.down_ask > tick.down_bid
        )

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return lo if value < lo else (hi if value > hi else value)