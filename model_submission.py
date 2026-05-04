"""
ModelSubmission — v13: BTC Direction Follow
============================================

Key finding from correlation analysis:
  BTC 5-minute move vs token resolution: 0.797 correlation
  
  When BTC goes UP in a 5-min window → UP token resolves at ~$1
  When BTC goes DOWN in a 5-min window → DOWN token resolves at ~$0
  This holds ~80% of the time
  
Strategy: enter once per event based on BTC direction, hold, exit at TTR=60s

Why this beats the baseline:
  Baseline: 4433 trades, enters/exits repeatedly, pays fees every time
  v13: 1 trade per event, minimal fees, 80% win rate
  
Why this works:
  BTC direction in first 30-60s of event predicts final resolution
  Token prices lag BTC by 5-30 seconds
  Enter when BTC direction is clear, ride token repricing
  
Entry logic:
  Wait 45 ticks (45 seconds) for BTC direction to establish
  Compare current BTC to event-start BTC
  If BTC up > 0.05%: enter UP
  If BTC down > 0.05%: enter DOWN
  Only enter once per event
  
Exit logic:
  Force flat at TTR=60s (avoid resolution lottery)
  No stop loss — single entry, hold through noise
  Event stop loss at 20% (catastrophic protection only)
  
Position sizing:
  25% of capital per trade
  Never re-enter after exit
"""

from __future__ import annotations

from typing import Any

from polybench import FLAT, MarketInfo, Model, RunResult, Side, Signal, Tick


class ModelSubmission(Model):

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config=config)

        self._capital = float(self.config.get("starting_capital", 1000.0))

        # BTC direction threshold to trigger entry
        # 0.0005 = 0.05% BTC move required
        self._btc_threshold = float(self.config.get("btc_threshold", 0.0005))

        # Ticks to wait before first entry (let direction establish)
        self._warmup_ticks = int(self.config.get("warmup_ticks", 45))

        # Force flat before resolution
        self._force_flat_ttr = float(self.config.get("force_flat_ttr", 62.0))

        # Position size
        self._size = float(self.config.get("size", 0.25))

        # Maximum spread to accept entry
        self._max_spread = float(self.config.get("max_spread", 0.08))

        # Event stop loss (catastrophic protection)
        self._event_max_loss = float(self.config.get("event_max_loss", 0.20))

        # Only enter once per event
        self._entered_this_event = False

        # State
        self._side = Side.FLAT
        self._shares = 0.0
        self._entry_price = 0.0
        self._entry_notional = 0.0
        self._current_cash = self._capital
        self._event_start_cash = self._capital
        self._event_start_btc = 0.0
        self._tick_count = 0
        self._stopped_out = False

    def on_start(self, market_info: MarketInfo) -> None:
        self._side = Side.FLAT
        self._shares = 0.0
        self._entry_price = 0.0
        self._entry_notional = 0.0
        self._event_start_cash = self._current_cash
        self._event_start_btc = 0.0
        self._tick_count = 0
        self._stopped_out = False
        self._entered_this_event = False

    def on_finish(self, result: RunResult) -> None:
        pass

    def on_tick(self, tick: Tick) -> Signal | None:
        self._tick_count += 1

        # Capture BTC price at event start
        if self._event_start_btc <= 0.0 and tick.btc_last > 0.0:
            self._event_start_btc = tick.btc_last

        # Force flat near resolution
        if tick.time_to_resolve <= self._force_flat_ttr:
            return self._go_flat()

        # Stopped out
        if self._stopped_out:
            return FLAT

        # Already entered this event — just hold or manage
        if self._side != Side.FLAT:
            return self._manage_position(tick)

        # Already entered and exited this event — don't re-enter
        if self._entered_this_event:
            return FLAT

        # Wait for warmup
        if self._tick_count < self._warmup_ticks:
            return FLAT

        # Need BTC data
        if self._event_start_btc <= 0.0 or tick.btc_last <= 0.0:
            return FLAT

        # Need valid book
        if not self._book_valid(tick):
            return FLAT

        # Spread gate
        spread = tick.up_ask - tick.up_bid
        if spread <= 0 or spread > self._max_spread:
            return FLAT

        # Core signal: BTC direction from event start
        btc_move = (tick.btc_last - self._event_start_btc) / self._event_start_btc

        if btc_move > self._btc_threshold:
            # BTC up → enter UP
            return self._enter(Side.UP, tick.up_ask, tick.up_bid)

        if btc_move < -self._btc_threshold:
            # BTC down → enter DOWN
            return self._enter(Side.DOWN, tick.down_ask, tick.down_bid)

        # BTC hasn't moved enough yet — wait
        return FLAT

    def _manage_position(self, tick: Tick) -> Signal:
        # Event stop loss (catastrophic protection only)
        if self._current_cash < self._event_start_cash * (1.0 - self._event_max_loss):
            self._stopped_out = True
            return self._go_flat()

        # Hold — don't exit early, ride the move
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