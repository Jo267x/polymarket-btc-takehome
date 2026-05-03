"""
ModelSubmission — Momentum with Risk Management v10
====================================================

Philosophy shift from v9:
Previous versions were too conservative — 2 trades in 600 ticks is
not a trading strategy. We were so focused on avoiding losses that
we forgot to make money.

New approach: baseline momentum signal + proper risk management
- Entry: similar to MomentumBaseline (follow momentum)
  but with spread quality gate and position size cap
- Risk: hard stop loss per trade (prevents -$13k disasters)  
- Exit: momentum reversal OR profit take at extreme prices
- Size: fixed fraction of capital, never leveraged

The baseline problem is NOT its signal — it's that it has no
stop loss and sizes up to 5x capital. When momentum reverses,
it bleeds out completely. Our fix: same signal, bounded risk.

Key differences from baseline:
1. Size capped at 0.30 of capital (baseline goes to 5x)
2. Hard stop loss: exit if trade loses > 25% of position value
3. Event stop: if down > 15% in event, sit out rest of event
4. Force flat at TTR < 60s (avoid resolution lottery)
5. Spread gate: only trade when spread < 0.08
6. No re-entry during cooldown after stop loss
"""

from __future__ import annotations

from typing import Any

from polybench import FLAT, MarketInfo, Model, RunResult, Side, Signal, Tick


class ModelSubmission(Model):

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config=config)

        self._capital = float(self.config.get("starting_capital", 1000.0))

        # --- Core momentum parameters (similar to baseline) ---
        self._lookback_s = int(self.config.get("lookback_s", 30))
        self._pm_threshold = float(self.config.get("pm_threshold", 0.012))
        self._pm_per_unit = float(self.config.get("pm_per_unit", 0.06))

        # --- Position sizing (HARD CAP — prevents leverage) ---
        self._max_size = float(self.config.get("max_size", 0.30))

        # --- Risk management ---
        self._stop_loss_pct = float(self.config.get("stop_loss_pct", 0.25))
        self._event_max_loss = float(self.config.get("event_max_loss", 0.15))
        self._force_flat_ttr = float(self.config.get("force_flat_ttr", 60.0))

        # --- Quality gates ---
        self._max_spread = float(self.config.get("max_spread", 0.08))
        self._min_ttr = float(self.config.get("min_ttr", 65.0))

        # --- Profit take at extreme prices ---
        self._profit_take_high = float(self.config.get("profit_take_high", 0.88))
        self._profit_take_low = float(self.config.get("profit_take_low", 0.12))

        # --- Cooldown after stop loss ---
        self._cooldown_after_stop = int(self.config.get("cooldown_after_stop", 20))

        # --- State ---
        self._side = Side.FLAT
        self._shares = 0.0
        self._entry_price = 0.0
        self._entry_notional = 0.0
        self._current_cash = self._capital
        self._event_start_cash = self._capital
        self._stopped_out = False
        self._ticks_since_stop = 999
        self._tick_count = 0

    def on_start(self, market_info: MarketInfo) -> None:
        self._side = Side.FLAT
        self._shares = 0.0
        self._entry_price = 0.0
        self._entry_notional = 0.0
        self._event_start_cash = self._current_cash
        self._stopped_out = False
        self._ticks_since_stop = 999
        self._tick_count = 0

    def on_finish(self, result: RunResult) -> None:
        pass

    def on_tick(self, tick: Tick) -> Signal | None:
        self._tick_count += 1

        if self._side == Side.FLAT:
            self._ticks_since_stop += 1

        # Force flat near resolution
        if tick.time_to_resolve <= self._force_flat_ttr:
            return self._go_flat(stop=False)

        # Stopped out for this event
        if self._stopped_out:
            return FLAT

        # Need valid book
        if not self._book_valid(tick):
            return self._go_flat(stop=True)

        # Spread gate
        spread = tick.up_ask - tick.up_bid
        if spread <= 0 or spread > self._max_spread:
            if self._side != Side.FLAT:
                return self._hold(tick)
            return FLAT

        # TTR gate
        if tick.time_to_resolve < self._min_ttr:
            return self._go_flat(stop=False)

        # Cooldown after stop loss
        if self._ticks_since_stop < self._cooldown_after_stop:
            if self._side != Side.FLAT:
                return self._manage_position(tick)
            return FLAT

        # Compute momentum signal (same as baseline)
        momentum_signal = self._polymarket_momentum(tick)

        # Manage existing position
        if self._side != Side.FLAT:
            return self._manage_position(tick)

        # Enter on momentum signal
        return self._try_enter(tick, momentum_signal)

    def _polymarket_momentum(self, tick: Tick) -> tuple[Side | None, float]:
        """
        Mirror of MomentumBaseline._polymarket_momentum but returns
        (side, magnitude) tuple instead of Signal directly.
        This lets us apply our own risk management on top.
        """
        window = tick.up_mid_recent
        if len(window) < 2:
            return (None, 0.0)

        offset = min(len(window) - 1, self._lookback_s)
        past = window[-1 - offset]
        now_price = window[-1]
        if past <= 0.0 or now_price <= 0.0:
            return (None, 0.0)

        move = now_price - past
        if abs(move) < self._pm_threshold:
            return (None, 0.0)

        magnitude = min(abs(move) / self._pm_per_unit, self._max_size)
        side = Side.UP if move > 0 else Side.DOWN
        return (side, magnitude)

    def _try_enter(self, tick: Tick, signal: tuple) -> Signal:
        side, magnitude = signal
        if side is None or magnitude <= 0:
            return FLAT

        # Profit take gates — don't enter if already at extreme
        if side == Side.UP and tick.up_mid > self._profit_take_high:
            return FLAT
        if side == Side.DOWN and tick.up_mid < self._profit_take_low:
            return FLAT

        if side == Side.UP:
            ask = tick.up_ask
            bid = tick.up_bid
        else:
            ask = tick.down_ask
            bid = tick.down_bid

        if ask <= 0 or bid <= 0 or ask <= bid:
            return FLAT

        # Size: use momentum magnitude but cap at max_size
        size = self._clamp(magnitude, 0.0, self._max_size)
        available = max(0.0, self._current_cash)
        notional = min(size * self._capital, available)
        if notional <= 0:
            return FLAT

        self._side = side
        self._shares = notional / ask
        self._entry_price = ask
        self._entry_notional = notional
        self._current_cash -= notional
        self._ticks_since_stop = 999

        return Signal(side=side, size=notional / self._capital, confidence=0.6)

    def _manage_position(self, tick: Tick) -> Signal:
        direction = 1.0 if self._side == Side.UP else -1.0

        # Profit take at extreme prices
        if self._side == Side.UP and tick.up_mid > self._profit_take_high:
            return self._go_flat(stop=False)
        if self._side == Side.DOWN and tick.up_mid < self._profit_take_low:
            return self._go_flat(stop=False)

        # Per-trade stop loss
        current_bid = tick.up_bid if self._side == Side.UP else tick.down_bid
        if self._entry_price > 0 and current_bid > 0:
            trade_pnl = self._shares * (current_bid - self._entry_price)
            if trade_pnl < -self._stop_loss_pct * self._entry_notional:
                return self._go_flat(stop=True)

        # Event stop loss
        if self._current_cash < self._event_start_cash * (1.0 - self._event_max_loss):
            self._stopped_out = True
            return self._go_flat(stop=True)

        # Check if momentum signal flipped direction
        momentum_signal = self._polymarket_momentum(tick)
        new_side, magnitude = momentum_signal
        if new_side is not None and new_side != self._side and magnitude > 0:
            # Momentum has reversed — exit
            return self._go_flat(stop=False)

        return self._hold(tick)

    def _hold(self, tick: Tick) -> Signal:
        ask = tick.up_ask if self._side == Side.UP else tick.down_ask
        if ask <= 0 or self._shares <= 0:
            return self._go_flat(stop=True)
        size = self._clamp((self._shares * ask) / self._capital, 0.0, 1.0)
        return Signal(side=self._side, size=size, confidence=0.6)

    def _go_flat(self, stop: bool = False) -> Signal:
        if self._side != Side.FLAT:
            self._current_cash += self._entry_notional
            if stop:
                self._ticks_since_stop = 0
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