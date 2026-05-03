"""
ModelSubmission — Generation 6: Mean Reversion + Momentum Hybrid
=================================================================

Core insight from research log:
  Win rate: 24%, Avg win: $35, Avg loss: $105, Ratio: 3:1 (bad)
  Need either higher win rate OR better risk/reward ratio
  
Mean reversion addresses both:
  - Higher win rate: prices tend to revert to 0.50 early in events
  - Better risk/reward: target 0.12 profit vs 0.08 stop = 1.5:1

Strategy:
  Phase 1 (TTR > 150s): MEAN REVERSION
    - Token market is uncertain, noise trader dominated
    - Prices oscillate around 0.50
    - Buy UP when up_mid < 0.40 (oversold), target 0.50
    - Buy DOWN when up_mid > 0.60 (overbought), target 0.50
    - Stop loss if move continues against us by 0.08
    
  Phase 2 (TTR 60-150s): MOMENTUM
    - Direction has established, follow the trend
    - Only enter on strong confirmed momentum
    - Tighter position size, faster exit
    
  Phase 3 (TTR < 60s): FLAT
    - Resolution lottery, insider-dominated
    - No exceptions

Risk/Reward Design:
  Mean reversion trade:
    Entry: up_mid = 0.62 (buying DOWN)
    Target: up_mid = 0.52 → profit = 0.10 per share
    Stop:   up_mid = 0.70 → loss = 0.08 per share
    R/R = 1.25:1 minimum, win rate target > 55%
    
  Momentum trade:
    Entry: confirmed trend with 30s lookback
    Target: profit-take at 0.82/0.18
    Stop: 20% of notional
    R/R = variable, win rate target > 45%
"""

from __future__ import annotations

from typing import Any

from polybench import FLAT, MarketInfo, Model, RunResult, Side, Signal, Tick


class ModelSubmission(Model):

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config=config)

        self._capital = float(self.config.get("starting_capital", 1000.0))

        # --- Phase boundaries ---
        self._mean_rev_ttr = float(self.config.get("mean_rev_ttr", 150.0))
        self._force_flat_ttr = float(self.config.get("force_flat_ttr", 60.0))

        # --- Mean reversion parameters ---
        # Entry: fade when token moves this far from 0.50
        self._mr_entry_threshold = float(self.config.get("mr_entry_threshold", 0.10))
        # Target: exit when token returns this close to 0.50
        self._mr_target = float(self.config.get("mr_target", 0.52))
        # Stop: exit if move continues this far beyond entry
        self._mr_stop_move = float(self.config.get("mr_stop_move", 0.08))
        # Size: fraction of capital for mean reversion trades
        self._mr_size = float(self.config.get("mr_size", 0.25))

        # --- Momentum parameters ---
        self._mom_lookback = int(self.config.get("mom_lookback", 30))
        self._mom_threshold = float(self.config.get("mom_threshold", 0.015))
        self._mom_size = float(self.config.get("mom_size", 0.20))
        self._mom_profit_take_high = float(self.config.get("mom_profit_take_high", 0.82))
        self._mom_profit_take_low = float(self.config.get("mom_profit_take_low", 0.18))

        # --- Shared risk parameters ---
        self._max_spread = float(self.config.get("max_spread", 0.08))
        self._stop_loss_pct = float(self.config.get("stop_loss_pct", 0.25))
        self._event_max_loss = float(self.config.get("event_max_loss", 0.15))
        self._warmup_ticks = int(self.config.get("warmup_ticks", 30))

        # --- Cooldowns ---
        self._cooldown_profit = int(self.config.get("cooldown_profit", 5))
        self._cooldown_stop = int(self.config.get("cooldown_stop", 20))

        # --- State ---
        self._side = Side.FLAT
        self._shares = 0.0
        self._entry_price = 0.0
        self._entry_notional = 0.0
        self._entry_up_mid = 0.0   # up_mid at entry (for mean reversion target)
        self._current_cash = self._capital
        self._event_start_cash = self._capital
        self._stopped_out = False
        self._ticks_since_exit = 999
        self._cooldown_required = 5
        self._tick_count = 0
        self._trade_type = ""  # "mr" or "mom"

    def on_start(self, market_info: MarketInfo) -> None:
        self._side = Side.FLAT
        self._shares = 0.0
        self._entry_price = 0.0
        self._entry_notional = 0.0
        self._entry_up_mid = 0.0
        self._event_start_cash = self._current_cash
        self._stopped_out = False
        self._ticks_since_exit = 999
        self._cooldown_required = self._cooldown_profit
        self._tick_count = 0
        self._trade_type = ""

    def on_finish(self, result: RunResult) -> None:
        pass

    def on_tick(self, tick: Tick) -> Signal | None:
        self._tick_count += 1

        if self._side == Side.FLAT:
            self._ticks_since_exit += 1

        # Phase 3: flat
        if tick.time_to_resolve <= self._force_flat_ttr:
            return self._go_flat(stop=False)

        # Event stopped out
        if self._stopped_out:
            return FLAT

        # Warmup
        if self._tick_count < self._warmup_ticks:
            return FLAT

        # Valid book
        if not self._book_valid(tick):
            return self._go_flat(stop=True)

        # Spread gate
        spread = tick.up_ask - tick.up_bid
        if spread <= 0 or spread > self._max_spread:
            if self._side != Side.FLAT:
                return self._hold(tick)
            return FLAT

        # Manage existing position
        if self._side != Side.FLAT:
            return self._manage_position(tick)

        # Cooldown
        if self._ticks_since_exit < self._cooldown_required:
            return FLAT

        # Phase selection
        in_mean_rev = tick.time_to_resolve > self._mean_rev_ttr
        in_momentum = self._force_flat_ttr < tick.time_to_resolve <= self._mean_rev_ttr

        if in_mean_rev:
            return self._mean_reversion_entry(tick)
        elif in_momentum:
            return self._momentum_entry(tick)

        return FLAT

    # ------------------------------------------------------------------ #
    #  Mean Reversion Entry                                                #
    # ------------------------------------------------------------------ #

    def _mean_reversion_entry(self, tick: Tick) -> Signal:
        """
        Fade moves away from 0.50 early in the event.
        
        Logic: Early in a 5-min event, the market is uncertain.
        Token prices oscillate around 0.50 as noise traders and 
        momentum followers push prices around. When price moves
        significantly from 0.50, fade it back.
        
        Key difference from momentum: we're betting ON reversion
        not on continuation. Higher win rate, lower individual gain.
        """
        deviation = tick.up_mid - 0.50

        # Token too high → fade with DOWN (expect reversion to 0.50)
        if deviation > self._mr_entry_threshold:
            # Don't fade if spread is too wide (low conviction market)
            if spread := (tick.up_ask - tick.up_bid) > self._max_spread * 0.7:
                return FLAT
            # Don't fade if we're already at extreme (might continue)
            if tick.up_mid > 0.75:
                return FLAT
            return self._enter_mr(
                Side.DOWN, tick.down_ask, tick.down_bid,
                tick.up_mid, self._mr_size
            )

        # Token too low → fade with UP (expect reversion to 0.50)
        if deviation < -self._mr_entry_threshold:
            if tick.up_mid < 0.25:
                return FLAT
            return self._enter_mr(
                Side.UP, tick.up_ask, tick.up_bid,
                tick.up_mid, self._mr_size
            )

        return FLAT

    # ------------------------------------------------------------------ #
    #  Momentum Entry                                                      #
    # ------------------------------------------------------------------ #

    def _momentum_entry(self, tick: Tick) -> Signal:
        """
        Follow confirmed momentum in the middle phase.
        Direction has established by now, ride the trend.
        """
        window = tick.up_mid_recent
        if len(window) < self._mom_lookback + 1:
            return FLAT

        past = window[-1 - self._mom_lookback]
        now = window[-1]
        if past <= 0 or now <= 0:
            return FLAT

        move = now - past
        if abs(move) < self._mom_threshold:
            return FLAT

        # Don't enter at extremes
        if now > 0.75 or now < 0.25:
            return FLAT

        if move > self._mom_threshold:
            return self._enter_mom(
                Side.UP, tick.up_ask, tick.up_bid,
                self._mom_size
            )
        if move < -self._mom_threshold:
            return self._enter_mom(
                Side.DOWN, tick.down_ask, tick.down_bid,
                self._mom_size
            )

        return FLAT

    # ------------------------------------------------------------------ #
    #  Position Management                                                 #
    # ------------------------------------------------------------------ #

    def _manage_position(self, tick: Tick) -> Signal:
        if self._trade_type == "mr":
            return self._manage_mr(tick)
        else:
            return self._manage_mom(tick)

    def _manage_mr(self, tick: Tick) -> Signal:
        """
        Mean reversion exit logic:
        - Take profit when price returns toward 0.50 (target)
        - Stop loss if price continues away from 0.50
        """
        # Profit take: price returned toward 0.50
        if self._side == Side.DOWN:
            # We shorted UP (bought DOWN) because up_mid was high
            # Take profit when up_mid falls back toward 0.50
            if tick.up_mid <= self._mr_target:
                return self._go_flat(stop=False)
            # Stop loss: up_mid continued higher
            if tick.up_mid > self._entry_up_mid + self._mr_stop_move:
                return self._go_flat(stop=True)

        elif self._side == Side.UP:
            # We bought UP because up_mid was low
            # Take profit when up_mid rises back toward 0.50
            if tick.up_mid >= (1.0 - self._mr_target):
                return self._go_flat(stop=False)
            # Stop loss: up_mid continued lower
            if tick.up_mid < self._entry_up_mid - self._mr_stop_move:
                return self._go_flat(stop=True)

        # Per-trade stop loss backstop
        current_bid = tick.up_bid if self._side == Side.UP else tick.down_bid
        if self._entry_price > 0 and current_bid > 0:
            trade_pnl = self._shares * (current_bid - self._entry_price)
            if trade_pnl < -self._stop_loss_pct * self._entry_notional:
                return self._go_flat(stop=True)

        # Event stop loss
        if self._current_cash < self._event_start_cash * (1.0 - self._event_max_loss):
            self._stopped_out = True
            return self._go_flat(stop=True)

        return self._hold(tick)

    def _manage_mom(self, tick: Tick) -> Signal:
        """
        Momentum exit logic:
        - Profit take at extreme prices
        - Stop loss on reversal
        """
        # Profit take
        if self._side == Side.UP and tick.up_mid > self._mom_profit_take_high:
            return self._go_flat(stop=False)
        if self._side == Side.DOWN and tick.up_mid < self._mom_profit_take_low:
            return self._go_flat(stop=False)

        # Momentum reversal exit
        window = tick.up_mid_recent
        if len(window) >= self._mom_lookback + 1:
            past = window[-1 - self._mom_lookback]
            now = window[-1]
            if past > 0 and now > 0:
                move = now - past
                direction = 1.0 if self._side == Side.UP else -1.0
                if direction * move < -self._mom_threshold:
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

        return self._hold(tick)

    # ------------------------------------------------------------------ #
    #  Entry helpers                                                       #
    # ------------------------------------------------------------------ #

    def _enter_mr(self, side: Side, ask: float, bid: float,
                  up_mid_at_entry: float, size: float) -> Signal:
        if ask <= 0 or bid <= 0 or ask <= bid:
            return FLAT
        size = self._clamp(size, 0.0, 0.40)
        available = max(0.0, self._current_cash)
        notional = min(size * self._capital, available)
        if notional <= 0:
            return FLAT
        self._side = side
        self._shares = notional / ask
        self._entry_price = ask
        self._entry_notional = notional
        self._entry_up_mid = up_mid_at_entry
        self._current_cash -= notional
        self._ticks_since_exit = 0
        self._trade_type = "mr"
        return Signal(side=side, size=notional / self._capital, confidence=0.6)

    def _enter_mom(self, side: Side, ask: float, bid: float,
                   size: float) -> Signal:
        if ask <= 0 or bid <= 0 or ask <= bid:
            return FLAT
        size = self._clamp(size, 0.0, 0.35)
        available = max(0.0, self._current_cash)
        notional = min(size * self._capital, available)
        if notional <= 0:
            return FLAT
        self._side = side
        self._shares = notional / ask
        self._entry_price = ask
        self._entry_notional = notional
        self._entry_up_mid = 0.0
        self._current_cash -= notional
        self._ticks_since_exit = 0
        self._trade_type = "mom"
        return Signal(side=side, size=notional / self._capital, confidence=0.6)

    def _hold(self, tick: Tick) -> Signal:
        ask = tick.up_ask if self._side == Side.UP else tick.down_ask
        if ask <= 0 or self._shares <= 0:
            return self._go_flat(stop=True)
        size = self._clamp((self._shares * ask) / self._capital, 0.0, 1.0)
        return Signal(side=self._side, size=size, confidence=0.6)

    def _go_flat(self, stop: bool = False) -> Signal:
        if self._side != Side.FLAT:
            self._current_cash += self._entry_notional
            self._ticks_since_exit = 0
            self._cooldown_required = (
                self._cooldown_stop if stop else self._cooldown_profit
            )
        self._side = Side.FLAT
        self._shares = 0.0
        self._entry_price = 0.0
        self._entry_notional = 0.0
        self._entry_up_mid = 0.0
        self._trade_type = ""
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