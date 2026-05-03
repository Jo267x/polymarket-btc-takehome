"""
Replay Analysis Tool
====================
Run after a replay to understand WHERE the model makes and loses money.
Gives you the same diagnostic instinct you'd use reading a stock trade log.

Usage:
    python analyze_replay.py runs/replay/ticks.parquet
    python analyze_replay.py runs/replay/ticks.parquet --verbose
"""

from __future__ import annotations

import sys
import math
from pathlib import Path


def analyze(parquet_path: str, verbose: bool = False) -> None:
    try:
        import pandas as pd
    except ImportError:
        print("pandas required: pip install pandas pyarrow")
        return

    df = pd.read_parquet(parquet_path)
    print(f"\n{'='*65}")
    print(f"REPLAY ANALYSIS: {parquet_path}")
    print(f"{'='*65}")
    print(f"Total rows:  {len(df)}")
    print(f"Events:      {df['event_id'].nunique()}")

    # Split active ticks from settlement rows
    active = df[df['resolved_outcome'].fillna('').eq('')]
    settlement = df[df['resolved_outcome'].fillna('').ne('')]

    print(f"Active ticks: {len(active)}")
    print(f"Settlement rows: {len(settlement)}")

    # ------------------------------------------------------------------ #
    # Per-event summary
    # ------------------------------------------------------------------ #
    print(f"\n{'─'*65}")
    print("PER-EVENT SUMMARY")
    print(f"{'─'*65}")
    print(f"{'Event':<35} {'Model PnL':>10} {'Base PnL':>10} {'Outcome':<8} {'Trades':>6}")
    print(f"{'─'*65}")

    event_results = []
    for event_id, g in df.groupby('event_id'):
        g = g.sort_values('ts')
        slug = g['slug'].iloc[0] if 'slug' in g.columns else str(event_id)[:30]

        # Model PnL for this event
        model_start = g['equity'].iloc[0]
        model_end = g['equity'].iloc[-1]
        model_pnl = model_end - model_start

        # Baseline PnL
        base_start = g['baseline_equity'].iloc[0]
        base_end = g['baseline_equity'].iloc[-1]
        base_pnl = base_end - base_start

        # Outcome
        outcome = g['resolved_outcome'].dropna().replace('', float('nan')).dropna()
        outcome_str = outcome.iloc[-1] if len(outcome) > 0 else 'PENDING'

        # Trades
        trades = int(g['fills_this_tick'].sum())

        event_results.append({
            'slug': slug[-35:],
            'model_pnl': model_pnl,
            'base_pnl': base_pnl,
            'outcome': outcome_str,
            'trades': trades,
            'beat_baseline': model_pnl > base_pnl,
        })

        marker = '✓' if model_pnl > base_pnl else '✗'
        print(
            f"{marker} {slug[-33:]:<33} "
            f"${model_pnl:>+8.2f} "
            f"${base_pnl:>+8.2f} "
            f"{outcome_str:<8} "
            f"{trades:>6}"
        )

    # ------------------------------------------------------------------ #
    # Signal distribution: what was the model actually doing?
    # ------------------------------------------------------------------ #
    print(f"\n{'─'*65}")
    print("SIGNAL DISTRIBUTION (what did the model emit?)")
    print(f"{'─'*65}")

    if 'signal_side' in active.columns:
        signal_counts = active['signal_side'].value_counts()
        total_active = len(active)
        for side, count in signal_counts.items():
            pct = count / total_active * 100
            bar = '█' * int(pct / 2)
            print(f"  {side:<8} {count:>5} ticks ({pct:>5.1f}%)  {bar}")

    # ------------------------------------------------------------------ #
    # Spread analysis: when did we trade vs when did we stay flat?
    # ------------------------------------------------------------------ #
    print(f"\n{'─'*65}")
    print("SPREAD ANALYSIS (volume proxy)")
    print(f"{'─'*65}")

    if 'up_bid' in active.columns and 'up_ask' in active.columns:
        active = active.copy()
        active['spread'] = active['up_ask'] - active['up_bid']

        trading_ticks = active[active['signal_side'].isin(['UP', 'DOWN'])]
        flat_ticks = active[active['signal_side'] == 'FLAT']

        if len(trading_ticks) > 0:
            print(f"  Avg spread when TRADING: {trading_ticks['spread'].mean():.4f}")
        if len(flat_ticks) > 0:
            print(f"  Avg spread when FLAT:    {flat_ticks['spread'].mean():.4f}")
        print(f"  Overall avg spread:      {active['spread'].mean():.4f}")
        print(f"  Spread > 0.07:           {(active['spread'] > 0.07).sum()} ticks "
              f"({(active['spread'] > 0.07).mean()*100:.1f}%)")

    # ------------------------------------------------------------------ #
    # TTR analysis: when in the event did we trade?
    # ------------------------------------------------------------------ #
    print(f"\n{'─'*65}")
    print("TTR ANALYSIS (when in the event did we trade?)")
    print(f"{'─'*65}")

    if 'time_to_resolve' in active.columns and 'signal_side' in active.columns:
        bins = [0, 45, 90, 135, 180, 225, 270, 300]
        labels = ['0-45s', '45-90s', '90-135s', '135-180s', '180-225s', '225-270s', '270-300s']
        active = active.copy()
        active['ttr_bucket'] = pd.cut(
            active['time_to_resolve'],
            bins=bins,
            labels=labels,
            right=True
        )
        ttr_signal = active.groupby('ttr_bucket', observed=True)['signal_side'].apply(
            lambda x: (x.isin(['UP', 'DOWN'])).mean()
        )
        print(f"  {'TTR Bucket':<12} {'Trade Rate':>12}")
        for bucket, rate in ttr_signal.items():
            bar = '█' * int(rate * 30)
            print(f"  {str(bucket):<12} {rate:>10.1%}  {bar}")

    # ------------------------------------------------------------------ #
    # Win/loss summary
    # ------------------------------------------------------------------ #
    print(f"\n{'─'*65}")
    print("OVERALL COMPARISON")
    print(f"{'─'*65}")

    if event_results:
        model_total = sum(e['model_pnl'] for e in event_results)
        base_total = sum(e['base_pnl'] for e in event_results)
        beat_count = sum(1 for e in event_results if e['beat_baseline'])
        n_events = len(event_results)

        print(f"  Model total PnL:    ${model_total:>+.2f}")
        print(f"  Baseline total PnL: ${base_total:>+.2f}")
        print(f"  Edge vs baseline:   ${model_total - base_total:>+.2f}")
        print(f"  Beat baseline:      {beat_count}/{n_events} events ({beat_count/n_events:.0%})")

        # Timeout check
        if 'timeout' in active.columns:
            timeout_rate = active['timeout'].mean()
            print(f"  Timeout rate:       {timeout_rate:.2%}")
            if timeout_rate > 0.01:
                print("  ⚠ HIGH TIMEOUT RATE — on_tick is too slow")

    # ------------------------------------------------------------------ #
    # Verbose: per-phase PnL breakdown
    # ------------------------------------------------------------------ #
    if verbose and 'time_to_resolve' in active.columns:
        print(f"\n{'─'*65}")
        print("VERBOSE: PnL BY PHASE")
        print(f"{'─'*65}")

        phases = [
            ('Phase 3 (0-45s)',   active['time_to_resolve'] <= 45),
            ('Phase 2 (45-180s)', (active['time_to_resolve'] > 45) & (active['time_to_resolve'] <= 180)),
            ('Phase 1 (180s+)',   active['time_to_resolve'] > 180),
        ]

        active_copy = active.copy()
        active_copy['equity_delta'] = active_copy['equity'].diff().fillna(0)

        for phase_name, mask in phases:
            phase_df = active_copy[mask]
            if len(phase_df) == 0:
                continue
            pnl = phase_df['equity_delta'].sum()
            trades = int(phase_df['fills_this_tick'].sum())
            trade_pct = phase_df['signal_side'].isin(['UP', 'DOWN']).mean()
            print(f"  {phase_name:<20} PnL: ${pnl:>+7.2f}  Trades: {trades:>4}  Active: {trade_pct:.0%}")

    print(f"\n{'='*65}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_replay.py <ticks.parquet> [--verbose]")
        sys.exit(1)

    path = sys.argv[1]
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    if not Path(path).exists():
        print(f"File not found: {path}")
        sys.exit(1)

    analyze(path, verbose=verbose)