"""
Parameter Sweep — Fast Offline Iteration
=========================================
Runs the model against the replay fixture across a grid of parameters.
Use this to find good hyperparameters before going live.

Usage:
    python sweep_params.py --fixture tests/fixtures/recorded_event.parquet
    python sweep_params.py --fixture tests/fixtures/recorded_event.parquet --top 5
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import math
from itertools import product
from pathlib import Path


def load_model_class(path: str):
    spec = importlib.util.spec_from_file_location("model_submission", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ModelSubmission


def run_replay(model_class, config: dict, fixture_path: Path, tmp_path: Path):
    from polybench.replay import ReplayConfig, replay
    model = model_class(config=config)
    cfg = ReplayConfig(
        starting_capital=1000.0,
        slippage_bps=50.0,
        fee_rate=0.072,
        output_dir=tmp_path / "out",
        scratch_dir=tmp_path / "scratch",
    )
    result = replay(model, fixture_path, cfg)
    return result


def primary_score(result) -> float:
    m = result.metrics
    pnl = float(m.get("pnl_total", 0))
    sharpe = max(float(m.get("sharpe", 0)), 0)
    mdd = float(m.get("max_drawdown", 0))
    if sharpe == 0:
        return 0.0
    return pnl * sharpe * max(0.0, 1.0 - mdd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", default="tests/fixtures/recorded_event.parquet")
    parser.add_argument("--model", default="model_submission.py")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--quick", action="store_true",
                        help="Run a smaller grid for faster iteration")
    args = parser.parse_args()

    fixture = Path(args.fixture)
    if not fixture.exists():
        print(f"Fixture not found: {fixture}")
        print("Run: python replay.py --model-file examples/model_submission.py ...")
        sys.exit(1)

    model_class = load_model_class(args.model)

    # ------------------------------------------------------------------ #
    # Parameter grid
    # Focused on the parameters most likely to matter based on our
    # three-phase hypothesis
    # ------------------------------------------------------------------ #

    if args.quick:
        # Smaller grid for rapid iteration (~15 combinations)
        grid = {
            "force_flat_ttr":           [30.0, 45.0, 60.0],
            "phase2_entry_ttr":         [150.0, 180.0],
            "momentum_entry_threshold": [0.006, 0.010],
            "spread_tight":             [0.04],
            "spread_medium":            [0.07],
            "base_size":                [0.40],
            "mean_revert_threshold":    [0.06],
        }
    else:
        # Full grid (~96 combinations, takes a few minutes)
        grid = {
            "force_flat_ttr":           [30.0, 45.0, 60.0],
            "phase2_entry_ttr":         [150.0, 180.0, 210.0],
            "momentum_entry_threshold": [0.005, 0.008, 0.012],
            "spread_tight":             [0.03, 0.04, 0.05],
            "spread_medium":            [0.06, 0.08],
            "base_size":                [0.35, 0.50],
            "mean_revert_threshold":    [0.05, 0.07],
        }

    keys = list(grid.keys())
    values = list(grid.values())
    combinations = list(product(*values))

    print(f"Running {len(combinations)} parameter combinations...")
    print(f"Fixture: {fixture}")
    print()

    import tempfile
    results = []

    for i, combo in enumerate(combinations):
        config = dict(zip(keys, combo))

        with tempfile.TemporaryDirectory() as tmp:
            try:
                result = run_replay(model_class, config, fixture, Path(tmp))
                score = primary_score(result)
                m = result.metrics
                results.append({
                    "config": config,
                    "score": score,
                    "pnl": float(m.get("pnl_total", 0)),
                    "sharpe": float(m.get("sharpe", 0)),
                    "drawdown": float(m.get("max_drawdown", 0)),
                    "trades": int(m.get("n_trades", 0)),
                    "timeout_rate": float(m.get("timeout_rate", 0)),
                })
            except Exception as e:
                results.append({
                    "config": config,
                    "score": float("-inf"),
                    "error": str(e),
                })

        # Progress
        if (i + 1) % 10 == 0 or (i + 1) == len(combinations):
            best_so_far = max(r["score"] for r in results if math.isfinite(r["score"]))
            print(f"  {i+1}/{len(combinations)} done. Best score so far: {best_so_far:.4f}")

    # ------------------------------------------------------------------ #
    # Results
    # ------------------------------------------------------------------ #
    valid = [r for r in results if math.isfinite(r["score"])]
    valid.sort(key=lambda r: r["score"], reverse=True)

    print(f"\n{'='*70}")
    print(f"TOP {args.top} CONFIGURATIONS")
    print(f"{'='*70}")

    for rank, r in enumerate(valid[:args.top], 1):
        print(f"\n#{rank}  Score: {r['score']:.4f}")
        print(f"     PnL: ${r['pnl']:+.2f}  "
              f"Sharpe: {r['sharpe']:.3f}  "
              f"MaxDD: {r['drawdown']:.2%}  "
              f"Trades: {r['trades']}")
        print(f"     Config: {r['config']}")

    # Also show baseline for comparison
    print(f"\n{'─'*70}")
    print("BASELINE COMPARISON (best result above)")
    if valid:
        best = valid[0]
        with tempfile.TemporaryDirectory() as tmp:
            result = run_replay(model_class, best["config"], fixture, Path(tmp))
            bm = result.baseline_metrics
            base_score = max(bm.get("sharpe", 0), 0) * bm.get("pnl_total", 0) * max(0, 1 - bm.get("max_drawdown", 0))
            print(f"  Baseline score:  {base_score:.4f}")
            print(f"  Baseline PnL:    ${bm.get('pnl_total', 0):+.2f}")
            print(f"  Baseline Sharpe: {bm.get('sharpe', 0):.3f}")
            print(f"  Baseline MaxDD:  {bm.get('max_drawdown', 0):.2%}")
            print(f"\n  Your model edge: {best['score'] - base_score:+.4f}")

    print(f"\n{'='*70}")
    print("SUGGESTED NEXT CONFIG TO TRY:")
    if valid:
        best_config = valid[0]["config"]
        print(f"  {best_config}")
    print()


if __name__ == "__main__":
    main()