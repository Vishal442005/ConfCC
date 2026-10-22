"""
confcc/sweep.py — Parameter sweep runner for the Windows ConfCC simulator (W1).

Runs the full factorial of:
  controller × trace × target_rtt × alpha × gamma × window × seed

Uses multiprocessing for parallelism (no GNU parallel dependency on Windows).

Usage:
    # Dry-run (print config only)
    python -m confcc.sweep --dry-run

    # Quick test (2 seeds)
    python -m confcc.sweep --seeds 2 --jobs 4

    # Full sweep (30 seeds, all params)
    python -m confcc.sweep --seeds 30 --jobs 0

Results are saved to:
    results/confcc/summary.csv              (one row per run)
    results/confcc/<scheme>/<trace>_s<seed>.csv  (per-ACK records)
"""
from __future__ import annotations

import argparse
import csv
import itertools
import multiprocessing as mp
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

# Relative imports when called as a module
import sys, os
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from windows_sim.common.traces import load_all_traces, Trace
from windows_sim.confcc.controllers import make_controller, ALL_CONTROLLERS
from windows_sim.confcc.simulator import ConfCCSimulator

RESULTS_ROOT = Path(__file__).parent.parent / "results" / "confcc"


# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------
ALL_SCHEMES = ["confcc", "delay_target_no_aci", "cubic", "bbr", "vegas",
               "newreno", "copa"]

PARAM_GRID = {
    "alpha":      [0.01, 0.05, 0.10],
    "gamma":      [0.005, 0.01, 0.05],
    "window":     [100, 500, 2000],
    "target_rtt": [20, 50, 100, 200],
}


# ---------------------------------------------------------------------------
# Worker function (must be top-level for multiprocessing on Windows)
# ---------------------------------------------------------------------------
@dataclass
class SweepTask:
    scheme: str
    trace_name: str
    seed: int
    alpha: float = 0.05
    gamma: float = 0.01
    window: int = 500
    target_rtt: float = 50.0
    duration_ms: float = 60_000.0
    save_per_ack: bool = True


def _run_one(task: SweepTask) -> dict:
    """Execute one simulation run. Must be picklable for mp.Pool."""
    from windows_sim.common.traces import generate_synthetic_traces
    from windows_sim.confcc.controllers import make_controller
    from windows_sim.confcc.simulator import ConfCCSimulator

    traces = generate_synthetic_traces()
    trace = traces.get(task.trace_name, list(traces.values())[0])

    ctrl_kwargs = dict(
        target_rtt_ms=task.target_rtt,
        alpha=task.alpha,
        gamma=task.gamma,
        window=task.window,
    )
    # Non-ConfCC controllers ignore extra kwargs gracefully via **kwargs
    if task.scheme not in ("confcc", "delay_target_no_aci"):
        ctrl_kwargs = {}

    ctrl = make_controller(task.scheme, **ctrl_kwargs)
    sim = ConfCCSimulator(ctrl, trace, duration_ms=task.duration_ms, seed=task.seed)
    records = sim.run()
    summ = sim.summary()

    # Save per-ACK CSV
    if task.save_per_ack and records:
        out_dir = RESULTS_ROOT / task.scheme
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"{task.trace_name}_s{task.seed:03d}.csv"
        sim.save_csv(csv_path)

    return dict(
        scenario="python_sim",
        scheme=task.scheme,
        trace=task.trace_name,
        seed=task.seed,
        alpha=task.alpha,
        gamma=task.gamma,
        window=task.window,
        target_rtt_ms=task.target_rtt,
        **summ,
    )


# ---------------------------------------------------------------------------
# Build task list
# ---------------------------------------------------------------------------
def build_tasks(
    schemes: List[str],
    trace_names: List[str],
    seeds: int,
    alpha_values: Optional[List[float]] = None,
    gamma_values: Optional[List[float]] = None,
    window_values: Optional[List[int]] = None,
    target_rtts: Optional[List[float]] = None,
    param_sweep: bool = False,
) -> List[SweepTask]:
    tasks = []
    # Baseline run (default params)
    for scheme, trace, seed in itertools.product(schemes, trace_names, range(1, seeds + 1)):
        tasks.append(SweepTask(
            scheme=scheme,
            trace_name=trace,
            seed=seed,
        ))

    if param_sweep:
        # Full parameter grid for ConfCC only
        for alpha, gamma, window, tgt, trace, seed in itertools.product(
            alpha_values or PARAM_GRID["alpha"],
            gamma_values or PARAM_GRID["gamma"],
            window_values or PARAM_GRID["window"],
            target_rtts or PARAM_GRID["target_rtt"],
            trace_names,
            range(1, seeds + 1),
        ):
            tasks.append(SweepTask(
                scheme="confcc",
                trace_name=trace,
                seed=seed,
                alpha=alpha,
                gamma=gamma,
                window=window,
                target_rtt=tgt,
            ))
    return tasks


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_sweep(tasks: List[SweepTask], jobs: int, dry: bool) -> List[dict]:
    if dry:
        print(f"[dry-run] Would execute {len(tasks)} tasks")
        for t in tasks[:5]:
            print(f"  {t}")
        if len(tasks) > 5:
            print(f"  ... and {len(tasks)-5} more")
        return []

    n_jobs = jobs if jobs > 0 else max(1, mp.cpu_count() - 1)
    print(f"[sweep] Running {len(tasks)} tasks on {n_jobs} workers …")
    t0 = time.perf_counter()
    with mp.Pool(n_jobs) as pool:
        results = pool.map(_run_one, tasks)
    elapsed = time.perf_counter() - t0
    print(f"[sweep] Done in {elapsed:.1f}s")
    return results


def save_summary(results: List[dict], path: Path):
    if not results:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(results[0].keys())
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"[sweep] Summary written to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ConfCC Python sweep (W1)")
    parser.add_argument("--schemes",    nargs="+", default=ALL_SCHEMES)
    parser.add_argument("--traces",     nargs="+", default=["synth", "verizon", "att", "tmobile"])
    parser.add_argument("--seeds",      type=int, default=30)
    parser.add_argument("--jobs",       type=int, default=0,
                        help="Workers (0 = all cores - 1)")
    parser.add_argument("--param-sweep", action="store_true",
                        help="Run full parameter grid for ConfCC")
    parser.add_argument("--dry-run",    action="store_true")
    args = parser.parse_args()

    tasks = build_tasks(
        schemes=args.schemes,
        trace_names=args.traces,
        seeds=args.seeds,
        param_sweep=args.param_sweep,
    )
    results = run_sweep(tasks, args.jobs, args.dry_run)
    if results:
        save_summary(results, RESULTS_ROOT / "summary.csv")


if __name__ == "__main__":
    main()
