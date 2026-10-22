"""
common/traces.py — Trace library for Windows simulation.

Loads Mahimahi-style .rates files produced by the Linux side
(tools/traces/*.rates) and also generates synthetic traces for
offline testing when real Linux CSVs are not yet available.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np

# Root of the ConfCC project tree
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_TRACES_DIR = _REPO_ROOT / "tools" / "traces"
_RESULTS_DIR = _REPO_ROOT / "ns-3-dev" / "results" / "ns3" / "confcc"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class RatePoint:
    """One time-bin entry from a .rates file."""
    t_ms: float      # bin start time in milliseconds
    rate_mbps: float  # capacity in Mbit/s


@dataclass
class Trace:
    name: str
    points: List[RatePoint] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return (self.points[-1].t_ms / 1000.0) if self.points else 0.0

    def capacity_at(self, t_ms: float) -> float:
        """Binary search for the capacity at simulation time t_ms."""
        lo, hi = 0, len(self.points) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.points[mid].t_ms <= t_ms:
                lo = mid
            else:
                hi = mid - 1
        return self.points[lo].rate_mbps

    def iter_windows(self, bin_ms: float = 100.0) -> Iterator[RatePoint]:
        yield from self.points


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_rates_file(path: Path) -> Trace:
    """
    Load a Mahimahi-style .rates file.
    Expected format (CSV or space-separated):  t_ms  rate_mbps
    """
    path = Path(path)
    points: List[RatePoint] = []
    with open(path, newline="") as fh:
        # Detect delimiter
        sample = fh.read(512)
        fh.seek(0)
        delimiter = "," if "," in sample else None  # None = whitespace
        reader = csv.reader(fh, delimiter=delimiter) if delimiter else \
            (row.split() for row in fh if row.strip() and not row.startswith("#"))
        for row in reader:
            try:
                t_ms = float(row[0])
                rate = float(row[1])
                # Convert bps to Mbps if raw values are in bps (> 1000)
                if rate > 1000.0:
                    rate = rate / 1e6
                points.append(RatePoint(t_ms=t_ms, rate_mbps=rate))
            except (ValueError, IndexError):
                continue
    return Trace(name=path.stem, points=points)


def load_all_traces(traces_dir: Optional[Path] = None) -> dict[str, Trace]:
    """Load all .rates files found in the traces directory."""
    d = Path(traces_dir) if traces_dir else _TRACES_DIR
    traces: dict[str, Trace] = {}
    if d.exists():
        for f in sorted(d.glob("*.rates")):
            traces[f.stem] = load_rates_file(f)
    if not traces:
        print("[traces] No .rates files found; generating synthetic fallbacks.")
        traces = generate_synthetic_traces()
    return traces


# ---------------------------------------------------------------------------
# Synthetic trace generation (fallback when Linux files absent)
# ---------------------------------------------------------------------------
def _sinusoidal_capacity(
    t_ms: float,
    base_mbps: float,
    variation: float,
    period_s: float = 10.0,
) -> float:
    t_s = t_ms / 1000.0
    return max(0.1, base_mbps * (1 + variation * math.sin(2 * math.pi * t_s / period_s)))


def generate_synthetic_trace(
    name: str,
    capacity_mbps: float = 12.0,
    duration_s: float = 60.0,
    variation: float = 0.3,
    bin_ms: float = 100.0,
    rng: Optional[np.random.Generator] = None,
) -> Trace:
    if rng is None:
        rng = np.random.default_rng(42)
    points: List[RatePoint] = []
    t = 0.0
    while t <= duration_s * 1000:
        cap = _sinusoidal_capacity(t, capacity_mbps, variation)
        noise = rng.normal(0, 0.05 * cap)
        cap = max(0.1, cap + noise)
        points.append(RatePoint(t_ms=t, rate_mbps=cap))
        t += bin_ms
    return Trace(name=name, points=points)


def generate_synthetic_traces() -> dict[str, Trace]:
    profiles = {
        "verizon": dict(capacity_mbps=8.0,  variation=0.6),
        "att":     dict(capacity_mbps=12.0, variation=0.4),
        "tmobile": dict(capacity_mbps=20.0, variation=0.25),
        "synth":   dict(capacity_mbps=12.0, variation=0.3),
    }
    rng = np.random.default_rng(0)
    return {
        name: generate_synthetic_trace(name, rng=rng, **params)
        for name, params in profiles.items()
    }


# ---------------------------------------------------------------------------
# Linux results loader
# ---------------------------------------------------------------------------
def load_linux_summary(results_dir: Optional[Path] = None):
    """
    Load the summary.csv written by the Linux ns-3 runs.
    Returns a pandas DataFrame, or None if the file does not exist yet.
    """
    try:
        import pandas as pd
    except ImportError:
        return None

    d = Path(results_dir) if results_dir else _RESULTS_DIR
    summary = d / "summary.csv"
    if not summary.exists():
        return None
    return pd.read_csv(summary)


if __name__ == "__main__":
    # Quick self-test
    traces = load_all_traces()
    for name, tr in traces.items():
        print(f"{name}: {len(tr.points)} bins, {tr.duration_s:.1f}s, "
              f"mean={sum(p.rate_mbps for p in tr.points)/len(tr.points):.2f} Mbps")
