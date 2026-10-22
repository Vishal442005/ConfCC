#!/usr/bin/env python3
"""
trace_to_rates.py
-----------------
Convert a Mahimahi-format trace (one integer millisecond timestamp per 1500-byte
delivery opportunity) into a two-column rate-bin file for the ns-3 trace-driven
bottleneck scenario (S2).

Usage:
    python3 trace_to_rates.py <input.trace> <output.rates> [--bin-ms 100]

Output format (space-separated, no header beyond the comment line):
    t_ms  rate_bps
"""

import argparse
import sys
import numpy as np
from pathlib import Path


PACKET_BYTES = 1500


def trace_to_rates(trace_path: str, output_path: str, bin_ms: int = 100) -> None:
    ts = np.loadtxt(trace_path, dtype=np.int64)
    if ts.ndim == 0:
        ts = ts.reshape(1)

    n_bins = int(ts.max()) // bin_ms + 1
    counts = np.bincount(ts // bin_ms, minlength=n_bins)
    rate_bps = counts * PACKET_BYTES * 8 / (bin_ms / 1000.0)

    t_ms = np.arange(n_bins, dtype=np.int64) * bin_ms
    out = np.column_stack([t_ms, rate_bps.astype(np.int64)])

    np.savetxt(
        output_path,
        out,
        fmt="%d",
        header="t_ms rate_bps",
        comments="# ",
    )
    print(
        f"[trace_to_rates] {n_bins} bins, "
        f"mean rate={rate_bps.mean()/1e6:.2f} Mbps, "
        f"peak={rate_bps.max()/1e6:.2f} Mbps → {output_path}"
    )


def generate_synthetic_trace(output_path: str, capacity_mbps: float = 12.0,
                              duration_s: int = 60, variation: float = 0.3) -> None:
    """
    Generate a synthetic Mahimahi-style trace for testing without real traces.
    Models a bursty link with sinusoidal capacity variation.
    """
    rng = np.random.default_rng(42)
    capacity_bps = capacity_mbps * 1e6
    pkts_per_ms = capacity_bps / (PACKET_BYTES * 8 * 1000)

    timestamps = []
    t_ms = 0
    end_ms = duration_s * 1000

    # Sinusoidal variation + noise
    while t_ms < end_ms:
        phase = 2 * np.pi * t_ms / 20000  # 20-second period
        local_rate = pkts_per_ms * (1 + variation * np.sin(phase))
        local_rate = max(0.1, local_rate) * rng.lognormal(0, 0.1)
        # inter-arrival in ms
        iat = 1.0 / local_rate
        t_ms += iat
        if t_ms < end_ms:
            timestamps.append(int(t_ms))

    np.savetxt(output_path, np.array(timestamps, dtype=np.int64), fmt="%d")
    print(f"[generate_synthetic] {len(timestamps)} packets → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mahimahi trace → ns-3 rate bins")
    sub = parser.add_subparsers(dest="cmd")

    p_conv = sub.add_parser("convert", help="Convert existing Mahimahi trace")
    p_conv.add_argument("input", help="Input .trace file")
    p_conv.add_argument("output", help="Output .rates file")
    p_conv.add_argument("--bin-ms", type=int, default=100, help="Bin width in ms")

    p_gen = sub.add_parser("generate", help="Generate synthetic trace")
    p_gen.add_argument("output", help="Output .trace file")
    p_gen.add_argument("--capacity-mbps", type=float, default=12.0)
    p_gen.add_argument("--duration-s", type=int, default=60)
    p_gen.add_argument("--variation", type=float, default=0.3,
                       help="Fractional sinusoidal variation (0=flat)")

    args = parser.parse_args()
    if args.cmd == "convert":
        trace_to_rates(args.input, args.output, args.bin_ms)
    elif args.cmd == "generate":
        generate_synthetic_trace(args.output, args.capacity_mbps,
                                 args.duration_s, args.variation)
    else:
        parser.print_help()
        sys.exit(1)
