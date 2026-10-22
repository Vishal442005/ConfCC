"""
confcc/simulator.py — SimPy-based ConfCC network simulator (W1).

Models a single-bottleneck dumbbell topology:
  Sender → [bottleneck link with time-varying capacity] → Receiver → ACK path

The simulator:
  1. Reads a Trace (capacity vs time) from the Linux tools or synthetic fallback.
  2. Drains packets through a FIFO bottleneck queue at the current capacity.
  3. For each ACK, calls the congestion controller's on_ack().
  4. Writes per-ACK records to a results CSV.

Design notes:
  - Closed-loop RTT feedback: we explicitly implement a "score arrives
    one RTT later" queue to test the delayed-feedback analysis (W1 task 5).
  - Loss is triggered when the bottleneck queue exceeds max_queue_packets.
  - This is intentionally a *fluid model*; the authoritative numbers come from ns-3.
"""
from __future__ import annotations

import csv
import time as wall_time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import simpy
import numpy as np

from ..common.traces import Trace, generate_synthetic_trace
from .controllers import Controller, make_controller

MSS = 1448       # bytes per segment
BITS_PER_BYTE = 8


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------
@dataclass
class AckRecord:
    t_ms: float
    rtt_ms: float
    cwnd: float
    qdelay_ms: float
    upper_bound_ms: float
    alpha_t: float
    covered: int
    goodput_bps: float
    flow_id: int = 0


# ---------------------------------------------------------------------------
# Bottleneck queue
# ---------------------------------------------------------------------------
class BottleneckQueue:
    """
    A SimPy Resource that drains packets at the current trace capacity.
    Capacity is updated every bin_ms (default 100 ms).
    """

    def __init__(self, env: simpy.Environment, trace: Trace,
                 prop_delay_ms: float = 10.0, max_queue_ms: float = 200.0):
        self.env = env
        self.trace = trace
        self.prop_delay_ms = prop_delay_ms
        self.max_queue_ms = max_queue_ms
        self._queue_bytes: float = 0.0
        self._draining = env.process(self._drain())

    def _capacity_bps(self) -> float:
        cap_mbps = self.trace.capacity_at(self.env.now)
        return cap_mbps * 1e6

    def _max_queue_bytes(self) -> float:
        return self._capacity_bps() * self.max_queue_ms / 1000.0 / BITS_PER_BYTE

    def _drain(self):
        TICK_MS = 1.0  # drain every 1 ms
        while True:
            yield self.env.timeout(TICK_MS)
            cap_bps = self._capacity_bps()
            drained = min(self._queue_bytes,
                          cap_bps * TICK_MS / 1000.0 / BITS_PER_BYTE)
            self._queue_bytes = max(0.0, self._queue_bytes - drained)

    def enqueue(self, bytes_: float) -> bool:
        """Return True if packet enqueued (not dropped)."""
        if self._queue_bytes + bytes_ > self._max_queue_bytes():
            return False
        self._queue_bytes += bytes_
        return True

    def qdelay_ms(self) -> float:
        cap_bps = self._capacity_bps()
        if cap_bps <= 0:
            return self.max_queue_ms
        return self._queue_bytes * BITS_PER_BYTE / cap_bps * 1000.0

    def propagation_delay_ms(self) -> float:
        return self.prop_delay_ms


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------
class ConfCCSimulator:
    """
    Main SimPy simulation loop for one (controller, trace, seed) run.
    """

    def __init__(
        self,
        controller: Controller,
        trace: Trace,
        duration_ms: float = 60_000.0,
        prop_delay_ms: float = 10.0,
        max_queue_ms: float = 200.0,
        seed: int = 1,
        flow_id: int = 0,
        delayed_feedback_rtt_ratio: float = 1.0,  # 1.0 = normal; >1 = extra delay
    ):
        self.controller = controller
        self.trace = trace
        self.duration_ms = duration_ms
        self.prop_delay_ms = prop_delay_ms
        self.max_queue_ms = max_queue_ms
        self.seed = seed
        self.flow_id = flow_id
        self.delayed_feedback_rtt_ratio = delayed_feedback_rtt_ratio
        self._records: List[AckRecord] = []
        self._rng = np.random.default_rng(seed)

    def run(self) -> List[AckRecord]:
        env = simpy.Environment()
        queue = BottleneckQueue(env, self.trace, self.prop_delay_ms, self.max_queue_ms)
        self.controller.reset()
        env.process(self._sender(env, queue))
        env.run(until=self.duration_ms)
        return self._records

    def _sender(self, env: simpy.Environment, queue: BottleneckQueue):
        """Generates packet bursts each RTT-sized slot."""
        cwnd = self.controller.cwnd
        total_bytes_sent = 0.0
        total_bytes_acked = 0.0
        t_start = env.now

        while env.now < self.duration_ms:
            # Send cwnd worth of packets
            bytes_per_window = cwnd * MSS
            sent_this_window = 0.0
            cap_bps = queue._capacity_bps()

            for _ in range(int(cwnd)):
                pkt = MSS
                dropped = not queue.enqueue(pkt)
                if not dropped:
                    sent_this_window += pkt
                    total_bytes_sent += pkt

            # Simulate RTT: prop delay × 2 + queueing delay
            q_delay = queue.qdelay_ms()
            rtt_ms = 2 * queue.propagation_delay_ms() + q_delay
            # Add small random jitter
            rtt_ms += self._rng.exponential(0.5)

            # Delayed feedback: score arrives after rtt * ratio ms
            feedback_delay = rtt_ms * self.delayed_feedback_rtt_ratio
            yield env.timeout(max(feedback_delay, 1.0))

            total_bytes_acked += sent_this_window
            elapsed_ms = env.now - t_start
            goodput_bps = (total_bytes_acked * BITS_PER_BYTE /
                           max(elapsed_ms / 1000.0, 1e-9))

            # Get upper bound for logging
            ub_val = float("nan")
            alpha_t = float("nan")
            if hasattr(self.controller, "_aci") and self.controller._aci is not None:
                ub_val = self.controller._aci.current_upper if self.controller._aci.current_upper is not None else (rtt_ms + 10.0)
                alpha_t = self.controller._aci.alpha_t
            elif hasattr(self.controller, "_pred") and self.controller._pred is not None:
                y_hat = self.controller._ewma.predict() if hasattr(self.controller, "_ewma") else rtt_ms
                ub_val = self.controller._pred.compute_upper(y_hat)
                alpha_t = getattr(self.controller._pred, "alpha", float("nan"))

            if not np.isnan(ub_val):
                covered = int(rtt_ms <= ub_val)
            else:
                covered = 1

            # Packet loss → trigger congestion signal
            if dropped or q_delay > self.max_queue_ms * 0.9:
                cwnd = self.controller.on_loss()
            else:
                cwnd = self.controller.on_ack(rtt_ms, cap_bps / 1e6, env.now)

            rec = AckRecord(
                t_ms=env.now,
                rtt_ms=rtt_ms,
                cwnd=cwnd,
                qdelay_ms=q_delay,
                upper_bound_ms=ub_val,
                alpha_t=alpha_t,
                covered=covered,
                goodput_bps=goodput_bps,
                flow_id=self.flow_id,
            )
            self._records.append(rec)

    def summary(self) -> dict:
        """Compute summary statistics from the records list."""
        if not self._records:
            return {}
        rtts = np.array([r.rtt_ms for r in self._records])
        cov  = np.array([r.covered for r in self._records])
        bps  = np.array([r.goodput_bps for r in self._records])
        return dict(
            goodput_mbps=float(bps[-1] / 1e6) if len(bps) else 0.0,
            p50_ms=float(np.percentile(rtts, 50)),
            p95_ms=float(np.percentile(rtts, 95)),
            p99_ms=float(np.percentile(rtts, 99)),
            coverage=float(np.mean(cov)),
            violation_rate=float(np.mean(1 - cov)),
            delayed_feedback_ratio=float(self.delayed_feedback_rtt_ratio),
            n_records=len(self._records),
        )

    def save_csv(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self._records:
            return
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(asdict(self._records[0]).keys()))
            writer.writeheader()
            for r in self._records:
                writer.writerow(asdict(r))


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from ..common.traces import generate_synthetic_trace

    trace = generate_synthetic_trace("synth", capacity_mbps=12.0)
    ctrl = make_controller("confcc", target_rtt_ms=50.0)

    t0 = wall_time.perf_counter()
    sim = ConfCCSimulator(ctrl, trace, duration_ms=30_000, seed=1)
    records = sim.run()
    elapsed = wall_time.perf_counter() - t0
    print(f"ConfCC: {len(records)} ack events in {elapsed:.2f}s (wall)")
    print(sim.summary())
