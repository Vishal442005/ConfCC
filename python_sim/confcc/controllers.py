"""
confcc/controllers.py — Congestion-window controllers (Python model).

Implements Python models of:
  ConfCC       — ACI-wrapped delay-target controller
  NewReno      — AIMD: +1/RTT on ACK, halve on loss/delay signal
  CUBIC        — cubic window growth function
  Vegas        — compare expected vs actual rate; adjust by alpha/beta
  BBR          — simplified: max-BW filter + min-RTT filter + pacing gain
  Copa         — simplified: target rate = 1/(delta * queuing delay)
  DelayTarget  — same control law as ConfCC without ACI (ablation)

All controllers expose the same interface:
    on_ack(rtt_ms, capacity_mbps, t_ms)  → new cwnd (segments)
    reset()

Note: these are *models* for fast parameter sweeps and ablations.
Real ns-3 / Pantheon results from Linux are authoritative for headline numbers.
"""
from __future__ import annotations

from typing import Deque, List, Optional
from collections import deque
import math

import numpy as np

from .predictors import (
    ACIBoundsTracker,
    EWMAPredictor,
    FixedQuantilePredictor,
    SplitConformalPredictor,
)

# Segment size (bytes) used to convert cwnd (bytes) ↔ segments
MSS = 1448


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class Controller:
    """Abstract controller interface."""

    name: str = "base"

    def on_ack(self, rtt_ms: float, capacity_mbps: float,
               t_ms: float) -> float:
        """
        Process one ACK / RTT sample.

        Parameters
        ----------
        rtt_ms       : measured RTT in ms
        capacity_mbps: current link capacity (for emulation; not all schemes use it)
        t_ms         : simulation clock in ms

        Returns
        -------
        cwnd in segments (integer)
        """
        raise NotImplementedError

    def on_loss(self) -> float:
        """React to a packet loss event. Returns new cwnd."""
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}()"


# ---------------------------------------------------------------------------
# ConfCC — ACI-wrapped controller
# ---------------------------------------------------------------------------
class ConfCC(Controller):
    """
    ConfCC congestion controller.

    cwnd_{t+1} = cwnd_t × min(1, T / U_t)

    where U_t is the ACI upper bound from ACIBoundsTracker and
    T is the target RTT.
    """

    name = "confcc"

    def __init__(
        self,
        target_rtt_ms: float = 50.0,
        alpha: float = 0.05,
        eta: float = 0.01,
        gamma: float = 0.1,
        window: int = 500,
        init_cwnd: float = 10.0,
        max_cwnd: float = 1000.0,
    ):
        self.target_rtt_ms = target_rtt_ms
        self.max_cwnd = max_cwnd
        self.cwnd = init_cwnd
        self._ewma = EWMAPredictor(gamma=gamma, init_rtt=target_rtt_ms)
        self._aci  = ACIBoundsTracker(alpha=alpha, eta=eta, window=window)
        # Logging
        self._log: List[dict] = []

    def on_ack(self, rtt_ms: float, capacity_mbps: float, t_ms: float) -> float:
        y_hat = self._ewma.predict()
        ub = self._aci.compute_upper(self._ewma.predict())

        # Window control law:
        # Below target delay: grow cwnd (+1 packet per RTT epoch -> +1/cwnd per ACK)
        # Above target delay: multiplicative delay reduction
        if ub <= self.target_rtt_ms:
            self.cwnd = float(np.clip(self.cwnd + 1.0 / max(1.0, self.cwnd), 1.0, self.max_cwnd))
        else:
            ratio = self.target_rtt_ms / max(ub, 1.0)
            self.cwnd = float(np.clip(self.cwnd * ratio, 1.0, self.max_cwnd))

        # Update state
        self._aci.update(rtt_ms, y_hat)
        self._ewma.update(rtt_ms)

        self._log.append(dict(
            t_ms=t_ms,
            rtt_ms=rtt_ms,
            cwnd=self.cwnd,
            upper_bound_ms=ub,
            alpha_t=self._aci.alpha_t,
            covered=int(rtt_ms <= ub),
        ))
        return self.cwnd

    def on_loss(self) -> float:
        self.cwnd = max(1.0, self.cwnd * 0.5)
        return self.cwnd

    def reset(self):
        self.cwnd = 10.0
        self._ewma.reset(self.target_rtt_ms)
        self._aci.reset()
        self._log.clear()

    @property
    def log(self) -> List[dict]:
        return self._log


# ---------------------------------------------------------------------------
# NewReno / AIMD
# ---------------------------------------------------------------------------
class NewReno(Controller):
    """
    Simplified NewReno: AIMD in congestion avoidance.
    +1/cwnd per ACK (≈ +1 per RTT), halve on loss.
    """

    name = "newreno"

    def __init__(self, init_cwnd: float = 10.0, max_cwnd: float = 1000.0,
                 ssthresh: float = 32.0):
        self.cwnd = init_cwnd
        self.ssthresh = ssthresh
        self.max_cwnd = max_cwnd

    def on_ack(self, rtt_ms: float, capacity_mbps: float, t_ms: float) -> float:
        if self.cwnd < self.ssthresh:
            self.cwnd = min(self.cwnd + 1.0, self.max_cwnd)   # slow start
        else:
            self.cwnd = min(self.cwnd + 1.0 / self.cwnd, self.max_cwnd)
        return self.cwnd

    def on_loss(self) -> float:
        self.ssthresh = max(self.cwnd / 2.0, 2.0)
        self.cwnd = 1.0
        return self.cwnd

    def reset(self):
        self.cwnd = 10.0
        self.ssthresh = 32.0


# ---------------------------------------------------------------------------
# CUBIC
# ---------------------------------------------------------------------------
class CUBIC(Controller):
    """
    CUBIC window growth function.
    W(t) = C(t - K)³ + W_max   with K = (W_max β / C)^(1/3)
    β = 0.7, C = 0.4
    """

    name = "cubic"

    C: float = 0.4
    BETA: float = 0.7

    def __init__(self, init_cwnd: float = 10.0, max_cwnd: float = 1000.0):
        self.cwnd = init_cwnd
        self.max_cwnd = max_cwnd
        self._w_max: float = init_cwnd
        self._t_epoch: float = 0.0    # ms since last loss
        self._k: float = 0.0

    def _cubic_w(self, t_since_loss_s: float) -> float:
        return self.C * (t_since_loss_s - self._k) ** 3 + self._w_max

    def on_ack(self, rtt_ms: float, capacity_mbps: float, t_ms: float) -> float:
        t_since_loss_s = (t_ms - self._t_epoch) / 1000.0
        w_cubic = self._cubic_w(t_since_loss_s)
        self.cwnd = float(np.clip(w_cubic, 1.0, self.max_cwnd))
        return self.cwnd

    def on_loss(self) -> float:
        self._w_max = self.cwnd
        self._k = (self._w_max * self.BETA / self.C) ** (1.0 / 3.0)
        self.cwnd = max(1.0, self.cwnd * (1 - self.BETA))
        self._t_epoch = 0.0
        return self.cwnd

    def reset(self):
        self.cwnd = 10.0
        self._w_max = 10.0
        self._t_epoch = 0.0
        self._k = 0.0


# ---------------------------------------------------------------------------
# Vegas
# ---------------------------------------------------------------------------
class Vegas(Controller):
    """
    Simplified Vegas: compare expected and actual send rate.
    Increase window if actual_rate > expected * (1 - alpha_v),
    decrease if actual_rate < expected * (1 - beta_v).
    """

    name = "vegas"

    def __init__(self, alpha_v: float = 2.0, beta_v: float = 4.0,
                 init_cwnd: float = 10.0, max_cwnd: float = 1000.0):
        self.cwnd = init_cwnd
        self.max_cwnd = max_cwnd
        self.alpha_v = alpha_v
        self.beta_v = beta_v
        self._min_rtt: float = float("inf")

    def on_ack(self, rtt_ms: float, capacity_mbps: float, t_ms: float) -> float:
        self._min_rtt = min(self._min_rtt, rtt_ms)
        if self._min_rtt == float("inf") or self._min_rtt <= 0:
            return self.cwnd

        expected = self.cwnd / (self._min_rtt / 1000.0)   # segments/s
        actual   = self.cwnd / (rtt_ms / 1000.0)

        diff = (expected - actual) * (self._min_rtt / 1000.0)
        if diff < self.alpha_v:
            self.cwnd = min(self.cwnd + 1.0, self.max_cwnd)
        elif diff > self.beta_v:
            self.cwnd = max(self.cwnd - 1.0, 1.0)
        return self.cwnd

    def on_loss(self) -> float:
        self.cwnd = max(self.cwnd / 2.0, 1.0)
        return self.cwnd

    def reset(self):
        self.cwnd = 10.0
        self._min_rtt = float("inf")


# ---------------------------------------------------------------------------
# BBR (simplified model)
# ---------------------------------------------------------------------------
class BBR(Controller):
    """
    Simplified BBR model.
    - max-BW filter over last 10 RTTs
    - min-RTT filter (windowed, 10 s)
    - pacing gain cycle: [1.25, 0.75, 1, 1, 1, 1, 1, 1]
    - cwnd cap = BDP × cwnd_gain (2.0)

    This is a *fluid model*, not the full BBR state machine.
    """

    name = "bbr"

    PACING_GAIN_CYCLE = [1.25, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    CWND_GAIN = 2.0
    BW_FILTER_LEN = 10
    RTT_FILTER_MS = 10_000  # 10 seconds

    def __init__(self, init_cwnd: float = 10.0, max_cwnd: float = 1000.0):
        self.cwnd = init_cwnd
        self.max_cwnd = max_cwnd
        self._bw_samples: Deque[float] = deque(maxlen=self.BW_FILTER_LEN)
        self._rtt_history: Deque[tuple] = deque()  # (t_ms, rtt_ms)
        self._cycle: int = 0

    def _max_bw(self) -> float:
        return max(self._bw_samples) if self._bw_samples else 1.0

    def _min_rtt(self, t_ms: float) -> float:
        cutoff = t_ms - self.RTT_FILTER_MS
        while self._rtt_history and self._rtt_history[0][0] < cutoff:
            self._rtt_history.popleft()
        if not self._rtt_history:
            return 20.0
        return min(r for _, r in self._rtt_history)

    def on_ack(self, rtt_ms: float, capacity_mbps: float, t_ms: float) -> float:
        # Update BW estimate (delivered bytes / RTT)
        bw = (self.cwnd * MSS * 8) / max(rtt_ms / 1000.0, 1e-6) / 1e6   # Mbps
        self._bw_samples.append(bw)
        self._rtt_history.append((t_ms, rtt_ms))

        pacing_gain = self.PACING_GAIN_CYCLE[self._cycle % len(self.PACING_GAIN_CYCLE)]
        self._cycle = (self._cycle + 1) % len(self.PACING_GAIN_CYCLE)

        max_bw = self._max_bw()
        min_rtt = self._min_rtt(t_ms)
        bdp = max_bw * 1e6 * (min_rtt / 1000.0) / (MSS * 8)   # segments
        self.cwnd = float(np.clip(self.CWND_GAIN * bdp * pacing_gain,
                                  1.0, self.max_cwnd))
        return self.cwnd

    def on_loss(self) -> float:
        return self.cwnd   # BBR does not react to loss directly

    def reset(self):
        self.cwnd = 10.0
        self._bw_samples.clear()
        self._rtt_history.clear()
        self._cycle = 0


# ---------------------------------------------------------------------------
# Copa (simplified)
# ---------------------------------------------------------------------------
class Copa(Controller):
    """
    Simplified Copa: target rate = 1 / (delta * queueing delay).
    cwnd = target_rate * RTT_min
    Competitive-mode switching is omitted (label as model in paper).
    """

    name = "copa"

    def __init__(self, delta: float = 0.5, init_cwnd: float = 10.0,
                 max_cwnd: float = 1000.0):
        self.delta = delta
        self.cwnd = init_cwnd
        self.max_cwnd = max_cwnd
        self._min_rtt: float = float("inf")

    def on_ack(self, rtt_ms: float, capacity_mbps: float, t_ms: float) -> float:
        self._min_rtt = min(self._min_rtt, rtt_ms)
        q_delay = max(rtt_ms - self._min_rtt, 1e-3)
        target_rate_pkt_per_s = 1.0 / (self.delta * q_delay / 1000.0)
        bdp = target_rate_pkt_per_s * (rtt_ms / 1000.0)
        self.cwnd = float(np.clip(bdp, 1.0, self.max_cwnd))
        return self.cwnd

    def on_loss(self) -> float:
        self.cwnd = max(self.cwnd / 2.0, 1.0)
        return self.cwnd

    def reset(self):
        self.cwnd = 10.0
        self._min_rtt = float("inf")


# ---------------------------------------------------------------------------
# DelayTarget (ConfCC without ACI — ablation)
# ---------------------------------------------------------------------------
class DelayTargetNoACI(Controller):
    """
    Same window-control law as ConfCC but with a fixed-quantile (no ACI) bound.
    Exact ablation for the paper.
    """

    name = "delay_target_no_aci"

    def __init__(self, target_rtt_ms: float = 50.0, gamma: float = 0.1,
                 margin_ms: float = 10.0, init_cwnd: float = 10.0,
                 max_cwnd: float = 1000.0):
        self.target_rtt_ms = target_rtt_ms
        self.max_cwnd = max_cwnd
        self.cwnd = init_cwnd
        self._ewma = EWMAPredictor(gamma=gamma, init_rtt=target_rtt_ms)
        self._pred = FixedQuantilePredictor(margin_ms=margin_ms)

    def on_ack(self, rtt_ms: float, capacity_mbps: float, t_ms: float) -> float:
        y_hat = self._ewma.predict()
        ub = self._pred.compute_upper(y_hat)
        if ub <= self.target_rtt_ms:
            self.cwnd = float(np.clip(self.cwnd + 1.0 / max(1.0, self.cwnd), 1.0, self.max_cwnd))
        else:
            ratio = self.target_rtt_ms / max(ub, 1.0)
            self.cwnd = float(np.clip(self.cwnd * ratio, 1.0, self.max_cwnd))
        self._ewma.update(rtt_ms)
        return self.cwnd

    def on_loss(self) -> float:
        self.cwnd = max(self.cwnd / 2.0, 1.0)
        return self.cwnd

    def reset(self):
        self.cwnd = 10.0
        self._ewma.reset(self.target_rtt_ms)


# ---------------------------------------------------------------------------
# SplitConformalCC (Static calibration set — ablation)
# ---------------------------------------------------------------------------
class SplitConformalCC(Controller):
    """
    Delay-target controller with fixed Split-Conformal calibration.
    Unlike ACI, calibration is frozen after warmup, leading to miscoverage
    under distribution shift.
    """

    name = "split_conformal"

    def __init__(self, target_rtt_ms: float = 50.0, alpha: float = 0.05,
                 window: int = 200, gamma: float = 0.1, init_cwnd: float = 10.0,
                 max_cwnd: float = 1000.0, **kwargs):
        self.target_rtt_ms = target_rtt_ms
        self.max_cwnd = max_cwnd
        self.cwnd = init_cwnd
        self.alpha = alpha
        self._ewma = EWMAPredictor(gamma=gamma, init_rtt=target_rtt_ms)
        self._pred = SplitConformalPredictor(alpha=alpha, calib_window=window)

    def on_ack(self, rtt_ms: float, capacity_mbps: float, t_ms: float) -> float:
        y_hat = self._ewma.predict()
        self._pred.add_calibration(rtt_ms, y_hat)
        ub = self._pred.compute_upper(y_hat)
        if ub <= self.target_rtt_ms:
            self.cwnd = float(np.clip(self.cwnd + 1.0 / max(1.0, self.cwnd), 1.0, self.max_cwnd))
        else:
            ratio = self.target_rtt_ms / max(ub, 1.0)
            self.cwnd = float(np.clip(self.cwnd * ratio, 1.0, self.max_cwnd))
        self._ewma.update(rtt_ms)
        return self.cwnd

    def on_loss(self) -> float:
        self.cwnd = max(self.cwnd / 2.0, 1.0)
        return self.cwnd

    def reset(self):
        self.cwnd = 10.0
        self._ewma.reset(self.target_rtt_ms)
        self._pred.reset()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
ALL_CONTROLLERS = {
    "confcc":               lambda **kw: ConfCC(**kw),
    "newreno":              lambda **kw: NewReno(**kw),
    "cubic":                lambda **kw: CUBIC(**kw),
    "vegas":                lambda **kw: Vegas(**kw),
    "bbr":                  lambda **kw: BBR(**kw),
    "copa":                 lambda **kw: Copa(**kw),
    "delay_target_no_aci":  lambda **kw: DelayTargetNoACI(**kw),
    "split_conformal":      lambda **kw: SplitConformalCC(**kw),
}


import inspect as _inspect


def make_controller(name: str, **kwargs) -> Controller:
    """
    Create a controller by name, automatically filtering kwargs to only those
    accepted by the concrete class so callers can always pass the full ConfCC
    kwarg set without worrying about incompatible parameters.
    """
    if name not in ALL_CONTROLLERS:
        raise ValueError(f"Unknown controller '{name}'. "
                         f"Available: {list(ALL_CONTROLLERS)}")
    # Build a dummy instance to discover which kwargs are accepted
    factory = ALL_CONTROLLERS[name]
    # Peek at the lambda's closure to get the real class
    cls_map = {
        "confcc":               ConfCC,
        "newreno":              NewReno,
        "cubic":                CUBIC,
        "vegas":                Vegas,
        "bbr":                  BBR,
        "copa":                 Copa,
        "delay_target_no_aci":  DelayTargetNoACI,
        "split_conformal":      SplitConformalCC,
    }
    cls = cls_map.get(name)
    if cls is not None:
        accepted = set(_inspect.signature(cls.__init__).parameters) - {"self"}
        kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    return factory(**kwargs)


if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(1)
    for ctrl_name in ["confcc", "cubic", "bbr", "vegas", "newreno", "copa"]:
        ctrl = make_controller(ctrl_name, target_rtt_ms=50.0)
        ctrl.reset()
        cwnds = []
        for t in range(500):
            rtt = 30 + 10 * rng.standard_normal() + (20 if t > 300 else 0)
            rtt = max(1.0, rtt)
            cwnds.append(ctrl.on_ack(rtt, 12.0, t * 100.0))
        print(f"{ctrl_name:25s}: mean_cwnd={np.mean(cwnds):.1f}, "
              f"final={cwnds[-1]:.1f}")
