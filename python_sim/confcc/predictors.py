"""
confcc/predictors.py — RTT predictors and Adaptive Conformal Interval (ACI).

Implements:
  - EWMAPredictor       : exponentially weighted moving average (base predictor)
  - ACIBoundsTracker    : Adaptive Conformal Inference upper-bound tracker
  - SplitConformalPredictor : fixed-calibration-set split-conformal (ablation)
  - FixedQuantilePredictor  : uncalibrated fixed-quantile baseline

The ACI algorithm follows Gibbs & Candès (2021) "Adaptive Conformal Inference
Under Distribution Shift".
"""
from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Base RTT predictor
# ---------------------------------------------------------------------------
class EWMAPredictor:
    """
    Exponentially Weighted Moving Average predictor.
    y_hat_{t+1} = (1 - gamma) * y_hat_t + gamma * RTT_t
    """

    def __init__(self, gamma: float = 0.1, init_rtt: float = 20.0):
        assert 0 < gamma <= 1.0, "gamma must be in (0, 1]"
        self.gamma = gamma
        self._y_hat: float = init_rtt

    def predict(self) -> float:
        """Return current one-step-ahead prediction."""
        return self._y_hat

    def update(self, rtt: float) -> None:
        """Ingest new RTT observation and update internal state."""
        self._y_hat = (1 - self.gamma) * self._y_hat + self.gamma * rtt

    def reset(self, init_rtt: float = 20.0):
        self._y_hat = init_rtt


# ---------------------------------------------------------------------------
# ACI Bounds Tracker
# ---------------------------------------------------------------------------
class ACIBoundsTracker:
    """
    Adaptive Conformal Interval (ACI) upper-bound tracker.

    At each ACK:
      1. Compute non-conformity score s_t = RTT_t - y_hat_t
      2. Store in sliding calibration window S_W (size W)
      3. Upper bound: U_t = y_hat_{t+1} + Quantile(S_W, 1 - alpha_t)
      4. Observe actual RTT; update:
           alpha_{t+1} = clip(alpha_t + eta * (alpha - I{RTT_t > U_{t-1}}))

    Coverage guarantee (long-run): empirical coverage → 1 - alpha_target.
    """

    def __init__(
        self,
        alpha: float = 0.05,   # target miscoverage rate (1 - alpha = 95% coverage)
        eta: float = 0.01,     # step-size for alpha_t update
        window: int = 500,     # calibration window size W
        alpha_min: float = 1e-4,
        alpha_max: float = 0.5,
    ):
        self.alpha_target = alpha
        self.eta = eta
        self.window = window
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha_t: float = alpha
        self._scores: Deque[float] = deque(maxlen=window)
        self._last_upper: Optional[float] = None

    def compute_upper(self, y_hat_next: float) -> float:
        """
        Compute the ACI upper bound given the next-step prediction.
        Returns y_hat_next if the calibration window is still empty.
        """
        if not self._scores:
            self._last_upper = y_hat_next * 2.0   # conservative warm-up
            return self._last_upper
        q = float(np.quantile(list(self._scores), 1.0 - self.alpha_t))
        self._last_upper = y_hat_next + q
        return self._last_upper

    def update(self, rtt_observed: float, y_hat_at_t: float) -> None:
        """
        Call after observing RTT_t.
        y_hat_at_t is the prediction that was made *for* this step.
        """
        s_t = rtt_observed - y_hat_at_t
        self._scores.append(s_t)

        # Update adaptive alpha
        covered = 1.0 if (self._last_upper is not None
                          and rtt_observed <= self._last_upper) else 0.0
        self.alpha_t = float(
            np.clip(
                self.alpha_t + self.eta * (self.alpha_target - (1.0 - covered)),
                self.alpha_min,
                self.alpha_max,
            )
        )

    @property
    def current_upper(self) -> Optional[float]:
        return self._last_upper

    def reset(self):
        self.alpha_t = self.alpha_target
        self._scores.clear()
        self._last_upper = None


# ---------------------------------------------------------------------------
# Split-Conformal (fixed calibration — ablation)
# ---------------------------------------------------------------------------
class SplitConformalPredictor:
    """
    Standard split-conformal predictor with a fixed held-out calibration set.
    Unlike ACI, alpha does NOT adapt — coverage degrades under distribution shift.
    Used as an ablation to quantify the benefit of ACI.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        calib_window: int = 500,
    ):
        self.alpha = alpha
        self.calib_window = calib_window
        self._calib_scores: List[float] = []
        self._frozen = False
        self._quantile: float = 0.0

    def add_calibration(self, rtt: float, y_hat: float) -> None:
        """Add a calibration-set observation (non-conformity score)."""
        if not self._frozen:
            self._calib_scores.append(rtt - y_hat)
            if len(self._calib_scores) >= self.calib_window:
                self._freeze()

    def _freeze(self):
        self._frozen = True
        self._quantile = float(
            np.quantile(self._calib_scores, 1.0 - self.alpha)
        )

    def compute_upper(self, y_hat_next: float) -> float:
        if not self._frozen:
            return y_hat_next * 2.0
        return y_hat_next + self._quantile

    def reset(self):
        self._calib_scores.clear()
        self._frozen = False
        self._quantile = 0.0


# ---------------------------------------------------------------------------
# Fixed quantile (uncalibrated — baseline)
# ---------------------------------------------------------------------------
class FixedQuantilePredictor:
    """
    No conformal calibration — simply adds a fixed margin above the EWMA.
    Serves as the 'delay-target controller, no conformal predictor' ablation.
    """

    def __init__(self, margin_ms: float = 10.0):
        self.margin_ms = margin_ms

    def compute_upper(self, y_hat_next: float) -> float:
        return y_hat_next + self.margin_ms

    def update(self, rtt_observed: float, y_hat_at_t: float) -> None:
        pass   # stateless

    def reset(self):
        pass


if __name__ == "__main__":
    # Quick smoke-test
    import numpy as np

    rng = np.random.default_rng(42)
    ewma = EWMAPredictor(gamma=0.1, init_rtt=20.0)
    aci  = ACIBoundsTracker(alpha=0.05, eta=0.01, window=200)

    covered = 0
    n = 500
    for i in range(n):
        y_hat = ewma.predict()
        ub = aci.compute_upper(y_hat)
        rtt = 20 + 5 * rng.standard_normal() + (10 if i > 300 else 0)
        aci.update(rtt, y_hat)
        ewma.update(rtt)
        if rtt <= ub:
            covered += 1

    print(f"ACI empirical coverage: {covered/n:.3f}  (target 0.95)")
    print(f"Final alpha_t: {aci.alpha_t:.4f}")
