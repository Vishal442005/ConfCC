"""
common/stats.py — Shared statistical utilities.

Provides bootstrap confidence intervals, Wilcoxon signed-rank tests with
Holm–Bonferroni correction, coverage calculations, and Jain's fairness index.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------
def bootstrap_ci(
    data: Sequence[float],
    stat_fn=np.mean,
    n_boot: int = 2000,
    ci: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """Return (lower, upper) bootstrap confidence interval."""
    if rng is None:
        rng = np.random.default_rng(0)
    arr = np.asarray(data, dtype=float)
    boot_stats = [stat_fn(rng.choice(arr, size=len(arr), replace=True))
                  for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    return float(np.quantile(boot_stats, alpha)), float(np.quantile(boot_stats, 1 - alpha))


# ---------------------------------------------------------------------------
# Wilcoxon with Holm–Bonferroni correction
# ---------------------------------------------------------------------------
def pairwise_wilcoxon_holm(
    data: Dict[str, Sequence[float]],
    alternative: str = "two-sided",
) -> Dict[Tuple[str, str], dict]:
    """
    Perform all pairwise Wilcoxon signed-rank tests and apply Holm–Bonferroni
    correction to control the family-wise error rate.

    Parameters
    ----------
    data : dict mapping scheme_name → list of per-seed metric values
           All lists must have the same length (paired observations).
    alternative : 'two-sided' | 'less' | 'greater'

    Returns
    -------
    Dict of (scheme_a, scheme_b) → {p_raw, p_adj, reject, statistic}
    """
    keys = list(data.keys())
    pairs = list(itertools.combinations(keys, 2))
    raw: List[Tuple[Tuple[str, str], float, float]] = []

    for a, b in pairs:
        x = np.asarray(data[a], dtype=float)
        y = np.asarray(data[b], dtype=float)
        if len(x) != len(y):
            continue
        res = scipy_stats.wilcoxon(x, y, alternative=alternative,
                                   zero_method="wilcox")
        raw.append(((a, b), res.statistic, res.pvalue))

    # Holm–Bonferroni correction
    raw.sort(key=lambda t: t[2])
    m = len(raw)
    results: Dict[Tuple[str, str], dict] = {}
    reject_all = False
    for rank, (pair, stat, p_raw) in enumerate(raw):
        threshold = 0.05 / (m - rank)
        reject = (not reject_all) and (p_raw <= threshold)
        if not reject:
            reject_all = True  # once we fail to reject, all subsequent also fail
        p_adj = min(1.0, p_raw * (m - rank))
        results[pair] = dict(statistic=stat, p_raw=p_raw,
                             p_adj=p_adj, reject=reject)
    return results


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------
def empirical_coverage(
    rtts: Sequence[float],
    upper_bounds: Sequence[float],
) -> float:
    """Fraction of RTT samples that fall within their upper bound."""
    arr_rtt = np.asarray(rtts, dtype=float)
    arr_ub  = np.asarray(upper_bounds, dtype=float)
    return float(np.mean(arr_rtt <= arr_ub))


# ---------------------------------------------------------------------------
# Jain's fairness index
# ---------------------------------------------------------------------------
def jain_fairness(throughputs: Sequence[float]) -> float:
    """Jain's Fairness Index: (Σxᵢ)² / (n · Σxᵢ²)."""
    x = np.asarray(throughputs, dtype=float)
    n = len(x)
    if n == 0 or np.sum(x ** 2) == 0:
        return float("nan")
    return float((np.sum(x) ** 2) / (n * np.sum(x ** 2)))


# ---------------------------------------------------------------------------
# Pareto dominance helpers
# ---------------------------------------------------------------------------
def is_dominated(point: Tuple[float, float],
                 frontier: List[Tuple[float, float]]) -> bool:
    """
    Check if `point` is dominated by any point in `frontier`.
    Assumes (throughput, latency) where higher throughput and lower latency
    are better.
    """
    tp, lat = point
    for fp, fl in frontier:
        if fp >= tp and fl <= lat and (fp > tp or fl < lat):
            return True
    return False


def pareto_frontier(
    points: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """Return the Pareto-optimal points (max throughput, min latency)."""
    frontier = []
    for p in points:
        if not is_dominated(p, points):
            frontier.append(p)
    return sorted(frontier, key=lambda x: x[0])


if __name__ == "__main__":
    # Quick self-test
    import numpy as np
    rng = np.random.default_rng(1)
    data = {
        "confcc":  list(rng.normal(20, 2, 30)),
        "cubic":   list(rng.normal(18, 2, 30)),
        "bbr":     list(rng.normal(19, 2, 30)),
    }
    res = pairwise_wilcoxon_holm(data)
    for pair, r in res.items():
        print(f"{pair}: p_adj={r['p_adj']:.4f}, reject={r['reject']}")

    ci = bootstrap_ci(data["confcc"])
    print(f"confcc CI: {ci}")
    print(f"Jain: {jain_fairness([10, 20, 30]):.4f}")
