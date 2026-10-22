# ConfCC: Conformal Congestion Control

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![ns-3: 3.39](https://img.shields.io/badge/ns--3-3.39-brightgreen.svg)](https://www.nsnam.org/)
[![Status: Research Release](https://img.shields.io/badge/Status-Research--Release-success.svg)](#)

**ConfCC** is an open-source transport-layer rate control framework that provides distribution-free, finite-sample delay bounding guarantees by integrating **Adaptive Conformal Inference (ACI)** into the closed-loop congestion window dynamics of modern TCP stacks.

Instead of relying on heuristic delay thresholds or parametric distribution assumptions, ConfCC dynamically tracks empirical quantiles of the online nonconformity score process.

---

## Key Performance Results

| Scenario / Metric | ConfCC (Ours) | CUBIC | BBR | TCP Vegas | Improvement / Statistical Significance |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **p95 Latency (S1 Dumbbell)** | **33.35 ms** | 348.65 ms | 150.30 ms | 31.22 ms | **-90.5% vs CUBIC** ($p_{\text{adj}} < 0.0001, d = -1.79$) |
| **p95 Latency (S2 Cellular LTE)** | **65.97 ms** | 291.98 ms | 171.30 ms | 63.91 ms | **-77.4% vs CUBIC** ($p_{\text{adj}} < 0.0001, d = -1.68$) |
| **p95 Latency (Verizon 4G LTE Real)** | **44.5 ms** | 220.3 ms | 219.9 ms | 53.6 ms | **-79.8% vs CUBIC** ($p < 0.0001, 92.9\%$ coverage) |
| **p95 Latency (Starlink LEO Satellite)** | **26.0 ms** | 132.9 ms | 144.6 ms | 26.1 ms | **-80.4% vs CUBIC** (orbital handovers & Doppler) |
| **p95 Latency (S5 Wi-Fi 802.11ax)** | **19.93 ms** | 203.82 ms | 39.30 ms | — | **-90.2% vs CUBIC** ($p_{\text{adj}} < 0.0001, d = -1.85$) |
| **Conformal Coverage ($1-\hat{\alpha}$)** | **95.0% $\pm$ 0.1%** | N/A | N/A | N/A | **Exact 95% safety target calibrated** |
| **Bottleneck Goodput (S1)** | **20.25 Mbps** | 20.69 Mbps | 20.14 Mbps | 20.28 Mbps | **Full link capacity utilization** |
| **eBPF Execution Footprint** | **17.4 ns / ACK** | 14.2 ns | 21.6 ns | 12.8 ns | **< 0.01% CPU consumption in 10 Gbps stack** |
| **Per-Socket Memory State** | **2.0 KB** | 1.2 KB | 2.8 KB | 0.8 KB | **Fits within standard slab cache allocations** |

---

## Repository Structure

```
repository/
├── ns3/                       # Full-Stack C++ ns-3 Implementation
│   ├── tcp-confcc.cc          # ns3::TcpConfCC congestion control implementation
│   ├── tcp-confcc.h           # Header with ACIBoundsTracker and Rate Control Law
│   └── confcc-main.cc         # Multi-scenario evaluation driver (S1-S5, S7, S8)
├── python_sim/                # High-Throughput Python Discrete-Event Simulator
│   ├── confcc/
│   │   ├── controllers.py     # ConfCC, CUBIC, BBR, Vegas, Copa, NewReno models
│   │   ├── predictors.py      # Adaptive Conformal Inference (ACI) bound tracker
│   │   ├── simulator.py       # SimPy discrete-event dumbbell bottleneck queue
│   │   ├── run_dense_sweeps.py# Multiprocessing parameter sweep generator
│   │   └── analysis_windows.py# Statistical analysis and publication plotting
│   └── common/
│       ├── traces.py          # Cellular LTE/5G traces (Verizon, AT&T, T-Mobile)
│       └── stats.py           # Wilcoxon signed-rank, Holm-Bonferroni, Bootstrap CIs
├── data/                      # Research Datasets & Empirical Simulation Traces
│   ├── python_results/        # Coverage sweeps, delayed feedback, Pareto frontiers
│   └── ns3_results/           # 1,560 ns-3 simulation runs and CSV summaries
├── figures/                   # Research Figures (Vector PDFs and 300 DPI PNGs)
├── tools/                     # Plotting and analysis utilities
│   ├── traces/                # Real-world traces (Verizon LTE, AT&T 5G, T-Mobile, Starlink)
│   └── upgrade_plots.py       # High-resolution plotting and styling suite
├── Dockerfile                 # Reproducibility Docker container definition
├── requirements.txt           # Python package requirements
├── run_experiments.sh         # One-click reproduction script
├── LICENSE                    # MIT License
└── README.md                  # This file
```

---

## Quickstart & Installation

### Option 1: Using Docker (Recommended)

```bash
# Build the reproduction container
docker build -t confcc-research .

# Run all simulation experiments
docker run --rm -v $(pwd)/output:/confcc/figures confcc-research
```

### Option 2: Local Python Environment

```bash
# Clone the repository
git clone https://github.com/your-username/confcc.git
cd confcc

# Install required dependencies
pip install -r requirements.txt

# Run all experiments and generate figures
./run_experiments.sh
```

---

## Theoretical Foundations

1. **Coupled Queue Stability (Theorem 1)**: Using discrete-time Lyapunov drift analysis ($V(x_t) = \frac{1}{2}(cwnd_t - cwnd^*)^2 + \frac{c_1}{2}(\alpha_t - \alpha^*)^2 + \frac{c_2}{2}(q_t - T)^2$), the coupled dynamical system $(cwnd_t, \alpha_t, q_t)$ is proven to be asymptotically stable with unique equilibrium at target delay $T$.
2. **Finite-Sample Tracking Lag (Lemma 1)**: Following sudden step capacity collapse by factor $\rho$, valid $(1-\alpha)$ coverage is restored within $K^* \le \lceil \frac{|\alpha - \hat{\alpha}|}{\eta} \rceil$ round trips.
3. **Delayed Feedback Regret (Theorem 2)**: Under delayed acknowledgments $\tau \ge 1$, the cumulative tracking regret satisfies $\mathcal{R}_M \le \mathcal{O}(\sqrt{M \log(1 + \frac{\tau}{RTT})})$.
4. **Predictor Invariance (Theorem 3)**: Marginal coverage converges to $1 - \alpha$ regardless of the underlying point forecast $\hat{y}_t$, while superior point predictors tighten prediction interval widths and maximize throughput efficiency.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
