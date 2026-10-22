# Research Figures and Tables Index

This directory (`plots/` / `I:\Research1\Plots\`) contains all research figures, plots, and tables generated across the **ConfCC** workspace. Both vector PDFs (for LaTeX inclusion) and high-resolution 300 DPI PNGs (for visual inspection and presentations) are provided.

---

## 1. Master Audit of All Research Figures

| Figure File | High-Res PNG | PDF Size | Drawing Elements | Description & Visual Density |
| :--- | :--- | :---: | :---: | :--- |
| **`fig1_goodput_bars.pdf`** | `fig1_goodput_bars.png` | 63.1 KB | **2,375 ops** | **Mean Goodput Across ns-3 Scenarios (S1–S5, S7, S8)**:<br>Grouped bars for ConfCC, CUBIC, BBR, Vegas, NewReno with 95% bootstrap confidence intervals, individual seed jitter scatter clouds ($n=30$ runs per group), numerical goodput callouts, and multi-flow aggregate goodput. |
| **`fig2_latency_cdf_s1_dumbbell.pdf`** | `fig2_latency_cdf_s1_dumbbell.png` | 35.3 KB | **2,329 ops** | **Tail Latency Reduction for S1 Dumbbell (2 Panels)**:<br>**(a)** High-resolution empirical CDF comparing ConfCC with baseline queueing tails.<br>**(b)** Tail latency breakdown ($p_{50}, p_{95}, p_{99}$) with error whiskers and statistically verified $90.5\%$ latency reduction ($p < 0.0001, d = -1.79$). |
| **`fig2_latency_cdf_s2_trace.pdf`** | `fig2_latency_cdf_s2_trace.png` | 36.3 KB | **2,632 ops** | **Tail Latency Reduction for S2 Cellular Mahimahi LTE Trace (2 Panels)**:<br>Demonstrates tight queue bounding under real wireless channel dynamics without bufferbloat. |
| **`fig3_aci_coverage.pdf`** | `fig3_aci_coverage.png` | 128.0 KB | **8,482 ops** | **Adaptive Conformal Inference Dynamics (3 Panels, 14,222 ACKs)**:<br>**(a)** Per-ACK RTT scatter cloud, EWMA forecast, and dynamic ACI upper-bound envelope.<br>**(b)** Online calibration parameter $\alpha_t$ responding to channel jitter.<br>**(c)** Rolling empirical coverage convergence to target 95%. |
| **`fig5_fairness.pdf`** | `fig5_fairness.png` | 44.8 KB | **2,213 ops** | **Fairness Across Multi-flow Topologies (2 Panels)**:<br>**(a)** Jain's Fairness Index across multi-flow scenarios (S3, S4, S8) with individual seed swarms.<br>**(b)** Queue discipline impact: DropTail competitive concession vs. FQ-CoDel ideal fair sharing ($J = 0.985$). |
| **`fig6_coverage.pdf`** | `fig6_coverage.png` | 55.0 KB | **4,125 ops** | **Conformal Coverage Calibration (2 Panels, 240 Simulation Runs)**:<br>**(a)** Scenario-by-scenario empirical coverage distribution (Violin plot + boxplot + seed scatter clouds) tightly clustered around nominal 95% target.<br>**(b)** Finite-sample asymptotic convergence $|(1-\alpha) - \hat{C}_n|$ obeying theoretical $\mathcal{O}(1/\sqrt{n})$ bound. |
| **`win_c1_coverage_study.pdf`** | `win_c1_coverage_study.png` | 85.5 KB | **2,696 ops** | **Target Calibration Study (2 Panels, 2,160 Runs)**:<br>**(a)** Target sweep $1-\alpha \in [0.85, 0.99]$ calibration curves with 95% bootstrap bands.<br>**(b)** Dispersion cloud at 95% nominal target comparing ConfCC against Split-Conformal and No-ACI. |
| **`win_c2_pareto.pdf`** | `win_c2_pareto.png` | 44.8 KB | **1,996 ops** | **Throughput-Delay Pareto Frontier**:<br>Sweeping delay targets $T \in [20, 60]$ ms, Copa $\delta \in [0.05, 2.0]$, Vegas, CUBIC, BBR, NewReno with seed scatter, error whiskers, and trade-off envelopment. |
| **`win_c3_predictor_sensitivity.pdf`** | `win_c3_predictor_sensitivity.png` | 57.3 KB | **4,062 ops** | **Predictor Sensitivity & Coverage Invariance (2 Panels, W1.2 Task C3)**:<br>**(a)** Conformal coverage holding at 95% under severe step collapse even with a deliberately bad constant predictor.<br>**(b)** Predictive efficiency trade-off: Base predictor quality minimizes interval width and maximizes goodput. |
| **`win_c4_delayed_feedback.pdf`** | `win_c4_delayed_feedback.png` | 86.3 KB | **2,990 ops** | **Delayed Feedback Robustness (2 Panels, 900 Runs)**:<br>**(a)** Coverage error sweep across delay ratios $\tau/\text{RTT} \in [0.5, 10.0]$ with theoretical regret bound $\mathcal{O}(\sqrt{\log(1+\tau/\text{RTT})})$ and green safe operating zone.<br>**(b)** Goodput scaling under feedback delays. |
| **`win_c5_py_vs_ns3.pdf`** | `win_c5_py_vs_ns3.png` | 32.7 KB | **1,824 ops** | **Cross-Simulator Validation (2 Panels)**:<br>Linear correlation ($R^2$, Spearman $\rho$) for Goodput and 95th-percentile tail latency between Python model and native C++ `ns-3`. |

---

## 2. Research Tables

All tables are provided in LaTeX (`.tex`), clean tabular formats (`.csv`), formatted Markdown (`.md`), and rendered graphics (`.png`).

| Table File | Alternate Formats | Description |
| :--- | :--- | :--- |
| `table1_main_results.tex` | `table1_main_results.png`<br>`table1_main_results.csv`<br>`table1_main_results.md`<br>`table1_goodput_results.csv`<br>`table1_latency_p95_results.csv` | **Main Paper Results (Table 1)**:<br>Mean Goodput (Mbps) and p95 RTT (ms) across all scenarios over 30 seeds with 95% confidence intervals. Fully populated with zero `--` placeholders. |
| `table2_wilcoxon_stats.tex` | `table2_wilcoxon_stats.png`<br>`table2_wilcoxon_stats.csv`<br>`table2_wilcoxon_stats.md` | **Statistical Significance (Table 2)**:<br>Pairwise Wilcoxon signed-rank tests, Holm-Bonferroni adjusted $p$-values ($p_{\text{adj}} < 0.0001$), and large Cohen's $d$ effect sizes ($d = -1.79$) for 95th-percentile latency. |
| `win_c6_statistics.txt` | `win_c6_statistics.png`<br>`win_c6_statistics.csv`<br>`win_c6_statistics.md` | **Windows Emulation Statistics**:<br>Pairwise Wilcoxon tests and Holm corrections across Python controller sweeps. |
| `table_summary_ns3.csv` | CSV | **Master ns-3 Summary Data**: Tabular summary across all 1,560 simulation runs with true empirical coverage populated from ACK traces. |
| `table_summary_windows_sim.csv` | CSV | **Master Windows Simulator Summary Data**: Tabular summary of Python simulator runs. |

---

## 3. Full Paper Manuscript

- `paper_full.pdf`: Complete compiled publication manuscript (6 pages exact) with all figures and tables embedded.
