#!/usr/bin/env bash
# ConfCC Research Experiment Reproduction Runner
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  ConfCC: Conformal Congestion Control Research Experiments  "
echo "============================================================"

# 1. Run Python Dense Sweeps
echo "[*] Step 1/2: Running dense simulation sweeps ..."
PYTHONPATH="$SCRIPT_DIR/python_sim:$SCRIPT_DIR" python3 python_sim/confcc/run_dense_sweeps.py

# 2. Generate Research Figures
echo "[*] Step 2/2: Generating research figures ..."
PYTHONPATH="$SCRIPT_DIR/python_sim:$SCRIPT_DIR" python3 python_sim/confcc/analysis_windows.py

echo "============================================================"
echo "  Experiments completed. Figures saved to figures/           "
echo "============================================================"
