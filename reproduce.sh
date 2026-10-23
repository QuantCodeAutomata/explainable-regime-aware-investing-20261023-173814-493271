#!/usr/bin/env bash
# Reproduce the "Explainable Regime-Aware Investing" experiments.
set -o pipefail

python -m pytest -q || exit 1
python -m exp.exp1_wasserstein_hmm || exit 1
python -m exp.exp2_knn_mvo || exit 1

echo "Done. See results/RESULTS.md and the results/ directory for outputs."
