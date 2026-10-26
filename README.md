# Exploiting All Ancilla Outcomes in Linear Combinations of Unitaries

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20024365-blue)](https://doi.org/10.5281/zenodo.20024365)

This repository accompanies the paper

> **"Exploiting all ancilla outcomes in linear combinations of unitaries: low‑rank recovery and quantum trapdoor functions"** [pdf from research gate](https://www.researchgate.net/publication/404400359_Exploiting_all_ancilla_outcomes_in_linear_combinations_of_unitaries_low-rank_recovery_and_quantum_trapdoor_functions) [arxiv](https://doi.org/10.48550/arXiv.2605.02986)
> *Ammar Daskin* (2026)

It contains a research report with the main findings and their status, the scripts that verify the identities and bounds of the paper, and the code and data of the recovery experiments.

## Contents

- `theory_checks/` – verification scripts (numpy only): `verify_identities.py` (outcome formulas, frame, Walsh diagonalisation, real-amplitude conditions, block-unitary view), `apn_design.py` (APN phase register, Appendix B), `fisher_bound_check.py`, `k_resolved_variance.py`, `secret_mixing_*.py` and `complex_identifiability_multistart.py` (attack and identifiability evidence), `gram_shot_cost_check.py`.
- `quantum_circuit.py` – `AlternativeLCU` class that builds the Hadamard‑mixed model circuit (PennyLane) and computes the exact output matrix.
- `lcu_recovery.py`, `lcu_recovery_noisy.py` – recovery experiments (SVP vs. factorized recovery of the output matrix \(\Phi\) in the amplitude-access model); `data/` holds their results; `plot_recovery_figures.py` produces the figures in `figures/`.
- `comparison_to_standard_lcu.py` – post-selection probability of the model circuit against standard LCU (first version of the paper; not used in the current text).

## Usage

```bash
# verification scripts (each prints PASS/FAIL per check)
for f in theory_checks/*.py; do python3 -I $f; done

# recovery experiments (argument: number of system qubits, N = 2**n; paper uses 8 and 10); results go to data/
python lcu_recovery.py 8;        python lcu_recovery.py 10
python lcu_recovery_noisy.py 8;  python lcu_recovery_noisy.py 10
python plot_recovery_figures.py   # writes figures/fig_recovery_exact.pdf and figures/fig_recovery_noise.pdf
```

The recovery experiments need `numpy`, `scipy`, `matplotlib` and `pennylane`; the verification scripts need `numpy` only.
