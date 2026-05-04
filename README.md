# Exploiting All Ancilla Outcomes in Linear Combinations of Unitaries

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20024365-blue)](https://doi.org/10.5281/zenodo.20024365)

This repository contains the simulation code for the paper

> **"Exploiting all ancilla outcomes in linear combinations of unitaries: low‑rank recovery and quantum trapdoor functions"** [pdf from research gate](https://www.researchgate.net/publication/404400359_Exploiting_all_ancilla_outcomes_in_linear_combinations_of_unitaries_low-rank_recovery_and_quantum_trapdoor_functions) 
> *Ammar Daskin* (2026)

The code reproduces the numerical experiments of Section 4, comparing singular value projection (SVP) and factorized recovery of the full output matrix \(\Phi\) from partial and noisy observations.

## Contents

- `quantum_circuit.py` – `AlternativeLCU` class that builds the Hadamard‑mixed LCU circuit (PennyLane) and computes the exact output matrix.
- `lcu_recovery.py` – Matrix completion experiment: SVP vs. factorized recovery as a function of the fraction of observed entries.
- `lcu_recovery_noisy.py` – Noise‑resilience experiment: recovery under additive Gaussian noise with fixed observation fraction.
- `comparison_to_standard_lcu.py` – Success‑probability comparison between standard LCU and the alternative circuit.


## Usage

Run the scripts directly:

```bash
python lcu_recovery.py          # exact amplitudes, varying observation fraction
python lcu_recovery_noisy.py    # noisy observations, fixed observation fraction
python comparison_to_standard_lcu.py   # probability comparison plot
```
