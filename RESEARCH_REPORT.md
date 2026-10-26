# Research report: what the discarded LCU outcomes contain

Companion to the paper (arXiv 2605.02986, revised version): main findings with their status, honest bottom lines, corrections relative to the first arXiv version, open problems and a simulation plan.

In the revised paper, F1-F6 are in Secs. 2-4, F7-F12 in Appendix B and F13-F14 in Sec. 5.

Status labels: **proved** (proof in the paper or its appendix), **checked** (exact or numerical verification script in `theory_checks/`, no full proof), **conjecture**, **open**.

## 1. Question and framing

Standard LCU applies `T = sum_t alpha_t U_t` by post-selecting the ancilla and throws every other outcome away. Question: what do the discarded outcomes contain and what can be done with them?

Framing rules:

* We do **not** propose an alternative LCU. The Hadamard-mixing circuit with one rotation qubit is a *model* that makes every outcome an explicit combination of the same vectors `U_t|psi>`; its post-selected success probability never exceeds standard LCU (subnormalisation `K max alpha >= sum alpha`).
* Every claim has to be stated in the *physical* measurement model (computational-basis counts), not an amplitude oracle.
* Any statement about advantage is made against the strongest natural baseline and with its scope.

## 2. Main findings (current state)

| # | Finding | Status | Where in paper |
| --- | --- | --- | --- |
| F1 | Outcome states `phi_{i,rho} = (1/K) sum_t s_{it} rho_t(rho) U_t psi`; matrix `Phi = C X`, `C` real `2K x K` tight frame (`C^T C = I/K`) | proved | Sec. 2-4 |
| F2 | Block structure: `V ~ [[A,B],[B,-A]]`, `A = (H x I) diag(w_t U_t) (H x I)`, singular values `w_t`, `r_t` (CSD); `spec(V) = +-spec(U_t)`, `V^2 = (H x I)(+ I_2 x U_t^2)(H x I)`; `p_00 = ||T psi||^2/(Kc)^2 <= p_std` | proved | Sec. 3 |
| F3 | Standard LCU outcomes are `K` independent combinations (nothing to cross-check, phase-blind); the rotation qubit makes the `2K` outcomes lie in a `K`-dim space. Target recoverable from any `K` other outcomes (generic angles; `K <= 8` computer-assisted, all `K` conditional on Conj. 3) | proved / checked | Sec. 4 |
| F4 | A computational-basis record sees only `R(k) = Re[x(k) x(k)^dagger]`; invariant under `x -> e^{i phi} x` and `x -> conj x`; Walsh diagonalisation of the record | proved | Sec. 4.2 |
| F5 | Real amplitudes: record determines `x(k)` up to sign iff `C` has the complement property; `K = 2, 4` explicit conditions, `K = 8` computer-assisted; equal angles fail. Then every real LCU `beta` can be evaluated *after* the experiment | proved (`K <= 4`) / checked | Sec. 4.2 |
| F6 | Complex amplitudes: no uniform guarantee for `K = 4` (pencil argument); almost-every identifiability conjectured | proved (no-go) / conjecture | Sec. 4.2, App. A |
| F7 | Linear threshold: linear identifiability of `R(k)` needs `>= K(K+1)/2` real directions; standard LCU has `K`, the rotation-qubit model `2K`: below threshold for `K >= 4` | proved | App. B |
| F8 | **APN phase register**: `m = log2 K` extra qubits holding an almost-perfect-nonlinear `phi(t)` give `K^2` outcomes whose pair characters are orthogonal; unbiased estimator of every `Re <x_a|x_b>` (and per `k`); APN is necessary and sufficient; `m` qubits minimal for `m >= 3` | proved (+ cited Nyberg bound) | App. B |
| F9 | Gram shot bound `K^2 ln(K(K-1)/eta)/(2 eps^2)`, uniform over all `beta`; `Im G` with `log2 K` extra settings | proved | App. B |
| F10 | **Lower bound**: any ancilla-only Select-based measurement has Fisher trace `<= 2` over all `K(K-1)/2` overlaps (one Hadamard test: 1); APN attains `2(K-1)/K`; van Trees gives `S >= 3K(K-1)/(32 eps^2)` for `eps <~ 1/K^2` | proved (local, MSE, ancilla-only) | App. B |
| F11 | `k`-resolved post-hoc LCU: unbiased estimator of `f_beta(k)` for all `beta`, `sum_k MSE <= (K^2/(4S) + K/S') ||beta||_1^4` (loose by 10-30x in exact enumeration) | proved (bound loose) | App. B |
| F12 | Joint measurements are different: for diagonal `U_t`, sampling the system has Fisher trace `P`, so F10 cannot extend beyond ancilla-only records | proved (instance) | App. B |
| F13 | Secret coefficients: public-`H` hiding is broken by undo-`H` (observer Q); system marginal is key-independent (S); known-input attack recovers the key (R); secret generic mixing `F` does not help against known inputs (numerical) | proved / checked | Sec. 5 |
| F14 | (Kimi) Square standard complex LCU (`C` invertible `K x K`) has a diagonal-phase gauge ambiguity `x' = C^{-1} D(k) C x`; a redundant secret complex frame needs `M >= 4K-4` outcomes (complex phase-retrieval threshold) | checked (see caveat below) | Sec. 5 |

### Honest bottom line on advantage

* For estimating the Gram matrix of the branch vectors / post-hoc LCU, **there is no asymptotic shot advantage over one Hadamard test per pair**: `Theta(K^2/eps^2)` both ways; the best possible gain for ancilla-only measurements is a factor `<= 2` in total Fisher information, and the APN design attains it.
* What the record does give: one fixed circuit built only from `Select` (no controlled `U_b^dagger U_a`, no pair-dependent circuits), post-hoc evaluation of *every* real LCU from one dataset (pays off once more than `~K` coefficient vectors are needed), and an exact structural description of what discarded outcomes can and cannot determine.
* A speed-up would have to come from regimes the lower bound does not cover: joint ancilla-system or multi-copy measurements for non-commuting `U_t` (no bound, no protocol yet), or structured/low-rank Gram matrices.
* For a *single* target the all-outcomes estimator is worse than post-selection by `~K`.

### Honest bottom line on the trapdoor

No trapdoor function is established. Observer Q breaks the public-`H` version with one copy; S learns a key-independent marginal; R recovers the key from known inputs (rotation-angle key and APN key alike); square complex coefficients lose the redundancy. What survives is a one-time, computational, classical-record candidate (Conj. in Sec. 7) that needs a redundant secret frame. The title word "trapdoor" is kept for continuity with arXiv 2605.02986, but the abstract states the limits.

## 3. Corrections relative to the first arXiv version

* "Rank `<= K`" holds for every prepare-select-unprepare LCU; the special part is the `2K`-row redundancy.
* The "coherence" explanation of SVP failure is wrong (row leverage is `<= 2`); the obstacles are the degree-of-freedom count and that only intensities are observed.
* `V` does not have only `+-1` eigenvalues unless `U_t^2 = I`; `V^2` needs the outer Hadamards.
* `p_00 + p_01 ~ 1/K` only on average over `i`; exact value `(||D_w psi||^2 + ||D_r psi||^2)/K^2`.
* The recovery experiments used exact amplitudes plus complex Gaussian noise (amplitude oracle), not counts; the factorised recovery's "drop to machine precision" was a threshold effect (all columns need `>= K` observed rows). The experiments were re-run with the original scripts (`lcu_recovery.py`, `lcu_recovery_noisy.py`, seed 42; results in `results_*.npz`) and re-plotted at 14 pt by `plot_recovery_figures.py` as two 2-panel figures. The re-run reproduces the original numbers and shows that two statements of the first revision draft were too optimistic: (i) on the grid `p <= 0.9` factorised recovery does **not** reach machine precision (mean error `4e-3` at `N=256`, `1e-2` at `N=1024`, dominated by the few columns with fewer than `K` observed entries; SVP is better up to `p = 0.8`); (ii) at 70% observed entries both methods sit on a floor of ~0.16 (SVP) and ~0.24 (factorised) for all `sigma <~ 1e-4`, not `< 1%` for `sigma <~ 1e-3`. `main.tex` Sec. 4.4 now says exactly this; the figures illustrate the observation threshold, they are not evidence for count-based recovery.
* The trapdoor was broken in the quantum-output model; the "information-theoretic ambiguity" and "Unique-Games" remarks were unsupported and are gone; the one-wayness theorem is now a conjecture with an explicit model.
* "Complex coefficients by two rotation qubits": not proved; extension only.

## 4. Caveats on related notes

* **Standard complex coefficients (reviewer note):** the gauge ambiguity `x'(k) = C^{-1} D(k) C x(k)` is exact when the branch vectors `x(k)` are treated as *arbitrary* per `k`. In the actual model `x_t = U_t psi` with public `U_t`, so `x'` must come from some `psi'`; the ambiguity then need not survive, and with `K N` intensities against `2N` real unknowns the key holder may well be able to invert generically. The paper therefore states the result for unstructured branch vectors (and for the redundancy count), not as a proof that the key holder can never invert. Settling the structured case is open.
* **Stable rotation designs (reviewer note):** the stable `O(K)`-setting rotation design is compatible with, and capped by, the Fisher bound (trace `<= 2` per shot for any such design); it cannot beat the APN rate.
* **Fisher bound (F10):** local (small `eps`), mean-square error, ancilla-only. Do not quote as "no scheme can beat K^2".

## 5. Open problems, in priority order

1. Joint ancilla-system and multi-copy measurements for non-commuting `U_t`: bound or protocol (F12 shows both regimes exist).
2. Tight variance for the `k`-resolved estimator (flat inputs behave like `||beta||_2^4`, worst case `||beta||_1^4`).
3. Conj. 3 (Walsh full spark, all `K`): hinge for all-`K` statements in F3, F5.
4. Conj. 1 (complex-amplitude identifiability); stability constant of the record map.
5. Structured trapdoor correctness (key-holder inversion with public `U_t`), then a security definition and a reduction; otherwise leave as remark.
6. Use of low-rank structure of `G` (rank `<= N`) or non-uniform index amplitudes to beat `K^2`.

## 6. Simulation plan (after the theory is accepted)

All simulations of the first version are retired. New ones, all with shot sampling of `(i, rho, k)` from the exact probabilities and recorded seeds:

* APN estimator vs Hadamard tests: total shots to a given max-entry error, `K = 4..32`; check the factor `~2` and the `eps`-dependence of Thm. (Fisher bound).
* `k`-resolved post-hoc LCU: error of `f_beta(k)` vs shots for many `beta`; break-even number of coefficient vectors vs post-selection.
* Rotation-qubit model: honest finite-shot recovery of `R(k)` with the real-amplitude decoder, conditioning vs `K`.
* Attack experiments for the one-time hiding conjecture (multi-start alternating minimisation) vs honest decoding time.

## 7. File map

* `RESEARCH_REPORT.md`: this report.
* `theory_checks/`: `verify_identities.py` (54 identity checks), `apn_design.py`, `fisher_bound_check.py`, `k_resolved_variance.py`, `secret_mixing_*.py`, `complex_identifiability_multistart.py`, `gram_shot_cost_check.py`.
* `lcu_recovery.py`, `lcu_recovery_noisy.py`, `plot_recovery_figures.py`, `data/`: recovery experiments, their results and the figure script (figures go to `figures/`).
