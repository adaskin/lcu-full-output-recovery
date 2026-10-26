"""
Multi-start search for ambiguities of the complex phase-retrieval map x -> |C x|^2 (C real, 2K x K).
Evidence for Conjecture "almost-every identifiability up to phase and conjugation"; not a proof.
Counts exact fits (residual < 1e-10) that are NOT equal to the truth up to global phase / conjugation.
"""
import numpy as np
from scipy.linalg import hadamard
from scipy.optimize import least_squares

rng = np.random.default_rng(11)


def C_of(th):
    K = len(th); S = hadamard(K).astype(float)
    return np.vstack([S * np.cos(th), S * np.sin(th)]) / K


def run(K, n_inst, n_starts):
    tot_fit = tot_bad = 0
    for _ in range(n_inst):
        C = C_of(rng.uniform(0.15, 1.4, K))
        x = rng.normal(size=K) + 1j * rng.normal(size=K)
        p = np.abs(C @ x) ** 2
        f = lambda z: np.abs(C @ (z[:K] + 1j * z[K:])) ** 2 - p
        for _ in range(n_starts):
            z = least_squares(f, rng.normal(size=2 * K), xtol=1e-14, ftol=1e-14, gtol=1e-14).x
            if np.linalg.norm(f(z)) > 1e-10:
                continue
            y = z[:K] + 1j * z[K:]
            d = min(np.linalg.norm(y * np.vdot(y, v) / abs(np.vdot(y, v)) - v) for v in (x, x.conj()))
            tot_fit += 1; tot_bad += d > 1e-5
    return tot_fit, tot_bad

for K, ni, ns in [(2, 20, 20), (4, 20, 100), (8, 10, 400)]:
    fit, bad = run(K, ni, ns)
    print(f"K={K}: instances={ni}, starts each={ns}, exact fits={fit}, fits not equivalent to truth={bad}")
