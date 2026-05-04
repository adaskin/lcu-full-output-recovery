import numpy as np
from numpy.linalg import svd, norm, solve
from scipy.linalg import hadamard as hadamard_scipy
import matplotlib.pyplot as plt
from quantum_circuit import AlternativeLCU

# ----------------------------------------------------------------------
# Helper functions
def random_unitary(dim, seed=None):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    Q, _ = np.linalg.qr(Z)
    return Q

def build_C(alpha, K):
    w = alpha
    r = np.sqrt(1 - w**2)
    H_raw = hadamard_scipy(K)
    C = np.zeros((2*K, K), dtype=complex)
    for i in range(K):
        for t in range(K):
            C[i, t]    = (w[t] / K) * H_raw[i, t]
            C[K+i, t]  = (r[t] / K) * H_raw[i, t]
    return C

def factorized_recovery(true_Phi, C, mask, lam=1e-12):
    """Recover Phi = C X from observed entries, column‑wise LS."""
    K = C.shape[1]
    N = true_Phi.shape[1]
    X_rec = np.zeros((K, N), dtype=complex)
    for col in range(N):
        rows = np.where(mask[:, col])[0]
        if len(rows) < K:          # not enough equations – leave column as zero
            continue
        C_sub = C[rows, :]
        b = true_Phi[rows, col]
        A = C_sub.conj().T @ C_sub + lam * np.eye(K)
        rhs = C_sub.conj().T @ b
        X_rec[:, col] = solve(A, rhs)
    return C @ X_rec, X_rec

def svp_recovery(true_Phi, mask, K, max_iter=200):
    """Iterative hard thresholding (SVP) for matrix completion."""
    observed = np.zeros_like(true_Phi, dtype=complex)
    observed[mask] = true_Phi[mask]
    Phi_rec = observed.copy()
    for it in range(max_iter):
        Phi_rec[mask] = observed[mask]
        U, s, Vh = svd(Phi_rec, full_matrices=False)
        Phi_rec_new = (U[:, :K] * s[:K]) @ Vh[:K, :]
        delta = norm(Phi_rec_new - Phi_rec) / norm(Phi_rec)
        Phi_rec = Phi_rec_new
        if delta < 1e-12:
            break
    return Phi_rec

# ----------------------------------------------------------------------
# Parameters
K = 4
n_sys = 8            # system qubits → N = 256
N = 2**n_sys
max_iter = 200

ntrials = 10         # different random circuits/states
nmask   = 5          # masks per percentage per trial
fractions = np.linspace(0.1, 0.9, 9)

# Storage: shape (len(fractions), ntrials * nmask)
err_svp_Phi = np.zeros((len(fractions), ntrials * nmask))
err_svp_phi = np.zeros_like(err_svp_Phi)
err_fac_Phi = np.zeros_like(err_svp_Phi)
err_fac_phi = np.zeros_like(err_svp_Phi)

rng = np.random.default_rng(42)

for trial in range(ntrials):
    # Generate random instance
    alpha = rng.uniform(0, 1.0, size=K)
    U_list = [random_unitary(N, seed=rng.integers(0, 1e7)) for _ in range(K)]
    psi = rng.standard_normal(N)
    psi = psi / norm(psi)

    lcu = AlternativeLCU(K, alpha, U_list, psi=psi,
                         single_channel=True, output="state",
                         compute_outputs_via_matrices=True)
    lcu.compute_outputs_via_matrices()
    true_Phi = lcu.true_output_matrix2KN
    true_phi = true_Phi[0, :]          # rot=0, index=0 row
    C = build_C(alpha, K)

    for frac_idx, p in enumerate(fractions):
        for mask_idx in range(nmask):
            mask = rng.random((2*K, N)) < p

            # SVP
            Phi_svp = svp_recovery(true_Phi, mask, K, max_iter)
            err_svp_Phi[frac_idx, trial*nmask + mask_idx] = (
                norm(Phi_svp - true_Phi) / norm(true_Phi)
            )
            err_svp_phi[frac_idx, trial*nmask + mask_idx] = (
                norm(Phi_svp[0,:] - true_phi) / norm(true_phi)
            )

            # Factorized
            Phi_fac, _ = factorized_recovery(true_Phi, C, mask)
            err_fac_Phi[frac_idx, trial*nmask + mask_idx] = (
                norm(Phi_fac - true_Phi) / norm(true_Phi)
            )
            err_fac_phi[frac_idx, trial*nmask + mask_idx] = (
                norm(Phi_fac[0,:] - true_phi) / norm(true_phi)
            )

    print(f"Trial {trial+1}/{ntrials} complete")

# Compute statistics (mean ± std) across all trial×mask repetitions
mean_svp_Phi = np.mean(err_svp_Phi, axis=1)
std_svp_Phi  = np.std(err_svp_Phi, axis=1)
mean_svp_phi = np.mean(err_svp_phi, axis=1)
std_svp_phi  = np.std(err_svp_phi, axis=1)

mean_fac_Phi = np.mean(err_fac_Phi, axis=1)
std_fac_Phi  = np.std(err_fac_Phi, axis=1)
mean_fac_phi = np.mean(err_fac_phi, axis=1)
std_fac_phi  = np.std(err_fac_phi, axis=1)

# Print final numbers
for i, p in enumerate(fractions):
    print(f"{p:.1f}: SVP Φ {mean_svp_Phi[i]:.2e}±{std_svp_Phi[i]:.2e}   "
          f"Fact Φ {mean_fac_Phi[i]:.2e}±{std_fac_Phi[i]:.2e}")
    print(f"        SVP φ {mean_svp_phi[i]:.2e}±{std_svp_phi[i]:.2e}   "
          f"Fact φ {mean_fac_phi[i]:.2e}±{std_fac_phi[i]:.2e}")

# ----- Plot with error bars -----
# Plot
plt.rcParams.update({
'figure.figsize': (7, 5), 'font.size': 14, 'axes.labelsize': 14,
'axes.titlesize': 14, 'legend.fontsize': 12, 'xtick.labelsize': 14,
'ytick.labelsize': 14, 'lines.linewidth': 2, 
'savefig.dpi': 300, 'savefig.bbox': 'tight', 'text.usetex': False,
})
plt.figure(figsize=(8,5))
plt.errorbar(fractions, mean_svp_Phi, yerr=std_svp_Phi, fmt='o-', capsize=3, label='SVP Φ')
plt.errorbar(fractions, mean_svp_phi, yerr=std_svp_phi, fmt='s--', capsize=3, label='SVP φ')
plt.errorbar(fractions, mean_fac_Phi, yerr=std_fac_Phi, fmt='d-', capsize=3, label='Factorized Φ')
plt.errorbar(fractions, mean_fac_phi, yerr=std_fac_phi, fmt='x--', capsize=3, label='Factorized φ')
plt.yscale('log')
plt.xlabel('Fraction of observed entries')
plt.ylabel('Relative error')
plt.legend()
plt.grid(True, alpha=0.3)
plt.title(f'Matrix completion: SVP vs. factorized (K={K}, N={N}, {ntrials} trials × {nmask} masks)')
plt.tight_layout()
plt.savefig(f'recovery_svp_vs_factorizedK{K}-N{N}.pdf')
plt.savefig(f'recovery_svp_vs_factorizedK{K}-N{N}.png')
plt.show()