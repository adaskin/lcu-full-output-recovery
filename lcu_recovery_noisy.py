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

def factorized_recovery(observed, C, mask, lam=1e-12):
    """Recover Phi = C X from noisy observed entries."""
    K = C.shape[1]
    N = observed.shape[1]
    X_rec = np.zeros((K, N), dtype=complex)
    for col in range(N):
        rows = np.where(mask[:, col])[0]
        if len(rows) < K:
            continue
        C_sub = C[rows, :]
        b = observed[rows, col]          # noisy values only
        A = C_sub.conj().T @ C_sub + lam * np.eye(K)
        rhs = C_sub.conj().T @ b
        X_rec[:, col] = solve(A, rhs)
    return C @ X_rec, X_rec

def svp_recovery(observed, mask, K, max_iter=200):
    """Iterative hard thresholding using only observed entries."""
    Phi_rec = observed.copy()            # zeros at unobserved entries
    for it in range(max_iter):
        Phi_rec[mask] = observed[mask]   # restore observed entries
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
n_sys = 8                # N = 256
N = 2**n_sys
max_iter = 200

ntrials = 10             # number of random circuit instances
nmask   = 5              # independent masks (& noise) per trial & noise level
noise_levels = np.logspace(-5, -2, 10)   # std of complex Gaussian noise
observation_fraction = 0.7                 # fraction of entries observed

# Storage: shape (len(noise_levels), ntrials * nmask)
err_svp_Phi = np.zeros((len(noise_levels), ntrials * nmask))
err_svp_phi = np.zeros_like(err_svp_Phi)
err_fac_Phi = np.zeros_like(err_svp_Phi)
err_fac_phi = np.zeros_like(err_svp_Phi)

rng = np.random.default_rng(42)

for trial in range(ntrials):
    # Generate random instance
    alpha = rng.uniform(0, 1.0, size=K)
    U_list = [random_unitary(N, seed=rng.integers(0, 1e7)) for _ in range(K)]
    psi = rng.random(N) + 1j * rng.random(N)
    psi = psi / norm(psi)

    lcu = AlternativeLCU(K, alpha, U_list, psi=psi,
                         single_channel=True, output="state",
                         compute_outputs_via_matrices=True)
    lcu.compute_outputs_via_matrices()
    true_Phi = lcu.true_output_matrix2KN
    true_phi = true_Phi[0, :]          # rot=0, index=0 row
    C = build_C(alpha, K)

    for nidx, noise_std in enumerate(noise_levels):
        for m in range(nmask):
            # 1. Random observation mask (same shape as true_Phi)
            mask = rng.random(true_Phi.shape) < observation_fraction
            observed = np.zeros_like(true_Phi, dtype=complex)
            # 2. Add noise only to observed entries
            noise = (rng.normal(0, noise_std, true_Phi.shape) +
                     1j * rng.normal(0, noise_std, true_Phi.shape))
            observed[mask] = true_Phi[mask] + noise[mask]

            # 3. SVP recovery from noisy partial observations
            Phi_svp = svp_recovery(observed, mask, K, max_iter)
            idx = trial * nmask + m
            err_svp_Phi[nidx, idx] = norm(Phi_svp - true_Phi) / norm(true_Phi)
            err_svp_phi[nidx, idx] = norm(Phi_svp[0,:] - true_phi) / norm(true_phi)

            # 4. Factorized recovery (uses mask and known C)
            Phi_fac, _ = factorized_recovery(observed, C, mask)
            err_fac_Phi[nidx, idx] = norm(Phi_fac - true_Phi) / norm(true_Phi)
            err_fac_phi[nidx, idx] = norm(Phi_fac[0,:] - true_phi) / norm(true_phi)

    print(f"Trial {trial+1}/{ntrials} complete")

# Means and standard deviations
mean_svp_Phi = np.mean(err_svp_Phi, axis=1)
std_svp_Phi  = np.std(err_svp_Phi, axis=1)
mean_svp_phi = np.mean(err_svp_phi, axis=1)
std_svp_phi  = np.std(err_svp_phi, axis=1)

mean_fac_Phi = np.mean(err_fac_Phi, axis=1)
std_fac_Phi  = np.std(err_fac_Phi, axis=1)
mean_fac_phi = np.mean(err_fac_phi, axis=1)
std_fac_phi  = np.std(err_fac_phi, axis=1)

# Print summary
print("\nNoise std       SVP Φ error             Fact Φ error")
for i, nl in enumerate(noise_levels):
    print(f"{nl:.4f}           {mean_svp_Phi[i]:.4e}±{std_svp_Phi[i]:.4e}    "
          f"{mean_fac_Phi[i]:.4e}±{std_fac_Phi[i]:.4e}")

# ----- Plot -----
# Plot
plt.rcParams.update({
'figure.figsize': (7, 5), 'font.size': 14, 'axes.labelsize': 14,
'axes.titlesize': 14, 'legend.fontsize': 12, 'xtick.labelsize': 14,
'ytick.labelsize': 14, 'lines.linewidth': 2, 
'savefig.dpi': 300, 'savefig.bbox': 'tight', 'text.usetex': False,
})
plt.figure(figsize=(8,5))
plt.errorbar(noise_levels, mean_svp_Phi, yerr=std_svp_Phi, fmt='o-', capsize=3, label='SVP Φ')
plt.errorbar(noise_levels, mean_svp_phi, yerr=std_svp_phi, fmt='s--', capsize=3, label='SVP φ')
plt.errorbar(noise_levels, mean_fac_Phi, yerr=std_fac_Phi, fmt='d-', capsize=3, label='Factorized Φ')
plt.errorbar(noise_levels, mean_fac_phi, yerr=std_fac_phi, fmt='x--', capsize=3, label='Factorized φ')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Noise standard deviation (σ)')
plt.ylabel('Relative error')
plt.legend()
plt.grid(True, alpha=0.3)
plt.title(f'Shot‑noise recovery (obs. frac={observation_fraction}) '
          f'K={K}, N={N}, {ntrials} trials × {nmask} masks')
plt.tight_layout()
plt.savefig(f'recovery_svp_vs_factorized_noisy_maskK{K}-N{N}.pdf')
plt.savefig(f'recovery_svp_vs_factorized_noisy_maskK{K}-N{N}.png')
plt.show()