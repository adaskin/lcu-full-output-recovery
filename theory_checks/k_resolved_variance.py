"""
Checks for the k-resolved estimator of Sec. VI and for the remark on joint measurements.

(1) Estimator of the unnormalised output distribution f_beta(k) = |sum_t beta_t x_t(k)|^2 from the APN record:
    per shot (i, rho, k_s):  g_k = (K/2) 1[k_s = k] ((sum_a beta_a chi_a)^2 - |beta|^2)  gives the off-diagonal part,
    and the diagonal part sum_a beta_a^2 |x_a(k)|^2 comes from the Select-then-measure-index setting.
    Checked here: exact unbiasedness, and  sum_k E[g_k^2] <= (K^2/4) |beta|_1^4  (exact enumeration, no sampling).
(2) Joint measurement: for the diagonal instance of Thm. 19(c), sampling the system register has Fisher trace P
    (versus <= 4 for any ancilla-only measurement).
Run:  python3 -I theory_checks/k_resolved_variance.py   (numpy only)
"""
import numpy as np

rng = np.random.default_rng(3)
ok_all = []
IRRED = {2: 0b111, 3: 0b1011, 4: 0b10011}


def check(name, ok, detail=""):
    ok_all.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def par(x):
    return bin(int(x)).count("1") & 1


def gf_mul(x, y, m):
    r = 0
    while y:
        if y & 1:
            r ^= x
        y >>= 1
        x <<= 1
        if x >> m:
            x ^= IRRED[m]
    return r


def cube(m):
    return [gf_mul(gf_mul(t, t, m), t, m) for t in range(2 ** m)]


def haar(N):
    Z = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    Q, R = np.linalg.qr(Z)
    return Q * (np.diag(R) / np.abs(np.diag(R)))


for m in (2, 3):
    K, N = 2 ** m, 16
    phi = cube(m)
    psi = rng.normal(size=N) + 1j * rng.normal(size=N)
    psi /= np.linalg.norm(psi)
    X = np.array([haar(N) @ psi for _ in range(K)])
    chi = np.array([[[(-1) ** (par(i & t) ^ par(r & phi[t])) for t in range(K)] for r in range(K)] for i in range(K)])
    P = np.abs(np.einsum("irt,tk->irk", chi, X)) ** 2 / K ** 3          # P[i, rho, k]
    for trial in range(3):
        beta = rng.uniform(0.1, 1.0, K) * rng.choice([-1, 1], K)
        W = np.einsum("irt,t->ir", chi, beta) ** 2                      # (sum beta chi)^2 per outcome
        g = (K / 2) * (W - beta @ beta)                                 # per-shot value, times 1[k_s = k]
        mean_k = np.einsum("ir,irk->k", g, P)
        f_off = np.abs(beta @ X) ** 2 - (beta ** 2) @ (np.abs(X) ** 2)  # f_beta(k) - diagonal part
        check(f"K={K}: unbiased off-diagonal part of f_beta(k), all k", np.allclose(mean_k, f_off, atol=1e-13))
        second = np.einsum("ir,irk->", g ** 2, P)                      # sum_k E[g_k^2]
        bound = (K ** 2 / 4) * np.sum(np.abs(beta)) ** 4
        check(f"K={K}: sum_k E[g_k^2] = {second:.3f} <= (K^2/4)|beta|_1^4 = {bound:.1f}", second <= bound)

# joint measurement: diagonal instance of Thm. 19(c); Fisher information of sampling the system register k
for m in (2, 3):
    K = 2 ** m
    phi = cube(m)
    pairs = [(a, b) for a in range(K) for b in range(a + 1, K)]
    Pn = len(pairs)
    ks = [(k1, k2) for k1 in range(K) for k2 in range(K)]
    gam = np.array([[(-1) ** (par((a ^ b) & k1) ^ par((phi[a] ^ phi[b]) & k2)) for (k1, k2) in ks] for (a, b) in pairs])
    q0 = np.full(K * K, 1.0 / K ** 2)
    F = np.zeros((Pn, Pn))
    for k in range(K * K):
        dp = gam[:, k] / K ** 2
        F += np.outer(dp, dp) / q0[k]
    check(f"K={K}: diagonal instance: system sampling has tr F = {np.trace(F):.1f} = P = {Pn}", abs(np.trace(F) - Pn) < 1e-9)

print("ALL PASS" if all(ok_all) else "SOME FAILED")
