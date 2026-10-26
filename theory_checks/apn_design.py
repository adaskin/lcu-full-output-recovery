"""
Checks for the 'linear record' section of paper/main.tex (APN phase-register design).

Circuit: index register |t> (m qubits) and phase register (m qubits); |t>|0> -> |t>|phi(t)>, Select, then H on both
registers, measure (i, rho, k).  P(i,rho,k) = K^-3 |sum_t chi_t x_t(k)|^2, chi_t = (-1)^{i.t + rho.phi(t)}.
Run:  python3 theory_checks/apn_design.py   (numpy only)
"""
import numpy as np

rng = np.random.default_rng(11)
ok_all = []
IRRED = {2: 0b111, 3: 0b1011, 4: 0b10011, 5: 0b100101}  # x^2+x+1, x^3+x+1, x^4+x+1, x^5+x^2+1


def check(name, ok, detail=""):
    ok_all.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def gf_mul(a, b, m):
    r = 0
    while b:
        if b & 1:
            r ^= a
        b >>= 1
        a <<= 1
        if a >> m:
            a ^= IRRED[m]
    return r


def cube(m):
    return np.array([gf_mul(gf_mul(t, t, m), t, m) for t in range(2 ** m)])


def par(x):
    return bin(int(x)).count("1") & 1


def diff_uniformity(phi):
    K = len(phi)
    best = 0
    for u in range(1, K):
        cnt = {}
        for t in range(K):
            v = int(phi[t]) ^ int(phi[t ^ u])
            cnt[v] = cnt.get(v, 0) + 1
        best = max(best, max(cnt.values()))
    return best


def haar(N):
    Z = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    Q, R = np.linalg.qr(Z)
    return Q * (np.diag(R) / np.abs(np.diag(R)))


def chi_vec(i, rho, phi):
    return np.array([(-1) ** (par(i & t) ^ par(rho & int(phi[t]))) for t in range(len(phi))])


def probs(X, phi):
    """P[i, rho, k] from the amplitude formula, X: (K,N)."""
    K, N = X.shape
    P = np.zeros((K, K, N))
    for i in range(K):
        for rho in range(K):
            P[i, rho] = np.abs(chi_vec(i, rho, phi) @ X) ** 2 / K ** 3
    return P


def statevector_probs(X, phi, m):
    """Independent check: build the state (index, phase register, system), apply the Hadamards."""
    K, N = X.shape
    psi = np.zeros((K, K, N), dtype=complex)
    for t in range(K):
        psi[t, phi[t]] = X[t] / np.sqrt(K)
    H1 = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    H = H1
    for _ in range(m - 1):
        H = np.kron(H, H1)
    psi = np.einsum("ia,ajk->ijk", H, psi)
    psi = np.einsum("jb,ibk->ijk", H, psi)
    return np.abs(psi) ** 2


for m in (2, 3, 4, 5):
    check(f"x^3 over GF(2^{m}) has differential uniformity 2 (APN)", diff_uniformity(cube(m)) == 2)

for m in (2, 3):
    K, N = 2 ** m, 8
    phi = cube(m)
    psi = rng.normal(size=N) + 1j * rng.normal(size=N)
    psi /= np.linalg.norm(psi)
    X = np.array([haar(N) @ psi for _ in range(K)])
    P = probs(X, phi)
    check(f"K={K}: amplitude formula equals state-vector simulation", np.allclose(P, statevector_probs(X, phi, m)))
    check(f"K={K}: probabilities sum to 1", abs(P.sum() - 1) < 1e-12)
    # unbiasedness: E[Z_ab 1_k] = (2/K) Re(x_a(k) conj x_b(k)) for all a != b, every k
    err = 0.0
    for a in range(K):
        for b in range(K):
            if a == b:
                continue
            EZ = np.zeros(N)
            for i in range(K):
                for rho in range(K):
                    z = (-1) ** (par(i & (a ^ b)) ^ par(rho & (int(phi[a]) ^ int(phi[b]))))
                    EZ += z * P[i, rho]
            err = max(err, np.abs(EZ - (2 / K) * np.real(X[a] * np.conj(X[b]))).max())
    check(f"K={K}: E[Z_ab 1_k] = (2/K) Re x_a(k) conj x_b(k), all pairs, all k", err < 1e-13, f"(max err {err:.1e})")
    # target-norm identity
    beta = rng.uniform(0.2, 1.0, K)
    EW = sum((beta @ chi_vec(i, rho, phi)) ** 2 * P[i, rho].sum() for i in range(K) for rho in range(K))
    est = beta @ beta + (K / 2) * (EW - beta @ beta)
    check(f"K={K}: ||T psi||^2 = ||b||^2 + (K/2)(E[(sum b chi)^2] - ||b||^2)", abs(est - np.linalg.norm(beta @ X) ** 2) < 1e-12)

# non-APN phase functions mix pairs
for m in (3, 4):
    K = 2 ** m
    for name, phi in (("identity", np.arange(K)), ("square", np.array([gf_mul(t, t, m) for t in range(K)]))):
        check(f"K={K}: phi = {name} is not APN (uniformity {diff_uniformity(phi)} > 2)", diff_uniformity(phi) > 2)

# linear-identifiability threshold: the 2K directions of the rotation-qubit model span only 2K of K(K+1)/2 dimensions
for m in (1, 2, 3, 4):
    K = 2 ** m
    th = rng.uniform(0.2, 1.3, K)
    S = np.array([[(-1) ** par(i & t) for t in range(K)] for i in range(K)])
    C = np.vstack([S * np.cos(th) / K, S * np.sin(th) / K])
    iu = np.triu_indices(K)
    wgt = np.where(iu[0] == iu[1], 1.0, 2.0)
    A = np.array([np.outer(c, c)[iu] * wgt for c in C])
    r = np.linalg.matrix_rank(A)
    check(f"K={K}: record directions of the rotation-qubit model have rank {r} on Sym(K) (full is {K*(K+1)//2})",
          r == min(2 * K, K * (K + 1) // 2))

# APN design: the K(K-1)/2 pair characters are orthogonal on the K^2 outcomes
for m in (2, 3, 4):
    K = 2 ** m
    phi = cube(m)
    pairs = [(a, b) for a in range(K) for b in range(a + 1, K)]
    M = np.zeros((K * K, len(pairs)))
    for j, (a, b) in enumerate(pairs):
        for i in range(K):
            for rho in range(K):
                M[i * K + rho, j] = (-1) ** (par(i & (a ^ b)) ^ par(rho & (int(phi[a]) ^ int(phi[b]))))
    check(f"K={K}: the {len(pairs)} APN pair characters are orthogonal, norm^2 K^2", np.allclose(M.T @ M, K * K * np.eye(len(pairs))))

# complex overlaps: phase gate diag(1,i) on bit j of the index gives (2/K) Re[i^{a_j-b_j} G_ab]; Im G_ab for a_j != b_j
m = 3
K, N = 2 ** m, 8
phi = cube(m)
psi = rng.normal(size=N) + 1j * rng.normal(size=N)
psi /= np.linalg.norm(psi)
X = np.array([haar(N) @ psi for _ in range(K)])
G = X @ X.conj().T                      # G[a,b] = sum_k x_a(k) conj x_b(k)
err = 0.0
for j in range(m):
    Xj = np.array([1j ** ((t >> j) & 1) * X[t] for t in range(K)])
    P = probs(Xj, phi).sum(axis=2)
    for a in range(K):
        for b in range(K):
            if ((a >> j) & 1) != ((b >> j) & 1):
                EZ = sum((chi_vec(i, r, phi)[a] * chi_vec(i, r, phi)[b]) * P[i, r] for i in range(K) for r in range(K))
                sign = 1 if ((a >> j) & 1) > ((b >> j) & 1) else -1       # i^{a_j-b_j} = sign * i
                err = max(err, abs(EZ - (2 / K) * (-sign) * G[a, b].imag))
check("K=8: bit-j phase gate settings give Im G_ab for every pair differing in bit j", err < 1e-13, f"(max err {err:.1e})")

print("ALL PASS" if all(ok_all) else "SOME FAILED")
