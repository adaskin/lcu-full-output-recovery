"""
Checks for the Fisher-information bound of Sec. VI (lower bound for ancilla-only Select-based measurements).

Setting: ancilla state sum_t sqrt(a_t)|t>|chi_t>, system Gram matrix G(theta) = I + sum_{a<b} theta_ab (E_ab + E_ba),
ancilla state rho(theta) = V diag(sqrt a) G diag(sqrt a) V^dag with V = sum_t |t,chi_t><t|.
Claim: for any rank-one POVM, the trace of the Fisher matrix w.r.t. (theta_ab) is <= 2 at theta = 0 and <= 2/(1-c)
whenever G >= (1-c) I.  The APN phase-register measurement attains 2 (K-1)/K.
Run:  python3 -I theory_checks/fisher_bound_check.py   (numpy only)
"""
import numpy as np

rng = np.random.default_rng(5)
ok_all = []


def check(name, ok, detail=""):
    ok_all.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def par(x):
    return bin(int(x)).count("1") & 1


def fisher_trace(V, a, theta, povm):
    """povm: list of (weight w_j, unit vector e_j) on the ancilla space; returns tr F and p."""
    K = len(a)
    pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
    G = np.eye(K)
    for (i, j), th in zip(pairs, theta):
        G[i, j] = G[j, i] = th
    rho = V @ np.diag(np.sqrt(a)) @ G @ np.diag(np.sqrt(a)) @ V.conj().T
    tr = 0.0
    for w, e in povm:
        p = np.real(w * e.conj() @ rho @ e)
        if p < 1e-14:
            continue
        for (i, j) in pairs:
            # d p / d theta_ij = 2 w sqrt(a_i a_j) Re( <e|V_i><V_j|e> ) with V_i = V e_i
            dp = 2 * w * np.sqrt(a[i] * a[j]) * np.real((e.conj() @ V[:, i]) * (V[:, j].conj() @ e))
            tr += dp ** 2 / p
    return tr


def random_povm(D):
    """random orthonormal basis of C^D, weights 1 (sum_j |e_j><e_j| = I)."""
    Z = rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D))
    Q, _ = np.linalg.qr(Z)
    return [(1.0, Q[:, j]) for j in range(D)]


for K, aux in ((4, 1), (4, 3), (8, 2)):
    worst0 = worstc = 0.0
    for _ in range(30):
        D = K * aux
        chi = rng.normal(size=(K, aux)) + 1j * rng.normal(size=(K, aux))
        chi /= np.linalg.norm(chi, axis=1, keepdims=True)
        V = np.zeros((D, K), dtype=complex)
        for t in range(K):
            V[t * aux:(t + 1) * aux, t] = chi[t]
        a = rng.uniform(0.2, 1, K)
        a /= a.sum()
        povm = random_povm(D)
        worst0 = max(worst0, fisher_trace(V, a, np.zeros(K * (K - 1) // 2), povm))
        P = K * (K - 1) // 2
        theta = rng.uniform(-1, 1, P) / (2 * P)       # sum|theta| <= 1/2, so G >= I/2
        worstc = max(worstc, fisher_trace(V, a, theta, povm))
    check(f"K={K}, aux dim {aux}: tr F <= 2 at theta=0 for random designs and POVMs", worst0 <= 2 + 1e-9, f"(max {worst0:.3f})")
    check(f"K={K}, aux dim {aux}: tr F <= 4 for G >= I/2", worstc <= 4 + 1e-9, f"(max {worstc:.3f})")

# the APN phase-register design saturates the bound: tr F = 2 (K-1)/K
IRRED = {2: 0b111, 3: 0b1011, 4: 0b10011}


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


for m in (2, 3, 4):
    K = 2 ** m
    phi = [gf_mul(gf_mul(t, t, m), t, m) for t in range(K)]
    D = K * K
    V = np.zeros((D, K), dtype=complex)
    for t in range(K):
        V[t * K + phi[t], t] = 1.0
    # Hadamard basis on both registers: e_{(i,rho)}(t, phi) = (-1)^{i.t + rho.phi} / K
    povm = []
    for i in range(K):
        for rho in range(K):
            e = np.array([(-1) ** (par(i & t) ^ par(rho & f)) for t in range(K) for f in range(K)], dtype=complex) / K
            povm.append((1.0, e))
    tr = fisher_trace(V, np.full(K, 1.0 / K), np.zeros(K * (K - 1) // 2), povm)
    check(f"K={K}: APN measurement has tr F = 2(K-1)/K", abs(tr - 2 * (K - 1) / K) < 1e-9, f"(tr F = {tr:.4f})")

print("ALL PASS" if all(ok_all) else "SOME FAILED")
