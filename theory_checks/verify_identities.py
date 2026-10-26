"""
Numerical checks of the identities and claims stated in the theory sections of paper/main.tex.
Each check prints PASS/FAIL; nothing here produces a figure of the paper.

Conventions (match the paper): K = 2^m, S[i,t] = (-1)^{popcount(i&t)}, w_t = cos th_t, r_t = sin th_t,
x_t = U_t psi, Phi[(rho,i),k] = (1/K) sum_t S[i,t] rho_t x_t[k], rho_t(0)=w_t, rho_t(1)=r_t.
Run:  python3 theory_checks/verify_identities.py
"""
import itertools
import numpy as np
from scipy.linalg import hadamard, block_diag

rng = np.random.default_rng(7)
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def haar(N):
    Z = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    Q, R = np.linalg.qr(Z)
    return Q * (np.diag(R) / np.abs(np.diag(R)))


def rand_psi(N):
    p = rng.normal(size=N) + 1j * rng.normal(size=N)
    return p / np.linalg.norm(p)


def instance(K, N, theta=None, orthonormal=False):
    th = rng.uniform(0.15, 1.4, K) if theta is None else theta
    w, r = np.cos(th), np.sin(th)
    if orthonormal:  # x_t pairwise orthogonal: take columns of one Haar unitary
        Q = haar(N)
        X = Q[:, :K].T
        Us, psi = None, None
    else:
        Us = [haar(N) for _ in range(K)]
        psi = rand_psi(N)
        X = np.array([U @ psi for U in Us])
    S = hadamard(K).astype(float)
    Phi = np.vstack([S * w, S * r]) @ X / K
    return th, w, r, S, X, Phi, Us, psi


# ---------------------------------------------------------------- Lemma 0: C^T C, POVM, total probability
for K in (2, 4, 8, 16):
    th = rng.uniform(0.1, 1.4, K); w, r = np.cos(th), np.sin(th); S = hadamard(K).astype(float)
    C = np.vstack([S * w, S * r]) / K
    check(f"C^T C = I/K (K={K})", np.allclose(C.T @ C, np.eye(K) / K))
    check(f"POVM sum_j K c_j c_j^T = I (K={K})", np.allclose(sum(K * np.outer(c, c) for c in C), np.eye(K)))
K, N = 4, 16
th, w, r, S, X, Phi, Us, psi = instance(K, N)
P = np.abs(Phi) ** 2
check("sum of all outcome probabilities = 1", np.isclose(P.sum(), 1.0))
check("P(k) = (1/K) sum_t |x_t(k)|^2 (key independent)", np.allclose(P.sum(0), (np.abs(X) ** 2).sum(0) / K))
check("P(rho=0,k) = (1/K) sum_t w_t^2 |x_t(k)|^2", np.allclose(P[:K].sum(0), (w[:, None] ** 2 * np.abs(X) ** 2).sum(0) / K))
check("P(rho=0) = mean(w^2)", np.isclose(P[:K].sum(), np.mean(w ** 2)))

# ---------------------------------------------------------------- reduced system state is (1/K) sum U_t psi psi^dag U_t^dag
rhoS = sum(np.outer(Phi[j], Phi[j].conj()) for j in range(2 * K))
check("system marginal = (1/K) sum_t x_t x_t^dag", np.allclose(rhoS, sum(np.outer(x, x.conj()) for x in X) / K))

# ---------------------------------------------------------------- Prop A2 / accounting
alpha = rng.uniform(0.2, 1.0, K)
c = alpha.max(); Tpsi = sum(a * x for a, x in zip(alpha, X))
p00 = np.linalg.norm(Tpsi) ** 2 / (c ** 2 * K ** 2)
p_std = np.linalg.norm(Tpsi) ** 2 / alpha.sum() ** 2
Phi_a = np.vstack([S * (alpha / c), S * np.sqrt(1 - (alpha / c) ** 2)]) @ X / K
check("p_{0,0} = ||T psi||^2/(c^2 K^2)", np.isclose(np.sum(np.abs(Phi_a[0]) ** 2), p00))
check("p00/p_std = (||a||_1/(K||a||_inf))^2", np.isclose(p00 / p_std, (alpha.sum() / (K * c)) ** 2))

# ---------------------------------------------------------------- Throughput with orthonormal branches
th, w, r, S, X, Phi, _, _ = instance(8, 64, orthonormal=True)[:8]
pj = (np.abs(Phi) ** 2).sum(1)
check("orthonormal branches: p_{i,0} = ||w||^2/K^2, p_{i,1} = ||r||^2/K^2",
      np.allclose(pj[:8], (w ** 2).sum() / 64) and np.allclose(pj[8:], (r ** 2).sum() / 64))

# ---------------------------------------------------------------- Walsh / Gram identity (Thm: ancilla-only statistics)
for K in (2, 4, 8):
    N = 8
    th, w, r, S, X, Phi, _, _ = instance(K, N)[:8]
    P = np.abs(Phi) ** 2                      # (2K, N)
    Pa = P.sum(1)                              # ancilla-only record
    G = X.conj() @ X.T                         # G[t,t'] = <x_t | x_t'>; only the real part enters
    ok = True
    for rho_idx, rho in enumerate((w, r)):
        for u in range(K):
            lhs = sum(S[i, u] * Pa[rho_idx * K + i] for i in range(K))
            rhs = sum(rho[t] * rho[t ^ u] * np.real(G[t, t ^ u]) for t in range(K)) / K
            ok &= np.isclose(lhs, rhs)
    check(f"Walsh transform of ancilla record = XOR-weighted Gram sums (K={K})", ok)
    # per-k version
    ok = True
    for rho_idx, rho in enumerate((w, r)):
        for u in range(K):
            lhs = np.array([sum(S[i, u] * P[rho_idx * K + i, k] for i in range(K)) for k in range(N)])
            rhs = sum(rho[t] * rho[t ^ u] * np.real(X[t] * X[t ^ u].conj()) for t in range(K)) / K
            ok &= np.allclose(lhs, rhs)
    check(f"per-k Walsh identity (K={K})", ok)

# generic rank for M >= K/4 settings (class systems), K = 8, 16
for K in (4, 8, 16):
    M = max(1, K // 4)
    ths = rng.uniform(0.1, 1.4, (M, K))
    ok = True
    for u in range(1, K):
        pairs = [(t, t ^ u) for t in range(K) if t < (t ^ u)]
        rows = []
        for s in range(M):
            w, r = np.cos(ths[s]), np.sin(ths[s])
            rows.append([w[a] * w[b] for a, b in pairs]); rows.append([r[a] * r[b] for a, b in pairs])
        ok &= np.linalg.matrix_rank(np.array(rows), tol=1e-9) == K // 2
    check(f"M=ceil(K/4)={M} settings determine all Re<x_t|x_t'> classes (K={K})", ok)

# ---------------------------------------------------------------- amplitude inverse and redundancy
K, N = 8, 32
th, w, r, S, X, Phi, _, _ = instance(K, N)[:8]
C = np.vstack([S * w, S * r]) / K
check("X = K C^T Phi", np.allclose(X, K * C.T @ Phi))
check("X = diag(1/w) S^T Phi_0", np.allclose(X, (S.T @ Phi[:K]) / w[:, None]))
G_ = S @ np.diag(r / w) @ S.T / K
check("Phi_1 = G Phi_0, G = (1/K) S diag(tan th) S^T", np.allclose(Phi[K:], G_ @ Phi[:K]))
U_, s_, _ = np.linalg.svd(Phi, full_matrices=False)
lev = np.sum(np.abs(U_[:, :K]) ** 2, 1)
check("row leverage = mean(w^2) / mean(r^2)", np.allclose(lev[:K], np.mean(w ** 2)) and np.allclose(lev[K:], np.mean(r ** 2)))

# ---------------------------------------------------------------- block-encoding view: V = H M H similar to M
K, N = 4, 4
th = rng.uniform(0.15, 1.4, K); w, r = np.cos(th), np.sin(th)
Us = [haar(N) for _ in range(K)]
Hn = hadamard(K) / np.sqrt(K)
M = block_diag(*[np.kron(np.array([[w[t], r[t]], [r[t], -w[t]]]), Us[t]) for t in range(K)])
HH = np.kron(Hn, np.eye(2 * N)); V = HH @ M @ HH
check("H^2 = I and V = H M H^{-1}", np.allclose(Hn @ Hn, np.eye(K)) and np.allclose(V, HH @ M @ np.linalg.inv(HH)))
ev_V = np.sort_complex(np.linalg.eigvals(V)); ev_M = np.sort_complex(np.linalg.eigvals(M))
check("spec(V) = spec(M) = union_t {+-mu : mu in spec U_t}", np.allclose(ev_V, ev_M, atol=1e-8))
spec_pm = np.sort_complex(np.concatenate([np.concatenate([np.linalg.eigvals(U), -np.linalg.eigvals(U)]) for U in Us]))
check("... equals explicit +-spec(U_t)", np.allclose(ev_V, spec_pm, atol=1e-8))
check("V^2 = (H x I)(+_t I_2 x U_t^2)(H x I)", np.allclose(V @ V, HH @ block_diag(*[np.kron(np.eye(2), U @ U) for U in Us]) @ HH))
check("eigenvalues of V are NOT all +-1 for generic U_t", not np.allclose(np.abs(ev_V.real), 1))
# block-encoding normalisation
blk = V[:N, :N]
D_w = sum(w[t] * Us[t] for t in range(K))
check("<0,0|V|0,0> = D_w / K", np.allclose(blk, D_w / K))

# ---------------------------------------------------------------- hiding: undo-H attack, known-plaintext key recovery
K, N = 4, 8
th, w, r, S, X, Phi, Us, psi = instance(K, N)
# build the full output state: index (x) rot (x) sys, then undo public H and measure index
rho_vec = np.stack([w, r], 1)                              # |rho_t> = w|0> + r|1>
branch = np.zeros((K, 2, N), complex)
for t in range(K):
    branch[t] = np.outer(rho_vec[t], Us[t] @ psi) / np.sqrt(K)
Hn = hadamard(K) / np.sqrt(K)
out = np.einsum("it,trn->irn", Hn, branch)                 # output of the circuit before measurement
undo = np.einsum("it,trn->irn", Hn, out)                   # adversary applies public H
ok = True
for t in range(K):
    s_ = undo[t].reshape(2, N)                             # post-measurement (unnormalised) state given index t
    rec0 = (Us[t].conj().T @ s_[0]); rec1 = (Us[t].conj().T @ s_[1])
    # state is |rho_t> (x) U_t psi / sqrt(K): applying U_t^dag gives |rho_t> (x) psi / sqrt(K)
    ok &= np.allclose(rec0, w[t] * psi / np.sqrt(K)) and np.allclose(rec1, r[t] * psi / np.sqrt(K))
check("undo-H attack returns psi exactly, no key needed (model Q)", ok)

# known-plaintext: P(rho=0,k) = (1/K) sum_t w_t^2 |x_t(k)|^2  -> linear LS for w_t^2
N = 64
th, w, r, S, X, Phi, Us, psi = instance(K, N)
P = np.abs(Phi) ** 2
A = (np.abs(X) ** 2).T / K
w2, *_ = np.linalg.lstsq(A, P[:K].sum(0), rcond=None)
check("known-plaintext attack recovers w_t^2 by N x K least squares", np.allclose(w2, w ** 2, atol=1e-10))

# ---------------------------------------------------------------- identifiability checks (real amplitudes)
def n_solutions_real(K, w, r, a):
    S = hadamard(K).astype(float)
    C = np.vstack([S * w, S * r]) / K
    p = (C @ a) ** 2
    sols = []
    for sg in itertools.product([1, -1], repeat=2 * K - 1):
        y = np.r_[1, sg] * np.sqrt(p)
        xs = np.linalg.lstsq(C, y, rcond=None)[0]
        if np.linalg.norm(C @ xs - y) < 1e-9:
            sols.append(xs * np.sign(xs[np.argmax(np.abs(xs))]))
    return len(np.unique(np.round(sols, 6), axis=0))

for K in (2, 4):
    th = rng.uniform(0.15, 1.4, K)
    a = rng.normal(size=K)
    check(f"real amplitudes, generic angles: unique up to sign (K={K})",
          n_solutions_real(K, np.cos(th), np.sin(th), a) == 1)
    th_eq = np.full(K, 0.7)
    n_eq = n_solutions_real(K, np.cos(th_eq), np.sin(th_eq), a)
    check(f"real amplitudes, equal angles: ambiguous (K={K})", n_eq > 1, f"({n_eq} solutions)")

# K=4 closed form: classes u determine off-diagonal products by 2x2 systems, det = w_a w_b r_c r_d - w_c w_d r_a r_b
K = 4
th = rng.uniform(0.15, 1.4, K); w, r = np.cos(th), np.sin(th)
a = rng.normal(size=K)
S = hadamard(K).astype(float); C = np.vstack([S * w, S * r]) / K
p = (C @ a) ** 2
ok = True
for u in (1, 2, 3):
    pairs = [(t, t ^ u) for t in range(K) if t < (t ^ u)]
    # Walsh transform of the data over i, for each rho
    d0 = sum(S[i, u] * p[i] for i in range(K)) * K            # = sum_t w_t w_{t+u} a_t a_{t+u}
    d1 = sum(S[i, u] * p[K + i] for i in range(K)) * K
    Mx = np.array([[2 * w[x] * w[y] for x, y in pairs], [2 * r[x] * r[y] for x, y in pairs]])
    sol = np.linalg.solve(Mx, np.array([d0, d1]))
    ok &= np.allclose(sol, [a[x] * a[y] for x, y in pairs])
check("K=4: each XOR class (2 pairs) solved by a 2x2 system for the products a_t a_t'", ok)

# full spark for generic angles, K=4,8 (exhaustive) and K=16 (random subsets)
for K in (4, 8):
    th = rng.uniform(0.1, 1.4, K); w, r = np.cos(th), np.sin(th); S = hadamard(K).astype(float)
    C = np.vstack([S * w, S * r]) / K
    mind = min(abs(np.linalg.det(C[list(R)])) for R in itertools.combinations(range(2 * K), K))
    check(f"C is full spark for generic angles (K={K}, exhaustive)", mind > 1e-12, f"(min |minor| = {mind:.2e})")
K = 16
th = rng.uniform(0.1, 1.4, K); w, r = np.cos(th), np.sin(th); S = hadamard(K).astype(float)
C = np.vstack([S * w, S * r]) / K
bad = sum(np.linalg.slogdet(K * C[rng.choice(2 * K, K, replace=False)])[0] == 0 for _ in range(20000))
check("C is full spark for generic angles (K=16, 20000 random subsets)", bad == 0)

# ---------------------------------------------------------------- complex case: no uniform injectivity at K=4 (pencil argument)
K = 4
th = rng.uniform(0.15, 1.4, K); w, r = np.cos(th), np.sin(th); S = hadamard(K).astype(float)
C = np.vstack([S * w, S * r]) / K
idx = [(i, j) for i in range(K) for j in range(i, K)]
def F_of(Y):  # data of a symmetric matrix Y: c_j^T Y c_j
    return np.array([c @ Y @ c for c in C])
Fmat = np.array([[c[i] * c[j] * (1 if i == j else 2) for (i, j) in idx] for c in C])      # 8 x 10
_, _, Vt = np.linalg.svd(Fmat); ker = Vt[np.linalg.matrix_rank(Fmat):]
def to_sym(v):
    Z = np.zeros((K, K))
    for val, (i, j) in zip(v, idx):
        Z[i, j] = Z[j, i] = val
    return Z
A_, B_ = to_sym(ker[0]), to_sym(ker[1])
found = None
for phi in np.linspace(0, np.pi, 4001):
    Z = np.cos(phi) * A_ + np.sin(phi) * B_
    ev = np.linalg.eigvalsh(Z)
    if (ev > 1e-9).sum() <= 2 and (ev < -1e-9).sum() <= 2 and np.abs(ev).max() > 1e-6:
        found = Z; break
if found is not None:
    ev, Ev = np.linalg.eigh(found)
    Pp = (Ev[:, ev > 1e-9] * ev[ev > 1e-9]) @ Ev[:, ev > 1e-9].T
    Nn = -(Ev[:, ev < -1e-9] * ev[ev < -1e-9]) @ Ev[:, ev < -1e-9].T
    same = np.allclose(F_of(Pp), F_of(Nn), atol=1e-12)
    check("K=4: two distinct PSD matrices of rank<=2 give identical data (no uniform injectivity)",
          same and not np.allclose(Pp, Nn), f"(ranks {np.linalg.matrix_rank(Pp)},{np.linalg.matrix_rank(Nn)})")
else:
    check("K=4 pencil contains an element of signature <=(2,2)", False)

# ---------------------------------------------------------------- complex case: local injectivity (Jacobian of x -> |Cx|^2)
for K in (2, 4, 8, 16):
    th = rng.uniform(0.15, 1.4, K); S = hadamard(K).astype(float)
    C = np.vstack([S * np.cos(th), S * np.sin(th)]) / K
    ranks = []
    for _ in range(3):
        x = rng.normal(size=K) + 1j * rng.normal(size=K)
        z = np.r_[x.real, x.imag]
        f = lambda zz: np.abs(C @ (zz[:K] + 1j * zz[K:])) ** 2
        J = np.zeros((2 * K, 2 * K)); e = 1e-6
        for j in range(2 * K):
            d = np.zeros(2 * K); d[j] = e
            J[:, j] = (f(z + d) - f(z - d)) / (2 * e)
        ranks.append(np.linalg.matrix_rank(J, tol=1e-6 * np.linalg.norm(J, 2)))
    check(f"Jacobian of x -> |Cx|^2 has rank 2K-1 (K={K})", all(r_ == 2 * K - 1 for r_ in ranks), f"(ranks {ranks})")

# ---------------------------------------------------------------- combinatorial condition of the full-spark lemma (all (I0, I1))
def lemma_condition_holds(K):
    S = hadamard(K).astype(float)
    rows = range(K)
    for a in range(K + 1):
        b = K - a
        for I0 in itertools.combinations(rows, a):
            for I1 in itertools.combinations(rows, b):
                ok = False
                for T in itertools.combinations(range(K), b):
                    Tc = [t for t in range(K) if t not in T]
                    d1 = np.linalg.det(S[np.ix_(I1, T)]) if b else 1.0
                    d0 = np.linalg.det(S[np.ix_(I0, Tc)]) if a else 1.0
                    if abs(d1) > 1e-9 and abs(d0) > 1e-9:
                        ok = True; break
                if not ok:
                    return False
    return True
for K in (2, 4, 8):
    check(f"combinatorial condition of the full-spark lemma for all (I0,I1), K={K}", lemma_condition_holds(K))

# K=2 boundary: theta = (0, pi/2) loses the sign of a0 a1
C2 = np.vstack([hadamard(2) * np.cos([0, np.pi / 2]), hadamard(2) * np.sin([0, np.pi / 2])]) / 2
a = np.array([0.7, 1.3])
check("K=2, theta=(0,pi/2): (a0,a1) and (a0,-a1) give identical data", np.allclose((C2 @ a) ** 2, (C2 @ (a * [1, -1])) ** 2))

print(f"\n{sum(results)}/{len(results)} checks passed")
raise SystemExit(0 if all(results) else 1)
