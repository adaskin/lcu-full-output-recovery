"""Secret generic mixing unitary F (instead of the public Hadamard) in the classical-record model.

Record: P(j,rho,k) = |sum_t F_jt rho_t x_t(k)|^2 / K,  x_t = U_t psi, public U_t, secret (theta, F).
Checks
 (1) the (rho,k) and k marginals do not depend on F;
 (2) known-input attack: angles from the (rho,k) marginal by linear least squares,
     then each row of F by phase retrieval with KNOWN sensing vectors.
"""
import numpy as np
rng = np.random.default_rng(7)

def haar(d):
    z = (rng.normal(size=(d, d)) + 1j*rng.normal(size=(d, d)))/np.sqrt(2)
    q, r = np.linalg.qr(z)
    return q*(np.diag(r)/abs(np.diag(r)))

def record(F, theta, X):                      # X: K x N branch matrix
    K = len(theta)
    w, r = np.cos(theta), np.sin(theta)
    P = np.zeros((2, K, X.shape[1]))
    for rho, v in enumerate((w, r)):
        P[rho] = np.abs(F @ (v[:, None]*X))**2/K
    return P                                  # P[rho, j, k]

K, n = 8, 6
N = 2**n
U = [haar(N) for _ in range(K)]
theta = rng.uniform(0, np.pi/2, K)
F = haar(K)
H = np.array([[(-1)**bin(i & t).count("1") for t in range(K)] for i in range(K)])/np.sqrt(K)

def branches(psi): return np.array([u@psi for u in U])
psi = haar(N)[:, 0]
X = branches(psi)

# (1) marginals independent of the mixing unitary
P_F, P_H = record(F, theta, X), record(H, theta, X)
print("max |P(rho,k;F)-P(rho,k;H)|  :", np.abs(P_F.sum(1)-P_H.sum(1)).max())
print("max |P(k;F)-(1/K)sum|x_t|^2| :", np.abs(P_F.sum((0, 1)) - (np.abs(X)**2).sum(0)/K).max())
print("max |P(i,rho,k;F)-P(i,rho,k;H)| (full record differs):", np.abs(P_F-P_H).max())

# (2) known-input attack. m known inputs
m = 4
Xs = [branches(haar(N)[:, 0]) for _ in range(m)]
Pk = [record(F, theta, x) for x in Xs]
# angles: P(rho,k) = (1/K) sum_t rho_t^2 |x_t(k)|^2 -> least squares for w^2
A = np.vstack([(np.abs(x)**2).T/K for x in Xs])             # (m N) x K
b0 = np.concatenate([p[0].sum(0) for p in Pk])
b1 = np.concatenate([p[1].sum(0) for p in Pk])
w2 = np.linalg.lstsq(A, b0, rcond=None)[0]
r2 = np.linalg.lstsq(A, b1, rcond=None)[0]
th_hat = np.arctan2(np.sqrt(np.clip(r2, 0, None)), np.sqrt(np.clip(w2, 0, None)))
print("angle error (max)           :", np.abs(th_hat-theta).max(), "| w2+r2-1:", np.abs(w2+r2-1).max())

# rows of F: |<f_j, y>|^2 = K P, sensing vectors y_{rho,k} = rho_t x_t(k) (known now)
Y = np.vstack([np.hstack([np.cos(th_hat)[:, None]*x, np.sin(th_hat)[:, None]*x]).T for x in Xs])  # (m*2N) x K
def phase_retrieval(Y, meas, iters=3000, starts=6):
    best = None
    for s in range(starts):
        f = rng.normal(size=K) + 1j*rng.normal(size=K); f /= np.linalg.norm(f)
        for _ in range(iters):                              # alternating projections (Gerchberg-Saxton)
            z = Y @ f
            z = np.sqrt(meas)*np.exp(1j*np.angle(z))
            f = np.linalg.lstsq(Y, z, rcond=None)[0]
        res = np.linalg.norm(np.abs(Y@f)**2-meas)/np.linalg.norm(meas)
        if best is None or res < best[0]: best = (res, f)
    return best
overlaps = np.zeros((K, K))
resid = []
for j in range(K):
    meas = np.concatenate([np.concatenate([Pk[i][0, j], Pk[i][1, j]]) for i in range(m)])*K
    res, f = phase_retrieval(Y, meas)
    resid.append(res)
    f = f/np.linalg.norm(f)
    overlaps[j] = np.abs(np.array([np.vdot(F[l], f) for l in range(K)]))
print("max relative residual of the phase-retrieval fits :", max(resid))
print("recovered row j has overlap with true row j        :", np.round(np.diag(overlaps), 6))
print("largest overlap with any OTHER row (sanity)        :", np.round(overlaps[~np.eye(K, dtype=bool)].max(), 6))

# (3) observer Q with a Haar-random secret F: single copy averages to (I/K) (x) omega  (Monte Carlo, K=4, N=2)
Kq, Nq, M = 4, 2, 40000
Uq = [haar(Nq) for _ in range(Kq)]
thq = rng.uniform(0, np.pi/2, Kq)
psiq = haar(Nq)[:, 0]
Om = np.zeros((Kq, 2, Nq), complex)
for t in range(Kq):
    Om[t, 0] = np.cos(thq[t])*Uq[t]@psiq/np.sqrt(Kq)
    Om[t, 1] = np.sin(thq[t])*Uq[t]@psiq/np.sqrt(Kq)
Om = Om.reshape(Kq, 2*Nq)
avg = np.zeros((Kq*2*Nq,)*2, complex)
for _ in range(M):
    v = (haar(Kq) @ Om).reshape(-1)
    avg += np.outer(v, v.conj())
avg /= M
omega = sum(np.outer(Om[t], Om[t].conj()) for t in range(Kq))
target = np.kron(np.eye(Kq)/Kq, omega)
print("Q average vs (I/K) x omega: max deviation", np.abs(avg-target).max(), "of scale", np.abs(target).max(), "(Monte Carlo error ~", 1/np.sqrt(M), ")")
