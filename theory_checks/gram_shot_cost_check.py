# Crude check of the worst-entry shot cost of the rotation-qubit Gram estimator (Thm. 7) vs Hadamard tests.
# Random search over angles, one Haar instance, delta-method covariance. Evidence only, not a proof. Run: python3 -I theory_checks/gram_shot_cost_check.py
import numpy as np, itertools
rng=np.random.default_rng(1)
def haar(K,N):
    A=rng.normal(size=(N,N))+1j*rng.normal(size=(N,N)); Q,R=np.linalg.qr(A); return Q*(np.diag(R)/abs(np.diag(R)))
def setup(K,N):
    psi=rng.normal(size=N)+1j*rng.normal(size=N); psi/=np.linalg.norm(psi)
    X=np.array([haar(K,N)@psi for _ in range(K)])
    return X
def S_(K): return np.array([[(-1)**bin(i&t).count('1') for t in range(K)] for i in range(K)])
def Cmat(th):
    K=len(th); S=S_(K); return np.vstack([S*np.cos(th)/K, S*np.sin(th)/K])  # rows (rho,i)
pairs=lambda K:[(a,b) for a in range(K) for b in range(a+1,K)]
def design(ths,K):
    # linear map: p_(rho,i) summed over k = c^T ReG c ; unknown offdiag ReG
    pr=pairs(K); rows=[]
    for th in ths:
        C=Cmat(th); 
        A=np.array([[2*C[r,a]*C[r,b] for (a,b) in pr] for r in range(2*K)])
        rows.append(A)
    return rows,pr
def stds(ths,X,K):
    # per-setting prob vector, then estimator g=pinv(Astack) (p - diag part)
    rows,pr=design(ths,K); M=len(ths); A=np.vstack(rows)
    # shots split equally: S/M each; covariance of p̂_s = (diag p - pp^T)/(S/M)
    ps=[]
    for th in ths:
        C=Cmat(th); P=np.abs(C@X)**2; ps.append(P.sum(1))
    B=np.linalg.pinv(A)
    cov=np.zeros((A.shape[0],)*2)
    for s in range(M):
        n=2*K; p=ps[s]; cs=(np.diag(p)-np.outer(p,p))*M
        cov[s*n:(s+1)*n,s*n:(s+1)*n]=cs
    V=B@cov@B.T   # times 1/S
    return np.sqrt(np.diag(V)), np.linalg.matrix_rank(A)==len(pr)
for K,M in [(4,1),(8,2),(16,4)]:
    N=64; X=setup(K,N); X=X/1
    best=None
    for it in range(4000):
        ths=[rng.uniform(0.05,np.pi/2-0.05,K) for _ in range(M)]
        sd,ok=stds(ths,X,K)
        if not ok: continue
        m=sd.max()
        if best is None or m<best[0]: best=(m,sd.mean())
    npairs=K*(K-1)//2
    # Hadamard test: each pair needs S_h shots, std<=1/sqrt(S_h); total shots for std eps: npairs/eps^2
    # ours: std=best/sqrt(S) => S=best^2/eps^2
    print(K,M,"our max-std*sqrt(S)=%.2f -> S=%.1f/eps^2 ; Hadamard total %d/eps^2"%(best[0],best[0]**2,npairs))
