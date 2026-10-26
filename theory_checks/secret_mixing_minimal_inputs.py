"""How many KNOWN inputs does the known-input key recovery of Prop. 11(c) need? (exact probabilities, noiseless)

Each input gives 2N sensing vectors per row of F; complex phase retrieval needs about 4K-4, so about ceil((4K-4)/(2N)) inputs."""
import numpy as np
rng = np.random.default_rng(11)
def haar(d):
    z=(rng.normal(size=(d,d))+1j*rng.normal(size=(d,d)))/np.sqrt(2); q,r=np.linalg.qr(z); return q*(np.diag(r)/abs(np.diag(r)))
def run(K, N, m, starts=8, iters=2500):
    U=[haar(N) for _ in range(K)]; th=rng.uniform(0,np.pi/2,K); F=haar(K)
    w,r=np.cos(th),np.sin(th)
    Xs=[np.array([u@haar(N)[:,0] for u in U]) for _ in range(m)]
    # exact record P[rho][j,k]
    Pk=[[np.abs(F@(v[:,None]*x))**2/K for v in (w,r)] for x in Xs]
    A=np.vstack([(np.abs(x)**2).T/K for x in Xs])
    b0=np.concatenate([p[0].sum(0) for p in Pk]); b1=np.concatenate([p[1].sum(0) for p in Pk])
    ok_angles = (A.shape[0] >= K) and np.linalg.matrix_rank(A)==K
    if ok_angles:
        w2=np.linalg.lstsq(A,b0,rcond=None)[0]; r2=np.linalg.lstsq(A,b1,rcond=None)[0]
        th_hat=np.arctan2(np.sqrt(np.clip(r2,0,None)),np.sqrt(np.clip(w2,0,None))); aerr=np.abs(th_hat-th).max()
    else:
        return dict(K=K,N=N,m=m,angles="rank-deficient")
    Y=np.vstack([np.hstack([np.cos(th_hat)[:,None]*x, np.sin(th_hat)[:,None]*x]).T for x in Xs])
    worst=0
    for j in range(K):
        meas=np.concatenate([np.concatenate([Pk[i][0][j],Pk[i][1][j]]) for i in range(m)])*K
        best=None
        for s in range(starts):
            f=rng.normal(size=K)+1j*rng.normal(size=K); f/=np.linalg.norm(f)
            for _ in range(iters):
                z=Y@f; z=np.sqrt(meas)*np.exp(1j*np.angle(z)); f=np.linalg.lstsq(Y,z,rcond=None)[0]
            res=np.linalg.norm(np.abs(Y@f)**2-meas)/np.linalg.norm(meas)
            if best is None or res<best[0]: best=(res,f)
        f=best[1]/np.linalg.norm(best[1])
        worst=max(worst,1-abs(np.vdot(F[j],f)))
    return dict(K=K,N=N,m=m,angle_err=float(aerr),row_err=float(worst),n_unknown_real=K*K+K,n_numbers=m*2*K*N)
for K,N,m in [(4,4,1),(4,4,2),(4,16,1),(8,8,1),(8,8,2),(8,8,3),(8,16,1),(8,16,2),(8,64,1)]:
    print(run(K,N,m),flush=True)
