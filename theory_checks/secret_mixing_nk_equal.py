"""Known-input key recovery for N = K (Prop. 11(c)): angles from the square marginal system, rows of F by phase retrieval."""
import numpy as np, sys
rng = np.random.default_rng(5)
def haar(d):
    z=(rng.normal(size=(d,d))+1j*rng.normal(size=(d,d)))/np.sqrt(2); q,r=np.linalg.qr(z); return q*(np.diag(r)/abs(np.diag(r)))
def run(K, N, m, starts=10, iters=1500):
    U=[haar(N) for _ in range(K)]; th=rng.uniform(0,np.pi/2,K); F=haar(K)
    w,r=np.cos(th),np.sin(th)
    Xs=[np.array([u@haar(N)[:,0] for u in U]) for _ in range(m)]
    Pk=[[np.abs(F@(v[:,None]*x))**2/K for v in (w,r)] for x in Xs]
    A=np.vstack([(np.abs(x)**2).T/K for x in Xs])
    rank=np.linalg.matrix_rank(A)
    if rank<K: return dict(K=K,N=N,m=m,angle_rank=int(rank),note="angles not determined")
    b0=np.concatenate([p[0].sum(0) for p in Pk]); b1=np.concatenate([p[1].sum(0) for p in Pk])
    w2=np.linalg.lstsq(A,b0,rcond=None)[0]; r2=np.linalg.lstsq(A,b1,rcond=None)[0]
    th_hat=np.arctan2(np.sqrt(np.clip(r2,0,None)),np.sqrt(np.clip(w2,0,None)))
    aerr=float(np.abs(th_hat-th).max())
    Y=np.vstack([np.hstack([np.cos(th_hat)[:,None]*x, np.sin(th_hat)[:,None]*x]).T for x in Xs]); Yp=np.linalg.pinv(Y)
    worst=0
    for j in range(K):
        meas=np.concatenate([np.concatenate([Pk[i][0][j],Pk[i][1][j]]) for i in range(m)])*K
        best=None
        for s in range(starts):
            f=rng.normal(size=K)+1j*rng.normal(size=K); f/=np.linalg.norm(f)
            for _ in range(iters):
                z=Y@f; f=Yp@(np.sqrt(meas)*np.exp(1j*np.angle(z)))
            res=np.linalg.norm(np.abs(Y@f)**2-meas)/np.linalg.norm(meas)
            if best is None or res<best[0]: best=(res,f)
            if res<1e-10: break
        f=best[1]/np.linalg.norm(best[1]); worst=max(worst,1-abs(np.vdot(F[j],f)))
    return dict(K=K,N=N,m=m,angle_err=aerr,row_err=float(worst))
for K in (4,8,16,32):
    for m in (1,2,3):
        print(run(K,K,m),flush=True)
