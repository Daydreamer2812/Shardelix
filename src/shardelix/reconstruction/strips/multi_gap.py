"""Missing-strip recovery with calibrated endpoint continuity."""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter1d
from shardelix.reconstruction.common import Pose

def solve_multigap_strips(pieces,shape,grid,rotations):
    """Conservative missing strips with several contiguous holes.

    Long-axis endpoint signatures are compared with both sides' orientation.
    A calibrated lag curve estimates separation, then reciprocal unique
    joins produce a single acyclic path. Exact public missing budget required;
    endpoint missing/flat/weakly textured cases explicitly abstain.
    """
    n=len(pieces);missing=grid[0]*grid[1]-n;horizontal=grid[1]==1
    h,w=shape[0]//grid[0],shape[1]//grid[1];width=h if horizontal else w
    fail=lambda reason,**kw:({},{"multigap_status":reason,**kw})
    if not(6<=n<=768 and 1<=missing<=48 and 2<=width<=32):return fail("not-applicable")
    ims=[];turn=[]
    for p in pieces:
        k=0 if p.rgba.shape[:2]==(h,w) else 1
        if k and not rotations:return fail("orientation")
        a=p.rotated(k)[:,:,:3]
        if a.shape[:2]!=(h,w):return fail("shape")
        if horizontal:a=a.transpose(1,0,2)
        a=gaussian_filter1d(a.astype(np.float32)/255,.65,axis=0)
        if len(a)>768:a=a[::max(1,len(a)//768)]
        ims.append(a);turn.append(k)
    # Estimate the visible within-strip correlation decay at every lag.
    def corr(a,b):
        a=a-a.mean(axis=0);b=b-b.mean(axis=0)
        return float(np.sum(a*b)/max(1e-8,np.linalg.norm(a)*np.linalg.norm(b)))
    lag=[]
    for d in range(1,min(width,9)):
        vals=[corr(a[:,j],a[:,j+d]) for a in ims for j in range(0,width-d,max(1,width//5))]
        lag.append(float(np.median(vals)))
    if not lag or lag[0]<.72:return fail("low-continuity")
    # Estimate Gaussian covariance length, bounded to avoid unidentifiable gaps.
    fit=[-np.log(max(.005,min(.999,l)))/(d+1)**2 for d,l in enumerate(lag) if .10<l<.998]
    decay=float(np.median(fit)) if fit else .10
    if not .0005<decay<.5:return fail("unreliable-lag")
    ends=np.stack([v for a in ims for v in (a[:,0],a[:,-1])])
    means=ends.mean(axis=1);X=(ends-means[:,None,:]).reshape(2*n,-1)
    norm=np.linalg.norm(X,axis=1);X/=np.maximum(norm[:,None],1e-8)
    Y=X.reshape(2*n,-1,3)[:,::-1].reshape(2*n,-1)
    ix=np.arange(2*n);same=(ix[:,None]%2)==(ix[None,:]%2)
    sim=np.where(same,X@Y.T,X@X.T)-.35*np.abs(means[:,None]-means[None,:]).mean(axis=2)
    sim[ix[:,None]//2==ix[None,:]//2]=-np.inf
    sim[norm<.3,:]=-np.inf;sim[:,norm<.3]=-np.inf
    if not rotations:sim[same]=-np.inf
    direct=max(.60,lag[0]-.15)
    pred=[float(np.exp(-decay*(1+g*width)**2)) for g in range(1,min(missing,4)+1)]
    # No trustworthy gap-length information past the observable covariance.
    if max(pred)<.17:return fail("gap-outside-correlation")
    order=np.argsort(-sim,axis=1,kind="stable");edges={};parent=list(range(n))
    def root(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    def add(a,b,g):
        x,y=root(a//2),root(b//2)
        if x==y:return False
        parent[x]=y;edges[a]=(b,g);edges[b]=(a,g);return True
    for a in range(2*n):
        b=int(order[a,0]);s=float(sim[a,b])
        if a<b and order[b,0]==a and s>=direct and min(s-sim[a,order[a,1]],s-sim[b,order[b,1]])>=.035:add(a,b,0)
    direct_links=len(edges)//2;budget=0
    # Search only remaining endpoints; possible gap lengths by calibrated lag.
    for _ in range(n):
        free=[i for i in range(2*n) if i not in edges]
        if len(free)<=2:break
        sub=sim[np.ix_(free,free)].copy()
        roots=np.array([root(i//2) for i in free]);sub[roots[:,None]==roots[None,:]]=-np.inf
        rank=np.argsort(-sub,axis=1,kind="stable");choices=[]
        for a,i in enumerate(free):
            b=int(rank[a,0]);j=free[b];s=float(sub[a,b])
            if a>=b or rank[b,0]!=a or not .17<=s<direct:continue
            margin=min(s-sub[a,rank[a,1]],s-sub[b,rank[b,1]])
            if margin<.075:continue
            distances=sorted((abs(s-c),g+1) for g,c in enumerate(pred) if c>=.10)
            if not distances or distances[0][0]>.16:continue
            if len(distances)>1 and distances[1][0]-distances[0][0]<.075:continue
            choices.append((s,i,j,distances[0][1]))
        if not choices:break
        changed=0
        for s,i,j,g in sorted(choices,reverse=True):
            if i in edges or j in edges or budget+g>missing:continue
            if add(i,j,g):budget+=g;changed+=1
        if not changed:break
    if len(edges)!=2*(n-1) or budget!=missing:return fail("ambiguous-gap-budget",multigap_direct_links=direct_links,multigap_inferred_missing=budget)
    free=[i for i in range(2*n) if i not in edges]
    at=min(free) if rotations else next((i for i in free if i%2==0),-1)
    if at<0:return fail("orientation")
    out={};cell=0
    while pieces[at//2].id not in out:
        i,rev=divmod(at,2);k=(turn[i]+rev*2)%4
        out[pieces[i].id]=Pose(0,cell*h,k) if horizontal else Pose(cell*w,0,k)
        cell+=1
        if at^1 not in edges:break
        at,g=edges[at^1];cell+=g
    if len(out)!=n or cell!=grid[0]*grid[1]:return fail("path")
    return out,{"multigap_status":"resolved-conditional-on-endpoints","multigap_direct_links":direct_links,
                "multigap_inferred_missing":budget,"multigap_decay":decay,"multigap_expected_similarity":pred}
