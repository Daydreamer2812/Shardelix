"""Conservative missing-strip bridging using observable long-axis correlation."""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter1d
from shardelix.reconstruction.common import Pose

def solve_thin_missing(pieces,shape,grid,rotations):
    """Independent endpoint graph that separates real seams from missing gaps.

    Uses within-piece lag-one similarity as a classical calibration. Returns
    a complete proposal only when all actual seams and single-cell gaps form
    one unambiguous path with an exhausted missing budget.
    """
    n=len(pieces);missing=grid[0]*grid[1]-n;horizontal=grid[1]==1
    h,w=shape[0]//grid[0],shape[1]//grid[1];width=h if horizontal else w
    diag={"thin_gap_status":"not-applicable","thin_gap_added":0}
    if not(4<=n<=1024 and 1<=missing<=64 and 1<=width<=4):return {},diag
    images=[];turns=[]
    for p in pieces:
        k=0 if p.rgba.shape[:2]==(h,w) else 1
        if k and not rotations:return {},diag
        a=p.rotated(k)[:,:,:3]
        if a.shape[:2]!=(h,w):return {},diag
        if horizontal:a=a.transpose(1,0,2)
        a=gaussian_filter1d(a.astype(np.float32)/255,.7,axis=0)
        if len(a)>768:a=a[::max(1,len(a)//768)]
        images.append(a);turns.append(k)
    ends=np.stack([x for a in images for x in (a[:,0],a[:,-1])])
    means=ends.mean(axis=1);X=ends-means[:,None,:]
    X=X.reshape(len(X),-1);norm=np.linalg.norm(X,axis=1);X/=np.maximum(norm[:,None],1e-6)
    Y=X.reshape(len(X),-1,3)[:,::-1].reshape(len(X),-1)
    direct=X@X.T;reverse=X@Y.T
    parity=np.arange(2*n)%2;same=parity[:,None]==parity[None,:]
    score=np.where(same,reverse,direct)-.45*np.abs(means[:,None]-means[None,:]).mean(axis=2)
    score[(norm<.3),:]=-np.inf;score[:,norm<.3]=-np.inf
    score[np.arange(2*n)[:,None]//2==np.arange(2*n)[None,:]//2]=-np.inf
    if not rotations:score[same]=-np.inf
    if width>=2:
        samples=[]
        for a in images:
            p=a[:,:-1]-a[:,:-1].mean(axis=0);q=a[:,1:]-a[:,1:].mean(axis=0)
            den=float(np.linalg.norm(p)*np.linalg.norm(q))
            if den>1e-6:samples.append(float(np.sum(p*q)/den))
        lag=float(np.median(samples)) if samples else .85
    else:lag=.85
    if lag<.65:return {},{**diag,"thin_gap_status":"low-continuity"}
    direct_min=max(.55,lag-.16)
    order=np.argsort(-score,axis=1,kind="stable");best=order[:,0];runner=order[:,1]
    edges={}
    for i,j in enumerate(best):
        v=score[i,j]
        if best[j]==i and v>=direct_min and min(v-score[i,runner[i]],v-score[j,runner[j]])>=.035:
            edges[i]=(int(j),0)
    # True adjacency chains have exactly n-(missing+1) links if every gap is
    # a single missing cell and neither endpoint was removed.
    links=len(edges)//2;runs=n-links
    diag.update(thin_gap_direct_links=links,thin_gap_runs=runs,thin_gap_lag1=lag,thin_gap_direct_threshold=direct_min)
    if runs!=missing+1:return {},{**diag,"thin_gap_status":"gap-budget-ambiguous"}
    free=np.array([i for i in range(2*n) if i not in edges])
    if len(free)!=2*runs:return {},diag
    local=score[np.ix_(free,free)].copy()
    # Prevent bridging endpoints in the same already-connected chain.
    parents=list(range(n))
    def root(i):
        while parents[i]!=i:i=parents[i]
        return i
    for i,(j,g) in edges.items():
        a,b=root(i//2),root(j//2)
        if a!=b:parents[a]=b
    roots=np.array([root(int(i)//2) for i in free]);local[roots[:,None]==roots[None,:]]=-np.inf
    order=np.argsort(-local,axis=1,kind="stable")
    bridges=0;confidence=[]
    for i in range(len(free)):
        j=int(order[i,0]);v=local[i,j]
        if int(order[j,0])!=i or v<.20 or v>=direct_min:continue
        if min(v-local[i,order[i,1]],v-local[j,order[j,1]])<.08:continue
        edges[int(free[i])]=(int(free[j]),1)
        if i<j:bridges+=1;confidence.append(float(v))
    if bridges!=missing:return {},{**diag,"thin_gap_status":"endpoint-ambiguity","thin_gap_bridges":bridges}
    endpoint=[i for i in range(2*n) if i not in edges]
    if len(endpoint)!=2:return {},diag
    at=min(endpoint) if rotations else next((i for i in endpoint if i%2==0),-1)
    if at<0:return {},diag
    seen=set();out={};cell=0
    while at//2 not in seen:
        i,rev=divmod(at,2);seen.add(i);k=(turns[i]+2*rev)%4
        out[pieces[i].id]=Pose(0,cell*h,k) if horizontal else Pose(cell*w,0,k)
        cell+=1
        if at^1 not in edges:break
        at,g=edges[at^1];cell+=g
    if len(out)!=n or cell!=grid[0]*grid[1]:return {},diag
    return out,{**diag,"thin_gap_status":"resolved-single-gap-budget","thin_gap_bridges":bridges,
                "thin_gap_min_similarity":min(confidence)}
