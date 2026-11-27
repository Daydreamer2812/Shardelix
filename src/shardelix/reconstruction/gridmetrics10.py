"""Memory-bounded rectangular diagnostics and evaluation helpers.

Only geometry and the supplied placements are used for the blind objective.
Evaluation functions are called separately with truth after solving.
"""
from __future__ import annotations
import numpy as np
from .common import Pose

def grid_state(pieces,poses,shape,grid):
    r,c=grid;ph,pw=shape[0]//r,shape[1]//c
    lookup={p.id:i for i,p in enumerate(pieces)}
    ids=np.full((r,c),-1,np.int32);kmap=np.zeros((r,c),np.uint8);collisions=0;invalid=0
    for key,p in poses.items():
        idx=lookup.get(key)
        if idx is None:invalid+=1;continue
        if p.x<0 or p.y<0 or p.x%pw or p.y%ph or p.y//ph>=r or p.x//pw>=c or (pieces[idx].rgba.shape[1::-1] if p.k%2 else pieces[idx].rgba.shape[:2])!=(ph,pw):
            invalid+=1;continue
        y,x=p.y//ph,p.x//pw
        if ids[y,x]>=0:collisions+=1
        ids[y,x]=idx;kmap[y,x]=p.k%4
    return ids,kmap,collisions,invalid

def grid_objective(pieces,poses,shape,grid):
    ids,ks,coll,invalid=grid_state(pieces,poses,shape,grid)
    filled=ids>=0;total=0;count=0
    r,c=grid;ph,pw=shape[0]//r,shape[1]//c
    # One oriented view per cell, in bounded row blocks. The old version
    # repeatedly rotated/sliced both cells for every seam.
    height=max(1,4096//c)
    for lo in range(0,r,height):
        hi=min(r,lo+height);end=min(r,hi+1);nr=end-lo
        t=np.zeros((nr,c,pw,3),np.uint8);b=np.zeros_like(t)
        l=np.zeros((nr,c,ph,3),np.uint8);q=np.zeros_like(l)
        yy,xx=np.where(filled[lo:end])
        for y,x in zip(yy,xx):
            z=pieces[int(ids[y+lo,x])].rotated(int(ks[y+lo,x]))[:,:,:3]
            t[y,x]=z[0];b[y,x]=z[-1];l[y,x]=z[:,0];q[y,x]=z[:,-1]
        v=filled[lo:end];h0=hi-lo
        mask=v[:h0,:-1]&v[:h0,1:]
        aa=q[:h0,:-1][mask].astype(np.int16);bb=l[:h0,1:][mask].astype(np.int16)
        total+=int(np.abs(aa-bb).sum());count+=aa.size
        mask=v[:-1]&v[1:]
        aa=b[:-1][mask].astype(np.int16);bb=t[1:][mask].astype(np.int16)
        total+=int(np.abs(aa-bb).sum());count+=aa.size
    return float(filled.mean()-2*(coll+invalid)/ids.size-.1*total/(255*max(1,count)))

def evaluate_grid(pieces,poses,truth,shape,grid,rotations):
    """Strict quarter-turn grid metric computed on cells, no full image needed.

    Equivalent to pixel-label DPA/F1/coverage for opaque, uniform rectangles.
    Does not report pixel accuracy (requires rendering actual values).
    """
    r,c=grid
    if shape[0]%r or shape[1]%c or len(pieces)!=r*c or len(truth)!=len(pieces) or any(not p.mask.all() for p in pieces):
        raise ValueError("Expected complete opaque uniform grid")
    ids,k,coll,invalid=grid_state(pieces,poses,shape,grid)
    gt,tk,tc,ti=grid_state(pieces,truth,shape,grid)
    if tc or ti or (gt<0).any():raise ValueError("Expected complete uniform grid truth")
    n=len(truth);r,c=grid
    best=(-1,ids,k)
    for turn in ((0,1,2,3) if r==c and shape[0]==shape[1] else (0,2)) if rotations else (0,):
        a=np.rot90(ids,-turn);b=(np.rot90(k,-turn)+turn)%4
        score=int(((a==gt)&(b==tk)&(a>=0)).sum())
        if score>best[0]:best=(score,a,b)
    count,a,b=best
    def edges(z):
        keys=[]
        for x,y in ((z[:,:-1],z[:,1:]),(z[:-1,:],z[1:,:])):
            ok=(x>=0)&(y>=0)
            xx=x[ok].astype(np.int64);yy=y[ok].astype(np.int64)
            keys.append(np.minimum(xx,yy)*n+np.maximum(xx,yy))
        return np.unique(np.concatenate(keys))
    ap,at=edges(a),edges(gt);tp=len(np.intersect1d(ap,at,assume_unique=True))
    f1=2*tp/max(1,len(ap)+len(at)) if len(ap)+len(at) else 1.
    cov=float((ids>=0).mean());dpa=count/n
    return {"piece_count":n,"placed":int((ids>=0).sum()),"direct_position_accuracy":dpa,
            "adjacency_f1":f1,"adjacency_precision":tp/max(1,len(ap)),"adjacency_recall":tp/max(1,len(at)),
            "coverage":cov,"collision_pixels":coll*(shape[0]//r)*(shape[1]//c),"invalid_poses":invalid,
            "pass_95":bool(dpa>.95 and f1>.95 and cov>.95 and not coll and not invalid),
            "metric_engine":"exact-uniform-grid-cell","raw_position_accuracy":float(((ids==gt)&(k==tk)&(ids>=0)).sum()/n)}
