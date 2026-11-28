"""Complementary arc retrieval with classical RGB seam descriptors.

Repeated geometric paths (especially straight cuts) require image
evidence. Group by exact complementary shape, then retrieve nearest appearance
descriptors with a KD-tree; only pose-consistent votes survive aggregation.
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree

from shardelix.reconstruction.irregular.lattice import R, contours

INWARD=np.array([[0,0],[-1,0],[-1,-1],[0,-1]],dtype=int)


def content_candidates(pieces,k=32,rotations=False,neighbors=6,contrast_min=.023,degree_limit=18):
    index=defaultdict(list)
    ns=9
    samples=np.linspace(0,k-1,ns,dtype=int)
    for i,p in enumerate(pieces):
        for v,d in contours(p.mask):
            n=len(d)
            if n<=k:continue
            pp=v+INWARD[d]
            colors=p.rgba[pp[:,1],pp[:,0],:3].astype(np.float32)/255
            dirs=np.r_[d,d[:k-1]]
            colors=np.concatenate([colors,colors[:k-1]])
            # Every start position is indexed so opposing windows need not align
            # to a subsampling phase. Suppress near-constant colour windows:
            # these provide essentially no evidence to resolve repeated shape.
            from numpy.lib.stride_tricks import sliding_window_view
            parts=colors[np.arange(n)[:,None]+samples]
            keep=np.ptp(parts,axis=1).max(axis=1)>=contrast_min
            starts=np.flatnonzero(keep)
            if not len(starts):continue
            ds=sliding_window_view(dirs,k)[starts];rev=(ds[:,::-1]+2)%4
            a=((ds.astype(np.int16)-ds[:,:1])%4 if rotations else ds).astype(np.uint8)
            b=((rev.astype(np.int16)-rev[:,:1])%4 if rotations else rev).astype(np.uint8)
            for row,q in enumerate(starts):
                key=a[row].tobytes();rk=b[row].tobytes()
                index[key].append((i,v[q],v[(q+k)%n],int(ds[row,0]),int(rev[row,0]),rk,parts[q].ravel()))
    accum=defaultdict(lambda:[0,0.0])
    for key,items in index.items():
        rk=items[0][5]
        other=index.get(rk)
        if not other or key>rk:continue
        aa=np.stack([q[-1] for q in items])
        bb=np.stack([q[-1].reshape(ns,3)[::-1].ravel() for q in other])
        kk=min(neighbors,len(other))
        if len(items)*len(other)<=256:
            from scipy.spatial.distance import cdist
            allcost=cdist(aa,bb,"euclidean")
            idx=np.argsort(allcost,axis=1,kind="stable")[:,:kk]
            dist=np.take_along_axis(allcost,idx,axis=1)
        else:
            tree=cKDTree(bb);dist,idx=tree.query(aa,k=kk,workers=1)
            if kk==1:dist=dist[:,None];idx=idx[:,None]
        for row,(part,ds,js) in enumerate(zip(items,dist,idx)):
            i,a,_,da,_,_,_=part
            for rank,(d,jj) in enumerate(zip(ds,js)):
                j,_,b,_,db,_,_=other[int(jj)]
                if i==j:continue
                cost=float(d)/np.sqrt(ns*3)
                if cost>.18:continue
                r=(da-db)%4
                if r and not rotations:continue
                bx,by=map(int,b)
                rx,ry=((bx,by),(-by,bx),(-bx,-by),(by,-bx))[r]
                tx,ty=int(a[0])-rx,int(a[1])-ry
                key2=(i,j,r,tx,ty)
                if i>j:
                    rr=(-r)%4;xx,yy=((tx,ty),(-ty,tx),(-tx,-ty),(ty,-tx))[rr]
                    key2=(j,i,rr,-xx,-yy)
                v=accum[key2];v[0]+=1
                v[1]+=(1/(1+rank))*(.035/(cost+.015))
    ranked=[]
    for key,(n,score) in accum.items():
        if n>=2:
            ranked.append((float(score),n,key))
    ranked.sort(reverse=True)
    # Limit hypotheses per piece to avoid overwhelming geometric graph with
    # weak alternative offsets; good hypotheses are ranked by consensus.
    counts=defaultdict(int);out=[]
    for score,n,key in ranked:
        i,j,*_=key
        if counts[i]>=degree_limit or counts[j]>=degree_limit:continue
        counts[i]+=1;counts[j]+=1
        out.append((max(2,int(score)),*key))
    return out
