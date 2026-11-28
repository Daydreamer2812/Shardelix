"""Width-constrained oriented block exchanges for long image/document strips.

A long chain can have almost all correct neighbors but a poor absolute
placement. Refine permutations of entire chains, not isolated fragments.
All candidate moves preserve exact public slot thicknesses.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist
from shardelix.reconstruction.common import Pose
from shardelix.reconstruction.rotation import _canonical_tiles
from shardelix.reconstruction.tiles.straight import _grid_bounds


def _lcp(a,b):
    """Exact longest common prefix table, including one-past-end sentinel."""
    n=len(a);m=len(b)
    out=np.zeros((n+1,m+1),np.int16)
    for i in range(n-1,-1,-1):
        out[i,:m]=np.where(a[i]==b,1+out[i+1,1:],0)
    return out


def _features(canonical,axis,samples):
    first=[];last=[];second=[];penultimate=[];width=[]
    for p in canonical:
        a=p.rgba[:,:,:3].astype(np.float32)/255
        for q in (a,a[::-1,::-1]):
            if axis==0:q=q.transpose(1,0,2)
            ix=np.linspace(0,q.shape[1]-1,min(q.shape[1],samples)).round().astype(int)
            first.append(q[0,ix].ravel());last.append(q[-1,ix].ravel())
            second.append(q[min(1,len(q)-1),ix].ravel());penultimate.append(q[max(0,len(q)-2),ix].ravel())
            width.append(len(q))
    return [np.asarray(x) for x in (first,last,second,penultimate,width)]


def _costs(features,curvature=.9):
    F,L,S,P,W=features;n=len(F);dim=F.shape[1]
    # Edge cost plus the second derivative at both sides if an interior
    # neighbor exists. For one-pixel strips the triple term below is needed.
    pair=cdist(L,F,'cityblock')/dim
    da=cdist(2*L-P,F,'cityblock')/dim
    db=cdist(L,2*F-S,'cityblock')/dim
    pair+=curvature*(da*(W[:,None]>1)+db*(W[None,:]>1))
    out=np.zeros((n+1,n+1),np.float32);out[:n,:n]=pair
    return out


def _energy(order,pair,features,curvature):
    F,L,S,P,W=features
    val=float(pair[order[:-1],order[1:]].sum())
    ix=np.flatnonzero(W[order[1:-1]]==1)+1
    if len(ix):
        a,b,c=order[ix-1],order[ix],order[ix+1]
        val+=curvature*float(np.abs(L[a]-2*F[b]+F[c]).mean(axis=1).sum())
    return val


def _adjacent_sweep(order,pair,features,curvature,passes=4):
    """All adjacent equal-width swaps, tested on local triple energy."""
    F,L,S,P,W=features
    out=order.copy();moves=0;n=len(out)
    for _ in range(passes):
        changed=0
        for i in range(n-1):
            if W[out[i]]!=W[out[i+1]]:continue
            a=max(0,i-2);b=min(n,i+4)
            q=out[a:b].copy();before=_energy(q,pair,features,curvature)
            q[i-a],q[i-a+1]=q[i-a+1],q[i-a]
            after=_energy(q,pair,features,curvature)
            if after<before-1e-8:
                out[a:b]=q;changed+=1
        moves+=changed
        if not changed:break
    return out,moves


def _move(order,i,j,k,rev):
    block=(order[i:j][::-1]^(rev==1)) if rev else order[i:j]
    if k<i:return np.concatenate((order[:k],block,order[k:i],order[j:]))
    return np.concatenate((order[:i],order[j:k],block,order[k:]))


def refine_strip_blocks(pieces,poses,shape,grid,rotations,rounds=12,samples=192,curvature=.9,boundaries=70):
    if len(poses)!=len(pieces) or len(pieces)<4:return poses,{"block_moves":0}
    n=len(pieces);axis=0 if grid[0]==1 else 1
    canonical,turns,ambiguous=_canonical_tiles(pieces,shape,grid)
    if ambiguous:return poses,{"block_moves":0}
    ys,xs=_grid_bounds(*shape,*grid);slots=np.diff(xs) if axis==0 else np.diff(ys)
    ids={p.id:i for i,p in enumerate(pieces)}
    key=(lambda id:poses[id].x) if axis==0 else (lambda id:poses[id].y)
    order=np.array([2*ids[id]+int(((poses[id].k-turns[ids[id]])%4)//2) for id in sorted(poses,key=key)])
    features=_features(canonical,axis,samples)
    if not np.array_equal(features[-1][order],slots):return poses,{"block_moves":0}
    pair=_costs(features,curvature)
    lcp=_lcp(slots,slots);back=_lcp(slots,slots[::-1])
    initial_energy=_energy(order,pair,features,curvature)
    moves=0;swaps=0
    if not rotations:
        # The order still permits global block relocations, no orientation flip.
        can_reverse=False
    else:can_reverse=True
    for _ in range(rounds):
        order,changes=_adjacent_sweep(order,pair,features,curvature,passes=2);swaps+=changes
        score=_energy(order,pair,features,curvature)
        e=pair[order[:-1],order[1:]].copy()
        from scipy.ndimage import median_filter
        relative=e/(.0008+median_filter(e,size=9,mode='nearest'))
        # High absolute and unusually bad relative seams are both candidates.
        kk=min(boundaries//2,n-1)
        b=np.unique(np.r_[0,n,np.argsort(e)[-kk:]+1,np.argsort(relative)[-kk:]+1])
        # Include nearby boundaries and thickness-pattern periods at high
        # discontinuities, so true blocks are not excluded by weak seams.
        period=next((k for k in range(1,n+1) if np.array_equal(slots[k:],slots[:-k])),n)
        if period<=16:
            b=np.unique(np.r_[b,np.clip(b-period,0,n),np.clip(b+period,0,n)])
        # Limit enumeration independently of n; 100 boundaries => 1M triples.
        if len(b)>110:
            base=np.unique(np.r_[0,n,np.argsort(e)[-28:]+1,np.argsort(relative)[-28:]+1])
            extra=np.setdiff1d(b,base);b=np.unique(np.r_[base,extra[np.argsort(np.r_[e,0][np.maximum(extra-1,0)])[-(110-len(base)):]]])
        i,j,k=np.meshgrid(b,b,b,indexing='ij')
        ok=(i<j)&((k<i)|(k>j))
        i,j,k=i[ok],j[ok],k[ok]
        candidates=[]
        ext=np.r_[2*n,order,2*n]
        reverse_prefix=np.r_[0.,np.cumsum(pair[order[1:],order[:-1]]-e,dtype=np.float64)]
        for rev in (0,1,2) if can_reverse else (0,2):
            length=j-i;target=np.where(k<i,k,k-length)
            valid=(back[target,n-j]>=length if rev else lcp[i,target]>=length)
            left=k<i
            # Avoid out-of-range indexing for the inactive branch.
            valid &= np.where(left,lcp[k,np.minimum(k+length,n)]>=i-k,lcp[j,i]>=k-j)
            ii,jj,kkk=i[valid],j[valid],k[valid]
            if not len(ii):continue
            a,f,l,c,u,v=ext[ii],ext[ii+1],ext[jj],ext[jj+1],ext[kkk],ext[kkk+1]
            nf,nl=(l^(rev==1),f^(rev==1)) if rev else (f,l)
            delta=pair[a,c]+pair[u,nf]+pair[nl,v]-pair[a,f]-pair[l,c]-pair[u,v]
            if rev==2:delta+=reverse_prefix[jj-1]-reverse_prefix[ii]
            limit=min(32,len(delta));ix=np.argpartition(delta,limit-1)[:limit]
            candidates.extend((float(delta[q]),int(ii[q]),int(jj[q]),int(kkk[q]),rev) for q in ix)
        for rev in (1,2) if can_reverse else (2,):
            ii,jj=np.meshgrid(b,b,indexing='ij');valid=(jj>ii)&(back[ii,n-jj]>=jj-ii)
            ii,jj=ii[valid],jj[valid]
            a,f,l,c=ext[ii],ext[ii+1],ext[jj],ext[jj+1]
            delta=pair[a,l^(rev==1)]+pair[f^(rev==1),c]-pair[a,f]-pair[l,c]
            if rev==2:delta+=reverse_prefix[jj-1]-reverse_prefix[ii]
            limit=min(16,len(delta))
            if limit:
                ix=np.argpartition(delta,limit-1)[:limit]
                candidates.extend((float(delta[q]),int(ii[q]),int(jj[q]),-1,rev) for q in ix)
        best=order;best_score=score
        for _,i0,j0,k0,rev in sorted(candidates)[:64]:
            if k0==-1:
                q=order.copy();q[i0:j0]=q[i0:j0][::-1]^(rev==1)
            else:q=_move(order,i0,j0,k0,rev)
            if not np.array_equal(features[-1][q],slots):continue
            s=_energy(q,pair,features,curvature)
            if s<best_score-1e-7:best_score=s;best=q
        if best is order:
            if not changes:break
        else:order=best;moves+=1
    final_energy=_energy(order,pair,features,curvature)
    # Guaranteed feasibility and no increase in the fixed smoothness energy.
    if final_energy>initial_energy+1e-7:return poses,{"block_moves":0}
    out={pieces[s//2].id:Pose(int(xs[x]),int(ys[y]),int((turns[s//2]+2*(s%2))%4)) for (y,x),s in zip(np.ndindex(grid),order)}
    return out,{"block_moves":moves,"adjacent_swaps":swaps,"strip_energy_before":initial_energy,"strip_energy_after":final_energy}
