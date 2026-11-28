"""Bounded exact-candidate recovery for low-consensus small grids.

Compute MGC in row blocks: exact all-edge search without retaining a dense
matrix. Only the sparse top-five candidates survive.
"""
import time
import numpy as np
from scipy.spatial.distance import cdist
from shardelix.reconstruction.tiles.consensus import profiles
from shardelix.reconstruction.tiles.sparse import assemble

def exact_mgc_candidates(pieces,shape,grid,rotations,batch=128):
    desc=profiles(pieces,depth=2);m=len(desc)
    lens=np.array([len(e) for e,g in desc]);out=np.full((m,5),-1,np.int32);costs=np.full((m,5),np.inf,np.float32)
    for length in np.unique(lens):
        ii=np.flatnonzero(lens==length);e=np.stack([desc[i][0] for i in ii]);g=np.stack([desc[i][1] for i in ii])
        f=e.reshape(len(ii),-1);rev=e[:,::-1].copy().reshape(len(ii),-1)
        scale=np.sqrt(np.mean(g*g,axis=1)+.00012)
        den=np.maximum(.01,np.sqrt(np.mean(g*g,axis=(1,2))))
        for lo in range(0,len(ii),batch):
            hi=min(len(ii),lo+batch);rows=ii[lo:hi];size=hi-lo
            base=cdist(f[lo:hi],rev,"cityblock")/f.shape[1]
            base/=np.sqrt(den[lo:hi,None]*den[None,:])
            score=.35*base;del base
            for ch in range(3):
                q=cdist(e[lo:hi,:,ch]+g[lo:hi,:,ch],e[:,::-1,ch],"cityblock")/length
                rr=cdist(e[:,:,ch]+g[:,:,ch],e[lo:hi,::-1,ch],"cityblock").T/length
                score+=(.65/6)*(q/scale[lo:hi,ch,None]+rr/scale[None,:,ch])
            score=score.astype(np.float32)
            score[rows[:,None]//4==ii[None,:]//4]=np.inf
            if not rotations:score[(rows[:,None]%4+2)%4!=ii[None,:]%4]=np.inf
            k=min(5,len(ii))
            # Deterministic order for exact ties: stable sort by index.
            idx=np.argsort(score,axis=1,kind="stable")[:,:k];v=np.take_along_axis(score,idx,axis=1)
            out[rows,:k]=np.where(np.isfinite(v),ii[idx],-1);costs[rows,:k]=v
    return out,costs

def solve_dense_fallback(pieces,shape,grid,rotations,report_components=True):
    if len(pieces)>1024:raise ValueError("Exact recovery is bounded to 1024 observed pieces")
    t=time.perf_counter();idx,cost=exact_mgc_candidates(pieces,shape,grid,rotations)
    retrieval=time.perf_counter()-t
    p,d=assemble(pieces,shape,grid,rotations,idx,cost,report_components)
    d.update(algorithm="blocked-mgc-sparse-graph-v11",index_mode="dense-mgc",retrieval_seconds=retrieval,assembly_seconds=time.perf_counter()-t-retrieval)
    return p,d
