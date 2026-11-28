"""Deterministic coarse-to-fine edge retrieval (CPU only).

Mid-seam extrapolation is symmetric: at a true join each of the two predicted
half-pixel boundary colors should coincide. Low-dimensional indices propose
candidates; the calling solver verifies the full uncompressed edge profile.
This module does not use labels, reference images, or train a model.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree

def pooled(a, samples):
    cuts=np.linspace(0,a.shape[1],min(samples,a.shape[1])+1,dtype=int)
    return np.stack([a[:,lo:hi].mean(axis=1) for lo,hi in zip(cuts[:-1],cuts[1:])],axis=1).reshape(len(a),-1)

def query_edges(eu,gu,ev,gv,*,k,workers,mode="midpoint",samples=4):
    """Return a bounded shortlist, using compact int32 IDs.

    Mixed mode adds a separate color-only index. A separate, validated
    legacy index remains available to the caller for ambiguous solutions.
    """
    k=min(k,len(ev))
    extra=[]
    if mode not in {"midpoint","mixed"}:raise ValueError("Unknown index mode")
    # Gradient points outward (boundary minus next inward pixel).
    # Extrapolating half a pixel from either piece gives the same seam point.
    a=pooled(eu+.5*gu,samples)
    b=pooled((ev+.5*gv)[:,::-1],samples)
    tree=cKDTree(b,leafsize=32)
    _,ids=tree.query(a,k=k,eps=.55,workers=workers)
    extra.append(np.asarray(ids,dtype=np.int32).reshape(len(eu),k))
    if mode=="mixed":
        # Independent retrieval protects against inaccurate edge derivatives.
        a=pooled(eu,samples)
        b=pooled(ev[:,::-1],samples)
        tree=cKDTree(b,leafsize=32)
        _,ids=tree.query(a,k=k,eps=.55,workers=workers)
        extra.append(np.asarray(ids,dtype=np.int32).reshape(len(eu),k))
    return np.concatenate(extra,axis=1) if len(extra)>1 else extra[0]
