"""Vectorized 2x2 loop voting for sparse grid graphs.

The scalar V10 routine is retained as a reference in sparse_grid. Bounded
batches avoid an N-by-N matrix and only count observed shortlist matches.
"""
import numpy as np

def _loop_votes_reference(idx,batch=65536):
    m,keep=idx.shape
    support=np.zeros((m,keep),np.int32)
    def cast(x):return x.astype(np.int64,copy=False)
    def add(a,b):
        # Each directed candidate receives the undirected pair's count, if
        # present in that direction. Reciprocal link construction reads once.
        for u,v in ((a,b),(b,a)):
            ri,ci=np.nonzero(idx[u]==v[:,None])
            if len(ri):np.add.at(support,(u[ri],ci),1)
    for lo in range(0,m,batch):
        a0=np.arange(lo,min(lo+batch,m),dtype=np.int64);i=a0//4;s0=a0%4;c=4*i+(s0+1)%4
        for rb in range(min(2,keep)):
            b0=idx[a0,rb]
            for re in range(min(2,keep)):
                e0=idx[c,re];valid=(b0>=0)&(e0>=0)&(b0//4!=e0//4)
                if not valid.any():continue
                a=a0[valid];b=cast(b0[valid]);e=cast(e0[valid]);s=s0[valid];cc=c[valid];ii=i[valid]
                j,t=b//4,b%4;l,u=e//4,e%4
                rj=(s+2-t)%4;js=4*j+(s+1-rj)%4
                rl=(s+3-u)%4;ls=4*l+(s-rl)%4
                for rf in range(min(2,keep)):
                    f=cast(idx[js,rf]);z,v=f//4,f%4
                    rz=(s+3-v)%4;g=4*z+(s+2-rz)%4
                    ok=(f>=0)&(z!=ii)&(z!=j)&(z!=l)&np.any(idx[ls,:2]==g[:,None],axis=1)
                    if not ok.any():continue
                    for x,y in ((a,b),(cc,e),(js,f),(ls,g)):add(x[ok],y[ok])
    return support

def loop_votes(idx,batch=65536):
    from ..native_core import loop_votes as accelerated
    result=accelerated(idx)
    return _loop_votes_reference(idx,batch) if result is None else result

def reciprocal_links(idx,costs,rows,margin):
    support=loop_votes(idx);m,keep=idx.shape
    aa=np.arange(m,dtype=np.int64);links=[]
    for rank in range(keep):
        bb=idx[:,rank];valid=bb>aa
        if not valid.any():continue
        a=aa[valid];b=bb[valid].astype(np.int64)
        check=idx[b]==a[:,None];rec=check.any(axis=1)
        a,b=a[rec],b[rec];r=check[rec].argmax(axis=1)
        lp=support[a,rank]
        conf=np.where((rank==0)&(r==0),np.minimum(margin[a],margin[b]),0.)
        # Same numeric ordering as scalar implementation; float64 here to
        # preserve ties/rounding in old Python strength calculation.
        conf=conf.astype(np.float64);strength=conf+.45*np.minimum(lp,4)-.12*(rank+r)
        good=(strength>0)&((rank+r==0)|(lp>0))
        links.extend(zip(strength[good].tolist(),a[good].tolist(),b[good].tolist(),conf[good].tolist(),lp[good].tolist()))
    links.sort(key=lambda t:(-t[0],t[1],t[2]))
    return links
