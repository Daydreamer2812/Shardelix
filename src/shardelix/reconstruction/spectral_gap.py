"""Coarse configuration-space search of abraded pieces using FFT correlations.

No reference image, network or learned features. Sum-of-squared RGB error,
band support and foreground collision are computed for all translations on a
coarse grid, then the caller verifies the best SE(2) hypotheses at full scale.
"""
from __future__ import annotations
from collections import defaultdict
import math
import numpy as np
from scipy import fft, ndimage as ndi

from .scan import rotation
from .continuation import _sample


def _raster(shape,field,xy,R,center,band):
    # xy are scene offsets; sample a rotated moving fragment about its center.
    q=xy@R+center
    qf=q+field.pad
    valid=(qf[...,0]>=.5)&(qf[...,1]>=.5)&(qf[...,0]<field.rgb.shape[1]-.5)&(qf[...,1]<field.rgb.shape[0]-.5)
    inside=_sample(shape.fragment.mask.astype(np.float32),q,mode="constant")>.5
    dist=_sample(field.distance,qf,mode="constant")
    rgb=_sample(field.rgb,qf,mode="constant")/255.
    strength=np.clip(_sample(field.strength,qf,mode="constant")/8.,0,1)
    # Use a mask band on observed pixels rather than the whole interior.
    # Preceding interior distance is computed separately for each shape.
    core=ndi.distance_transform_edt(shape.fragment.mask).astype(np.float32)
    dep=_sample(core,q,mode="constant")
    obs=inside*(dep<=9.5)*valid
    ext=(~inside)*valid*(dist>.1)*(dist<=band)*np.exp(-dist/max(4.,band*.75))
    obs=obs.astype(np.float32);ext=ext.astype(np.float32)
    def channels(w):
        return np.concatenate([w[None], (rgb*w[...,None]).transpose(2,0,1),
                               (w*np.sum(rgb*rgb,axis=-1))[None],
                               (w*strength)[None]],axis=0)
    return np.concatenate([inside[None].astype(np.float32),channels(obs),channels(ext)],axis=0)


def _spectrum(shape,field,xy,R,center,band):
    return fft.rfft2(_raster(shape,field,xy,R,center,band),axes=(-2,-1))


def _correlation(a,b,N):
    # a,b frequency channels: original mask, observed[6], extrapolated[6].
    # Return simultaneous bidirectional observed/extrapolated support and
    # squared color difference, texture evidence and physical overlap.
    aobs,aext=a[:,1:7],a[:,7:13]
    bobs,bext=b[1:7],b[7:13]
    def pair(A,B):
        return (A[:,0]*B[0].conj(),
                A[:,4]*B[0].conj()+A[:,0]*B[4].conj()-2*np.sum(A[:,1:4]*B[1:4].conj(),axis=1),
                A[:,5]*B[0].conj()+A[:,0]*B[5].conj())
    sa,ea,ga=pair(aext,bobs)
    sb,eb,gb=pair(aobs,bext)
    freq=np.stack([sa+sb,ea+eb,ga+gb,a[:,0]*b[0].conj()],axis=1)
    return fft.irfft2(freq,s=(N,N),axes=(-2,-1))


def fft_proposals(shapes,fields,pairs,max_gap,update=lambda _:None,*,limit=14):
    """All translations at 24 angles for a bounded set of piece pairs."""
    if not pairs:return {}
    centers=[s.points.mean(0) for s in shapes]
    extent=max(float(np.linalg.norm(s.points-c,axis=1).max()) for s,c in zip(shapes,centers))
    step=max(2.8,(4*(extent+max_gap+5))/96.)
    band=min(18.,max(8.,max_gap*.72))
    N=fft.next_fast_len(int(math.ceil(4*(extent+band+4)/step))+2,real=True)
    # Padding supports linear (rather than wrapped) correlation over all
    # meaningful translations. Coarse step grows for large scan fragments.
    if N>112:
        step*=N/112
        N=fft.next_fast_len(int(math.ceil(4*(extent+band+4)/step))+2,real=True)
    y,x=np.indices((N,N),dtype=np.float32)
    xy=np.stack([(x-N//2)*step,(y-N//2)*step],axis=-1)
    by_moving=defaultdict(list)
    for i,j in pairs:
        if i!=j:by_moving[max(i,j)].append(min(i,j))
    fixed=sorted({i for js in by_moving.values() for i in js})
    spectra={i:_spectrum(shapes[i],fields[i],xy,np.eye(2),centers[i],band) for i in fixed}
    out=defaultdict(list)
    for ordinal,(j,near) in enumerate(sorted(by_moving.items())):
        near=sorted(set(near))
        A=np.stack([spectra[i] for i in near])
        area=np.array([shapes[i].fragment.mask.sum() for i in near])
        denom=np.minimum(area,shapes[j].fragment.mask.sum())/step**2
        for k in range(24):
            theta=2*np.pi*k/24-np.pi
            R=rotation(theta);B=_spectrum(shapes[j],fields[j],xy,R,centers[j],band)
            values=_correlation(A,B,N)
            support=np.maximum(values[:,0],0)
            error=np.sqrt(np.maximum(values[:,1],0)/np.maximum(support*3,1.e-6))*255.
            texture=np.maximum(values[:,2],0)/np.maximum(support,1.e-6)
            overlap=np.maximum(values[:,3],0)/np.maximum(denom[:,None,None],1.)
            # Moderate support threshold and texture weighting; full-raster
            # verification is independent of this approximate squared error.
            rank=np.sqrt(support)*np.exp(-error/25.)*(.20+.80*np.minimum(texture,1.))
            rank[(support<max(4.,15/step))|(overlap>.035)]=0.
            # Best 2 distinct translation peaks for each rotation.
            flat=rank.reshape(len(near),-1)
            for _ in range(2):
                ids=np.argmax(flat,axis=1)
                scores=flat[np.arange(len(near)),ids]
                yy,xx=np.unravel_index(ids,(N,N))
                for z,i in enumerate(near):
                    if scores[z]<.30:continue
                    dx=int(xx[z]);dy=int(yy[z])
                    if dx>N//2:dx-=N
                    if dy>N//2:dy-=N
                    t=centers[i]+step*np.array([dx,dy])-R@centers[j]
                    out[i,j].append((float(scores[z]),theta,t))
                    # Suppress neighboring translations around the first peak.
                    flat[z,((yy[z]+np.arange(-2,3))%N)[:,None]*N+((xx[z]+np.arange(-2,3))%N)[None,:]]=0.
        # Keep peak diversity rather than 14 near-identical angles/translations.
        if ordinal%max(1,len(by_moving)//12)==0:
            update(f"Spectral configuration search {ordinal+1}/{len(by_moving)}")
    result={}
    for pair,items in out.items():
        ranked=[]
        for s,a,t in sorted(items,key=lambda q:-q[0]):
            if any(abs(math.atan2(math.sin(a-b),math.cos(a-b)))<.23 and np.linalg.norm(t-u)<6 for b,u,_,_ in ranked):continue
            ranked.append((a,t,2,float(1./max(s,1.e-6))))
            if len(ranked)>=limit:break
        if ranked:result[pair]=ranked
    return result
