"""Blind continuous rectangle rectification for digital document strips.

Estimate the common long axis and each variable width from observable alpha
moments; use the existing image-only foreground seam solver afterwards.
No canvas size, reference, hidden angle or original ordering is needed.
"""
from __future__ import annotations
import numpy as np
from .common import Fragment
from .pose9 import _rectify

def normalize_document_strips(pieces,sampling=1):
    if sampling not in (1,2,3,4):raise ValueError("Sampling must be 1–4")
    if not 1<=len(pieces)<=1024:raise ValueError("Document strip cap is 1024")
    dims=[]
    for p in pieces:
        a=p.rgba[:,:,3].astype(np.float64)/255
        yy,xx=np.nonzero(a>.005);v=a[yy,xx]
        mass=v.sum()
        if mass<12*sampling**2:raise ValueError("Insufficient alpha support for a digital strip")
        x=xx-np.average(xx,weights=v);y=yy-np.average(yy,weights=v)
        cov=np.array([[np.sum(v*x*x),np.sum(v*x*y)],[np.sum(v*x*y),np.sum(v*y*y)]])/mass
        eig=np.linalg.eigvalsh(cov)
        # Discrete uniform interval variance = (L²-1)/12.
        short,long=np.sqrt(np.maximum(1,12*eig+1))/sampling
        if long<1.5*short:raise ValueError("Unknown-width continuous mode expects elongated rectangular strips")
        dims.append((float(long),float(mass/sampling**2),float(short)))
    H=int(round(np.median([d[0] for d in dims])))
    if H<12 or any(abs(d[0]-H)>max(2,.012*H) for d in dims):
        raise ValueError("Continuous document strips must share a common long-axis length")
    widths=[];out=[];angles={};affines={}
    for p,(length,mass,short) in zip(pieces,dims):
        W=int(round(mass/H))
        if not 1<=W<=H or abs(W-short)>max(2,.10*W):
            raise ValueError("Strip alpha is not consistent with a digital rectangle")
        a,t,A=_rectify(p.rgba,(H,W),sampling,True)
        # Long rectangular mode is only justified if thresholded inverse area
        # matches the observed alpha. Geometric priors do not invent text.
        A[:,2]-=A[:,:2]@np.asarray(p.input_origin)
        out.append(Fragment(p.id,a));angles[p.id]=t;affines[p.id]=A.tolist();widths.append(W)
    return out,angles,affines,{"estimated_strip_height":H,"estimated_strip_widths":widths,
                               "document_pose_status":"estimated-digital-rectangles"}
