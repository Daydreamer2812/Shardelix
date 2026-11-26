"""Classical polygon-side retrieval for straight/slanted polygon regions."""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from .geometry import contours,R
from .content import INWARD

def _rdp(v,tol=1.1):
    """Return retained vertex indices for an open polyline."""
    keep={0,len(v)-1};todo=[(0,len(v)-1)]
    while todo:
        i,j=todo.pop()
        if j<=i+1:continue
        d=v[j]-v[i];norm=np.linalg.norm(d)
        if norm<1e-9:dist=np.linalg.norm(v[i+1:j]-v[i],axis=1)
        else:
            q=v[i+1:j]-v[i];dist=np.abs(q[:,0]*d[1]-q[:,1]*d[0])/norm
        k=int(np.argmax(dist))+i+1
        if dist[k-i-1]>tol:
            keep.add(k);todo.extend([(i,k),(k,j)])
    return sorted(keep)

def _segments(pieces,rotations,tol):
    segments=[]
    for i,p in enumerate(pieces):
        for v,d in contours(p.mask):
            if len(v)<8:continue
            # Split closed contour at two separated vertices.
            far=int(np.argmax(np.sum((v-v[0])**2,axis=1)))
            if not 0<far<len(v):continue
            vv=np.vstack([v,v[:1]])
            kee=_rdp(vv[:far+1],tol)+[far+j for j in _rdp(vv[far:],tol)[1:]]
            pixels=v+INWARD[d];rgb=p.rgba[pixels[:,1],pixels[:,0],:3].astype(np.float32)/255
            rgb=np.vstack([rgb,rgb[:1]])
            for a,b in zip(kee[:-1],kee[1:]):
                vector=vv[b]-vv[a]
                l=float(np.linalg.norm(vector))
                if l<6:continue
                theta=np.arctan2(vector[1],vector[0])%(2*np.pi)
                q=int(np.floor((theta+1e-10)/(np.pi/2)))%4 if rotations else 0
                # Canonical quarter-turn to a common positive quadrant.
                vc=R[-q%4]@vector
                ts=np.linspace(a,b-1,7)
                color=np.stack([np.interp(ts,np.arange(len(rgb)),rgb[:,c]) for c in range(3)],axis=1)
                segments.append((i,vv[a],vv[b],q,vc,color,l))
    return segments

def segment_candidates(pieces,rotations=False,tol=1.0,neighbors=12):
    seg=_segments(pieces,rotations,tol)
    if not seg:return []
    # Coordinates measured in pixels, plus independent RGB appearance.
    # Color is softly weighted so geometry remains useful on uniform textures.
    feat=np.stack([np.r_[s[4]/1.8,s[5].ravel()*5.] for s in seg])
    query=np.stack([np.r_[(s[4] if rotations else -s[4])/1.8,s[5][::-1].ravel()*5.] for s in seg])
    dist,idx=cKDTree(feat).query(query,k=min(neighbors,len(seg)))
    if idx.ndim==1:idx=idx[:,None];dist=dist[:,None]
    seen={}
    for a,(ds,js) in enumerate(zip(dist,idx)):
        i,p,q,ka,va,ca,la=seg[a]
        for D,b in zip(ds,js):
            j,u,v,kb,vb,cb,lb=seg[int(b)]
            if i>=j or abs(la-lb)>5 or D>7:continue
            r=(ka+2-kb)%4 if rotations else 0
            if np.linalg.norm((R[r]@(v-u))+(q-p))>4.:continue
            # Endpoint alignment plus small rasterization-independent integer
            # offsets; final hypotheses must pass whole-mask collision checks.
            shifts={tuple(p-R[r]@v),tuple(q-R[r]@u)}
            center=np.round((p+q-R[r]@(u+v))/2).astype(int)
            for dx in (-1,0,1):
                for dy in (-1,0,1):shifts.add(tuple(center+[dx,dy]))
            for x,y in shifts:
                key=(i,j,r,int(x),int(y))
                seen[key]=max(seen.get(key,0),int(min(la,lb)))
    return [(v,*k) for k,v in seen.items()]
