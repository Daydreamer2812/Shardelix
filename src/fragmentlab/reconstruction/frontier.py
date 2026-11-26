"""Anchor-preserving completion by exact empty-boundary arc retrieval.

Only visible occupied pixels, empty raster and remaining fragment masks.
Unlike whole-component packing, recompute frontier evidence after each batch.
"""
from collections import defaultdict, Counter
import numpy as np
from .common import Pose, render
from .geometry import R, contours, _vertex_to_pose, _keys
from .verified import exact_holes


def _arc_index(mask, k, rotations):
    out=defaultdict(list)
    for v,d in contours(mask):
        for key,_,a,_,da,_ in _keys(v,d,k,1,rotations):
            out[key].append((a,da))
    return out


def _border_pixels(a):
    mask=a[:,:,3]>0
    pad=np.pad(mask,1)
    result=[]
    for dx,dy,neighbor in ((0,-1,pad[:-2,1:-1]),(1,0,pad[1:-1,2:]),(0,1,pad[2:,1:-1]),(-1,0,pad[1:-1,:-2])):
        yy,xx=np.nonzero(mask&~neighbor)
        result.append((yy,xx,dx,dy))
    return result


def _fit(rgb,occ,a,borders,pose):
    h,w=a.shape[:2];H,W=occ.shape
    x,y=pose.x,pose.y
    if x<0 or y<0 or x+w>W or y+h>H:return None
    mask=a[:,:,3]>0
    if np.any(occ[y:y+h,x:x+w]&mask):return None
    n=0;wall=0;cost=0.;perim=0
    for yy,xx,dx,dy in borders:
        Y=yy+y+dy;X=xx+x+dx;perim+=len(yy)
        ok=(Y>=0)&(X>=0)&(Y<H)&(X<W);wall+=int((~ok).sum())
        Y,X,gy,gx=Y[ok],X[ok],yy[ok],xx[ok]
        keep=occ[Y,X]
        if not keep.any():continue
        n+=int(keep.sum())
        cost+=np.abs(a[gy[keep],gx[keep],:3].astype(np.int16)-rgb[Y[keep],X[keep],:3].astype(np.int16)).sum()
    if n<8:return None
    color=float(cost)/(n*3*255)
    fraction=(n+wall)/max(1,perim)
    return fraction,color,n,perim


def complete_frontier(pieces,poses,shape,rotations=False,rounds=14,bucket=96):
    """Propose exact nonoverlapping fits against the current empty region.

    Prefer long, photometrically smooth contacts; do not force ambiguous short
    contacts. All proposals are rechecked after each placement in a batch.
    """
    from .common import normalize_pose_set
    if not poses or len(poses)==len(pieces):return poses,{"frontier_added":0}
    _,extent=normalize_pose_set(pieces,poses)
    if tuple(extent)!=tuple(shape):return poses,{"frontier_added":0,"frontier_skipped":"unanchored"}
    poses=dict(poses);total=0;history=[]
    missing=[p for p in pieces if p.id not in poses]
    cache={(p.id,k):(p.rotated(k),_border_pixels(p.rotated(k))) for p in missing for k in range(4 if rotations else 1)}
    arcs={}
    for k in (18,13,9):
        arcs[k]={p.id:_arc_index(p.mask,k,rotations) for p in missing}
    for rnd in range(rounds):
        rgb,labels,collision=render(pieces,poses,shape);occ=labels>=0
        if collision:break
        proposals=[]
        for k in (18,13,9):
            targets=_arc_index(~occ,k,rotations)
            for p in missing:
                if p.id in poses:continue
                votes=Counter()
                for key,src in arcs[k][p.id].items():
                    trg=targets.get(key,[])
                    if len(trg)*len(src)>bucket or not trg:continue
                    for (ax,ay),a_dir in src:
                        for (bx,by),b_dir in trg:
                            r=(b_dir-a_dir)%4 if rotations else 0
                            t=np.array([bx,by])-R[r]@np.array([ax,ay])
                            q=_vertex_to_pose(p,r,t)
                            votes[q]+=1
                if not votes:continue
                fits=[]
                for q,v in votes.most_common(24):
                    a,b=cache[p.id,q.k]
                    fit=_fit(rgb,occ,a,b,q)
                    if fit is None:continue
                    frac,col,n,perim=fit
                    # Length + fraction, favour low seam discontinuity.
                    if frac<.16 or col>.135 or n<12:continue
                    score=(frac**.65)*(n**.35)/(.012+col)
                    fits.append((score,q,frac,col,n,v))
                if not fits:continue
                fits.sort(key=lambda f:-f[0])
                score,q,frac,col,n,v=fits[0]
                margin=score/max(fits[1][0],1e-9) if len(fits)>1 else 3.
                if (frac>=.57 and col<=.105) or (frac>=.22 and col<=.09 and margin>=1.13 and v>=2):
                    proposals.append((score*min(1.6,margin),p.id,q))
            if len(proposals)>=10:break
        changed=0
        for score,id,q in sorted(proposals,key=lambda f:-f[0]):
            if id in poses:continue
            a,b=cache[id,q.k];fit=_fit(rgb,occ,a,b,q)
            if fit is None:continue
            frac,col,n,perim=fit
            if col>.105 or frac<.22:continue
            h,w=a.shape[:2];m=a[:,:,3]>0
            occ[q.y:q.y+h,q.x:q.x+w][m]=True
            rgb[q.y:q.y+h,q.x:q.x+w][m]=a[m]
            poses[id]=q;changed+=1
        poses,exact=exact_holes(pieces,poses,shape,rotations)
        total+=changed+exact;history.append([changed,exact])
        if not (changed+exact) or len(poses)==len(pieces):break
    return poses,{"frontier_added":total,"frontier_batches":history}


def repair_small_gaps(pieces,poses,shape,rotations=False):
    """Near-complete mask assembly: reopen tiny impossible gaps once.

    Tiny residual regions can be caused by incorrect boundary placement.
    Reinsert only a bounded neighborhood; accept by the same blind coverage,
    overlap and seam objective, never by truth.
    """
    from scipy.ndimage import label,binary_dilation
    from .solver import objective
    if not .975*len(pieces)<=len(poses)<len(pieces):
        return poses,{"gap_repair_added":0}
    _,labels,collision=render(pieces,poses,shape)
    if collision:return poses,{"gap_repair_added":0}
    regs,n=label(labels<0);area=np.bincount(regs.ravel(),minlength=n+1)
    minimum=min(int(p.mask.sum()) for p in pieces)
    small=np.flatnonzero((area>0)&(area<.12*minimum))
    if not len(small):return poses,{"gap_repair_added":0}
    mask=np.isin(regs,small)&(regs>0)
    suspect=np.unique(labels[binary_dilation(mask,iterations=2)&(labels>=0)])
    if not 0<len(suspect)<=min(40,.04*len(pieces)):return poses,{"gap_repair_added":0}
    remove={pieces[int(i)].id for i in suspect}
    base={id:q for id,q in poses.items() if id not in remove}
    pp,diag=complete_frontier(pieces,base,shape,rotations)
    if objective(pieces,pp,shape)>objective(pieces,poses,shape):
        return pp,{"gap_repair_added":len(pp)-len(poses),"gap_reopened":len(remove),"gap_repair_batches":diag.get("frontier_batches",[])}
    return poses,{"gap_repair_added":0}
