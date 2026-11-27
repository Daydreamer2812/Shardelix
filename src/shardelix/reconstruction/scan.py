"""Experimental planar torn-paper scan reconstruction (no grid or ground truth).

All rigid transforms use image-coordinate pixel vertices, x right, y down.
Only a common scale is applied; physical DPI/perspective differences are not inferred.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageOps, ImageDraw
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from .common import Fragment, json_write, save_rgba
from .geometry import contours

MAX_SCAN_PIECES = 256
MAX_SCAN_SIDE = 1600
MAX_WORK_PIXELS = 4_000_000


def chroma_mask(a):
    """Detect magenta sentinel backgrounds with a mostly uniform outer rim.

    Preserve exact silhouettes rather than applying morphological closing to
    digitally keyed fragments. Return None for ordinary photographs/tiles.
    """
    rgb=np.asarray(a)[...,:3]
    key=(rgb[:,:,0]>=250)&(rgb[:,:,1]<=5)&(rgb[:,:,2]>=250)
    rim=np.r_[key[0],key[-1],key[:,0],key[:,-1]]
    if rim.mean()>=.90 and .02<=float((~key).mean())<=.95:
        return ~key
    return None


def preprocess(paths, target: Path, *, pixel_limit=24_000_000, preserve_opaque=False, piece_rasters=False, max_pieces=MAX_SCAN_PIECES, preserve_scale=False):
    """Split overview scans, or preserve each explicitly declared piece raster.

    An eroded piece may have multiple surviving foreground islands. Joining
    their bounding crop does not fill in the missing alpha pixels.
    """
    target.mkdir(parents=True, exist_ok=True)
    sizes = []
    for path in paths:
        with Image.open(path) as im:
            sizes.append((im.width, im.height))
    total = sum(w*h for w,h in sizes)
    if total > pixel_limit:
        raise ValueError("Decoded scans exceed pixel budget; scan at lower DPI or use native Studio.")
    scale = 1. if preserve_scale else min(1., MAX_SCAN_SIDE/max(max(s) for s in sizes), math.sqrt(MAX_WORK_PIXELS/total))
    pieces, scans, warnings = [], [], []
    for path in paths:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGBA")
            # Convert chroma key to alpha before resampling so the sentinel
            # color cannot bleed into the foreground boundary.
            from .input_detection import solid_background,scan_evidence
            is_opaque=im.getchannel("A").getextrema()[0]>=128
            keyed=chroma_mask(np.asarray(im)) if is_opaque else None
            key_bg=np.array([255,0,255]) if keyed is not None else None
            key_method="magenta-chroma-key"
            if is_opaque and keyed is None:
                generic=solid_background(np.asarray(im),exact=True)
                if generic is not None:keyed,key_bg=generic;key_method="solid-background"
            # For automatic mode, a textured, non-matte opaque edge is evidence
            # of a full raster piece, not permission to remove guessed pixels.
            keep_full=bool(preserve_opaque and is_opaque and keyed is None and scan_evidence(im) is None)
            if keyed is not None:im.putalpha(Image.fromarray(keyed.astype(np.uint8)*255))
            if scale < 1:
                im = im.resize(tuple(max(1, round(v*scale)) for v in im.size), Image.Resampling.LANCZOS)
            a = np.array(im)
        h,w = a.shape[:2]
        if min(h,w) < 8:
            scans.append({"file":path.name,"pieces":0,"warning":"Image too small"})
            warnings.append(f"{path.name}: image too small.")
            continue
        rgb = a[:,:,:3].astype(np.float32)
        method = "alpha"
        bg = None
        threshold = None
        if keyed is not None:
            method=key_method
            mask=a[:,:,3]>=128
            bg=key_bg
            threshold=5
        elif keep_full:
            method="opaque-raster-preserved"
            mask=np.ones((h,w),bool)
            warnings.append(f"{path.name}: no reliable background detected; preserving the complete opaque raster.")
        elif np.any(a[:,:,3]<128):
            mask = a[:,:,3]>=128
        else:
            method = "border-background"
            b = max(1,min(5,h//12,w//12))
            border = np.concatenate([rgb[:b].reshape(-1,3),rgb[-b:].reshape(-1,3),
                                     rgb[:, :b].reshape(-1,3),rgb[:, -b:].reshape(-1,3)])
            bg = np.median(border,axis=0)
            bd = np.linalg.norm(border-bg,axis=1)/math.sqrt(3)
            med = np.median(bd)
            threshold = float(max(8.,med+6*1.4826*np.median(abs(bd-med))))
            # Compensate gentle planar light falloff on a matte photo background
            # only if an independently fitted border plane clearly improves fit.
            yy,xx=np.indices((h,w),sparse=True)
            rim=(yy<b)|(yy>=h-b)|(xx<b)|(xx>=w-b)
            ry,rx=np.nonzero(rim)
            subs=slice(None,None,max(1,math.ceil(len(ry)/4096)))
            ry,rx=ry[subs],rx[subs]
            X=np.column_stack([np.ones(len(ry)),rx/max(1,w-1),ry/max(1,h-1)])
            C=rgb[ry,rx]
            good=np.linalg.norm(C-bg,axis=1)/math.sqrt(3)<=max(30,threshold)
            if good.sum()>=max(24,.6*len(good)):
                for _ in range(3):
                    coef=np.linalg.lstsq(X[good],C[good],rcond=None)[0]
                    err=np.linalg.norm(C-X@coef,axis=1)/math.sqrt(3)
                    med_err=np.median(err[good])
                    improved=err<=max(7.,med_err*3)
                    if improved.sum()<max(24,.6*len(good)):break
                    good=improved
                old=np.median(np.linalg.norm(C[good]-bg,axis=1)/math.sqrt(3))
                new=np.median(err[good])
                if old>5 and new<old*.65 and np.max(abs(coef[1:]))<80:
                    method="border-plane"
                    bg_field=coef[0]+xx[:,:,None]/max(1,w-1)*coef[1]+yy[:,:,None]/max(1,h-1)*coef[2]
                    dist=np.linalg.norm(rgb-bg_field,axis=2)/math.sqrt(3)
                    threshold=float(max(8.,new+6*1.4826*np.median(abs(err[good]-new))))
                else:dist=np.linalg.norm(rgb-bg,axis=2)/math.sqrt(3)
            else:dist=np.linalg.norm(rgb-bg,axis=2)/math.sqrt(3)
            if threshold > 42:
                warnings.append(f"{path.name}: uneven or occupied background; segmentation unreliable.")
            # Reject low-amplitude JPEG ringing around a strongly contrasting
            # sheet; preserve the noise-derived floor for faint foregrounds.
            threshold=max(threshold,min(35.,.22*float(np.percentile(dist,90))))
            # Closing bridges tiny scanner/JPEG breaks, not gaps between separate pieces.
            mask = ndi.binary_closing(dist > threshold, iterations=1, border_value=0)
            mask = ndi.binary_fill_holes(mask)
            # Re-add foreground touching image borders (closing erodes outermost pixels).
            edge = dist > threshold
            mask[[0,-1],:] |= edge[[0,-1],:]
            mask[:,[0,-1]] |= edge[:,[0,-1]]
        labels,n = ndi.label(mask)
        raw_components=int(n)
        if piece_rasters and n:
            labels=mask.astype(np.int32)
            n=1
            if raw_components>1:
                warnings.append(f"{path.name}: preserved disconnected foreground within one declared piece raster.")
        count = 0
        # Reject speckle, not all small components. Keep holes in explicit alpha scans.
        minimum = max(36,round(h*w*0.0002))
        for label,sl in enumerate(ndi.find_objects(labels),1):
            if sl is None: continue
            local = labels[sl] == label
            if np.count_nonzero(local)<minimum or min(local.shape)<5:continue
            if len(pieces)>=max_pieces:
                raise ValueError(f"Scan mode supports up to {max_pieces} detected pieces; separate speckle and use smaller sets.")
            y,x=sl
            if method!="opaque-raster-preserved" and (x.start==0 or y.start==0 or x.stop==w or y.stop==h):
                warnings.append(f"{path.name}: a piece touches the scan border; its edge may be clipped.")
            out=a[sl].copy()
            out[:,:,3]=local.astype(np.uint8)*255
            out[~local]=0
            frag=Fragment(f"scan_{len(pieces)+1:03d}",out,(x.start,y.start))
            save_rgba(target/(frag.id+".png"),out)
            pieces.append(frag);count+=1
            scans.append({"file":path.name,"id":frag.id,"method":method,"crop_xy":[x.start,y.start],
                          "crop_wh":[x.stop-x.start,y.stop-y.start],"mask_pixels":int(local.sum()),
                          "background_rgb":bg.tolist() if bg is not None else None,"threshold":threshold,
                          "foreground_components":raw_components,"piece_raster":bool(piece_rasters)})
        if count==0:
            warnings.append(f"{path.name}: no visible paper silhouette. Use a contrasting matte background and include a margin.")
    if not pieces:
        raise ValueError("No paper silhouettes detected. Use a flat, contrasting background with margins, or transparent PNG pieces.")
    return pieces,{"input_images":len(paths),"detected_pieces":len(pieces),"common_scale":scale,
                   "original_pixel_sizes":[list(s) for s in sizes],"pieces":scans,"warnings":list(dict.fromkeys(warnings)),
                   "piece_rasters":bool(piece_rasters)}


def rotation(theta):
    c,s=math.cos(theta),math.sin(theta)
    return np.array([[c,-s],[s,c]])


@dataclass
class Shape:
    fragment: Fragment
    points: np.ndarray
    tangent: np.ndarray
    tree: cKDTree
    core: np.ndarray
    samples: np.ndarray
    color: np.ndarray
    step: float


def _shape(p, step):
    loops=contours(p.mask)
    raw=np.asarray(max(loops,key=lambda l:len(l[0]))[0],float)
    # Smooth raster staircase before uniform Euclidean arc-length sampling.
    smooth=ndi.gaussian_filter1d(raw,1.5,axis=0,mode="wrap")
    d=np.linalg.norm(np.roll(smooth,-1,axis=0)-smooth,axis=1)
    lengths=np.r_[0,np.cumsum(d)]
    n=max(12,round(lengths[-1]/step))
    s=np.arange(n)*lengths[-1]/n
    v=np.stack([np.interp(s,lengths,np.r_[smooth[:,k],smooth[0,k]]) for k in range(2)],axis=1)
    t=np.roll(v,-2,axis=0)-np.roll(v,2,axis=0)
    t/=np.maximum(np.linalg.norm(t,axis=1,keepdims=True),1e-6)
    core=ndi.binary_erosion(p.mask,iterations=3,border_value=0)
    yy,xx=np.nonzero(core)
    stride=max(1,math.ceil(len(xx)/1800))
    samples=np.column_stack([xx[::stride]+.5,yy[::stride]+.5])
    inward=v+np.column_stack([-t[:,1],t[:,0]])*3
    color=np.stack([ndi.map_coordinates(p.rgba[:,:,i].astype(float),[inward[:,1]-.5,inward[:,0]-.5],order=1,mode="nearest") for i in range(3)],axis=1)
    return Shape(p,v,t,cKDTree(v),core,samples,color,float(lengths[-1]/n))


def _descriptor(v, k, reverse=False):
    n=len(v)
    if n<=k+5:return None
    ids=np.arange(n)
    direction=-1 if reverse else 1
    windows=v[(ids[:,None]+direction*np.rint(np.linspace(0,k,9)).astype(int))%n]
    rel=windows-windows[:,:1]
    chord=rel[:,-1]
    norm=np.linalg.norm(chord,axis=1)
    u=chord/np.maximum(norm[:,None],1e-6)
    x=np.sum(rel*u[:,None],axis=-1)
    y=np.sum(rel*np.column_stack([-u[:,1],u[:,0]])[:,None],axis=-1)
    # Chord-normalized orientation but not scale (the scans must share DPI).
    valid=(norm>max(8,k*.55)) & (np.max(abs(y),axis=1)>1.25)
    feat=np.concatenate([x[:,1:-1]/k,y[:,1:-1]/k,norm[:,None]/k],axis=1)
    return ids[valid],feat[valid],windows[valid]


def _fit(a,b):
    ac=a.mean(axis=0);bc=b.mean(axis=0)
    U,_,V=np.linalg.svd((b-bc).T@(a-ac))
    R=V.T@U.T
    if np.linalg.det(R)<0:
        V[-1]*=-1;R=V.T@U.T
    return R,ac-bc@R.T


def _overlap(a,b,R,t):
    # Bidirectional interior samples; thin edge gaps/overlaps are tolerated.
    def frac(core, pts):
        if not len(pts):return 0.
        x,y=np.floor(pts).astype(int).T;h,w=core.shape
        keep=(x>=0)&(x<w)&(y>=0)&(y<h)
        return np.count_nonzero(core[y[keep],x[keep]])/len(pts)
    return max(frac(a.core,b.samples@R.T+t),frac(b.core,(a.samples-t)@R))


def _contact(a,b,R,t,tol, refine=True):
    # Nearest complementary boundaries and opposing tangent direction.
    for it in range(3 if refine else 1):
        d,idx=a.tree.query(b.points@R.T+t)
        opposing=np.sum(a.tangent[idx]*(b.tangent@R.T),axis=1)<-.60
        good=(d<tol)&opposing
        if good.sum()<7:return None
        if it<2 and refine:
            R,t=_fit(a.points[idx[good]],b.points[good])
    # Require a continuous seam, not a collection of accidental point matches.
    doubled=np.r_[good,good]
    breaks=np.flatnonzero(np.r_[True,doubled[1:]!=doubled[:-1],True])
    runs=[(int(l),min(len(good),int(r-l))) for l,r in zip(breaks[:-1],breaks[1:]) if doubled[l]]
    start,longest=max(runs,key=lambda q:q[1],default=(0,0))
    length=longest*b.step
    total=good.sum()*b.step
    if length<max(18.,min(len(a.points)*a.step,len(b.points)*b.step)*.045):return None
    arc=b.points[(start+np.arange(longest))%len(good)]
    chord=arc[-1]-arc[0]
    dev=abs((arc-arc[0])@np.array([-chord[1],chord[0]])/max(np.linalg.norm(chord),1e-6))
    # A nearly straight edge can match almost any exterior paper edge. Abstain
    # unless the contiguous contact includes meaningful non-collinear shape.
    bend=float(np.sqrt(np.mean(dev**2)))
    if bend<max(2.2,tol*.8) or np.max(dev)<4.:return None
    overlap=_overlap(a,b,R,t)
    if overlap>.035:return None
    residual=float(np.median(d[good]))
    color=float(np.median(np.mean(abs(a.color[idx[good]]-b.color[good]),axis=1)))
    score=(length+.25*total)*(0.25+0.75*min(1.,bend/9.))*math.exp(-color/55)-7*residual-150*overlap
    return score,R,t,{"contact_length":float(length),"median_gap":residual,"interior_overlap":float(overlap),"color_difference":color,"seam_bend_rms":bend}


def _pair(a,b,desc_a,desc_b,tol):
    hypotheses=Counter();poses={}
    for da,db in zip(desc_a,desc_b):
        if da is None or db is None or not len(da[0]) or not len(db[0]):continue
        _,fa,wa=da;_,fb,wb=db
        d,ix=cKDTree(fb).query(fa,k=min(3,len(fb)))
        if d.ndim==1:d=d[:,None];ix=ix[:,None]
        ii,jj=np.nonzero(d<.55)
        for i,j in zip(ii,jj):
            A,B=wa[i],wb[ix[i,j]]
            va=A[-1]-A[0];vb=B[-1]-B[0]
            theta=math.atan2(va[1],va[0])-math.atan2(vb[1],vb[0])
            theta=(theta+math.pi)%(2*math.pi)-math.pi
            R=rotation(theta);t=A.mean(axis=0)-B.mean(axis=0)@R.T
            key=(round(theta/.065),round(t[0]/5),round(t[1]/5))
            hypotheses[key]+=1
            if key not in poses or d[i,j]<poses[key][0]:poses[key]=(d[i,j],R,t)
    options=[]
    # Straight, uninformative edges alone cannot vote. Keep bounded hypotheses.
    for key,votes in hypotheses.most_common(32):
        if votes<2:continue
        _,R,t=poses[key]
        c=_contact(a,b,R,t,tol)
        if c is not None:
            score,R,t,info=c
            score+=min(votes,30)*.35
            if score<9:continue
            options.append((score,R,t,{**info,"votes":votes}))
    # Distinct alternate poses are retained if a stronger local match would
    # collide with an existing component during global assembly.
    best=[]
    for c in sorted(options,key=lambda q:-q[0]):
        if any(np.linalg.norm(c[2]-v[2])<8 and np.linalg.norm(c[1]-v[1])<.12 for v in best):continue
        best.append(c)
        if len(best)==4:break
    return best


def _merge(shapes,candidates):
    groups={i:{i:(np.eye(2),np.zeros(2))} for i in range(len(shapes))}
    owners=list(range(len(shapes)));accepted=[]
    for score,i,j,R,t,info in sorted(candidates,key=lambda c:-c[0]):
        x,y=owners[i],owners[j]
        if x==y:continue
        A,B=groups[x],groups[y]
        Ri,ti=A[i];Rj,tj=B[j]
        Q=Ri@R@Rj.T;u=Ri@t+ti-Q@tj
        mapped={k:(Q@P,Q@q+u) for k,(P,q) in B.items()}
        collision=False
        for k,(P,p) in A.items():
            for l,(S,s) in mapped.items():
                if _overlap(shapes[k],shapes[l],P.T@S,P.T@(s-p))>.035:
                    collision=True;break
            if collision:break
        if collision:continue
        A.update(mapped)
        for k in B:owners[k]=x
        del groups[y]
        accepted.append({"a":shapes[i].fragment.id,"b":shapes[j].fragment.id,"score":float(score),**info})
    return sorted(groups.values(),key=lambda g:(-len(g),-sum(shapes[k].fragment.mask.sum() for k in g))),accepted


def _render(shapes, poses, out, pixel_cap):
    vertices=np.concatenate([shapes[i].points@R.T+t for i,(R,t) in poses.items()])
    # Choose a compact global gauge from outer points. No claim of upright orientation.
    from scipy.spatial import ConvexHull, QhullError
    try:
        hull=vertices[ConvexHull(vertices).vertices]
        edges=np.roll(hull,-1,axis=0)-hull
        angles=np.unique(np.round(np.arctan2(edges[:,1],edges[:,0])%(math.pi/2),4))
        theta=min(angles,key=lambda v:np.prod(np.ptp(hull@rotation(-v).T,axis=0)))
        Q=rotation(-theta)
    except (QhullError,ValueError):Q=np.eye(2)
    # The pose estimator uses the largest contour, but the renderer must
    # include ALL surviving mask islands. Otherwise a tear separating two
    # regions of a single raster can cause the small region to be clipped.
    low=[];high=[]
    for i,(R,t) in poses.items():
        mask=shapes[i].fragment.mask
        yy,xx=np.nonzero(mask&~ndi.binary_erosion(mask))
        v=(np.column_stack([xx,yy])+.5)@(Q@R).T+Q@t
        low.append(v.min(axis=0));high.append(v.max(axis=0))
    # Fixed margin includes pixel footprints and the seam overlay.
    lo=np.floor(np.min(low,axis=0))-3
    hi=np.ceil(np.max(high,axis=0))+3
    w,h=map(int,hi-lo)
    if w*h>pixel_cap:raise ValueError("Proposed scan assembly exceeds canvas budget; use smaller scans.")
    canvas=np.zeros((h,w,4),np.uint8);labels=np.full((h,w),-1,np.int32)
    collision=0;result={}
    swap=np.array([[0,1],[1,0]])
    for i,(R,t) in poses.items():
        P=Q@R;p=Q@t-lo
        frag=shapes[i].fragment
        fh,fw=frag.mask.shape
        corners=np.array([[0,0],[fw,0],[0,fh],[fw,fh]])@P.T+p
        x0,y0=np.maximum(0,np.floor(corners.min(axis=0)).astype(int))
        x1,y1=np.minimum([w,h],np.ceil(corners.max(axis=0)).astype(int))
        shape=(y1-y0,x1-x0)
        if min(shape)<=0:continue
        offset=swap@(P.T@(np.array([x0+.5,y0+.5])-p)-.5)
        matrix=swap@P.T@swap
        warped=np.stack([ndi.affine_transform(frag.rgba[:,:,k],matrix,offset,output_shape=shape,order=0,prefilter=False) for k in range(4)],axis=2)
        m=warped[:,:,3]>0
        region=labels[y0:y1,x0:x1];collision+=int((m&(region>=0)).sum())
        keep=m&(region<0);canvas[y0:y1,x0:x1][keep]=warped[keep];region[keep]=i
        result[frag.id]={"input_to_canvas_affine":np.column_stack([P,p]).tolist(),"crop_xy":list(frag.input_origin)}
    save_rgba(out/"assembly.png",canvas)
    boundary=np.zeros((h,w),bool)
    for i in poses:
        m=labels==i;boundary|=m&~ndi.binary_erosion(m)
    cut=np.zeros_like(canvas);cut[boundary]=[220,30,40,255]
    overlay=canvas.copy();overlay[boundary]=[220,30,40,255]
    save_rgba(out/"assembly_with_cuts.png",overlay);save_rgba(out/"cut_lines.png",cut)
    json_write(out/"placements.json",result)
    return collision,float((labels>=0).mean()),[w,h]


def _indexed_pairs(shapes, forward, reverse, scales, *, keep=10):
    """Retrieve sparse plausible partners using global multiscale arc evidence.

    The full rigid-pose, overlap and continuity checks still run for each
    shortlisted pair. Keep both incoming/outgoing nominations to avoid
    dropping a small fragment just because its neighbor is much larger.
    """
    n=len(shapes)
    votes=np.zeros((n,n),np.float64)
    for k,scale in enumerate(scales):
        def features(shape, desc, direction):
            if desc is None or not len(desc[0]):return None
            ids, geom, _=desc
            offsets=np.rint(np.linspace(0,scale,3)).astype(int)*direction
            rgb=shape.color[(ids[:,None]+offsets)%len(shape.points)].reshape(len(ids),-1)
            return np.concatenate([geom,.40*rgb/255.],axis=1)
        fs=[features(sh,forward[i][k],1) for i,sh in enumerate(shapes)]
        rs=[features(sh,reverse[i][k],-1) for i,sh in enumerate(shapes)]
        valid=[i for i,r in enumerate(rs) if r is not None]
        if not valid:continue
        pool=np.concatenate([rs[i] for i in valid])
        owners=np.concatenate([np.full(len(rs[i]),i,dtype=np.int32) for i in valid])
        tree=cKDTree(pool)
        neighbors=min(24,len(pool))
        for i,f in enumerate(fs):
            if f is None:continue
            d,ix=tree.query(f,k=neighbors)
            if neighbors==1:d=d[:,None];ix=ix[:,None]
            owner=owners[ix]
            good=(owner!=i)&(d<.65)
            # A single arc can vote for many near-identical windows of the
            # same piece. Retain at most its strongest vote for each owner.
            best=np.zeros((len(f),n))
            row=np.broadcast_to(np.arange(len(f))[:,None],owner.shape)
            np.maximum.at(best,(row[good],owner[good]),np.exp(-3*d[good]))
            votes[i]+=best.sum(0)/max(1,len(f))*(scale/12)**.5
    # Shortlist a bounded number per fragment. Neighbors with no evidence
    # are not invented merely to fill the quota.
    pairs=set()
    for i in range(n):
        score=votes[i]+votes[:,i]
        score[i]=0
        for j in np.argsort(-score,kind="stable")[:keep]:
            if score[j]>0:pairs.add((min(i,int(j)),max(i,int(j))))
    return sorted(pairs)


def reconstruct(pieces,out:Path,report,update=lambda _:None,*,pixel_cap=8_000_000,gap_recovery=True,deep_recovery=True):
    started=time.perf_counter()
    # Fixed sampling budget for all pieces, not per-piece rescaling.
    step=max(2.0, sum(2*sum(p.mask.shape) for p in pieces)/16000)
    shapes=[_shape(p,step) for p in pieces]
    scales=[12,20,32]
    forward=[[_descriptor(s.points,k) for k in scales] for s in shapes]
    reverse=[[_descriptor(s.points,k,True) for k in scales] for s in shapes]
    candidates=[];tol=max(2.7,step*1.35)
    n=len(shapes)
    indexed=n>64
    if indexed:
        update("Indexing multiscale scan contours and colors")
        pairs=_indexed_pairs(shapes,forward,reverse,scales,keep=10 if n<=128 else 8)
    else:
        pairs=[(i,j) for i in range(n) for j in range(i+1,n)]
    for z,(i,j) in enumerate(pairs):
        if z%max(1,len(pairs)//100)==0:
            update(f"Matching scan contours {z+1}/{len(pairs)} candidate pairs")
        for c in _pair(shapes[i],shapes[j],forward[i],reverse[j],tol):candidates.append((c[0],i,j,*c[1:]))
    groups,joins=_merge(shapes,candidates)
    exact_size=len(groups[0])
    recovery={"attempted":False,"selected":False,"exact_largest_component":exact_size}
    alternative=None
    recovered=[]
    # Preserve the exact-contact route on clean data. A sparse, disconnected
    # assembly can trigger bounded gap-aware recovery without a known canvas.
    # Small irregular puzzles can also have a large but weak exact-contact
    # tree. Search a second model when these joins have weak local evidence.
    weak_small = n<=16 and any(q["color_difference"]>18 or
                              q["contact_length"]<25 or q["median_gap"]>2
                              for q in joins)
    requested = gap_recovery=="always"
    if gap_recovery and 2<=n<=128 and (requested or weak_small or exact_size<max(2,math.ceil(.60*n))):
        from .eroded import gap_candidates, rank_components, refine_components
        update("Recovering abraded seams and incomplete components")
        recovered,stats=gap_candidates(shapes,update)
        recovery.update(stats,attempted=True,trigger="always" if requested else "weak-small" if weak_small else "sparse-exact")
        if recovered:
            ggroups,gjoins=_merge(shapes,recovered)
            ggroups=refine_components(shapes,ggroups,recovered)
            ggroups=rank_components(shapes,ggroups,gjoins)
            recovery.update(gap_largest_component=max(map(len,ggroups)),
                            gap_primary_component=len(ggroups[0]),
                            gap_nontrivial_components=sum(len(g)>1 for g in ggroups))
            # Very sparse exact matches are not evidence of an intact object.
            # Prefer an independently verified component; don't select by
            # number of placed fragments alone. Otherwise retain exact.
            if len(ggroups[0])>=2 and (exact_size<max(3,math.ceil(.12*n)) or len(ggroups[0])>=exact_size):
                alternative=(groups,joins,"exact-contact")
                groups,joins=ggroups,gjoins
                recovery["selected"]=True
            else:
                alternative=(ggroups,gjoins,"gap-aware")
    report["gap_recovery"]=recovery
    assembly_model="gap-aware" if recovery["selected"] else "exact-contact"
    deep={"attempted":False,"selected":False}
    other_model=None
    # This high-quality route is bounded and is activated automatically when
    # the normal solver remains sparse. Keep routine intact cases fast.
    if gap_recovery and deep_recovery and 2<=n<=128 and (
          deep_recovery=="always" or (recovery["attempted"] and (weak_small or exact_size<max(2,math.ceil(.60*n))))):
        from .continuation import recover
        cg,cj,stats=recover(shapes,recovered,update)
        deep.update(stats)
        if cg is not None and len(cg[0])>=2:
            members={shapes[i].fragment.id for i in cg[0]}
            supported=[j for j in cj if j["a"] in members and j["b"] in members
                       and (j.get("consensus_supported") or j.get("line_count",0)>=2 or j.get("contextual_growth"))]
            if supported and len(cg[0])>=len(groups[0]):
                other_model=alternative
                alternative=(groups,joins,assembly_model)
                groups,joins=cg,cj
                assembly_model="classical-continuation"
                deep["selected"]=True
            else:
                other_model=(cg,cj,"classical-continuation")
    report["deep_recovery"]=deep
    update("Rendering scan reconstruction")
    main=groups[0]
    if len(groups)>1:
        report["warnings"].append(f"Only the selected connected group ({len(main)}/{n} fragments) is rendered in the main image. Remaining groups are unanchored; missing material or distractors may prevent reliable joins.")
    collision,coverage,wh=_render(shapes,main,out,pixel_cap)
    json_write(out/"unanchored_components.json",[[shapes[i].fragment.id for i in g] for g in groups[1:]])
    def component_records(gs,js):
        records=[]
        for k,g in enumerate(gs):
            members={shapes[i].fragment.id for i in g}
            local_joins=[q for q in js if q["a"] in members and q["b"] in members]
            records.append({"id":k,"members":sorted(members),"joins":local_joins,
                            "poses":{shapes[i].fragment.id:{
                                "input_to_component_affine":np.column_stack([R,t]).tolist(),
                                "crop_xy":list(shapes[i].fragment.input_origin)}
                                for i,(R,t) in g.items()}})
        return records
    # Each connected component has an independent rigid gauge. Do not pack
    # unrelated components into a fake reconstruction of the missing canvas.
    comp={"model":assembly_model,"accuracy_verified":False,
          "coordinate_system":"independent per-component SE(2); no placement between components",
          "primary_component":0,"components":component_records(groups,joins)}
    if alternative is not None:
        ag,aj,am=alternative
        comp["alternative"]={"model":am,"components":component_records(ag,aj)}
    if other_model is not None:
        ag,aj,am=other_model
        comp["other_model"]={"model":am,"components":component_records(ag,aj)}
    json_write(out/"component_placements.json",comp)
    report.update({"pair_search":"indexed-contour-color" if indexed else "all-pairs","searched_pairs":len(pairs),"possible_pairs":n*(n-1)//2,"candidate_poses":len(candidates),"candidate_pairs":len({(c[1],c[2]) for c in candidates}),"accepted_joins":joins,"component_sizes":[len(g) for g in groups],
                   "canvas_wh":wh,"orientation_ambiguous":True,"accuracy_verified":False,
                   "coverage_definition":"occupied fraction of estimated bounding box, not recovered original area"})
    # Contact sheet gives an immediate visual check of segmentation quality.
    tile=140;cols=min(8,len(pieces));rows=math.ceil(len(pieces)/cols)
    preview=Image.new("RGB",(cols*tile,rows*tile),(225,231,228));draw=ImageDraw.Draw(preview)
    for i,p in enumerate(pieces):
        im=Image.fromarray(p.rgba);im.thumbnail((tile-12,tile-24))
        x=(i%cols)*tile+(tile-im.width)//2;y=(i//cols)*tile+(tile-20-im.height)//2
        preview.paste(im,(x,y),im);draw.text(((i%cols)*tile+6,(i//cols+1)*tile-17),p.id,fill=(40,55,45))
    preview.save(out/"fragments_preview.png")
    json_write(out/"scan_report.json",report)
    d={"algorithm":"physical-scan-classical-continuation" if deep["selected"] else ("physical-scan-gap-aware-se2" if recovery["selected"] else "physical-scan-se2-experimental"),"status":"unverified" if len(main)==len(pieces) else "partial",
       "placed":len(main),"piece_count":len(pieces),"coverage":coverage,"render_collision_pixels":collision,
       "available_piece_recovery":len(main)/len(pieces),"runtime_seconds":time.perf_counter()-started,
       "scan":{"kind":"irregular-fragments","gap_recovery":recovery,"deep_recovery":deep,"input_images":report["input_images"],"detected_pieces":len(pieces),"components":len(groups),"pair_search":report["pair_search"],"searched_pairs":len(pairs),"possible_pairs":n*(n-1)//2,
               "warnings":report["warnings"],"common_scale":report["common_scale"],"accuracy_verified":False}}
    json_write(out/"diagnostics.json",d)
    return d
