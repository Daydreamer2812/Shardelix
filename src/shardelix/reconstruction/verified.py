"""Verified complementary-arc retrieval and exact hole completion for blind masks."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from scipy.ndimage import label, find_objects
from .common import Pose,render
from .geometry import propose,assemble,_vertex_to_pose,contours


def _overlap(a,b,x,y):
    """Overlapping array windows for b translated by (x,y) relative to a."""
    h,w=a.shape[:2];hh,ww=b.shape[:2]
    x0=max(0,x);y0=max(0,y);x1=min(w,x+ww);y1=min(h,y+hh)
    if x1<=x0 or y1<=y0:return None
    return a[y0:y1,x0:x1],b[y0-y:y1-y,x0-x:x1-x]


def verify_pair(pieces,cache,candidate):
    """Measure entire contact, rejecting any intersecting foreground pixels."""
    votes,i,j,r,tx,ty=candidate
    a=cache[i,0];b=cache[j,r]
    p=_vertex_to_pose(pieces[j],r,np.array([tx,ty]))
    x,y=p.x,p.y
    windows=_overlap(a,b,x,y)
    if windows is not None:
        aa,bb=windows
        if np.any((aa[:,:,3]>0)&(bb[:,:,3]>0)):return None
    length=0;cost=0.;axes=0
    for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
        windows=_overlap(a,b,x+dx,y+dy)
        if windows is None:continue
        aa,bb=windows;contact=(aa[:,:,3]>0)&(bb[:,:,3]>0)
        count=int(contact.sum())
        if count:
            axes+=1;length+=count
            cost+=np.abs(aa[contact,:3].astype(np.int16)-bb[contact,:3].astype(np.int16)).sum()
    if length<4:return None
    return length,float(cost)/(length*3*255),axes


def verified_candidates(pieces,rotations,k=16,max_bucket=18,content=True,segments=False):
    from .content import content_candidates
    votes={}
    for win,bucket in [(k,max_bucket),(max(k+8,24),max(5,max_bucket//2)),(max(10,k-4),max_bucket)]:
        for v,i,j,r,x,y in propose(pieces,win,rotations,bucket):
            key=(i,j,r,x,y)
            votes[key]=max(votes.get(key,0),v)
    if content:
        for v,i,j,r,x,y in content_candidates(pieces,k=max(12,k),rotations=rotations,neighbors=8):
            key=(i,j,r,x,y);votes[key]=max(votes.get(key,0),v)
    if segments:
        from .segments import segment_candidates
        for v,i,j,r,x,y in segment_candidates(pieces,rotations):
            key=(i,j,r,x,y);votes[key]=max(votes.get(key,0),v)
    cache={(i,k):p.rotated(k) for i,p in enumerate(pieces) for k in range(4 if rotations else 1)}
    perim=np.array([sum(len(d) for v,d in contours(p.mask)) for p in pieces])
    allv=[]
    for key,v in votes.items():
        m=verify_pair(pieces,cache,(v,*key))
        if m is not None:
            length,cost,axes=m;i,j,*_=key
            # Prefer long whole seams, discount candidates supported only by
            # small smooth contacts. Appearance is an independent cue.
            relative=length/max(1,min(perim[i],perim[j]))
            strength=(length*relative**.4)*(1+.13*np.log1p(v)) / (.009+cost)**.70
            allv.append((float(strength),v,length,cost,axes,key))
    allv.sort(reverse=True)
    # Only retain best translation hypotheses for each pair and cap degree:
    # important for low-texture repetitive staircases on weak boundaries.
    pair_count=defaultdict(int);degree=defaultdict(int);out=[]
    for strength,v,length,cost,axes,key in allv:
        i,j,*_=key
        if pair_count[i,j]>=3 or degree[i]>=36 or degree[j]>=36:continue
        pair_count[i,j]+=1;degree[i]+=1;degree[j]+=1
        out.append((max(2,int(strength*10)),*key))
    return out,len(votes)


def _shape_key(mask):
    return mask.shape,np.packbits(mask.ravel()).tobytes()


def exact_holes(pieces,poses,shape,rotations=False,rounds=4):
    """Fill a fully enclosed empty region by exact shape equivalence.

    No inferred labels, source pixels or original cut plan. A shape match is
    accepted only with zero overlap, and ambiguity is resolved by visible seams.
    """
    poses=dict(poses);lookup={p.id:p for p in pieces}
    placed=0
    for _ in range(rounds):
        rgb,labels,collision=render(pieces,poses,shape)
        if collision:break
        empty=labels<0;regs,n=label(empty)
        if not n:break
        missing=defaultdict(list)
        for p in pieces:
            if p.id in poses:continue
            for k in range(4 if rotations else 1):
                missing[_shape_key(p.rotated(k)[:,:,3]>0)].append((p.id,k))
        changed=0
        for lab,sl in enumerate(find_objects(regs),1):
            if sl is None:continue
            sy,sx=sl;y,x=sy.start,sx.start
            # Only enclosed holes have a reliable absolute mask.
            if y==0 or x==0 or sy.stop==shape[0] or sx.stop==shape[1]:continue
            mask=regs[sl]==lab
            candidates=missing.get(_shape_key(mask),[])
            if not candidates:continue
            best=None
            for i,k in candidates:
                if i in poses:continue
                a=lookup[i].rotated(k)
                yy,xx=np.where(mask)
                cost=0.;num=0
                for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                    Y=yy+y+dy;X=xx+x+dx
                    good=(Y>=0)&(X>=0)&(Y<shape[0])&(X<shape[1])
                    Y,X,gy,gx=Y[good],X[good],yy[good],xx[good]
                    ok=labels[Y,X]>=0
                    if np.any(ok):
                        cost+=np.abs(a[gy[ok],gx[ok],:3].astype(np.int16)-rgb[Y[ok],X[ok],:3].astype(np.int16)).sum()
                        num+=ok.sum()
                score=cost/max(1,num)
                if best is None or score<best[0]:best=(score,i,k)
            if best is not None:
                _,i,k=best;poses[i]=Pose(x,y,k);changed+=1
        placed+=changed
        if not changed:break
    return poses,placed


def solve_verified(pieces,shape,rotations=False,k=16,max_bucket=18,content=True,pack=True,segments=False,return_groups=False):
    from .solver import orient_to_canvas
    candidates,total=verified_candidates(pieces,rotations,k,max_bucket,content,segments)
    poses,info=assemble(pieces,candidates,shape,2,return_groups=True)
    groups=info.pop("_groups")
    if rotations:poses=orient_to_canvas(pieces,poses,shape)
    poses,filled=exact_holes(pieces,poses,shape,rotations)
    packed=0
    if pack and len(poses)<len(pieces):
        poses,packed=pack_components(pieces,poses,groups,shape,rotations)
        poses,more=exact_holes(pieces,poses,shape,rotations);filled+=more
    if return_groups:info.update(_groups=groups,_verified_candidates=candidates)
    return poses,{**info,"algorithm":"verified-contact-graph","candidate_pool":total,
                 "verified_hypotheses":len(candidates),"exact_holes_filled":filled,"component_pack_filled":packed}


def _best_fit(rgb,labels,part,rotations,limit=20):
    """Collision-free translation search in connected empty regions (FFT)."""
    from scipy.signal import fftconvolve
    H,W=labels.shape
    # Boundary contact count with existing canvas and outer boundary.
    occ=labels>=0
    ep=np.pad(occ.astype(np.float32),1,constant_values=1)
    contacts=ep[:-2,1:-1]+ep[2:,1:-1]+ep[1:-1,:-2]+ep[1:-1,2:]
    regs,num=label(~occ);slices=find_objects(regs)
    area=np.bincount(regs.ravel())
    options=[]
    for k in range(4 if rotations else 1):
        im=np.rot90(part,-k);mask=im[:,:,3]>0;h,w=mask.shape
        count=int(mask.sum())
        padded=np.pad(mask,1)
        perimeter=int(sum(np.sum(mask&~v) for v in (
                         padded[:-2,1:-1],padded[2:,1:-1],padded[1:-1,:-2],padded[1:-1,2:])))
        if not perimeter:continue
        for rid,sl in enumerate(slices,1):
            if sl is None or area[rid]<count:continue
            sy,sx=sl
            if sy.stop-sy.start<h or sx.stop-sx.start<w:continue
            # Component must lie wholly inside this exact empty region.
            block=(regs[sl]!=rid).astype(np.float32)
            kernel=mask[::-1,::-1].astype(np.float32)
            err=fftconvolve(block,kernel,mode="valid")
            valid=err<.4
            if not valid.any():continue
            match=fftconvolve(contacts[sl],kernel,mode="valid")
            match[~valid]=-1
            bestval=float(match.max())
            if bestval<.34*perimeter:continue
            # Evaluate the whole visible contact on the strongest geometric fits.
            pool=np.flatnonzero((match>=max(.34*perimeter,bestval-.08*perimeter)).ravel())
            if len(pool)>limit:
                values=match.ravel()[pool]
                pool=pool[np.argpartition(values,-limit)[-limit:]]
            yy,xx=np.where(mask)
            for ix in pool:
                y,x=np.unravel_index(ix,match.shape);x+=sx.start;y+=sy.start
                if np.any(occ[y+yy,x+xx]):continue
                total=0.;n=0
                for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                    Y=yy+y+dy;X=xx+x+dx
                    ok=(Y>=0)&(X>=0)&(Y<H)&(X<W)
                    Y,X,ay,ax=Y[ok],X[ok],yy[ok],xx[ok]
                    ok=occ[Y,X]
                    if np.any(ok):
                        total+=np.abs(im[ay[ok],ax[ok],:3].astype(np.int16)-rgb[Y[ok],X[ok],:3].astype(np.int16)).sum()
                        n+=int(ok.sum())
                c=total/(max(1,n)*3*255)
                frac=float(match.ravel()[ix]/perimeter)
                # High geometric contact + low appearance discontinuity.
                score=frac/(.045+c)
                options.append((score,frac,c,int(x),int(y),k))
    if not options:return None
    options.sort(reverse=True)
    return options[0]


def pack_components(pieces,poses,groups,shape,rotations,passes=2):
    """Place disconnected groups, then individual masks, into the anchored canvas."""
    from .geometry import R
    from .common import normalize_pose_set
    poses=dict(poses);lookup={p.id:p for p in pieces};filled=0
    _,extent=normalize_pose_set(pieces,poses)
    if tuple(extent)!=tuple(shape):return poses,0
    base=[g for g in groups if not set(g)&set(poses)]
    for rnd in range(passes):
        rgb,labels,_=render(pieces,poses,shape)
        groups0=sorted(base,key=len,reverse=True) if rnd==0 else []
        if rnd>0:
            groups0=[{p.id:Pose(0,0)} for p in pieces if p.id not in poses]
            groups0.sort(key=lambda g:lookup[next(iter(g))].mask.sum(),reverse=True)
        for g in groups0:
            if set(g)&set(poses):continue
            sel=[lookup[i] for i in g]
            g,extent=normalize_pose_set(sel,g)
            part,_,_=render(sel,g,extent)
            best=_best_fit(rgb,labels,part,rotations,limit=12)
            if best is None:continue
            _,frac,c,x,y,k=best
            # Strong geometry guard, especially on repetitive/white images.
            if frac<(.50 if len(g)>1 else .62) or c>.12:continue
            h,w=part.shape[:2]
            # Map component-local vertices to the rotated component crop.
            basecorners=np.array([[0,0],[w,0],[0,h],[w,h]])
            origin=(basecorners@R[k].T).min(axis=0)
            new={}
            for i,p in g.items():
                hh,ww=lookup[i].rotated(p.k).shape[:2]
                corners=np.array([[p.x,p.y],[p.x+ww,p.y],[p.x,p.y+hh],[p.x+ww,p.y+hh]])
                at=(corners@R[k].T).min(axis=0)-origin+np.array([x,y])
                new[i]=Pose(int(at[0]),int(at[1]),(p.k+k)%4)
            # Re-render only accepted component into the canvas, with a unique
            # placed-ID label (label identity unimportant for empty-region tests).
            for i,p in new.items():
                a=lookup[i].rotated(p.k);hh,ww=a.shape[:2];mask=a[:,:,3]>0
                reg=labels[p.y:p.y+hh,p.x:p.x+ww]
                if np.any(reg[mask]>=0):raise RuntimeError("Mask completion collision")
                reg[mask]=1;rgb[p.y:p.y+hh,p.x:p.x+ww][mask]=a[mask]
            poses.update(new);filled+=len(new)
    return poses,filled
