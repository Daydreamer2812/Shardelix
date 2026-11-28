"""Robust edge verification for slightly damaged *digital* masks.

A bounded boundary-only overlap is permitted during graph assembly.
Interior overlap is rejected. Ownership conflicts are resolved explicitly and
reported; no content is synthesized and no ground truth is read.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from scipy.ndimage import binary_erosion
from shardelix.reconstruction.common import Fragment,Pose,normalize_pose_set
from shardelix.reconstruction.irregular.lattice import R,Component,_vertex_to_pose,propose,contours
from shardelix.reconstruction.irregular.verified import _overlap
from shardelix.reconstruction.irregular.content import content_candidates
from shardelix.reconstruction.quality import orient_to_canvas

def tolerant_candidates(pieces,rotations):
    votes={}
    for k,bucket in [(18,12),(24,8),(12,12)]:
        for v,*key in propose(pieces,k,rotations,bucket):
            key=tuple(key);votes[key]=max(v,votes.get(key,0))
    # Independent photometric descriptor for repeated/short arcs.
    for v,*key in content_candidates(pieces,k=16,rotations=rotations,neighbors=8):
        key=tuple(key);votes[key]=max(v,votes.get(key,0))
    cache={(i,k):p.rotated(k) for i,p in enumerate(pieces) for k in range(4 if rotations else 1)}
    perim=np.array([sum(len(d) for _,d in contours(p.mask)) for p in pieces])
    areas=np.array([p.mask.sum() for p in pieces])
    cores={(i,k):binary_erosion(a[:,:,3]>0) for (i,k),a in cache.items()}
    out=[]
    for key,v in votes.items():
        i,j,r,tx,ty=key;a=cache[i,0];b=cache[j,r]
        p=_vertex_to_pose(pieces[j],r,np.array([tx,ty]));x,y=p.x,p.y
        windows=_overlap(a,b,x,y);coll=0
        if windows is not None:
            aa,bb=windows;coll=int(((aa[:,:,3]>0)&(bb[:,:,3]>0)).sum())
            if coll>max(3,.018*min(areas[i],areas[j])):continue
            cc,dd=_overlap(cores[i,0],cores[j,r],x,y)
            if (cc&dd).any():continue
        length=0;cost=0.
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            win=_overlap(a,b,x+dx,y+dy)
            if win is None:continue
            aa,bb=win;m=(aa[:,:,3]>0)&(bb[:,:,3]>0);n=int(m.sum())
            if n:
                length+=n;cost+=float(np.abs(aa[m,:3].astype(np.int16)-bb[m,:3]).sum())
        if length<max(7,.065*min(perim[i],perim[j])) or coll>max(3,.18*length):continue
        cost/=length*3*255
        relative=length/max(1,min(perim[i],perim[j]))
        strength=length*relative**.5*(1+.16*np.log1p(v))/(.012+cost)**.7
        strength*=np.exp(-4*coll/max(1,length))
        out.append((float(strength),key))
    out.sort(reverse=True)
    degree=defaultdict(int);pair=defaultdict(int);final=[]
    for strength,key in out:
        i,j,*_=key
        if degree[i]>=40 or degree[j]>=40 or pair[i,j]>=3:continue
        degree[i]+=1;degree[j]+=1;pair[i,j]+=1;final.append((strength,*key))
    return final,len(votes)

def _combine(a,b,r,t,shape):
    bh,bw=b.mask.shape
    corners=np.array([[0,0],[bw,0],[0,bh],[bw,bh]])+b.origin
    bmin=(corners@R[r].T).min(axis=0)+t;bm=np.rot90(b.mask,-r%4)
    amin=a.origin;ah,aw=a.mask.shape;bh,bw=bm.shape
    lo=np.minimum(amin,bmin);hi=np.maximum(amin+[aw,ah],bmin+[bw,bh])
    width,height=(hi-lo).tolist();h,w=shape
    if not ((width<=w and height<=h) or (width<=h and height<=w)):return None
    ax,ay=(amin-lo).tolist();bx,by=(bmin-lo).tolist()
    ix0=max(ax,bx);iy0=max(ay,by);ix1=min(ax+aw,bx+bw);iy1=min(ay+ah,by+bh)
    if ix0<ix1 and iy0<iy1:
        aa=a.mask[iy0-ay:iy1-ay,ix0-ax:ix1-ax];bb=bm[iy0-by:iy1-by,ix0-bx:ix1-bx]
        overlap=aa&bb;count=int(overlap.sum())
        if count:
            if count>max(4,.01*min(a.mask.sum(),bm.sum())):return None
            # A core-to-core overlap signals false placement, not aliasing.
            ac=binary_erosion(a.mask)[iy0-ay:iy1-ay,ix0-ax:ix1-ax]
            bc=binary_erosion(bm)[iy0-by:iy1-by,ix0-bx:ix1-bx]
            if (ac&bc).any():return None
    # All collision and extent checks passed. Mutate only the owned destination
    # component; the caller consumes both components after a successful join.
    # Avoid copying a growing dictionary on every union-by-size merge.
    if ax==ay==0 and (height,width)==a.mask.shape:
        mask=a.mask
    else:
        mask=np.zeros((height,width),bool)
        mask[ay:ay+ah,ax:ax+aw]=a.mask
    mask[by:by+bh,bx:bx+bw]|=bm
    poses=a.poses
    for j,(s,u) in b.poses.items():poses[j]=((r+s)%4,R[r]@u+t)
    return Component(poses,mask,lo)

def solve_tolerant(pieces,shape,rotations,precomputed=None,return_groups=False):
    candidates,total=precomputed if precomputed is not None else tolerant_candidates(pieces,rotations)
    comps={i:Component({i:(0,np.zeros(2,dtype=int))},p.mask.copy(),np.zeros(2,dtype=int)) for i,p in enumerate(pieces)}
    owners=list(range(len(pieces)));accepted=rejected=0
    for strength,i,j,r,tx,ty in candidates:
        ai,bi=owners[i],owners[j]
        if ai==bi:continue
        a,b=comps[ai],comps[bi];ri,ti=a.poses[i];rj,tj=b.poses[j]
        rel=(ri+r-rj)%4;shift=R[ri]@np.array([tx,ty])+ti-R[rel]@tj
        if len(a.poses)<len(b.poses):
            ai,bi=bi,ai;a,b=b,a;rel=(-rel)%4;shift=-R[rel]@shift
        g=_combine(a,b,rel,shift,shape)
        if g is None:rejected+=1;continue
        for q in b.poses:owners[q]=ai
        comps[ai]=g;del comps[bi];accepted+=1
        if len(comps)==1:break
    largest=max(comps.values(),key=lambda c:len(c.poses))
    poses={pieces[i].id:_vertex_to_pose(pieces[i],r,t) for i,(r,t) in largest.poses.items()}
    poses,extent=normalize_pose_set(pieces,poses)
    if rotations:poses=orient_to_canvas(pieces,poses,shape)
    d=dict(algorithm="tolerant-mask-graph",components=len(comps),candidate_pool=total,
           verified_hypotheses=len(candidates),accepted=accepted,rejected=rejected,extent=list(extent))
    if return_groups:
        groups=[]
        for comp in comps.values():
            if comp is largest:continue
            q={pieces[i].id:_vertex_to_pose(pieces[i],r,t) for i,(r,t) in comp.poses.items()}
            q,_=normalize_pose_set(pieces,q);groups.append(q)
        d["_groups"]=groups
    return poses,d

def repair_boundary_ownership(pieces,poses,shape):
    """Allocate each collided boundary pixel to one owner by distance-to-edge.

    Keep the normalized crop shape and pose coordinate system unchanged.
    No attempt to inpaint missing pixels. Returns raw conflict count.
    """
    h,w=shape;owner=np.full(shape,-1,np.int32);confidence=np.zeros(shape,np.uint8);raw=0;cores=0
    cached=[]
    for j,p in enumerate(pieces):
        pose=poses.get(p.id)
        if pose is None:cached.append(None);continue
        a=p.rotated(pose.k);hh,ww=a.shape[:2];x,y=pose.x,pose.y
        if x<0 or y<0 or x+ww>w or y+hh>h:cached.append(None);continue
        m=a[:,:,3]>0;core=binary_erosion(m);value=m.astype(np.uint8)+core.astype(np.uint8)
        box=owner[y:y+hh,x:x+ww];c=confidence[y:y+hh,x:x+ww]
        conflict=m&(box>=0);raw+=int(conflict.sum());cores+=int((conflict&core&(c==2)).sum())
        win=m&((box<0)|(value>c));box[win]=j;c[win]=value[win];cached.append((a,pose))
    out=[]
    for j,(p,item) in enumerate(zip(pieces,cached)):
        if item is None:out.append(p);continue
        a,pose=item;a=a.copy();hh,ww=a.shape[:2];reg=owner[pose.y:pose.y+hh,pose.x:pose.x+ww]
        a[(a[:,:,3]>0)&(reg!=j)]=0
        out.append(Fragment(p.id,np.rot90(a,pose.k).copy()))
    return out,raw,cores
