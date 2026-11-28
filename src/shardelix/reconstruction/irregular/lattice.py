"""Blind exact complementary-boundary hashing and collision-free rigid assembly."""
from __future__ import annotations

from collections import defaultdict, Counter
from dataclasses import dataclass
import numpy as np

from shardelix.reconstruction.common import Fragment, Pose, normalize_pose_set

# Rotations are on pixel vertices, in image coordinates (x right, y down).
R=np.array([[[1,0],[0,1]],[[0,-1],[1,0]],[[-1,0],[0,-1]],[[0,1],[-1,0]]],dtype=int)
DIR=np.array([[1,0],[0,1],[-1,0],[0,-1]],dtype=int)


def _contours_reference(mask: np.ndarray) -> list[tuple[np.ndarray,np.ndarray]]:
    """Oriented unit vertex edges, clockwise around foreground, exact to one pixel."""
    padded=np.pad(mask,1)
    sides=[mask & ~padded[:-2,1:-1],mask & ~padded[1:-1,2:],
           mask & ~padded[2:,1:-1],mask & ~padded[1:-1,:-2]]
    offsets=[(0,0),(1,0),(1,1),(0,1)]
    edges={}
    for d,s in enumerate(sides):
        yy,xx=np.nonzero(s)
        ox,oy=offsets[d]
        for x,y in zip(xx+ox,yy+oy):
            key=(int(x),int(y))
            edges.setdefault(key,set()).add(d)
    result=[]
    while edges:
        start=next(iter(edges))
        p=start
        prev=None
        vs=[]; ds=[]
        while p in edges:
            avail=edges[p]
            d=min(avail) if prev is None else next(k for k in ((prev+1)%4,prev,(prev-1)%4,(prev+2)%4) if k in avail)
            avail.remove(d)
            if not avail: del edges[p]
            vs.append(p);ds.append(d)
            p=(p[0]+int(DIR[d,0]),p[1]+int(DIR[d,1]))
            prev=d
            if p==start: break
        if p==start and len(ds)>=4:
            result.append((np.array(vs,dtype=np.int32),np.array(ds,dtype=np.uint8)))
    return result


def contours(mask: np.ndarray):
    """Exact ordered contours, using C++ when available."""
    from shardelix.native_core import contours as accelerated
    result=accelerated(mask)
    return _contours_reference(mask) if result is None else result

def _keys(v: np.ndarray,d: np.ndarray,k: int,step: int,rotations: bool):
    """Vectorized rolling arc descriptors; same exact hashes as the legacy loop."""
    n=len(d)
    if n<k+1:return
    from numpy.lib.stride_tricks import sliding_window_view
    dd=np.r_[d,d[:k-1]]
    windows=sliding_window_view(dd,k)[::step]
    starts=np.arange(0,n,step)
    valid=np.count_nonzero(windows[:,1:]!=windows[:,:-1],axis=1)>=4
    if not valid.any():return
    w=windows[valid];starts=starts[valid]
    rev=(w[:,::-1]+2)%4
    norm=(w.astype(np.int16)-w[:,:1])%4 if rotations else w
    rnorm=(rev.astype(np.int16)-rev[:,:1])%4 if rotations else rev
    norm=norm.astype(np.uint8);rnorm=rnorm.astype(np.uint8)
    for j,i in enumerate(starts):
        yield (norm[j].tobytes(),rnorm[j].tobytes(),tuple(map(int,v[i])),
               tuple(map(int,v[(i+k)%n])),int(w[j,0]),int(rev[j,0]))


def propose(pieces: list[Fragment], k: int=24, rotations: bool=False,
            max_bucket: int=5) -> list[tuple[int,int,int,int,int,int]]:
    """Votes for (i,j,r,tx,ty) where q_i = R[r] q_j + t.

    Frequent patterns are ambiguous and skipped. Reverse complementary arc
    matching uses pixel vertices, not pixels, so adjacent masks do not overlap.
    """
    index=defaultdict(list)
    for i,p in enumerate(pieces):
        for v,d in contours(p.mask):
            for key,revkey,a,b,da,db in _keys(v,d,k,1,rotations):
                bucket=index[key]
                if len(bucket)<=max_bucket:
                    bucket.append((i,a,da,revkey,b,db))
    votes=Counter()
    for key,bucket in index.items():
        if len(bucket)>max_bucket: continue
        for i,a,da,revkey,_,_ in bucket:
            other=index.get(revkey,[])
            if len(other)>max_bucket:continue
            for j,_,_,_,b,db in other:
                if i>=j:continue
                r=(da-db)%4
                if r and not rotations:continue
                x,y=b;xx,yy=((x,y),(-y,x),(-x,-y),(y,-x))[r]
                votes[i,j,r,a[0]-xx,a[1]-yy]+=1
    return [(v,*key) for key,v in votes.most_common()]


def _vertex_to_pose(piece: Fragment,r: int,t: np.ndarray) -> Pose:
    h,w=piece.mask.shape
    corners=np.array([[0,0],[w,0],[0,h],[w,h]])
    mn=(corners@R[r].T).min(axis=0)+t
    return Pose(int(mn[0]),int(mn[1]),r)


@dataclass
class Component:
    # local piece -> (quarterturn, vertex origin)
    poses: dict
    mask: np.ndarray
    # occupied raster origin in component coordinates
    origin: np.ndarray


def _combine(a: Component,b: Component,r: int,t: np.ndarray, max_shape: tuple[int,int]) -> Component|None:
    bh,bw=b.mask.shape
    source_corners=np.array([[0,0],[bw,0],[0,bh],[bw,bh]])+b.origin
    bmin=(source_corners@R[r].T).min(axis=0)+t
    bm=np.rot90(b.mask,-r%4)
    amin=a.origin
    ah,aw=a.mask.shape; bh,bw=bm.shape
    lo=np.minimum(amin,bmin); hi=np.maximum(amin+[aw,ah],bmin+[bw,bh])
    width,height=(hi-lo).tolist()
    # Permit global quarter-turn ambiguity, but never a larger canvas.
    h,w=max_shape
    if not ((width<=w and height<=h) or (width<=h and height<=w)):
        return None
    ax,ay=(amin-lo).tolist(); bx,by=(bmin-lo).tolist()
    ix0=max(ax,bx); iy0=max(ay,by);ix1=min(ax+aw,bx+bw);iy1=min(ay+ah,by+bh)
    if ix0<ix1 and iy0<iy1 and np.any(a.mask[iy0-ay:iy1-ay,ix0-ax:ix1-ax]&bm[iy0-by:iy1-by,ix0-bx:ix1-bx]):
        return None
    # At least one true 4-neighbor boundary (no purely hypothetical join).
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
    for j,(s,u) in b.poses.items():
        poses[j]=((r+s)%4,R[r]@u+t)
    return Component(poses,mask,lo)


def assemble(pieces: list[Fragment], candidates, shape: tuple[int,int],
             min_votes: int=2,return_groups:bool=False) -> tuple[dict[str,Pose],dict]:
    comps={i:Component({i:(0,np.zeros(2,dtype=int))},p.mask.copy(),np.zeros(2,dtype=int)) for i,p in enumerate(pieces)}
    owners=list(range(len(pieces)))
    accepted=0; rejected=0
    for votes,i,j,r,tx,ty in candidates:
        if votes<min_votes: break
        ai,bi=owners[i],owners[j]
        if ai==bi: continue
        a,b=comps[ai],comps[bi]
        ri,ti=a.poses[i];rj,tj=b.poses[j]
        # q_A = R[ri] (R[r] q_j+t_ij)+ti; q_j = R[-rj](q_B-tj).
        rel=(ri+r-rj)%4
        shift=R[ri]@np.array([tx,ty])+ti-R[rel]@tj
        if len(a.poses)<len(b.poses):
            ai,bi=bi,ai;a,b=b,a
            rel=(-rel)%4;shift=-R[rel]@shift
        joined=_combine(a,b,rel,shift,shape)
        if joined is None:
            rejected+=1
            continue
        for q in b.poses:owners[q]=ai
        comps[ai]=joined;del comps[bi];accepted+=1
        if len(comps)==1:break
    largest=max(comps.values(),key=lambda c:len(c.poses))
    poses={pieces[i].id:_vertex_to_pose(pieces[i],r,t) for i,(r,t) in largest.poses.items()}
    poses,extent=normalize_pose_set(pieces,poses)
    diag={"algorithm":"blind-complementary-arc-graph","components":len(comps),
          "largest_component":len(poses),"accepted":accepted,"rejected":rejected,
          "extent":list(extent)}
    if return_groups:
        groups=[]
        for comp in comps.values():
            gp={pieces[i].id:_vertex_to_pose(pieces[i],r,t) for i,(r,t) in comp.poses.items()}
            groups.append(normalize_pose_set(pieces,gp)[0])
        diag["_groups"]=groups
    return poses,diag


def solve_geometry(pieces: list[Fragment],shape: tuple[int,int],rotations: bool=False,
                   k: int=24,min_votes:int=2,max_bucket:int=5,content:bool=True):
    candidates=propose(pieces,k,rotations,max_bucket)
    poses,diag=assemble(pieces,candidates,shape,min_votes)
    # Retrieve repeated shape/colour arcs only when shape evidence alone failed.
    # A complete mask solution is never modified by weaker image evidence.
    content_count=0
    if content and len(poses)<.96*len(pieces):
        from shardelix.reconstruction.irregular.content import content_candidates
        extra=content_candidates(pieces,k=max(20,k),rotations=rotations)
        content_count=len(extra)
        # Strong exact rare-shape hypotheses dominate weak repetitive arcs.
        mixed=sorted([(v*6,i,j,r,x,y) for v,i,j,r,x,y in candidates if v>=min_votes]+extra,reverse=True)
        alternative,info=assemble(pieces,mixed,shape,min_votes=2)
        if len(alternative)>len(poses):
            poses,diag=alternative,info
            diag["algorithm"]="blind-shape-content-graph"
    diag.update({"candidates":len(candidates),"content_candidates":content_count,
                 "arc_window":k,"min_votes":min_votes,"max_bucket":max_bucket})
    return poses,diag
