"""Sparse, metadata-free exact lattice and broad-phase SE(2) scan matching.

The lattice route is a fast proposal family for intact pixel-aligned tears;
the continuous solver remains available for rotated/damaged physical edges.
No known canvas, grid, ground truth or external source is used.
"""
from __future__ import annotations

from collections import defaultdict
import math
import numpy as np

from .geometry import R, propose, contours
from .resource_budget import resource_budget


def _integer_parts(p, q, r, t):
    """Place a rotated raster in p's coordinates (pixel-vertex convention)."""
    a=p.rgba
    b=np.rot90(q.rgba,-int(r)%4)
    h,w=q.rgba.shape[:2]
    x,y=(np.array([[0,0],[w,0],[0,h],[w,h]])@R[r].T).min(0)+t
    return a,b,int(x),int(y)


def _overlap_slice(a,b,x,y):
    h,w=a.shape[:2];bh,bw=b.shape[:2]
    l=max(0,x);top=max(0,y);right=min(w,x+bw);bottom=min(h,y+bh)
    if right<=l or bottom<=top:return None
    return a[top:bottom,l:right],b[top-y:bottom-y,l-x:right-x]


def _exact_contact(p,q,r,t,votes):
    """Verify zero mask collision and observed four-neighbor color continuity."""
    from ..native_core import lattice_contact
    native=lattice_contact(p.rgba,q.rgba,r,t)
    if native is False:return None
    if native is not None:
        length,med,upper=native
    else:
        a,b,x,y=_integer_parts(p,q,r,t)
        region=_overlap_slice(a,b,x,y)
        if region is not None and np.any((region[0][...,3]>0)&(region[1][...,3]>0)):
            return None
        errors=[]
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            region=_overlap_slice(a,b,x+dx,y+dy)
            if region is None:continue
            aa,bb=region;m=(aa[...,3]>0)&(bb[...,3]>0)
            if m.any():
                errors.append(np.mean(np.abs(aa[m,:3].astype(np.int16)-bb[m,:3]),axis=1))
        if not errors:return None
        errors=np.concatenate(errors)
        length=len(errors);med=float(np.median(errors));upper=float(np.percentile(errors,80))
    # Exact shape votes are necessary but not sufficient. Short repeated
    # patterns and high seam discontinuity are not reliable large-scale joins.
    if length<12 or votes<3 or med>32 or upper>55:return None
    score=(length+.7*min(votes,length))*math.exp(-med/32-upper/100)
    if score<12:return None
    return score,{"contact_length":float(length),"median_gap":0.0,
                  "interior_overlap":0.0,"color_difference":med,
                  "seam_bend_rms":0.0,"votes":int(votes),"exact_lattice":True}


def _packed_contours(pieces,budget):
    # Grow one contiguous byte buffer and expose it as a NumPy view. A list
    # of individual arrays followed by concatenate held *two* complete
    # contour streams and became the bottleneck on 100k fragments.
    data=bytearray()
    # A browser needs temporary bridge copies, but not a fixed third of RAM:
    # reserve raster storage and bridge copies first, then use the remaining
    # budget for index shards and votes.
    browser_pack_limit=max(0,(budget.memory_bytes-sum(p.rgba.nbytes for p in pieces)-16*1024**2)//4) if budget.browser else None
    for i,p in enumerate(pieces):
        for v,d in contours(p.mask):
            row=np.empty(2+3*len(d),np.int32)
            row[:2]=i,len(d);row[2:].reshape(-1,3)[:,:2]=v;row[4::3]=d
            needed=len(data)+row.nbytes
            # Account for bytearray growth; WASM additionally transfers the
            # whole contiguous stream to its separate linear memory.
            if (needed>browser_pack_limit if budget.browser else needed*1.15>budget.packed_bytes):
                raise MemoryError("Packed contours exceed available memory budget; reduce scan resolution or use a larger-memory runner")
            data.extend(memoryview(row).cast("B"))
    return np.frombuffer(data,dtype=np.int32)


def lattice_candidates(pieces, update=lambda _:None, *, degree=12, budget=None):
    """Compact sparse exact boundary votes; bounded memory and degree."""
    from ..native_core import arc_votes,arc_votes_available
    budget=budget or resource_budget()
    median=float(np.median([min(p.rgba.shape[:2]) for p in pieces]))
    windows=(32,20) if median>=36 else (24,14)
    native=arc_votes_available()
    record_cap,vote_cap=budget.arc_records,budget.vote_entries
    if native:
        packed=_packed_contours(pieces,budget)
        if budget.browser:
            remain=budget.memory_bytes-sum(p.rgba.nbytes for p in pieces)-3*packed.nbytes-24*1024**2
            if remain<8*1024**2:raise MemoryError("Browser contour transfer leaves insufficient memory for the arc index")
            record_cap=max(1024,min(record_cap,int(remain*.45)//48))
            vote_cap=max(1024,min(vote_cap,int(remain*.45)//72))
        shards=max(1,math.ceil((len(packed)/3)/(record_cap*.85)))
    elif sum(2*sum(p.rgba.shape[:2]) for p in pieces)*224>budget.memory_bytes//2:
        raise MemoryError("Python arc index would exceed the memory budget; build the native core or reduce input resolution")
    results=[];passes=[]
    for k in windows:
        update(f"Indexing exact torn edges (arc window {k})")
        if native and len(packed):
            for cap,sh in ((6,shards),(6,2*shards),(4,2*shards),(3,2*shards),(2,2*shards)):
                try:
                    rows=arc_votes(packed,k,cap,sh,record_cap,vote_cap);break
                except MemoryError:continue
            else:raise MemoryError("Exact arc index exhausted its adaptive record/vote budget")
            passes.append({"window":k,"shards":sh,"bucket":cap,"record_cap":record_cap,"vote_cap":vote_cap})
        elif native:
            rows=np.empty((0,6),np.int32)
        else:
            rows=np.asarray([q for q in propose(pieces,k,True,max_bucket=6) if q[0]>=3],np.int32).reshape(-1,6)
        results.append(rows)
    rows=np.concatenate(results)
    if len(rows):
        # Preserve deterministic legacy ordering, including top-degree ties.
        order=np.lexsort((-rows[:,0],rows[:,5],rows[:,4],rows[:,3],rows[:,2],rows[:,1]))
        rows=rows[order];rows=rows[np.r_[True,np.any(rows[1:,1:]!=rows[:-1,1:],axis=1)]]
        owners=np.r_[rows[:,1],rows[:,2]]
        ids=np.tile(np.arange(len(rows),dtype=np.int32),2)
        order=np.lexsort((-ids,-rows[ids,0],owners))
        owner=owners[order];idx=np.arange(len(order))
        starts=np.maximum.accumulate(np.where(np.r_[True,owner[1:]!=owner[:-1]],idx,0))
        chosen=np.unique(ids[order[idx-starts<degree]])
        chosen=chosen[np.lexsort((chosen,-rows[chosen,0]))]
    else:chosen=np.empty(0,np.int32)
    candidates=[]
    for idx in chosen:
        votes,i,j,r,tx,ty=map(int,rows[idx])
        t=np.array([tx,ty],float)
        match=_exact_contact(pieces[i],pieces[j],r,t,votes)
        if match:
            score,info=match
            candidates.append((score,i,j,R[r].astype(float),t,info))
    verified_pairs=len(np.unique(rows[chosen,1:3],axis=0))
    return candidates,{"arc_windows":list(windows),"arc_backend":"compact-native" if native else "python",
                       "arc_passes":passes,"packed_contour_bytes":packed.nbytes if native else None,"voted_poses":len(rows),"verified_poses":len(chosen),
                       "verified_pairs":verified_pairs,"accepted_candidates":len(candidates)}


def lattice_color_rescue(pieces,existing,update=lambda _:None,*,degree=12):
    """Resolve repeated shape arcs using their observed edge colors.

    This second index is attempted only for incomplete lattice assemblies.
    A different appearance vote is not sufficient unless the same exact
    zero-overlap raster and seam-color checks pass.
    """
    from .content import content_candidates
    median=float(np.median([min(p.rgba.shape[:2]) for p in pieces]))
    k=24 if median>=36 else 16
    update("Resolving repeated lattice arcs by observed edge color")
    known={(i,j,int(round(math.atan2(RR[1,0],RR[0,0])/(math.pi/2)))%4,*np.rint(t).astype(int))
           for _,i,j,RR,t,_ in existing}
    raw=content_candidates(pieces,k,rotations=True,neighbors=4,degree_limit=degree)
    result=[]
    for votes,i,j,r,tx,ty in raw:
        if (i,j,r,tx,ty) in known:continue
        if votes<3:continue
        t=np.array([tx,ty],float);match=_exact_contact(pieces[i],pieces[j],r,t,votes)
        if match:
            score,info=match
            result.append((score*.85,i,j,R[r].astype(float),t,{**info,"appearance_retrieval":True}))
    return result,{"window":k,"verified":len(raw),"accepted":len(result)}


def filter_continuous(shapes,candidates):
    """Require distinctive local evidence or an independent closed cycle.

    Competing poses on a different boundary arc are not rivals. This filter
    applies only to the large-set continuous fallback, not intact lattice
    joins or the previously validated small physical solver.
    """
    from .continuation import _closed_triangles
    if not candidates:return []
    # Long curves alone can have a high geometric score even when the
    # opposing colors vary in unrelated ways. Rank competitors using
    # observed along-seam covariance as well as the original geometry score.
    candidates=[(s*(.30+.70*max(0.,info["profile_similarity"]))
                  *math.exp(-info["color_difference"]/45.),i,j,R,t,
                  {**info,"raw_contact_score":float(s)}) for s,i,j,R,t,info in candidates]
    incident=defaultdict(list)
    for z,(score,i,j,R,t,info) in enumerate(candidates):
        for u,k in ((i,"a"),(j,"b")):
            incident[u].append((z,info["arc_"+k],info["arc_fraction_"+k]))
    loops=_closed_triangles(shapes,candidates)
    result=[]
    for z,(s,i,j,R,t,info) in enumerate(candidates):
        margin=[]
        for u,k in ((i,"a"),(j,"b")):
            n=len(shapes[u].points);arc=info["arc_"+k];fraction=info["arc_fraction_"+k]
            rival=1.
            for other,pos,frac in incident[u]:
                if other==z:continue
                os,oi,oj,orr,ot,_=candidates[other]
                if (i,j)==(oi,oj) and np.linalg.norm(R-orr)<.16 and np.linalg.norm(t-ot)<7.:continue
                d=abs(pos-arc);d=min(d,n-d)
                if d<n*max(.05,min(.25,.36*(fraction+frac))):rival=max(rival,os)
            margin.append(s/rival)
        uniqueness=min(margin)
        line_correlated=info["texture"]>=6 and info["profile_similarity"]>=.45 and info["color_difference"]<=34.
        shape_distinctive=(info["seam_bend_rms"]>=4. and info["contact_length"]>=32. and
                           info["median_gap"]<=1.5 and info["color_difference"]<=22.)
        reliable=(uniqueness>=1.08 and s>=5. and (line_correlated or shape_distinctive))
        corroborated=loops[z]>0 and uniqueness>=.95 and info["color_difference"]<=30.
        if reliable or corroborated:
            score=s*min(1.5,uniqueness)*(1.+.25*min(3,int(loops[z])))
            result.append((score,i,j,R,t,{**info,"seam_uniqueness":float(uniqueness),"loop_support":int(loops[z])}))
    return result


def full_mask_overlap(a,b,R,t):
    """Check observed foreground, including thin tears and surviving islands."""
    def fraction(source,target,Q,u):
        yy,xx=np.nonzero(source.fragment.mask)
        stride=max(1,math.ceil(len(xx)/1600))
        if not len(xx):return 0.
        xy=np.column_stack([xx[::stride]+.5,yy[::stride]+.5])@Q.T+u
        x,y=np.floor(xy).astype(int).T
        mask=target.fragment.mask;h,w=mask.shape
        good=(x>=0)&(y>=0)&(x<w)&(y<h)
        return np.count_nonzero(mask[y[good],x[good]])/len(x)
    return max(fraction(b,a,R,t),fraction(a,b,R.T,-R.T@t))


class SpatialBoxes:
    """Conservative spatial hash over full transformed raster bounding boxes.

    Large boxes use a separate list; no candidate overlap is skipped.
    """
    def __init__(self,cell,shapes):
        self.cell=max(8.,float(cell));self.shapes=shapes
        self.buckets=defaultdict(list);self.big=[];self.all=[]

    def bounds(self,i,R,t):
        h,w=self.shapes[i].fragment.rgba.shape[:2]
        p=np.array([[0,0],[w,0],[0,h],[w,h]])@R.T+t
        return p.min(0),p.max(0)

    def cells(self,box):
        a,b=box
        x,y=np.floor(a/self.cell).astype(int);u,v=np.floor(b/self.cell).astype(int)
        if (u-x+1)*(v-y+1)>256:return None
        return [(i,j) for i in range(x,u+1) for j in range(y,v+1)]

    def add(self,i,box):
        self.all.append(i)
        keys=self.cells(box)
        if keys is None:self.big.append(i)
        else:
            for key in keys:self.buckets[key].append(i)

    def query(self,box):
        keys=self.cells(box)
        if keys is None:return self.all
        hits=set(self.big)
        for key in keys:hits.update(self.buckets.get(key,()))
        return sorted(hits)


def strict_lattice_overlap(a,b,Rrel,t):
    """Exact foreground occupancy for two quarter-turn integer poses."""
    r=int(round(math.atan2(Rrel[1,0],Rrel[0,0])/(math.pi/2)))%4
    if not np.allclose(Rrel,R[r],atol=1.e-6) or not np.allclose(t,np.rint(t),atol=1.e-6):
        return None
    A,B,x,y=_integer_parts(a.fragment,b.fragment,r,np.rint(t))
    region=_overlap_slice(A,B,x,y)
    return bool(region is not None and np.any((region[0][...,3]>0)&(region[1][...,3]>0)))


def merge_spatial(shapes,candidates,overlap,*,strict=False):
    """Union by size, only check nearby cross-component occupied rasters."""
    n=len(shapes)
    owners=list(range(n))
    groups={i:{i:(np.eye(2),np.zeros(2))} for i in range(n)}
    cell=float(np.median([min(s.fragment.rgba.shape[:2]) for s in shapes]))
    indices={i:SpatialBoxes(cell,shapes) for i in range(n)}
    for i,ix in indices.items():ix.add(i,ix.bounds(i,*groups[i][i]))
    joins=[];checks=0;rejected=0
    for score,i,j,R,t,info in sorted(candidates,key=lambda c:-c[0]):
        x,y=owners[i],owners[j]
        if x==y:continue
        A,B=groups[x],groups[y]
        Ri,ti=A[i];Rj,tj=B[j]
        Q=Ri@R@Rj.T;u=Ri@t+ti-Q@tj
        if len(A)<len(B):
            x,y=y,x;A,B=B,A;Q,u=Q.T,-Q.T@u
        ix=indices[x]
        mapped={k:(Q@S,Q@p+u) for k,(S,p) in B.items()}
        bounds={l:ix.bounds(l,S,s) for l,(S,s) in mapped.items()}
        collision=False
        for l,(S,s) in mapped.items():
            for k in ix.query(bounds[l]):
                P,p=A[k];rr=P.T@S;tt=P.T@(s-p)
                checks+=1
                exact=strict_lattice_overlap(shapes[k],shapes[l],rr,tt) if strict else None
                collides=(exact if exact is not None else
                          overlap(shapes[k],shapes[l],rr,tt)>.035 or
                          full_mask_overlap(shapes[k],shapes[l],rr,tt)>.045)
                if collides:
                    collision=True;break
            if collision:break
        if collision:rejected+=1;continue
        A.update(mapped)
        for k in B:
            ix.add(k,bounds[k]);owners[k]=x
        del groups[y];del indices[y]
        joins.append({"a":shapes[i].fragment.id,"b":shapes[j].fragment.id,"score":float(score),**info})
        if len(groups)==1:break
    result=sorted(groups.values(),key=lambda g:(-len(g),-sum(shapes[k].fragment.mask.sum() for k in g)))
    return result,joins,{"spatial_overlap_tests":checks,"collision_rejections":rejected}


def _rot(r,x,y):
    return ((x,y),(-y,x),(-x,-y),(y,-x))[r]


def _quarter(matrix):
    x,y=matrix[0,0],matrix[1,0]
    return 0 if x>.5 else 1 if y>.5 else 2 if x<-.5 else 3


def lattice_box(piece,pose):
    r,x,y=pose;h,w=piece.rgba.shape[:2]
    a,b,c,d=((0,0,w,h),(-h,0,0,w),(-w,-h,0,0),(0,-w,h,0))[r]
    return x+a,y+b,x+c,y+d


def merge_lattice(pieces,candidates,*,pixel_cap=None):
    """Integer SE(2), union by size, exact overlap; allocate spatial bins lazily."""
    n=len(pieces);owners=list(range(n));groups={i:{i:(0,0,0)} for i in range(n)}
    cell=max(8,int(np.median([min(p.rgba.shape[:2]) for p in pieces])))
    boxes=[lattice_box(p,(0,0,0)) for p in pieces]
    extents=dict(enumerate(boxes))
    indices={};large_boxes={}
    joins=[];checks=0;rejected=0;canvas_rejections=0
    def cells(b):
        x,y,u,v=b;x//=cell;y//=cell;u=(u-1)//cell;v=(v-1)//cell
        if (u-x+1)*(v-y+1)>256:return None
        return [(a,b) for a in range(x,u+1) for b in range(y,v+1)]
    def add(owner,k,b):
        keys=cells(b)
        if keys is None:large_boxes.setdefault(owner,[]).append(k)
        else:
            for key in keys:indices[owner].setdefault(key,[]).append(k)
    for score,i,j,rr,tt,info in sorted(candidates,key=lambda c:-c[0]):
        x,y=owners[i],owners[j]
        if x==y:continue
        A,B=groups[x],groups[y];ri,xi,yi=A[i];rj,xj,yj=B[j]
        q=(ri+_quarter(rr)-rj)%4
        u,v=_rot(ri,int(round(tt[0])),int(round(tt[1])));a,b=_rot(q,xj,yj);u+=xi-a;v+=yi-b
        if len(A)<len(B):
            x,y=y,x;A,B=B,A;q=(-q)%4;u,v=_rot(q,-u,-v)
        mapped={}
        for k,(r,px,py) in B.items():
            ax,ay=_rot(q,px,py);mapped[k]=((q+r)%4,ax+u,ay+v)
        bb={k:lattice_box(pieces[k],pose) for k,pose in mapped.items()}
        E=extents[x]
        extent=(min(E[0],min(b[0] for b in bb.values())),min(E[1],min(b[1] for b in bb.values())),max(E[2],max(b[2] for b in bb.values())),max(E[3],max(b[3] for b in bb.values())))
        if pixel_cap is not None and (extent[2]-extent[0]+6)*(extent[3]-extent[1]+6)>pixel_cap:
            canvas_rejections+=1;continue
        if x not in indices:
            indices[x]={}
            for k in A:add(x,k,boxes[k])
        collision=False
        for l,box in bb.items():
            keys=cells(box)
            if keys is None:hits=A
            else:
                hits=set(large_boxes.get(x,()))
                for key in keys:hits.update(indices[x].get(key,()))
            for k in hits:
                old=boxes[k]
                lft,top=max(box[0],old[0]),max(box[1],old[1]);rt,bot=min(box[2],old[2]),min(box[3],old[3])
                if lft>=rt or top>=bot:continue
                checks+=1
                ma=np.rot90(pieces[k].rgba[...,3],-A[k][0]%4)[top-old[1]:bot-old[1],lft-old[0]:rt-old[0]]
                mb=np.rot90(pieces[l].rgba[...,3],-mapped[l][0]%4)[top-box[1]:bot-box[1],lft-box[0]:rt-box[0]]
                if np.any((ma>0)&(mb>0)):collision=True;break
            if collision:break
        if collision:rejected+=1;continue
        A.update(mapped);extents[x]=extent
        for k,b in bb.items():boxes[k]=b;add(x,k,b);owners[k]=x
        del groups[y];del extents[y];indices.pop(y,None);large_boxes.pop(y,None)
        joins.append({"a":pieces[i].id,"b":pieces[j].id,"score":float(score),**info})
        if len(groups)==1:break
    result=sorted(groups.values(),key=lambda g:(-len(g),-sum(np.count_nonzero(pieces[k].rgba[...,3]) for k in g)))
    result=[{k:(R[r].astype(float),np.array([x,y],float)) for k,(r,x,y) in g.items()} for g in result]
    return result,joins,{"spatial_overlap_tests":checks,"collision_rejections":rejected,"canvas_budget_rejections":canvas_rejections,"integer_merge":True}
