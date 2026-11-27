"""Sparse, metadata-free exact lattice and broad-phase SE(2) scan matching.

The lattice route is a fast proposal family for intact pixel-aligned tears;
the continuous solver remains available for rotated/damaged physical edges.
No known canvas, grid, ground truth or external source is used.
"""
from __future__ import annotations

from collections import defaultdict
import math
import numpy as np

from .geometry import R, propose


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


def lattice_candidates(pieces, update=lambda _:None, *, degree=12):
    """Get sparse rigid translations from complementary integer boundary arcs.

    Bounded arc buckets suppress frequent nondistinctive patterns. Votes from
    long and short windows are combined by maximum, not double-counted.
    """
    median=float(np.median([min(p.rgba.shape[:2]) for p in pieces]))
    windows=(32,20) if median>=36 else (24,14)
    hypotheses={}
    for k in windows:
        update(f"Indexing exact torn edges (arc window {k})")
        for votes,i,j,r,tx,ty in propose(pieces,k,True,max_bucket=6):
            if votes<3:break
            key=(i,j,r,tx,ty)
            hypotheses[key]=max(hypotheses.get(key,0),votes)
    # Bound verification to top incident hypotheses on *either* endpoint;
    # otherwise a large neighbor can starve a small fragment of candidates.
    top=defaultdict(list)
    for key,v in hypotheses.items():
        i,j,*_=key
        top[i].append((v,key));top[j].append((v,key))
    chosen={key for near in top.values() for _,key in sorted(near,reverse=True)[:degree]}
    candidates=[]
    for key in sorted(chosen,key=lambda key:(-hypotheses[key],key)):
        i,j,r,tx,ty=key
        t=np.array([tx,ty],float)
        match=_exact_contact(pieces[i],pieces[j],r,t,hypotheses[key])
        if match:
            score,info=match
            candidates.append((score,i,j,R[r].astype(float),t,info))
    return candidates,{"arc_windows":list(windows),"voted_poses":len(hypotheses),
                       "verified_poses":len(chosen),"verified_pairs":len({key[:2] for key in chosen}),"accepted_candidates":len(candidates)}


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
