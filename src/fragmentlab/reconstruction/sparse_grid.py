"""Sparse classical grid matching: indexed edge profiles and reciprocal groups.

Inspired by Paikin & Tal's best-buddy initialization and Son et al.'s local
cycle consensus. No image, truth, trained descriptor, or dense pair matrix.
Uniform public cells are required for this scalable path.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import time
import math
import numpy as np
from scipy.spatial import cKDTree
from .common import Pose
from .grid_consensus import profiles
from .straight import _grid_bounds

# Clockwise grid directions, x/y.
D=((0,-1),(1,0),(0,1),(-1,0))

def rotate(x,y,k):
    return ((x,y),(-y,x),(-x,-y),(y,-x))[k%4]

def eligible(shape,grid):
    return bool(grid and shape[0]%grid[0]==0 and shape[1]%grid[1]==0)

def _features(a, samples=6):
    """Low-frequency area-averaged color profiles for coarse retrieval."""
    n=a.shape[1]
    # Segment means rather than point sampling for very long strips.
    cuts=np.linspace(0,n,min(samples,n)+1,dtype=int)
    return np.stack([a[:,lo:hi].mean(axis=1) for lo,hi in zip(cuts[:-1],cuts[1:])],axis=1).reshape(len(a),-1)

def candidates(pieces,shape,grid,rotations, shortlist=40, keep=5, profile_samples=None, index_mode="legacy", stats=None):
    desc=profiles(pieces)
    query_seconds=rerank_seconds=0.;m=len(desc);lens=np.array([len(e) for e,g in desc]);idx=np.full((m,keep),-1,np.int32)
    costs=np.full((m,keep),np.inf,np.float32)
    ph,pw=shape[0]//grid[0],shape[1]//grid[1]
    # Strips have only two physically meaningful sides; don't compare end caps.
    axis=0 if grid[0]==1 else 1 if grid[1]==1 else None
    allowed=np.ones(m,bool)
    if axis is not None:
        for i,p in enumerate(pieces):
            hh,ww=p.rgba.shape[:2]
            for s in range(4):
                length=ww if s%2==0 else hh
                allowed[4*i+s]=length==(ph if axis==0 else pw)
    # If orientation is given, only compare opposite directions.
    for L in np.unique(lens):
        base=np.flatnonzero((lens==L)&allowed)
        for side in ([None] if rotations else range(4)):
            u=base if side is None else base[base%4==side]
            v=base if side is None else base[base%4==(side+2)%4]
            if not len(u) or not len(v):continue
            eu=np.stack([desc[i][0] for i in u]);gu=np.stack([desc[i][1] for i in u])
            ev=np.stack([desc[i][0] for i in v]);gv=np.stack([desc[i][1] for i in v])
            sample_count=profile_samples or (8 if len(pieces)<=2048 else 6)
            k=min(shortlist,len(v))
            epsilon=.45 if len(pieces)>32768 else .30 if len(pieces)>2048 else 0.0
            workers=4 if len(u)>=4096 and __import__("sys").platform!="emscripten" else 1
            tq=time.perf_counter()
            if index_mode=="legacy":
                target=_features(ev[:,::-1],sample_count);tree=cKDTree(target)
                q0=_features(eu,sample_count);q1=_features(eu+gu*.65,sample_count)
                _,all0=tree.query(q0,k=k,eps=epsilon,workers=workers)
                _,all1=tree.query(q1,k=k,eps=epsilon,workers=workers)
                allids=np.concatenate([np.asarray(all0,dtype=np.int32).reshape(len(u),k),
                                       np.asarray(all1,dtype=np.int32).reshape(len(u),k)],axis=1)
            else:
                from .index10 import query_edges
                allids=query_edges(eu,gu,ev,gv,k=k,workers=workers,mode=index_mode,samples=profile_samples or 4)
            query_seconds+=time.perf_counter()-tq
            tr=time.perf_counter()
            # Larger bounded vector batches reduce Python overhead.
            batch=min(512,max(32,8_000_000//max(1,allids.shape[1]*min(L,256)*3*4)))
            for lo in range(0,len(u),batch):
                hi=min(len(u),lo+batch);a=u[lo:hi]
                jj=allids[lo:hi].copy()
                jj.sort(axis=1)
                same=np.zeros(jj.shape,bool);same[:,1:]=jj[:,1:]==jj[:,:-1]
                # Downsample exact comparison only for very long strips.
                step=max(1,L//128);E=eu[lo:hi,::step];G=gu[lo:hi,::step]
                F=ev[jj,::-step];H=gv[jj,::-step]
                # When step>1, both have matching lengths; reverse before sampling.
                delta=E[:,None]-F
                basecost=np.mean(np.abs(delta),axis=(2,3))
                # Smooth-image prediction with robust per-channel-gradient scaling.
                # For 1-pixel strips no inward derivative exists.
                su=np.sqrt(np.mean(G*G,axis=1)+.00012)
                sv=np.sqrt(np.mean(H*H,axis=2)+.00012)
                pred=(np.mean(np.abs(delta+G[:,None]),axis=2)/su[:,None]+
                      np.mean(np.abs(-delta+H),axis=2)/sv).mean(axis=2)*.5
                scale=np.sqrt(np.maximum(.01,np.sqrt(np.mean(G*G,axis=(1,2))))[:,None]*
                              np.maximum(.01,np.sqrt(np.mean(H*H,axis=(2,3)))))
                cc=.45*basecost/scale+.55*pred
                cc[(a[:,None]//4==v[jj]//4)|same]=np.inf
                order=np.argsort(cc,axis=1,kind="stable")[:,:keep]
                jj=np.take_along_axis(jj,order,axis=1);val=np.take_along_axis(cc,order,axis=1)
                z=min(keep,jj.shape[1]);idx[a,:z]=np.where(np.isfinite(val),v[jj],-1)
                costs[a,:z]=val
            rerank_seconds+=time.perf_counter()-tr
    if stats is not None:stats.update(index_query_seconds=query_seconds,rerank_seconds=rerank_seconds)
    return idx,costs,allowed

def _links_reference(idx,costs):
    """Reciprocal candidates, then consistent 2x2 loops (both paths agree)."""
    m=len(idx);second=costs[:,1]
    with np.errstate(invalid="ignore",divide="ignore"):
        margin=np.where(np.isfinite(second),(second-costs[:,0])/np.maximum(second,1e-8),0.)
    margin=np.clip(margin,0,1)
    rows=[{int(j):(r,float(c)) for r,(j,c) in enumerate(zip(ii,vv)) if j>=0} for ii,vv in zip(idx,costs)]
    support=Counter()
    for a in range(m):
        i,s=divmod(a,4);c=4*i+(s+1)%4
        for b in idx[a,:2]:
            if b<0:continue
            j,t=divmod(int(b),4);rj=(s+2-t)%4
            js=4*j+(s+1-rj)%4
            for e in idx[c,:2]:
                if e<0:continue
                l,u=divmod(int(e),4)
                if l==j:continue
                rl=(s+3-u)%4;ls=4*l+(s-rl)%4
                for f in idx[js,:2]:
                    if f<0:continue
                    z,v=divmod(int(f),4)
                    if z in (i,j,l):continue
                    rz=(s+3-v)%4;g=4*z+(s+2-rz)%4
                    if g in rows[ls] and rows[ls][g][0]<2:
                        for x,y in ((a,int(b)),(c,int(e)),(js,int(f)),(ls,g)):
                            support[min(x,y),max(x,y)]+=1
    links=[]
    for a in range(m):
        for b,(rank,cost) in rows[a].items():
            if b<=a or a not in rows[b]:continue
            r,bc=rows[b][a];loop=support[a,b]
            conf=min(float(margin[a]),float(margin[b])) if r==rank==0 else 0.
            # Strong mutual matches, reinforced by independent loops.
            if rank+r>0 and loop==0:continue
            strength=conf+.45*min(loop,4)-.12*(rank+r)
            if strength<=0:continue
            links.append((strength,a,b,conf,loop))
    links.sort(key=lambda t:(-t[0],t[1],t[2]))
    return links,rows,margin

def _links(idx,costs):
    """Equivalent vectorized local cycle consensus on larger graphs."""
    if len(idx)<4096:return _links_reference(idx,costs)
    from .links11 import reciprocal_links
    second=costs[:,1]
    with np.errstate(invalid="ignore",divide="ignore"):
        margin=np.where(np.isfinite(second),(second-costs[:,0])/np.maximum(second,1e-8),0.)
    margin=np.clip(margin,0,1)
    rows=[{int(j):(r,float(c)) for r,(j,c) in enumerate(zip(ii,vv)) if j>=0} for ii,vv in zip(idx,costs)]
    return reciprocal_links(idx,costs,rows,margin),rows,margin

def assemble(pieces,shape,grid,rotations,idx,costs,report_components=False):
    n=len(pieces);rmax,cmax=grid;ph,pw=shape[0]//rmax,shape[1]//cmax
    links,rows,margin=_links(idx,costs)
    # Keep group occupancy and extent; merge small into large, avoiding O(N^2)
    # dictionary copying. The root frame is arbitrary, including quarter-turns.
    owner=list(range(n));poses={i:{i:(0,0,0)} for i in range(n)}
    occ={i:{(0,0):i} for i in range(n)}
    bounds={i:(0,0,0,0) for i in range(n)}
    dims=[p.rgba.shape[:2] for p in pieces]
    accepted=rejected=0
    for strength,a,b,confidence,loop in links:
        i,s=divmod(a,4);j,t=divmod(b,4);u,v=owner[i],owner[j]
        if u==v:continue
        x,y,k=poses[u][i];xx,yy,kk=poses[v][j]
        d=(s+k)%4;rr=(d+2-t-kk)%4
        dx,dy=D[d];ax,ay=rotate(xx,yy,rr);tx,ty=x+dx-ax,y+dy-ay
        if len(poses[u])<len(poses[v]):
            u,v=v,u;rr=(-rr)%4;tx,ty=rotate(-tx,-ty,rr)
        g,h=poses[u],poses[v];og=occ[u]
        moved={q:(a0+tx,b0+ty,(z+rr)%4) for q,(xx,yy,z) in h.items() for a0,b0 in [rotate(xx,yy,rr)]}
        if any((xx,yy) in og for xx,yy,k in moved.values()):rejected+=1;continue
        l,t0,r,b0=bounds[u];xxs, yys=zip(*[(xx,yy) for xx,yy,k in moved.values()])
        l=min(l,min(xxs));r=max(r,max(xxs));t0=min(t0,min(yys));b0=max(b0,max(yys))
        # Check whether the root frame can be rotated to the public tile/canvas.
        # All cells must have consistent aspect in that frame.
        root=next(iter(g));rot=g[root][2]
        hh,ww=dims[root];hh,ww=(ww,hh) if rot%2 else (hh,ww)
        sh,sw=b0-t0+1,r-l+1
        ok=(hh,ww)==(ph,pw) and sh<=rmax and sw<=cmax
        ok|=bool(rotations and (ww,hh)==(ph,pw) and sw<=rmax and sh<=cmax)
        if not ok or any((dims[q][::-1] if k%2 else dims[q])!=(hh,ww) for q,(xx,yy,k) in moved.items()):
            rejected+=1;continue
        cross=bad=0
        for q,(xx,yy,k) in moved.items():
            for d,(dx,dy) in enumerate(D):
                z=og.get((xx+dx,yy+dy))
                if z is None:continue
                cross+=1;edge=4*q+(d-k)%4;other=4*z+(d+2-g[z][2])%4
                if other not in rows[edge] and edge not in rows[other]:bad+=1
        if bad and (bad>=cross or bad>max(1,cross//3)):
            rejected+=1;continue
        if loop==0 and confidence<.04 and cross<2:continue
        for q,p in moved.items():g[q]=p;og[p[:2]]=q;owner[q]=u
        bounds[u]=(l,t0,r,b0);del poses[v],occ[v],bounds[v];accepted+=1
        if len(poses)==1:break
    largest=max(poses.values(),key=len)
    # Rotate/translate only to match the given dimensions. For partial groups,
    # use outward-edge mismatch evidence to select an origin near the frame.
    best_choice=None
    for k in range(4 if rotations else 1):
        g={q:(a,b,(s+k)%4) for q,(x,y,s) in largest.items() for a,b in [rotate(x,y,k)]}
        if any((dims[q][::-1] if s%2 else dims[q])!=(ph,pw) for q,(x,y,s) in g.items()):continue
        xs,ys=zip(*[(x,y) for x,y,s in g.values()]);lo_x,lo_y=min(xs),min(ys);w,h=max(xs)-lo_x+1,max(ys)-lo_y+1
        if w>cmax or h>rmax:continue
        # Check translations at the four descriptor-supported boundaries.
        for offx in {0,cmax-w}:
            for offy in {0,rmax-h}:
                gp={q:(x-lo_x+offx,y-lo_y+offy,s) for q,(x,y,s) in g.items()}
                border=0.
                for q,(x,y,s) in gp.items():
                    for d,(dx,dy) in enumerate(D):
                        if not (0<=x+dx<cmax and 0<=y+dy<rmax):
                            cost=costs[4*q+(d-s)%4,0]
                            border+=math.log1p(min(float(cost),10.))
                option=(border,k,offx,offy,gp)
                if best_choice is None or option[:4]>best_choice[:4]:best_choice=option
    if best_choice is None:return {},dict(components=len(poses),anchor_pieces=len(largest))
    group=best_choice[-1]
    # Sparse frontier using exact reciprocal shortlist contact; leave genuinely
    # unsupported holes rather than invent positions.
    occupied={(x,y):q for q,(x,y,s) in group.items()};used=set(group)
    import heapq
    heap=[];ticket=0
    def add(q):
        nonlocal ticket
        x,y,s=group[q]
        for d,(dx,dy) in enumerate(D):
            xx,yy=x+dx,y+dy
            if (xx,yy) in occupied or not (0<=xx<cmax and 0<=yy<rmax):continue
            e=4*q+(d-s)%4
            for b,(rank,cost) in rows[e].items():
                j,u=divmod(b,4);k=(d+2-u)%4
                if j in used or (not rotations and k) or (dims[j][::-1] if k%2 else dims[j])!=(ph,pw):continue
                # Evaluate every already placed adjacent cell, not just parent.
                agree=bad=0
                for p,(vx,vy) in enumerate(D):
                    other=occupied.get((xx+vx,yy+vy))
                    if other is None:continue
                    a=4*j+(p-k)%4;bb=4*other+(p+2-group[other][2])%4
                    if bb in rows[a] or a in rows[bb]:agree+=1
                    else:bad+=1
                if bad or (agree<2 and (rank>0 or e not in rows[b] or margin[e]<.08)):continue
                ticket+=1;heapq.heappush(heap,(-agree, cost, ticket,xx,yy,j,k))
    for q in list(group):add(q)
    while heap:
        _,_,_,x,y,j,k=heapq.heappop(heap)
        if j in used or (x,y) in occupied:continue
        # Revalidate after the frontier changed.
        good=True
        for d,(dx,dy) in enumerate(D):
            q=occupied.get((x+dx,y+dy))
            if q is None:continue
            a=4*j+(d-k)%4;b=4*q+(d+2-group[q][2])%4
            if b not in rows[a] and a not in rows[b]:good=False;break
        if not good:continue
        group[j]=(x,y,k);occupied[x,y]=j;used.add(j);add(j)
    packed=0;pack_batches=[]
    if min(grid)>1 and n<=32768 and len(group)<.998*n:
        from .gridpack11 import pack_grid_components
        packed,pack_batches=pack_grid_components(pieces,shape,grid,rotations,group,occupied,poses.values(),rows,margin)
        used=set(group)
    matches=reciprocal=total=0;gaps=[]
    for q,(x,y,k) in group.items():
        for d in (1,2):
            dx,dy=D[d];j=occupied.get((x+dx,y+dy))
            if j is None:continue
            total+=1;a=4*q+(d-k)%4;b=4*j+(d+2-group[j][2])%4
            if b in rows[a] and a in rows[b]:
                matches+=1
                if rows[a][b][0]==rows[b][a][0]==0:
                    reciprocal+=1;gaps.append(min(float(margin[a]),float(margin[b])))
    diag=dict(components=len(poses),anchor_pieces=len(largest),accepted=accepted,rejected=rejected,
        reciprocal_seam_fraction=reciprocal/max(1,total),shortlist_seam_fraction=matches/max(1,total),
        median_reciprocal_margin=float(np.median(gaps)) if gaps else 0., placed_fraction=len(group)/n, component_pack_added=packed, component_pack_batches=pack_batches)
    if report_components and len(used)<n:
        local=[]
        for comp in poses.values():
            left={q:v for q,v in comp.items() if q not in used}
            if not left:continue
            x0=min(v[0] for v in left.values());y0=min(v[1] for v in left.values())
            local.append({"count":len(left),"poses":{pieces[q].id:{"cell_x":x-x0,"cell_y":y-y0,"k":k} for q,(x,y,k) in left.items()}})
        local.sort(key=lambda x:-x["count"])
        diag.update(unanchored_component_count=len(local),_unanchored_components=local)
    return {pieces[q].id:Pose(x*pw,y*ph,k) for q,(x,y,k) in group.items()},diag

def solve_sparse_grid(pieces,shape,grid,rotations,shortlist=40,report_components=False,profile_samples=None,index_mode="legacy"):
    start=time.perf_counter()
    if not 8<=shortlist<=256:raise ValueError("shortlist must be in [8, 256]")
    if not eligible(shape,grid):raise ValueError("Sparse scaling requires uniform grid cells (canvas divisible by rows/columns)")
    stats={}
    idx,costs,allowed=candidates(pieces,shape,grid,rotations,shortlist,profile_samples=profile_samples,index_mode=index_mode,stats=stats)
    retrieval=time.perf_counter()-start
    p,d=assemble(pieces,shape,grid,rotations,idx,costs,report_components)
    dup=Counter(min(hashlib.sha256(str(q.rotated(k).shape).encode()+np.ascontiguousarray(q.rotated(k)).tobytes()).digest() for k in range(4 if rotations else 1)) for q in pieces)
    d.update(algorithm="sparse-edge-consensus-v10" if index_mode!="legacy" else "sparse-edge-consensus-v8",index_mode=index_mode,**stats,retrieval_seconds=retrieval,
             assembly_seconds=time.perf_counter()-start-retrieval,shortlist=shortlist,descriptor_samples=profile_samples or (4 if index_mode!="legacy" else 8 if len(pieces)<=2048 else 6),
             identical_piece_count=sum(v for v in dup.values() if v>1))
    return p,d

def solve_indexed_grid(pieces,shape,grid,rotations,shortlist=40,report_components=False,
                       profile_samples=None,index_mode="auto",auto_tune=True):
    """V10 adaptive coarse retrieval; never selects using evaluated accuracy."""
    if index_mode not in {"auto","legacy","midpoint","mixed"}:
        raise ValueError("index_mode must be auto, legacy, midpoint or mixed")
    size=len(pieces)
    # Long strips and small puzzles retain their calibrated V9 descriptors.
    initial=("midpoint" if min(grid)>1 and size>=2048 and profile_samples is None else "legacy") if index_mode=="auto" else index_mode
    p,d=solve_sparse_grid(pieces,shape,grid,rotations,shortlist,report_components,profile_samples,initial)
    def quality(poses,diag):
        return (len(poses)/size,diag.get("shortlist_seam_fraction",0),diag.get("reciprocal_seam_fraction",0))
    score=quality(p,d)
    trials=[{"index":initial,"quality":list(score),"seconds":d["retrieval_seconds"]+d["assembly_seconds"]}]
    # A bounded second trial only when local consensus/coverage are weak.
    weak=score[0]<.97 or score[1]<.95 or score[2]<.80 or d.get("median_reciprocal_margin",0)<.07
    if index_mode=="auto" and initial!="legacy" and auto_tune and weak:
        q,e=solve_sparse_grid(pieces,shape,grid,rotations,shortlist,report_components,profile_samples,"legacy")
        qs=quality(q,e)
        trials.append({"index":"legacy","quality":list(qs),"seconds":e["retrieval_seconds"]+e["assembly_seconds"]})
        # Coverage alone is not enough; compare normalized global seam quality.
        from .gridmetrics10 import grid_objective
        if grid_objective(pieces,q,shape,grid)>grid_objective(pieces,p,shape,grid):p,d=q,e
    d["index_trials"]=trials
    d["index_fallback_used"]=len(trials)>1
    return p,d
