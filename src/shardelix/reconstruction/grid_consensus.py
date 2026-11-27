"""Classical edge-consensus rectangular grid solver.

Edge descriptors compare real seam profiles. Reciprocal ambiguity margins,
closed 2x2 loops, dimension-constrained rigid groups and contextual assignment
are computed using only fragment pixels and public canvas/grid dimensions.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from .common import Pose
from .straight import _grid_bounds
from .geometry import R

D=np.array([[0,-1],[1,0],[0,1],[-1,0]],int)

def _profiles_reference(pieces,depth=1):
    out=[]
    for p in pieces:
        a=p.rgba[:,:,:3].astype(np.float32)/255.
        # decorrelated Y + chromatic channels; clockwise traversal
        r,g,b=np.moveaxis(a,-1,0)
        im=np.stack([.299*r+.587*g+.114*b,(r-g)*.65,(b-g)*.65],-1)
        for z in [im.transpose(1,0,2),im[:,::-1],im[::-1,::-1].transpose(1,0,2),im[::-1]]:
            e=z[:,0];q=z[:,min(1,z.shape[1]-1)]
            if depth>1 and z.shape[1]>1:
                # Robust low-frequency fit over multiple pixels inside each edge.
                # Estimate boundary value and outward gradient, reducing JPEG noise.
                from scipy.ndimage import gaussian_filter1d
                d=min(depth,z.shape[1]);t=np.arange(d,dtype=np.float32)
                y=z[:,:d];center=t.mean()
                slope=np.sum(y*(t-center)[None,:,None],axis=1)/max(float(np.sum((t-center)**2)),1e-8)
                g=-slope
                ehat=y.mean(axis=1)-center*slope
                e=gaussian_filter1d(ehat,.65,axis=0)
                q=e-gaussian_filter1d(g,.65,axis=0)
            out.append((e.copy(),np.ascontiguousarray(e-q)))
    return out

def profiles(pieces,depth=1):
    if depth==1:
        from ..native_core import profiles as accelerated
        result=accelerated(pieces)
        if result is not None:return result
    return _profiles_reference(pieces,depth)

def edge_costs(pieces,rotations,metric="hybrid",gradient=.35,depth=1):
    desc=profiles(pieces,depth);m=len(desc)
    lens=np.array([len(e) for e,g in desc]);raw=np.full((m,m),np.inf,np.float32)
    for length in np.unique(lens):
        ii=np.flatnonzero(lens==length)
        e=np.stack([desc[i][0] for i in ii]);g=np.stack([desc[i][1] for i in ii])
        f=e.reshape(len(ii),-1);b=e[:,::-1].reshape(len(ii),-1);d=g.reshape(len(ii),-1)
        n=f.shape[1];base=cdist(f,b,"cityblock")/n
        if metric=="l1":cost=base
        elif metric=="l2":cost=np.sqrt(cdist(f,b,"sqeuclidean")/n)
        else:
            predicted=cdist(f+d,b,"cityblock")/n
            predicted=(predicted+predicted.T)*.5
            if metric=="mgc":
                # Side-specific robust gradient scale; directional predictions.
                scale=np.sqrt(np.mean(g*g,axis=1)+.00012)
                norm=np.zeros_like(base)
                for ch in range(3):
                    q=cdist(e[:,:,ch]+g[:,:,ch],e[:,::-1,ch],"cityblock")/length
                    norm+=(q/scale[:,ch,None]+q.T/scale[None,:,ch])/6
                den=np.maximum(.01,np.sqrt(np.mean(g*g,axis=(1,2))))
                base2=base/np.sqrt(den[:,None]*den[None,:])
                cost=(1-gradient)*base2+gradient*norm
            else:
                # Along-edge first differences identify continuation of fine detail.
                dd=np.diff(e,axis=1,prepend=e[:,:1])
                rev=np.diff(e[:,::-1],axis=1,prepend=e[:,-1:])
                along=cdist(dd.reshape(len(ii),-1),rev.reshape(len(ii),-1),"cityblock")/n
                cost=(1-gradient)*base+gradient*predicted+.08*along
        cost=(cost+cost.T)*.5
        if not rotations:cost[(ii[:,None]%4+2)%4!=ii[None,:]%4]=np.inf
        cost[ii[:,None]//4==ii[None,:]//4]=np.inf
        raw[np.ix_(ii,ii)]=cost.astype(np.float32)
    return raw

def dimension_filter(raw,pieces,rh,cw,rotations):
    """Exclude relative edge pairings that cannot occur in the public grid."""
    dims=np.array([p.rgba.shape[:2] for p in pieces])
    ht=np.repeat(dims[:,0],4);wd=np.repeat(dims[:,1],4)
    sides=np.arange(4*len(pieces))%4
    length=np.where(sides%2==0,wd,ht);thick=np.where(sides%2==0,ht,wd)
    rh=np.asarray(rh);cw=np.asarray(cw)
    for side_length in np.unique(length):
        ix=np.flatnonzero(length==side_length);t=thick[ix];s=sides[ix]
        valid=np.zeros((len(ix),len(ix)),bool)
        for axis,along,across in [(0,cw,rh),(1,rh,cw)]:
            if side_length not in along:continue
            combos=set(zip(across[:-1],across[1:]))
            combos |= {(b,a) for a,b in combos}
            for a,b in combos:
                ok=(t[:,None]==a)&(t[None,:]==b)
                if not rotations:ok &= (s[:,None]%2==axis)&(s[None,:]%2==axis)
                valid |= ok
        block=raw[np.ix_(ix,ix)];block[~valid]=np.inf;raw[np.ix_(ix,ix)]=block
    return raw

def _neighbors(raw,k):
    n=len(raw);k=min(k,n-1)
    idx=np.argpartition(raw,k-1,axis=1)[:,:k]
    idx=np.take_along_axis(idx,np.argsort(np.take_along_axis(raw,idx,axis=1),axis=1,kind="stable"),axis=1)
    val=np.take_along_axis(raw,idx,axis=1)
    finite=raw[np.isfinite(raw)];scale=max(float(np.median(finite)),1e-8)
    second=np.maximum(val[:,min(1,k-1)],scale*.00001)
    # A tie has zero margin, even if both raw costs are zero.
    margin=np.maximum(0,(val[:,min(1,k-1)]-val[:,0])/np.maximum(val[:,min(1,k-1)],scale*.00001))
    margin[~np.isfinite(margin)]=0
    return idx,val,second,margin,scale

def loop_support(raw,idx,second,k=3):
    """Closed 2x2 loops in the local frame of a fragment, including rotations."""
    n=len(raw)//4;support={}
    def key(a,b):return (min(a,b),max(a,b))
    def q(a,b):return float(raw[a,b]/max(float(np.sqrt(second[a]*second[b])),1e-8))
    for i in range(n):
      for s in range(4):
        a=4*i+s;c=4*i+(s+1)%4;best=None
        for b in idx[a,:k]:
            j,t=divmod(int(b),4)
            if i==j or not np.isfinite(raw[a,b]):continue
            rj=(s+2-t)%4
            for e in idx[c,:k]:
                l,u=divmod(int(e),4)
                if l in (i,j) or not np.isfinite(raw[c,e]):continue
                rl=(s+3-u)%4
                js=4*j+(s+1-rj)%4;ls=4*l+(s-rl)%4
                # Both paths lead to the same diagonal piece and orientation.
                for f in idx[js,:k]:
                    z,v=divmod(int(f),4)
                    if z in (i,j,l):continue
                    rz=(s+3-v)%4
                    g=4*z+(s+2-rz)%4
                    if g not in idx[ls,:k]:continue
                    vals=[q(a,b),q(c,e),q(js,f),q(ls,g)]
                    score=max(vals)+.35*sum(vals)
                    if not np.isfinite(score):continue
                    if best is None or score<best[0]:best=(score,((a,b),(c,e),(js,f),(ls,g)))
        if best is not None:
            qloop=1./(1.+best[0])
            for a,b in best[1]:
                z=key(a,b);support[z]=max(qloop,support.get(z,0.))
    return support

def _fit(group,pieces,rh,cw,rotations,first=False):
    """Enumerate exact canvas-compatible rigid transforms (public dimensions only)."""
    rows,cols=len(rh),len(cw);out=[]
    uniform=len(set(rh))==len(set(cw))==1
    data=np.array(list(group.values()),int);ids=np.array(list(group))
    for k in range(4 if rotations else 1):
        xy=data[:,:2]@R[k].T
        lo=xy.min(0);hi=xy.max(0);span=hi-lo+1
        if span[0]>cols or span[1]>rows:continue
        turns=(data[:,2]+k)%4
        dims=np.array([pieces[int(i)].rgba.shape[:2] for i in ids])
        dims=np.where((turns%2)[:,None]!=0,dims[:,::-1],dims)
        if uniform:
            if not np.all(dims==[rh[0],cw[0]]):continue
            if first:return True
            for dy in range(-lo[1],rows-hi[1]):
                for dx in range(-lo[0],cols-hi[0]):out.append((k,int(dx),int(dy)))
        else:
            xshift=np.arange(-lo[0],cols-hi[0]);yshift=np.arange(-lo[1],rows-hi[1])
            if not len(xshift) or not len(yshift):continue
            xshift=xshift[np.all(cw[xy[:,0,None]+xshift[None,:]]==dims[:,1,None],axis=0)]
            yshift=yshift[np.all(rh[xy[:,1,None]+yshift[None,:]]==dims[:,0,None],axis=0)]
            if first and len(xshift) and len(yshift):return True
            for dy in yshift:
                for dx in xshift:out.append((k,int(dx),int(dy)))
    return False if first else out

def _transform(g,k,dx,dy):
    return {i:(int(R[k,0,0]*x+R[k,0,1]*y+dx),int(R[k,1,0]*x+R[k,1,1]*y+dy),(t+k)%4) for i,(x,y,t) in g.items()}

def _cross(a,b,cost):
    pos={(x,y):(i,k) for i,(x,y,k) in a.items()}
    vals=[]
    for i,(x,y,k) in b.items():
        for d,(dx,dy) in enumerate(D):
            q=pos.get((x+dx,y+dy))
            if q is None:continue
            j,t=q;vals.append(float(cost[4*i+(d-k)%4,4*j+(d+2-t)%4]))
    return vals

def _groups(pieces,rh,cw,rotations,raw,norm,idx,margin,support,k=4,strict=.18):
    n=len(pieces);cost=norm;groups={i:{i:(0,0,0)} for i in range(n)};owner=np.arange(n)
    links=[]
    for a in range(4*n):
        for rank,b0 in enumerate(idx[a,:k]):
            b=int(b0)
            if b<=a or a//4==b//4 or not np.isfinite(raw[a,b]):continue
            rr=np.flatnonzero(idx[b,:k]==a)
            if not len(rr):continue
            recip=int(rr[0]);loop=support.get((a,b),0)
            conf=min(margin[a],margin[b]) if rank==recip==0 else 0.
            # No confidence on identical-edge ties.
            strength=2*conf+1.15*loop-.16*(rank+recip)-.05*min(float(cost[a,b]),10.)
            links.append((strength,a,b,conf,loop,rank+recip))
    links.sort(reverse=True)
    accepted=0;rejected=0
    for phase in range(3):
        for strength,a,b,conf,loop,ranks in links:
            i,s=divmod(a,4);j,t=divmod(b,4)
            u,v=int(owner[i]),int(owner[j])
            if u==v:continue
            g,h=groups[u],groups[v]
            x,y,ki=g[i];xx,yy,kj=h[j]
            d=(s+ki)%4;r=(ki+s+2-t-kj)%4
            shift=np.array([x,y])+D[d]-R[r]@np.array([xx,yy])
            if phase==0 and not (conf>=strict or (loop>.4 and ranks<=1)):continue
            if phase==1 and not (conf>=strict*.35 or loop>.18):continue
            if phase==2 and not (conf>.015 or loop>.12):continue
            if len(g)<len(h):
                u,v=v,u;g,h=h,g;r=(-r)%4;shift=-R[r]@shift
            moved=_transform(h,r,*shift)
            if {(x,y) for x,y,k in g.values()}&{(x,y) for x,y,k in moved.values()}:continue
            cross=_cross(g,moved,cost)
            if not cross:continue
            # Multiple newly satisfied seams are independent evidence.
            if max(cross)>4.0 or (len(cross)>1 and np.mean(cross)>1.25):rejected+=1;continue
            if phase==2 and len(cross)==1 and conf<strict*.35 and loop<.35:continue
            if phase==1 and len(cross)==1 and conf<strict*.12 and loop<.25:continue
            joined={**g,**moved}
            if not _fit(joined,pieces,rh,cw,rotations,first=True):rejected+=1;continue
            groups[u]=joined;owner[list(h)]=u;del groups[v];accepted+=1
        if len(groups)==1:break
    return sorted(groups.values(),key=len,reverse=True),accepted,rejected

def _slot_cost(arr,y,x,states,norm):
    """Cost vector for candidate states 4*i+k, including all fixed neighbours."""
    rows,cols=arr.shape;out=np.zeros(len(states),np.float32);support=0
    for d,(dx,dy) in enumerate(D):
        yy,xx=y+dy,x+dx
        if not (0<=yy<rows and 0<=xx<cols) or arr[yy,xx]<0:continue
        q=int(arr[yy,xx]);a=4*(states//4)+(d-states%4)%4
        b=4*(q//4)+(d+2-q%4)%4
        out+=norm[a,b];support+=1
    return out,support

def _complete(pieces,shape,grid,groups,norm,raw,rotations,rh,cw,rounds):
    rows,cols=grid;n=len(pieces);anchor=groups[0]
    fits=_fit(anchor,pieces,rh,cw,rotations)
    if not fits:raise RuntimeError("Anchor no longer fits grid")
    # Boundary edges have no true match. Their high minimum seam cost
    # provides an observable weak clue for the anchor's translation.
    mincost=np.min(raw,axis=1);finite=mincost[np.isfinite(mincost)]
    scale=max(float(np.median(finite)),1e-6)
    bd=np.clip(mincost/scale,0,20)
    def border(g):
        score=0.
        for i,(x,y,k) in g.items():
            for d in range(4):
                dx,dy=D[d]
                if not (0<=x+dx<cols and 0<=y+dy<rows):
                    score+=np.log1p(bd[4*i+(d-k)%4])
        return score
    fit=max(fits,key=lambda k:border(_transform(anchor,*k)))
    a=_transform(anchor,*fit);arr=np.full(grid,-1,int)
    for i,(x,y,k) in a.items():arr[y,x]=4*i+k
    # Preserve a confident core; pack disjoint groups only if cross-contact
    # evidence supports their translation, otherwise break into individual tiles.
    packed=0
    for g in groups[1:]:
        if len(g)<2:continue
        best=None
        for f in _fit(g,pieces,rh,cw,rotations):
            moved=_transform(g,*f)
            if any(arr[y,x]>=0 for x,y,k in moved.values()):continue
            cross=_cross(a,moved,norm)
            if len(cross)<2:continue
            score=np.mean(cross)
            if max(cross)>3 or score>1.1:continue
            key=(score-.12*np.log1p(len(cross)), -len(cross))
            if best is None or key<best[0]:best=(key,moved)
        if best:
            a.update(best[1]);packed+=len(g)
            for i,(x,y,k) in best[1].items():arr[y,x]=4*i+k
    core=arr>=0
    allstates=np.array([4*i+k for i in range(n) for k in range(4 if rotations else 1)])
    sd=np.array([pieces[s//4].rotated(s%4).shape[:2] for s in allstates])
    used=set(a)
    # Greedy frontier, favor two/three/four independent neighbors. Global
    # assignment later resolves remaining one-to-one conflicts.
    while (arr<0).any():
        active=np.array([i//4 not in used for i in allstates])
        best=None
        for y,x in zip(*np.where(arr<0)):
            valid=active&(sd[:,0]==rh[y])&(sd[:,1]==cw[x]);options=allstates[valid]
            if not len(options):continue
            c,num=_slot_cost(arr,y,x,options,norm)
            if not num:continue
            # Best orientation per unique piece; runner-up must be another piece.
            order=np.argsort(c);s=int(options[order[0]]);e=float(c[order[0]])
            alt=next((float(c[j]) for j in order[1:] if options[j]//4!=s//4),e+1.)
            gap=max(0.,(alt-e)/max(alt,1e-6))
            key=(num+gap-.08*e/max(num,1),gap,-e)
            if best is None or key>best[0]:best=(key,int(y),int(x),s)
        if best is None:
            for y,x in zip(*np.where(arr<0)):
                valid=active&(sd[:,0]==rh[y])&(sd[:,1]==cw[x])
                if valid.any():best=((0,),int(y),int(x),int(allstates[valid][0]));break
        if best is None:raise ValueError("No dimension-compatible remaining tile")
        _,y,x,s=best;arr[y,x]=s;used.add(s//4)
    # One-to-one checkerboard refinement, using neighbors held fixed.
    changes=0
    for rnd in range(rounds):
      moved=0
      for parity in (0,1):
        y0,x0=np.where(np.indices(grid).sum(axis=0)%2==parity)
        for hh,ww in set(zip(rh[y0],cw[x0])):
            valid=(rh[y0]==hh)&(cw[x0]==ww)
            if rnd<2:valid &= ~core[y0,x0]
            ys,xs=y0[valid],x0[valid]
            if len(ys)<1:continue
            old=arr[ys,xs].copy();ids=old//4
            states=4*ids[:,None]+np.arange(4 if rotations else 1)[None,:]
            dims=np.array([pieces[int(s)//4].rotated(int(s)%4).shape[:2] for s in states.ravel()]).reshape(*states.shape,2)
            ok=(dims==[hh,ww]).all(-1);C=np.empty((len(ys),*states.shape),np.float32)
            for j,(y,x) in enumerate(zip(ys,xs)):
                c,num=_slot_cost(arr,y,x,states.ravel(),norm);C[j]=c.reshape(states.shape)
            C[:,~ok]=np.inf
            low=C.min(2);orient=C.argmin(2);low[~np.isfinite(low)]=1e8
            _,p=linear_sum_assignment(low)
            baseline=float(sum(C[j,j,int(s)%4] for j,s in enumerate(old)))
            if float(low[np.arange(len(p)),p].sum())<baseline-1e-6:
                new=states[p,orient[np.arange(len(p)),p]]
                moved+=int(np.count_nonzero(old!=new));arr[ys,xs]=new
      changes+=moved
      if rnd>=2 and moved==0:break
    return arr,len(anchor),packed,changes

def solve_consensus_grid(pieces,shape,grid,rotations=False,metric="mgc",gradient=.65,top_k=4,strict=.25,rounds=6,depth=1,dimensions=False,raw=None):
    n=len(pieces)
    if n==1:return {pieces[0].id:Pose(0,0)},{ "algorithm":"consensus-single"}
    ys,xs=_grid_bounds(*shape,*grid);rh,cw=np.diff(ys),np.diff(xs)
    if raw is None:raw=edge_costs(pieces,rotations,metric,gradient,depth)
    if dimensions:raw=dimension_filter(raw,pieces,rh,cw,rotations)
    idx,val,second,margin,scale=_neighbors(raw,max(top_k,3))
    den=np.sqrt(second[:,None]*second[None,:])
    with np.errstate(invalid="ignore",divide="ignore"):
        norm=raw/np.maximum(den,scale*.00001)
    norm[~np.isfinite(raw)]=np.inf
    support=loop_support(raw,idx,second,min(top_k,3))
    groups,accepted,rejected=_groups(pieces,rh,cw,rotations,raw,norm,idx,margin,support,top_k,strict)
    # Use ratio plus a modest absolute cost for context; ties remain ambiguous.
    context=(np.log1p(norm)+.15*raw/scale).astype(np.float32)
    arr,anchor,packed,changes=_complete(pieces,shape,grid,groups,context,raw,rotations,rh,cw,rounds)
    poses={pieces[s//4].id:Pose(int(xs[x]),int(ys[y]),int(s%4)) for y,x in np.ndindex(grid) for s in [int(arr[y,x])]}
    dup=Counter(hashlib.sha256(p.rgba.tobytes()).digest() for p in pieces)
    return poses,dict(algorithm="grid-edge-consensus",metric=metric,gradient=gradient,top_k=top_k,strict=strict,
                      components=len(groups),anchor_pieces=anchor,packed=packed,assignment_moves=changes,depth=depth,dimension_filter=dimensions,
                      accepted=accepted,rejected=rejected,loop_edges=len(support),
                      identical_piece_count=sum(v for v in dup.values() if v>1))

def score_grid(pieces,poses,shape,grid,rotations,raw=None):
    """Common blind objective for all candidate solvers (no ground truth)."""
    if raw is None:raw=edge_costs(pieces,rotations,"mgc",.65)
    idx,val,second,margin,scale=_neighbors(raw,3)
    ys,xs=_grid_bounds(*shape,*grid);arr=np.full(grid,-1,int)
    for i,p in enumerate(pieces):
        if p.id not in poses:continue
        q=poses[p.id];y=int(np.searchsorted(ys,q.y));x=int(np.searchsorted(xs,q.x))
        if not (0<=y<grid[0] and 0<=x<grid[1]) or ys[y]!=q.y or xs[x]!=q.x:continue
        if p.rotated(q.k).shape[:2]!=(ys[y+1]-ys[y],xs[x+1]-xs[x]):continue
        if arr[y,x]>=0:return float("inf")
        arr[y,x]=4*i+q.k
    if (arr<0).any():return float("inf")
    a=np.r_[arr[:,:-1].ravel(),arr[:-1,:].ravel()]
    b=np.r_[arr[:,1:].ravel(),arr[1:,:].ravel()]
    d=np.r_[np.ones(arr[:,:-1].size,int),np.full(arr[:-1,:].size,2,int)]
    u=4*(a//4)+(d-a%4)%4;v=4*(b//4)+(d+2-b%4)%4
    seam=raw[u,v]
    norm=seam/np.maximum(np.sqrt(second[u]*second[v]),scale*.00001)
    # A log ratio tempers isolated outliers; absolute term breaks ambiguity.
    return float(np.mean(np.log1p(norm)+.15*seam/scale))
