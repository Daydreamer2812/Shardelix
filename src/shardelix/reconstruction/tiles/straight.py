"""Classical straight-cut matching and assembly (no reference or truth)."""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from shardelix.reconstruction.common import Fragment, Pose


def _grid_bounds(h,w,rows,cols):
    ys=np.flatnonzero(np.r_[True,np.diff(np.arange(h)*rows//h)!=0,True])
    xs=np.flatnonzero(np.r_[True,np.diff(np.arange(w)*cols//w)!=0,True])
    return ys,xs


def _edge(a,side):
    rgb=a[:,:,:3].astype(np.float32)
    # Decorrelate luminance/color without a learned representation.
    r,g,b=np.moveaxis(rgb,-1,0)
    im=np.stack((.299*r+.587*g+.114*b,(r-g)*.65,(b-g)*.65),axis=-1)
    if side=="right": return im[:,::-1]
    if side=="left": return im
    if side=="bottom": return im[::-1].transpose(1,0,2)
    return im.transpose(1,0,2)


def _edge_stack(pieces,side,samples=0):
    arr=[_edge(p.rgba,side) for p in pieces]
    return arr


def _descriptor(a, samples, smooth=0):
    # a has shape (longitudinal edge, inward depth, colour).
    e=a[:,0]
    if a.shape[1]>1:
        g=e-a[:,1]
    else:
        g=np.zeros_like(e)
    if samples and len(e)>samples:
        # Area average provides a low-pass approximation without aliasing.
        from scipy.ndimage import gaussian_filter1d
        sig=max(0.,.45*len(e)/samples)
        e=gaussian_filter1d(e,sig,axis=0)
        g=gaussian_filter1d(g,sig,axis=0)
        ix=np.linspace(0,len(e)-1,samples)
        e=np.stack([np.interp(ix,np.arange(len(e)),e[:,c]) for c in range(3)],axis=-1)
        g=np.stack([np.interp(ix,np.arange(len(g)),g[:,c]) for c in range(3)],axis=-1)
    along=np.diff(e,axis=0,prepend=e[:1])
    return e,g,along,a.shape[1]>1


def _matrix_for_pair(arr_a,arr_b,ids_a,ids_b,samples,kind,gradient_weight):
    aa=[_descriptor(arr_a[i],samples) for i in ids_a]
    bb=[_descriptor(arr_b[i],samples) for i in ids_b]
    ea,ga,da,has_a=map(np.asarray,zip(*aa));eb,gb,db,has_b=map(np.asarray,zip(*bb))
    n=ea.shape[1]*ea.shape[2]
    def flat(a):return a.reshape(len(a),-1)
    x,y=flat(ea),flat(eb)
    gx,gy=flat(ga),flat(gb)
    if kind in {"l1","hybrid","normalized","mgc"}:
        # L1 is less sensitive to sharp edges than squared error.
        raw=cdist(x,y,"cityblock")/n
        if gradient_weight:
            pa=cdist(x+gx,y,"cityblock")/n
            pb=cdist(x,y+gy,"cityblock")/n
            pred=(pa*has_a[:,None]+pb*has_b[None,:])/np.maximum(1,has_a[:,None].astype(int)+has_b[None,:].astype(int))
            pred=np.where(has_a[:,None]|has_b[None,:],pred,raw)
        else:pred=raw
        along=cdist(flat(da),flat(db),"cityblock")/n if kind!="l1" else 0.
        mat=(1-gradient_weight)*raw+gradient_weight*pred
        if kind!="l1":mat=.80*mat+.20*along
        if kind in {"normalized","mgc"}:
            # Normalize by local directional gradient magnitude and a
            # regularized colour-dependent scale. Not trained on labels.
            saa=np.mean(np.abs(ga),axis=(1,2));sbb=np.mean(np.abs(gb),axis=(1,2))
            denom=np.maximum(.6, .5*(saa[:,None]+sbb[None,:]))
            if kind=="normalized":
                mat=mat/(denom**.35)
            else:
                # A side-specific diagonal covariance approximation, robust
                # against bright high-contrast textures and colour scale.
                sa=np.sqrt(np.mean(ga*ga,axis=1)+2.)
                sb=np.sqrt(np.mean(gb*gb,axis=1)+2.)
                mg=np.zeros_like(mat)
                for ch in range(3):
                    e1,e2=ea[:,:,ch],eb[:,:,ch]
                    u,v=ga[:,:,ch],gb[:,:,ch]
                    ca=cdist(e1+u,e2,"cityblock")/ea.shape[1]
                    cb=cdist(e1,e2+v,"cityblock")/ea.shape[1]
                    mg+=(ca/sa[:,ch,None]+cb/sb[None,:,ch])/6
                mat=.55*mat/np.sqrt(denom)+.45*mg
    else:
        raw=cdist(x,y,"sqeuclidean")/n
        pred=(cdist(x+gx,y,"sqeuclidean")+cdist(x,y+gy,"sqeuclidean"))/(2*n) if gradient_weight else raw
        mat=np.sqrt((1-gradient_weight)*raw+gradient_weight*pred)
    # Convert to visually interpretable near-0 scale like v2.
    return (mat/255).astype(np.float32)


def straight_costs(pieces,samples=0,kind="hybrid",gradient_weight=.35,axis_only=None):
    n=len(pieces); mats=[]
    for axis,(sa,sb) in enumerate((("right","left"),("bottom","top"))):
        mat=np.full((n,n),np.inf,np.float32)
        if axis_only is not None and axis!=axis_only:
            mats.append(mat);continue
        aa=_edge_stack(pieces,sa);bb=_edge_stack(pieces,sb)
        la=np.array([len(x) for x in aa]);lb=np.array([len(x) for x in bb])
        for ln in np.intersect1d(la,lb):
            ix=np.flatnonzero(la==ln);jx=np.flatnonzero(lb==ln)
            mat[np.ix_(ix,jx)]=_matrix_for_pair(aa,bb,ix,jx,samples,kind,gradient_weight)
        np.fill_diagonal(mat,np.inf);mats.append(mat)
    return mats


def normalized_costs(mats):
    """Dimensionless compatibility relative to competing outgoing/incoming edges."""
    out=[]
    for m in mats:
        # Second-best normalizer prevents near-zero first match from dominating.
        s=np.partition(m, min(1,len(m)-1),axis=1)[:,min(1,len(m)-1)]
        t=np.partition(m, min(1,len(m)-1),axis=0)[min(1,len(m)-1),:]
        floor=max(1e-6,float(np.nanmedian(m[np.isfinite(m)]))*.008) if np.any(np.isfinite(m)) else 1e-6
        s=np.maximum(s,floor);t=np.maximum(t,floor)
        with np.errstate(invalid="ignore",divide="ignore"):
            cost=m/np.sqrt(s[:,None]*t[None,:])
        cost[~np.isfinite(m)]=np.inf
        out.append(cost.astype(np.float32))
    return out


def _constrain_strip(m,widths,slots):
    allowed={(int(a),int(b)) for a,b in zip(slots[:-1],slots[1:])}
    m=m.copy()
    valid=np.zeros(m.shape,bool)
    for a,b in allowed:
        valid|=(widths[:,None]==a)&(widths[None,:]==b)
    m[~valid]=np.inf
    return m


def _cycle_components(succ):
    unseen=set(range(len(succ))); cycles=[]
    while unseen:
        j=min(unseen);path=[]
        while j in unseen:
            path.append(j);unseen.remove(j);j=int(succ[j])
        cycles.append(path)
    return cycles


def _strip_assignment(raw,widths,slots):
    """Min-cost path cover with a dummy endpoint, followed by cycle patching."""
    n=len(raw); raw=_constrain_strip(raw,widths,slots)
    m=normalized_costs([raw])[0]
    # Strip profiles are long; combine absolute and relative costs.
    finite=raw[np.isfinite(raw)]
    scale=max(float(np.median(finite)),1e-7) if finite.size else 1.
    cost=np.full((n+1,n+1),1e8)
    cost[:n,:n]=np.where(np.isfinite(m),np.log1p(m)+.6*raw/scale,1e8)
    cost[n,:n]=np.where(widths==slots[0],0.,1e8)
    cost[:n,n]=np.where(widths==slots[-1],0.,1e8)
    a,b=linear_sum_assignment(cost);succ=np.empty(n+1,int);succ[a]=b
    cycles=_cycle_components(succ)
    joined=0
    while len(cycles)>1:
        # Cheapest valid edge exchange joining the dummy path to a subtour.
        main=next(c for c in cycles if n in c)
        u=np.asarray(main);v=np.asarray([j for c in cycles if n not in c for j in c])
        du=succ[u];dv=succ[v]
        delta=cost[u[:,None],dv[None,:]]+cost[v[None,:],du[:,None]]-cost[u,du][:,None]-cost[v,dv][None,:]
        ix=np.unravel_index(np.argmin(delta),delta.shape)
        i,j=int(u[ix[0]]),int(v[ix[1]])
        succ[i],succ[j]=succ[j],succ[i];joined+=1
        cycles=_cycle_components(succ)
    order=[];i=int(succ[n])
    while i!=n:
        order.append(i);i=int(succ[i])
    order=np.array(order)
    if not np.array_equal(widths[order],slots):
        # Preserve valid dimensions even for nonperiodic unequal slot patterns.
        # Select the closest compatible ordering using a constrained assignment.
        d=np.abs(np.arange(n)[:,None]-np.arange(n)[None,:]).astype(float)
        d[slots[:,None]!=widths[order][None,:]]=1e8
        _,permutation=linear_sum_assignment(d);order=order[permutation]
    # Remove possible cyclic offset of a complete width-compatible chain.
    best=order;best_cost=float(cost[order[:-1],order[1:]].sum())
    for start in range(1,n):
        candidate=np.roll(order,-start)
        if np.array_equal(widths[candidate],slots):
            e=float(cost[candidate[:-1],candidate[1:]].sum())
            if e<best_cost:best_cost=e;best=candidate
    return best,joined


def _cycle_edges(costs,top_k):
    """Reward local 2x2 closed loops, not simply isolated attractive seams."""
    from shardelix.reconstruction.tiles.legacy import _grid_edges
    h,v=costs;n=len(h);k=min(top_k,n)
    hs=np.argsort(h,axis=1,kind="stable")[:,:k]
    vs=np.argsort(v,axis=1,kind="stable")[:,:k]
    support=[np.zeros((n,n),np.float32),np.zeros((n,n),np.float32)]
    for a in range(n):
        possibles=[]
        for b in hs[a]:
            for c in vs[a]:
                if b==c or not np.isfinite(h[a,b]+v[a,c]):continue
                ds=np.intersect1d(vs[b],hs[c],assume_unique=True)
                ds=ds[(ds!=a)&(ds!=b)&(ds!=c)]
                if not len(ds):continue
                e=h[a,b]+v[a,c]+v[b,ds]+h[c,ds]
                ix=np.argmin(e)
                if np.isfinite(e[ix]):possibles.append((float(e[ix]),int(b),int(c),int(ds[ix])))
        for e,b,c,d in sorted(possibles)[:2]:
            q=1./(1.+e)
            for ax,i,j in [(0,a,b),(1,a,c),(0,c,d),(1,b,d)]:
                support[ax][i,j]+=q
    edges=_grid_edges(costs,top_k)
    return sorted((p-.8*min(1.,support[a][i,j]),i,j,a) for p,i,j,a in edges)


def _assignment_refine(arr,costs,ph,pw,rh,cw,rounds=6):
    """Alternating checkerboard assignment, exact one-to-one per dimension."""
    rows,cols=arr.shape
    H,V=costs
    changes=0
    for _ in range(rounds):
        moved=0
        for parity in (0,1):
            yy,xx=np.where(np.indices(arr.shape).sum(axis=0)%2==parity)
            for ht,wd in set(zip(rh[yy],cw[xx])):
                mask=(rh[yy]==ht)&(cw[xx]==wd);y,x=yy[mask],xx[mask]
                if not len(y):continue
                candidates=arr[y,x].copy()
                c=np.zeros((len(y),len(y)),np.float64)
                for i,(r,s) in enumerate(zip(y,x)):
                    if s>0:c[i]+=H[arr[r,s-1],candidates]
                    if s<cols-1:c[i]+=H[candidates,arr[r,s+1]]
                    if r>0:c[i]+=V[arr[r-1,s],candidates]
                    if r<rows-1:c[i]+=V[candidates,arr[r+1,s]]
                c[~np.isfinite(c)]=1e8
                _,j=linear_sum_assignment(c)
                if c[np.arange(len(j)),j].sum()+1e-7<c.trace():
                    moved+=int(np.count_nonzero(j!=np.arange(len(j))))
                    arr[y,x]=candidates[j]
        changes+=moved
        if not moved:break
    return changes


def _energy(arr,costs):
    h,v=costs
    return float(h[arr[:,:-1],arr[:,1:]].sum()+v[arr[:-1,:],arr[1:,:]].sum())


def _roll_refine(arr,costs,rh,cw):
    best=arr.copy();e=_energy(best,costs)
    for y in range(len(rh)):
        if not np.array_equal(np.roll(rh,y),rh):continue
        for x in range(len(cw)):
            if not np.array_equal(np.roll(cw,x),cw):continue
            b=np.roll(arr,(y,x),(0,1));s=_energy(b,costs)
            if s<e-1e-7:best=b;e=s
    return best


def solve_straight(pieces,shape,grid,samples=0,kind="l2",gradient_weight=0.,
                   top_k=6,refine=5,cycle=True,normalize=True,threshold=None,raw_weight=0.):
    """Blind unrotated rectangular reconstruction with public shape only."""
    from shardelix.reconstruction.tiles.legacy import solve_grid
    rows,cols=grid;n=len(pieces)
    if rows*cols!=n:raise ValueError("Grid dimensions do not match count")
    if n==1:return {pieces[0].id:Pose(0,0)},{"algorithm":"straight-single"}
    ys,xs=_grid_bounds(*shape,*grid);rh,cw=np.diff(ys),np.diff(xs)
    ph=np.array([p.mask.shape[0] for p in pieces]);pw=np.array([p.mask.shape[1] for p in pieces])
    if rows==1 or cols==1:
        ax=0 if rows==1 else 1
        # Full-profile strips, with a modest memory/time bound for huge rasters.
        m=straight_costs(pieces,samples or 1024,kind,gradient_weight,axis_only=ax)[ax]
        order,patched=_strip_assignment(m,pw if ax==0 else ph,cw if ax==0 else rh)
        arr=order.reshape(rows,cols)
        diag={"algorithm":"straight-global-path","cycle_patches":patched}
    else:
        m=straight_costs(pieces,samples,kind,gradient_weight)
        costs=normalized_costs(m) if normalize else m
        if raw_weight:
            for a in range(2):
                scale=max(float(np.median(m[a][np.isfinite(m[a])])),1e-7)
                costs[a]=costs[a]+raw_weight*m[a]/scale
        edges=_cycle_edges(costs,top_k) if cycle else None
        if threshold is not None:
            from shardelix.reconstruction.tiles.legacy import _grid_edges
            edges=_grid_edges(costs,top_k) if edges is None else edges
            edges=[(p,i,j,a) for p,i,j,a in edges if costs[a][i,j]<threshold]
        poses,diag=solve_grid(pieces,shape,grid,top_k=top_k,refine=1,costs=costs,edge_list=edges,contact_check=normalize)
        # Guaranteed exact size-constrained completion if graph packing fails.
        arr=np.full(grid,-1,int);lookup={p.id:i for i,p in enumerate(pieces)}
        for k,p in poses.items():
            arr[int(np.searchsorted(ys,p.y)),int(np.searchsorted(xs,p.x))]=lookup[k]
        missing=list(set(range(n))-set(arr.ravel()))
        for y,x in zip(*np.where(arr<0)):
            ix=next(i for i in missing if ph[i]==rh[y] and pw[i]==cw[x])
            arr[y,x]=ix;missing.remove(ix)
        moves=_assignment_refine(arr,costs,ph,pw,rh,cw,refine)
        moves+=_swap_refine(arr,costs,rh,cw,rounds=refine)
        arr=_roll_refine(arr,costs,rh,cw)
        diag.update({"algorithm":"straight-cycle-grid","assignment_moves":moves,"compatibility_energy":_energy(arr,costs)})
    poses={pieces[int(i)].id:Pose(int(xs[x]),int(ys[y])) for y,x in np.ndindex(arr.shape) for i in [arr[y,x]]}
    duplicates=Counter((p.rgba.shape,hashlib.sha256(p.rgba.tobytes()).digest()) for p in pieces)
    return poses,{**diag,"kind":kind,"gradient_weight":gradient_weight,"samples":samples,
                  "identical_piece_groups":sum(v>1 for v in duplicates.values()),
                  "identical_piece_count":sum(v for v in duplicates.values() if v>1)}


def _swap_refine(arr,costs,rh,cw,rounds=8):
    """All-piece swaps across both checkerboard parities, accepted only by seam energy.

    Evaluate disjoint swaps in a batch, then recompute the full image costs.
    """
    rows,cols=arr.shape;n=arr.size
    H,V=costs
    dims=np.array([(rh[y],cw[x]) for y,x in np.ndindex(arr.shape)])
    coords=np.indices(arr.shape).reshape(2,-1).T
    total=0
    for _ in range(rounds):
        seq=arr.ravel().copy();C=np.zeros((n,n),np.float32)
        for i,(r,s) in enumerate(coords):
            if s>0:C[i]+=H[arr[r,s-1],seq]
            if s<cols-1:C[i]+=H[seq,arr[r,s+1]]
            if r>0:C[i]+=V[arr[r-1,s],seq]
            if r<rows-1:C[i]+=V[seq,arr[r+1,s]]
        diag=C.diagonal().copy();delta=C+C.T-diag[:,None]-diag[None,:]
        delta[(dims[:,None]!=dims[None,:]).any(axis=2)]=np.inf
        # Neighbouring exchange shares one seam, so the independent delta is invalid.
        for i,(r,s) in enumerate(coords):
            for rr,ss in ((r-1,s),(r+1,s),(r,s-1),(r,s+1)):
                if 0<=rr<rows and 0<=ss<cols:delta[i,rr*cols+ss]=np.inf
        delta[np.tril_indices(n)]=np.inf
        ii,jj=np.where(delta<-1e-6)
        if not len(ii):break
        ordered=np.argsort(delta[ii,jj]);blocked=set();moves=0
        for ix in ordered:
            i,j=int(ii[ix]),int(jj[ix])
            if i in blocked or j in blocked:continue
            y,x=coords[i];yy,xx=coords[j]
            arr[y,x],arr[yy,xx]=arr[yy,xx],arr[y,x]
            for r,s in [(y,x),(yy,xx)]:
                for rr,ss in [(r,s),(r-1,s),(r+1,s),(r,s-1),(r,s+1)]:
                    if 0<=rr<rows and 0<=ss<cols:blocked.add(int(rr*cols+ss))
            moves+=1
        total+=moves
        if not moves:break
    return total
