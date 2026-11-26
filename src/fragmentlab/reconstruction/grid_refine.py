"""Global row/column segment moves for grid layouts.

Exact one-to-one and dimension compatibility are preserved. Dynamic
programming chooses coordinated shifts across neighboring lines, avoiding a
local optimum where a big correctly matched area is displaced by a column.
"""
from __future__ import annotations
from functools import lru_cache
import numpy as np
from scipy.optimize import linear_sum_assignment
from .common import Pose
from .straight import _grid_bounds
from .grid_consensus import _neighbors, _slot_cost


def _energy(arr,cost):
    def edges(a,b,d):
        u=4*(a//4)+(d-a%4)%4
        v=4*(b//4)+(d+2-b%4)%4
        return cost[u,v]
    return float(edges(arr[:,:-1],arr[:,1:],1).sum()+edges(arr[:-1],arr[1:],2).sum())


@lru_cache(maxsize=32)
def _permutations(slot_tuple):
    slot=np.array(slot_tuple);n=len(slot);perms=[np.arange(n)]
    # Move one or two pieces across any contiguous interval, implementing
    # a left/right cyclic sub-line shift. More than one row may move together.
    for a in range(n-1):
        for b in range(a+2,n+1):
            for k in (1,-1,2,-2):
                if abs(k)>=b-a:continue
                p=np.arange(n);p[a:b]=np.roll(p[a:b],k)
                if np.array_equal(slot[p],slot):perms.append(p)
    return np.unique(np.asarray(perms),axis=0)

def _line_candidates(row,slot,cost,d,other_before,other_after,cross,cap):
    states=row[_permutations(tuple(slot))]
    u=4*(states[:,:-1]//4)+(d-states[:,:-1]%4)%4
    v=4*(states[:,1:]//4)+(d+2-states[:,1:]%4)%4
    internal=cost[u,v].sum(1)
    contextual=internal.copy()
    def dist(a,b):
        return cost[4*(a//4)+(cross-a%4)%4,4*(b//4)+(cross+2-b%4)%4].sum(-1)
    if other_before is not None:contextual+=dist(other_before[None,:],states)
    if other_after is not None:contextual+=dist(states,other_after[None,:])
    k=min(cap,len(states))
    selected=np.unique(np.r_[np.flatnonzero((states==row).all(1)),np.argsort(internal)[:k],np.argsort(contextual)[:k]])
    return states[selected],internal[selected]


def _lines(arr,cost,line_dims,axis,cap):
    if axis==1:
        a=arr;d=1;cross=2
    else:a=arr.T;d=2;cross=1
    choices=[];intra=[]
    for i,row in enumerate(a):
        v,c=_line_candidates(row,line_dims,cost,d,a[i-1] if i else None,a[i+1] if i+1<len(a) else None,cross,cap)
        choices.append(v);intra.append(c)
    dp=intra[0];back=[]
    for i in range(1,len(a)):
        p,q=choices[i-1],choices[i]
        u=4*(p//4)+(cross-p%4)%4;v=4*(q//4)+(cross+2-q%4)%4
        inter=cost[u[:,None,:],v[None,:,:]].sum(-1)
        val=dp[:,None]+inter
        back.append(np.argmin(val,axis=0));dp=intra[i]+np.min(val,axis=0)
    j=int(np.argmin(dp));out=[choices[-1][j]]
    for i in range(len(back)-1,-1,-1):
        j=int(back[i][j]);out.append(choices[i][j])
    out=np.asarray(out[::-1]);return out if axis==1 else out.T


def _assign(pieces,arr,rh,cw,cost,rotations,rounds=4):
    # Exact checkerboard matching: neighbors held fixed.
    out=arr.copy();total=0
    for _ in range(rounds):
        changed=0
        for parity in (0,1):
            yy,xx=np.where(np.indices(out.shape).sum(0)%2==parity)
            for h,w in set(zip(rh[yy],cw[xx])):
                m=(rh[yy]==h)&(cw[xx]==w);y,x=yy[m],xx[m];old=out[y,x].copy()
                ids=old//4;states=ids[:,None]*4+np.arange(4 if rotations else 1)[None,:]
                dims=np.array([pieces[s//4].rotated(s%4).shape[:2] for s in states.ravel()]).reshape(*states.shape,2)
                ok=(dims==[h,w]).all(-1)
                c=np.stack([_slot_cost(out,r,s,states.ravel(),cost)[0].reshape(states.shape) for r,s in zip(y,x)])
                c[:,~ok]=np.inf
                low=c.min(-1);orient=c.argmin(-1);low[~np.isfinite(low)]=1e8
                _,j=linear_sum_assignment(low);best=states[j,orient[np.arange(len(j)),j]]
                val=float(low[np.arange(len(j)),j].sum())
                base=float(c[np.arange(len(j)),np.arange(len(j)),old%4].sum())
                if val<base-1e-6:
                    out[y,x]=best;changed+=int(np.count_nonzero(best!=old))
        total+=changed
        if not changed:break
    return out,total


def refine_grid_lines(pieces,poses,shape,grid,rotations,raw,rounds=3,cap=50):
    if len(pieces)!=len(poses) or min(grid)<2:return poses,{"line_moves":0}
    ys,xs=_grid_bounds(*shape,*grid);rh,cw=np.diff(ys),np.diff(xs)
    a=np.full(grid,-1,int)
    for i,p in enumerate(pieces):
        pose=poses.get(p.id)
        if pose is None:return poses,{"line_moves":0}
        y,x=np.searchsorted(ys,pose.y),np.searchsorted(xs,pose.x)
        if y>=grid[0] or x>=grid[1] or ys[y]!=pose.y or xs[x]!=pose.x or p.rotated(pose.k).shape[:2]!=(rh[y],cw[x]) or a[y,x]>=0:return poses,{"line_moves":0}
        a[y,x]=4*i+pose.k
    if np.any(a<0):return poses,{"line_moves":0}
    _,_,second,_,scale=_neighbors(raw,3)
    with np.errstate(invalid='ignore',divide='ignore'):
        cost=np.log1p(raw/np.maximum(np.sqrt(second[:,None]*second[None,:]),scale*.00001))+.15*raw/scale
    cost[~np.isfinite(cost)]=1e7
    cost=cost.astype(np.float32)
    start=_energy(a,cost);moves=0;assignments=0
    for _ in range(rounds):
        prev=_energy(a,cost)
        if max(grid)<=64:
            for axis,dim in ((1,cw),(0,rh)):
                q=_lines(a,cost,dim,axis,cap)
                if _energy(q,cost)<_energy(a,cost)-1e-5:
                    moves+=int(np.count_nonzero(a!=q));a=q
        q,n=_assign(pieces,a,rh,cw,cost,rotations,4)
        if _energy(q,cost)<_energy(a,cost)-1e-5:a=q;assignments+=n
        if _energy(a,cost)>=prev-1e-5:break
    out={pieces[s//4].id:Pose(int(xs[x]),int(ys[y]),int(s%4)) for y,x in np.ndindex(grid) for s in [a[y,x]]}
    return out,{"line_moves":moves,"assignment_moves_v7":assignments,"grid_energy_before":start,"grid_energy_after":_energy(a,cost)}

def _repair(pieces,arr,mask,rh,cw,cost,rotations):
    """Destroy uncertain cells, reinsert by multi-neighbor margins, one-use only."""
    a=arr.copy();ids=a[mask]//4
    candidates=ids[:,None]*4+np.arange(4 if rotations else 1)[None,:]
    states=candidates.ravel()
    dims=np.array([pieces[s//4].rotated(s%4).shape[:2] for s in states])
    available=np.ones(len(states),bool)
    a[mask]=-1
    # Cache costs and update only those adjacent to newly placed cells.
    coords=np.column_stack(np.where(mask));m=len(coords)
    val=np.zeros((m,len(states)),np.float32);support=np.zeros(m,np.int16)
    same=(dims[None,:,:]==np.stack([rh[coords[:,0]],cw[coords[:,1]]],axis=1)[:,None,:]).all(-1)
    for q,(y,x) in enumerate(coords):
        val[q],support[q]=_slot_cost(a,y,x,states,cost)
    active=np.ones(m,bool);mp={tuple(v):i for i,v in enumerate(coords)}
    rows,cols=a.shape
    D=np.array([[0,-1],[1,0],[0,1],[-1,0]])
    # Evaluate best orientation per unique fragment; second-best must be
    # another piece. Dynamic score rewards multiple fixed neighbors.
    while active.any():
        pool=np.flatnonzero(active);valid=same[pool]&available[None,:]
        C=val[pool].copy();C[~valid]=1e8
        nturn=4 if rotations else 1
        low=C.reshape(len(pool),len(ids),nturn).min(-1)
        opt=np.argmin(C,axis=1);best=C[np.arange(len(pool)),opt]
        if len(ids)>1:second=np.partition(low,1,axis=1)[:,1]
        else:second=best+1
        gap=np.where(second<1e7,(second-best)/np.maximum(second,.01),1.)
        good=best<1e7
        quality=support[pool]+.8*gap-.18*best/np.maximum(support[pool],1)
        quality[~good]=-1e9
        if not good.any():return arr
        j=int(np.argmax(quality));q=pool[j];state=int(states[opt[j]]);y,x=coords[q]
        a[y,x]=state;active[q]=False;available[states//4==state//4]=False
        # For neighboring empty slot, new fixed state adds one contact term.
        for d,(dx,dy) in enumerate(D):
            r=mp.get((int(y+dy),int(x+dx)))
            if r is None or not active[r]:continue
            u=4*(states//4)+(d+2-states%4)%4
            v=4*(state//4)+(d-state%4)%4
            val[r]+=cost[u,v];support[r]+=1
    return a


def refine_grid_regions(pieces,poses,shape,grid,rotations,raw,iterations=8):
    """Deterministic bounded large-neighborhood search, blind energy acceptance."""
    from scipy.ndimage import gaussian_filter, binary_dilation
    if len(poses)!=len(pieces) or max(grid)>64 or len(pieces)<64:return poses,{"region_repairs":0}
    ys,xs=_grid_bounds(*shape,*grid);rh,cw=np.diff(ys),np.diff(xs)
    a=np.full(grid,-1,int)
    for i,p in enumerate(pieces):
        q=poses[p.id];y,x=np.searchsorted(ys,q.y),np.searchsorted(xs,q.x)
        if y>=grid[0] or x>=grid[1] or ys[y]!=q.y or xs[x]!=q.x or p.rotated(q.k).shape[:2]!=(rh[y],cw[x]) or a[y,x]>=0:return poses,{"region_repairs":0}
        a[y,x]=4*i+q.k
    if (a<0).any():return poses,{"region_repairs":0}
    _,_,second,_,scale=_neighbors(raw,3)
    with np.errstate(invalid='ignore',divide='ignore'):
        cost=np.log1p(raw/np.maximum(np.sqrt(second[:,None]*second[None,:]),scale*.00001))+.15*raw/scale
    cost[~np.isfinite(cost)]=1e7;cost=cost.astype(np.float32)
    init=_energy(a,cost);accepted=0
    rows,cols=grid;yy,xx=np.indices(grid);visited=np.zeros(grid,float)
    for i in range(iterations):
        err=np.zeros(grid,np.float32)
        for axis,d in ((1,1),(0,2)):
            u0=a[:,:-1] if axis==1 else a[:-1]
            v0=a[:,1:] if axis==1 else a[1:]
            u=4*(u0//4)+(d-u0%4)%4;v=4*(v0//4)+(d+2-v0%4)%4
            e=cost[u,v]
            if axis==1:err[:,:-1]+=e;err[:,1:]+=e
            else:err[:-1]+=e;err[1:]+=e
        # Several differently shaped repair neighborhoods, prioritizing
        # spatially concentrated high residuals rather than random swaps.
        if i%4==3:
            mask=binary_dilation(err>=np.quantile(err,.94),iterations=1)
        else:
            radius=max(2,int(min(grid)*(0.14 if i%3==0 else .21 if i%3==1 else .28)))
            score=gaussian_filter(err,float(radius)/2.)/(1.+visited*.9)
            y,x=np.unravel_index(np.argmax(score),grid)
            mask=(np.abs(yy-y)<=radius)&(np.abs(xx-x)<=radius)
            visited[mask]+=1
        if mask.sum()>min(len(pieces)*.45,400) or mask.sum()<4:continue
        q=_repair(pieces,a,mask,rh,cw,cost,rotations)
        q,_=_assign(pieces,q,rh,cw,cost,rotations,3)
        if _energy(q,cost)<_energy(a,cost)-1e-5:a=q;accepted+=1
    out={pieces[s//4].id:Pose(int(xs[x]),int(ys[y]),int(s%4)) for y,x in np.ndindex(grid) for s in [a[y,x]]}
    return out,{"region_repairs":accepted,"region_energy_before":init,"region_energy_after":_energy(a,cost)}


def refine_grid_phase(pieces,poses,shape,grid,rotations,raw):
    """Correct whole-grid cyclic phase/orientation under exact slot dimensions.

    A globally reversed unequal-slot grid can be locally coherent but one
    row off. Enumerate legal rotations and periodic shifts without using GT.
    The best toroidal cut is found analytically from row/column seam costs.
    """
    if len(poses)!=len(pieces):return poses,{"phase_shift":[0,0,0]}
    ys,xs=_grid_bounds(*shape,*grid);rh,cw=np.diff(ys),np.diff(xs)
    a=np.full(grid,-1,int)
    for i,p in enumerate(pieces):
        q=poses.get(p.id)
        if q is None:return poses,{"phase_shift":[0,0,0]}
        y,x=np.searchsorted(ys,q.y),np.searchsorted(xs,q.x)
        if y>=grid[0] or x>=grid[1] or ys[y]!=q.y or xs[x]!=q.x or p.rotated(q.k).shape[:2]!=(rh[y],cw[x]) or a[y,x]>=0:return poses,{"phase_shift":[0,0,0]}
        a[y,x]=4*i+q.k
    if np.any(a<0):return poses,{"phase_shift":[0,0,0]}
    _,_,second,_,scale=_neighbors(raw,3)
    with np.errstate(invalid='ignore',divide='ignore'):
        cost=np.log1p(raw/np.maximum(np.sqrt(second[:,None]*second[None,:]),scale*.00001))+.15*raw/scale
    cost[~np.isfinite(cost)]=1e7;cost=cost.astype(np.float32)
    dims=np.array([p.rotated(k).shape[:2] for p in pieces for k in range(4)])
    start=_energy(a,cost);best=(start,a,[0,0,0])
    for k in range(4 if rotations else 1):
        q=np.rot90(a,-k);q=4*(q//4)+(q%4+k)%4
        if q.shape!=a.shape:continue
        h,w=dims[q][...,0],dims[q][...,1]
        if not (np.all(h==h[:,:1]) and np.all(w==w[:1,:])):continue
        dy=[i for i in range(grid[0]) if np.array_equal(np.roll(h[:,0],i),rh)]
        dx=[i for i in range(grid[1]) if np.array_equal(np.roll(w[0],i),cw)]
        if not dy or not dx:continue
        def torus(d,axis):
            b=np.roll(q,-1,axis=axis)
            u=4*(q//4)+(d-q%4)%4;v=4*(b//4)+(d+2-b%4)%4
            return cost[u,v].sum(1-axis)
        row=torus(2,0);col=torus(1,1)
        sy=max(dy,key=lambda i:row[(grid[0]-1-i)%grid[0]])
        sx=max(dx,key=lambda i:col[(grid[1]-1-i)%grid[1]])
        moved=np.roll(q,(sy,sx),(0,1))
        score=_energy(moved,cost)
        if score<best[0]-1e-5:best=(score,moved,[k,int(sy),int(sx)])
    val,q,phase=best
    out={pieces[s//4].id:Pose(int(xs[x]),int(ys[y]),int(s%4)) for y,x in np.ndindex(grid) for s in [q[y,x]]}
    return out,{"phase_shift":phase,"phase_energy_before":start,"phase_energy_after":val}
