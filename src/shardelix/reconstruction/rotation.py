"""Blind quarter-turn alignment of long strips and rectangular tiles."""
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist
from .common import Fragment,Pose
from .straight import solve_straight
from .grid import solve_rotated_grid


def _signed_tree(same,flipped,neighbors=12):
    """Synchronize binary orientations using informative nearest-neighbor edges."""
    n=len(same)
    if n==1:return np.zeros(1,int)
    costs=np.minimum(same,flipped)
    np.fill_diagonal(costs,np.inf)
    scale=max(float(np.median(costs[np.isfinite(costs)])),1e-8)
    nn=np.argsort(costs,axis=1)[:,:min(neighbors,n-1)]
    links={}
    for i in range(n):
        for j in nn[i]:
            a,b=sorted((i,int(j)))
            if a==b or not np.isfinite(costs[a,b]):continue
            good=min(same[a,b],flipped[a,b]);bad=max(same[a,b],flipped[a,b])
            confidence=(1. if not np.isfinite(bad) else (bad-good)/(bad+scale*.01))/(1+good/(scale*.08))
            links[a,b]=(float(confidence),int(flipped[a,b]<same[a,b]))
    parent=list(range(n));parity=[0]*n
    def find(i):
        if parent[i]!=i:
            p=parent[i];root,s=find(p);parity[i]^=s;parent[i]=root
        return parent[i],parity[i]
    for (i,j),(c,flip) in sorted(links.items(),key=lambda q:-q[1][0]):
        a,x=find(i);b,y=find(j)
        if a!=b:parent[b]=a;parity[b]=x^y^flip
    signs=np.array([find(i)[1] for i in range(n)])
    # Coordinate-descent refinement of signed orientation agreement.
    graph=[[] for _ in range(n)]
    for (i,j),(c,f) in links.items():
        graph[i].append((j,c,f));graph[j].append((i,c,f))
    for _ in range(5):
        changed=0
        for i in range(n):
            gain=sum(c if (signs[i]^signs[j])!=f else -c for j,c,f in graph[i])
            if gain>1e-9:signs[i]^=1;changed+=1
        if not changed:break
    return signs



def _edge_pair_costs(pieces,axis,samples=512):
    """Best possible contact under same or reversed 180-degree orientation."""
    n=len(pieces)
    sides=[]
    for p in pieces:
        im=p.rgba[:,:,:3].astype(np.float32)/255.
        a,b=(im[:,0],im[:,-1]) if axis==0 else (im[0],im[-1])
        sides.append((a,b))
    lens=np.array([len(a) for a,b in sides])
    same=np.full((n,n),np.inf);flip=np.full((n,n),np.inf)
    for L in np.unique(lens):
        inds=np.flatnonzero(lens==L)
        x=np.linspace(0,L-1,min(samples,L))
        def descriptor(a):
            return np.stack([np.interp(x,np.arange(L),a[:,ch]) for ch in range(3)],axis=1)
        a=np.stack([descriptor(sides[i][0]) for i in inds])
        b=np.stack([descriptor(sides[i][1]) for i in inds])
        fa,fb=a.reshape(len(a),-1),b.reshape(len(b),-1)
        ca=cdist(fa,fb,"sqeuclidean")/fa.shape[1]
        # The same-order two endpoints must be opposite sides. If a piece
        # turns 180°, either equal side is valid only with reversed samples.
        cb=cdist(fa,a[:,::-1].reshape(len(a),-1),"sqeuclidean")/fa.shape[1]
        cc=cdist(fb,b[:,::-1].reshape(len(b),-1),"sqeuclidean")/fa.shape[1]
        same[np.ix_(inds,inds)]=np.minimum(ca,ca.T)
        flip[np.ix_(inds,inds)]=np.minimum(cb,cc)
    return same,flip

def _canonical_tiles(pieces,shape,grid):
    rows,cols=grid;h,w=shape
    from .straight import _grid_bounds
    y,x=_grid_bounds(h,w,rows,cols)
    allowed=set((int(a),int(b)) for a in np.diff(y) for b in np.diff(x))
    canonical=[];turns=[];ambiguous=False
    for p in pieces:
        possible=[k for k in (0,1) if p.rotated(k).shape[:2] in allowed]
        if not possible:raise ValueError("Piece dimensions are incompatible with the public grid")
        ambiguous|=len(possible)>1
        k=possible[0];turns.append(k);canonical.append(Fragment(p.id,p.rotated(k)))
    return canonical,np.array(turns),ambiguous


def _strip_cost_tensor(canonical,axis,samples,kind):
    from .straight import _edge,_descriptor
    left,right=("left","right") if axis==0 else ("top","bottom")
    A=np.stack([_descriptor(_edge(p.rgba,left),samples)[0] for p in canonical])
    B=np.stack([_descriptor(_edge(p.rgba,right),samples)[0] for p in canonical])
    a=A.reshape(len(A),-1);b=B.reshape(len(A),-1)
    ar=A[:,::-1].reshape(len(A),-1);br=B[:,::-1].reshape(len(A),-1)
    def dist(x,y):
        m=cdist(x,y,"cityblock" if kind=="l1" else "sqeuclidean")/x.shape[1]
        return (m if kind=="l1" else np.sqrt(m)).astype(np.float32)/255
    raw=np.empty((2,2,len(A),len(A)),np.float32)
    raw[0,0]=dist(b,a);raw[0,1]=dist(b,br);raw[1,0]=dist(ar,a);raw[1,1]=raw[0,0].T
    for x in raw:
        for m in x:np.fill_diagonal(m,np.inf)
    return raw


def _orientation_dp(raw,order):
    n=len(order);back=np.zeros((n,2),int);dp=np.zeros(2)
    for k,(i,j) in enumerate(zip(order[:-1],order[1:]),1):
        val=dp[:,None]+raw[:,:,i,j]
        back[k]=np.argmin(val,axis=0);dp=val.min(axis=0)
    s=np.zeros(n,int);s[-1]=np.argmin(dp)
    for k in range(n-1,0,-1):s[k-1]=back[k,s[k]]
    out=np.zeros(n,int);out[order]=s
    return out


def solve_rotated_strips(pieces,shape,grid,kind="l1",samples=1024,**kwargs):
    """Joint binary orientation and constrained Hamiltonian path refinement."""
    from .solver import objective
    from .straight import _grid_bounds,_strip_assignment
    canonical,turns,ambiguous=_canonical_tiles(pieces,shape,grid)
    if len(pieces)==1:
        return {pieces[0].id:Pose(0,0,int(turns[0]))},{"algorithm":"rotated-strip-single"}
    axis=0 if grid[0]==1 else 1
    raw=_strip_cost_tensor(canonical,axis,samples,kind)
    same=np.minimum(raw[0,0],raw[1,1]);rev=np.minimum(raw[0,1],raw[1,0])
    signs=_signed_tree(same,rev,neighbors=12)
    ys,xs=_grid_bounds(*shape,*grid)
    slots=np.diff(xs) if axis==0 else np.diff(ys)
    widths=np.array([p.mask.shape[1 if axis==0 else 0] for p in canonical])
    best=None;ids=np.arange(len(pieces))
    trials=0
    for flip in (0,1):
        s=signs^flip
        for _ in range(4):
            mat=raw[s[:,None],s[None,:],ids[:,None],ids[None,:]]
            order,patches=_strip_assignment(mat,widths,slots)
            # Optimize all rotations jointly for this order, then reconsider order.
            new_s=_orientation_dp(raw,order)
            for q in (s,new_s):
                poses={pieces[i].id:Pose(int(xs[x]),int(ys[y]),int((turns[i]+2*q[i])%4))
                       for (y,x),i in zip(np.ndindex(grid),order)}
                score=objective(pieces,poses,shape);trials+=1
                if best is None or score>best[0]:best=(score,poses,patches)
            if np.array_equal(s,new_s):break
            s=new_s
    return best[1],{"algorithm":"rotated-strip-joint-path","cycle_patches":best[2],
                    "orientation_trials":trials,"dimension_axis_ambiguous":bool(ambiguous),"kind":kind}

def solve_rotated_tiles(pieces,shape,grid,samples=24,gradient_weight=.25,top_k=3,**kwargs):
    """Rotation-aware tile graph, followed by a dimension-constrained upright solve.

    The graph determines relative rotations using pixels, then multiple
    canvas-compatible global orientations are tested. Missing orientation
    states use the permitted dimension orientation as an initialization.
    """
    from .solver import orient_to_canvas,objective
    canonical,turns,ambiguous=_canonical_tiles(pieces,shape,grid)
    from .straight import _grid_bounds
    ys,xs=_grid_bounds(*shape,*grid)
    allowed=set((int(a),int(b)) for a in np.diff(ys) for b in np.diff(xs))
    best=None
    candidates=[]
    if not ambiguous:
        s0,f0=_edge_pair_costs(canonical,0)
        s1,f1=_edge_pair_costs(canonical,1)
        # Calibrate the two directions separately before nearest-neighbor
        # orientation synchronization. This uses only edge cost statistics.
        def nrm(m,f):
            v=np.minimum(m,f);v=v[np.isfinite(v)&(v>0)]
            scale=max(float(np.median(v)),1e-8) if len(v) else 1.
            return m/scale,f/scale
        s0,f0=nrm(s0,f0);s1,f1=nrm(s1,f1)
        signs=_signed_tree(np.minimum(s0,s1),np.minimum(f0,f1),neighbors=8)
        for g in (0,1):candidates.append(((turns+2*(signs^g))%4).tolist())
    initial,info,state_costs,mapping=_oriented_grid_graph(pieces,shape,grid,kind="l1",top_k=top_k)
    initial=orient_to_canvas(pieces,initial,shape)
    best=(objective(pieces,initial,shape),initial,info)
    for g in range(4 if ambiguous else 2):
        ks=[]
        for p,k0 in zip(pieces,turns):
            k=(initial[p.id].k+g*(1 if ambiguous else 2))%4 if p.id in initial else int(k0)
            if p.rotated(k).shape[:2] not in allowed:k=int(k0)
            ks.append(k)
        candidates.append(ks)
    for ks in candidates:
        oriented=[Fragment(p.id,p.rotated(k)) for p,k in zip(pieces,ks)]
        try:pred,diag=solve_straight(oriented,shape,grid,kind="l1")
        except (ValueError,StopIteration):continue
        lookup={p.id:k for p,k in zip(pieces,ks)}
        poses={i:Pose(p.x,p.y,lookup[i]) for i,p in pred.items()}
        poses=_refine_grid_turns(pieces,poses,shape,grid,state_costs,mapping)
        score=objective(pieces,poses,shape)
        if score>best[0]:best=(score,poses,{**diag,"algorithm":"rotated-grid-sync-assignment"})
        # One more full assignment from the context-refined orientations.
        if len(poses)==len(pieces):
            ks2=[poses[p.id].k for p in pieces]
            if ks2!=list(ks):
                oriented2=[Fragment(p.id,p.rotated(k)) for p,k in zip(pieces,ks2)]
                try:pp,dd=solve_straight(oriented2,shape,grid,kind="l1")
                except (ValueError,StopIteration):continue
                lookup2={p.id:k for p,k in zip(pieces,ks2)}
                p2={i:Pose(p.x,p.y,lookup2[i]) for i,p in pp.items()}
                p2=_refine_grid_turns(pieces,p2,shape,grid,state_costs,mapping)
                val=objective(pieces,p2,shape)
                if val>best[0]:best=(val,p2,{**dd,"algorithm":"rotated-grid-context-refinement"})
    if len(best[1])==len(pieces):
        refined=_joint_grid_refine(pieces,best[1],shape,grid,state_costs,mapping)
        val=objective(pieces,refined,shape)
        if val>best[0]:best=(val,refined,{**best[2],"algorithm":"rotated-grid-joint-assignment"})
    return best[1],{**best[2],"orientation_graph_placed":len(initial),"dimension_axis_ambiguous":bool(ambiguous)}


def _oriented_grid_graph(pieces,shape,grid,kind="l1",top_k=4):
    """Graph with only physically permitted quarter-turn states, full edge profiles."""
    from .straight import straight_costs,normalized_costs,_cycle_edges,_grid_bounds
    from .geometry import R,assemble
    from .grid import _grid_edges
    ys,xs=_grid_bounds(*shape,*grid)
    allowed=set((int(a),int(b)) for a in np.diff(ys) for b in np.diff(xs))
    states=[];mapping=[]
    for i,p in enumerate(pieces):
        for k in range(4):
            a=p.rotated(k)
            if a.shape[:2] in allowed:
                states.append(Fragment(p.id,a));mapping.append((i,k))
    costs=straight_costs(states,samples=0,kind=kind,gradient_weight=0.)
    ids=np.array([i for i,k in mapping])
    for m in costs:m[ids[:,None]==ids[None,:]]=np.inf
    norms=normalized_costs(costs)
    edges=_cycle_edges(norms,top_k)
    seen={}
    def origin(piece,k):
        h,w=piece.rgba.shape[:2]
        return -(np.array([[0,0],[w,0],[0,h],[w,h]])@R[k].T).min(axis=0)
    for priority,a,b,axis in edges:
        i,ka=mapping[a];j,kb=mapping[b]
        if i==j:continue
        h,w=states[a].rgba.shape[:2]
        shift=np.array([w,0] if axis==0 else [0,h])
        r=(kb-ka)%4
        t=R[-ka%4]@(shift+origin(pieces[j],kb)-origin(pieces[i],ka))
        if i>j:i,j=j,i;r=-r%4;t=-R[r]@t
        key=(i,j,r,int(t[0]),int(t[1]))
        seen[key]=min(priority,seen.get(key,float("inf")))
    ranked=sorted(seen.items(),key=lambda x:x[1])
    votes=[(max(2,int(100000/(2+max(0,p)))),*k) for k,p in ranked]
    poses,diag=assemble(pieces,votes,shape,min_votes=1)
    return poses,{**diag,"algorithm":"oriented-full-profile-grid-graph"},costs,mapping


def _refine_grid_turns(pieces,poses,shape,grid,costs,mapping):
    """Optimize orientations for a fixed valid grid and return chosen state indices."""
    from .straight import _grid_bounds
    ys,xs=_grid_bounds(*shape,*grid)
    arr=np.full(grid,-1,int)
    ixs={p.id:i for i,p in enumerate(pieces)}
    mp={ik:j for j,ik in enumerate(mapping)}
    for id,p in poses.items():
        y=int(np.searchsorted(ys,p.y));x=int(np.searchsorted(xs,p.x))
        if y>=grid[0] or x>=grid[1] or xs[x]!=p.x or ys[y]!=p.y:return poses
        key=(ixs[id],p.k)
        if key not in mp:return poses
        arr[y,x]=mp[key]
    if (arr<0).any():return poses
    states={}
    for j,(i,k) in enumerate(mapping):states.setdefault(i,[]).append(j)
    for _ in range(4):
        changes=0
        for y,x in np.ndindex(grid):
            cur=arr[y,x];i,k=mapping[cur]
            options=[j for j in states[i] if pieces[i].rotated(mapping[j][1]).shape[:2]==(ys[y+1]-ys[y],xs[x+1]-xs[x])]
            def c(q):
                total=0.
                if x>0:total+=costs[0][arr[y,x-1],q]
                if x+1<grid[1]:total+=costs[0][q,arr[y,x+1]]
                if y>0:total+=costs[1][arr[y-1,x],q]
                if y+1<grid[0]:total+=costs[1][q,arr[y+1,x]]
                return total
            best=min(options,key=c)
            if c(best)<c(cur)-1e-8:arr[y,x]=best;changes+=1
        if not changes:break
    return {pieces[mapping[arr[y,x]][0]].id:Pose(int(xs[x]),int(ys[y]),int(mapping[arr[y,x]][1])) for y,x in np.ndindex(grid)}


def _joint_grid_refine(pieces,poses,shape,grid,costs,mapping,rounds=5):
    """Optimize both unique piece assignment and quarter-turn state per grid slot."""
    from scipy.optimize import linear_sum_assignment
    from .straight import _grid_bounds,normalized_costs
    ys,xs=_grid_bounds(*shape,*grid);rows,cols=grid;n=len(pieces)
    rh,cw=np.diff(ys),np.diff(xs)
    ixs={p.id:i for i,p in enumerate(pieces)};mi={ik:j for j,ik in enumerate(mapping)}
    arr=np.full(grid,-1,int)
    for i,p in poses.items():
        y=int(np.searchsorted(ys,p.y));x=int(np.searchsorted(xs,p.x))
        if y>=rows or x>=cols or ys[y]!=p.y or xs[x]!=p.x:return poses
        if (ixs[i],p.k) not in mi:return poses
        arr[y,x]=mi[ixs[i],p.k]
    if (arr<0).any():return poses
    # Relative compatibility tempers low-texture absolute seam bias.
    H,V=normalized_costs(costs)
    st_dims=np.array([pieces[i].rotated(k).shape[:2] for i,k in mapping])
    by_id=np.full((n,4),-1,int)
    for j,(i,k) in enumerate(mapping):by_id[i,k]=j
    ids=np.array([i for i,k in mapping])
    coords=np.indices(grid).reshape(2,-1).T
    for _ in range(rounds):
        moved=0
        for parity in (0,1):
            yy,xx=np.where(np.indices(grid).sum(axis=0)%2==parity)
            for h,w in set(zip(rh[yy],cw[xx])):
                valid=(rh[yy]==h)&(cw[xx]==w);y,x=yy[valid],xx[valid]
                if not len(y):continue
                current=arr[y,x].copy();tile_ids=ids[current]
                states=by_id[tile_ids]
                valid_s=(states>=0)&(st_dims[np.maximum(states,0)]==[h,w]).all(axis=-1)
                safe=np.maximum(states,0);C=np.zeros((len(y),len(y),4),np.float32)
                for j,(r,s) in enumerate(zip(y,x)):
                    if s>0:C[j]+=H[arr[r,s-1],safe]
                    if s+1<cols:C[j]+=H[safe,arr[r,s+1]]
                    if r>0:C[j]+=V[arr[r-1,s],safe]
                    if r+1<rows:C[j]+=V[safe,arr[r+1,s]]
                C[:,~valid_s]=np.inf
                best_state=np.argmin(C,axis=2);low=C.min(axis=2)
                low[~np.isfinite(low)]=1e8
                _,perm=linear_sum_assignment(low)
                new=safe[perm,best_state[np.arange(len(y)),perm]]
                # Valid current orientation and non-increasing true compatible cost.
                before=sum(C[j,j,mapping[s][1]] for j,s in enumerate(current))
                if float(low[np.arange(len(y)),perm].sum())+1e-6<before:
                    moved+=int(np.count_nonzero(new!=current));arr[y,x]=new
        if not moved:break
    return {pieces[mapping[arr[y,x]][0]].id:Pose(int(xs[x]),int(ys[y]),int(mapping[arr[y,x]][1])) for y,x in np.ndindex(grid)}


def solve_unknown_color_strips(pieces):
    """Blind opaque strips with a common long dimension and arbitrary widths.

    RGB/luminance-chrominance seam profiles and binary orientation synchronization
    operate on the canonical vertical strips. No uniform column grid is imposed.
    """
    from .document import chain
    canonical=[];turns=[]
    for p in pieces:
        k=int(p.rgba.shape[0]<p.rgba.shape[1])
        canonical.append(Fragment(p.id,p.rotated(k)));turns.append(k)
    n=len(pieces)
    if n==1:return {pieces[0].id:Pose(0,0,turns[0])},{"algorithm":"color-strip-single"}
    raw=_strip_cost_tensor(canonical,0,512,"l1")
    same=np.minimum(raw[0,0],raw[1,1]);flipped=np.minimum(raw[0,1],raw[1,0])
    signs=_signed_tree(np.minimum(same,same.T),np.minimum(flipped,flipped.T))
    ids=np.arange(n);best=None
    for seed in (signs,np.zeros(n,int)):
        s=seed
        for _ in range(4):
            mat=raw[s[:,None],s[None,:],ids[:,None],ids[None,:]]
            order=chain(mat,.35)
            q=_orientation_dp(raw,order)
            for v in (s,q):
                score=float(raw[v[order[:-1]],v[order[1:]],order[:-1],order[1:]].sum())
                if best is None or score<best[0]:best=(score,order.copy(),v.copy())
            if np.array_equal(s,q):break
            s=q
    score,order,signs=best
    poses={};x=0
    for i in order:
        k=int((turns[i]+2*signs[i])%4)
        poses[pieces[i].id]=Pose(x,0,k);x+=pieces[i].rotated(k).shape[1]
    return poses,{"algorithm":"unknown-width-color-strip-chain","blind_seam_energy":score,
                  "orientation_ambiguous":True,"document_missing_unknown":True}
