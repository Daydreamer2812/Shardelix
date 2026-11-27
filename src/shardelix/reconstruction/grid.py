"""Classical image-only grid and strip baseline: seam costs and constrained graph.

A grid with arbitrary texture cannot be solved uniquely for every image.
Parameters are exposed for validation/tuning rather than hard-coded per test.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from .common import Fragment,Pose

SIDES=("top","right","bottom","left")


def _feature(a,side,samples):
    rgb=a[:,:,:3].astype(np.float32)/255
    if side=="left": e=rgb[:,0]; q=rgb[:,min(1,rgb.shape[1]-1)]
    elif side=="right": e=rgb[:,-1];q=rgb[:,max(0,rgb.shape[1]-2)]
    elif side=="top": e=rgb[0];q=rgb[min(1,rgb.shape[0]-1)]
    else:e=rgb[-1];q=rgb[max(0,rgb.shape[0]-2)]
    t=np.linspace(0,len(e)-1,samples)
    def sample(x):
        return np.array([np.interp(t,np.arange(len(x)),x[:,i]) for i in range(3)]).T.ravel()
    return sample(e),sample(e-q),len(e)


def seam_costs(pieces: list[Fragment],samples: int=40,gradient_weight: float=.6):
    feats={s:[_feature(p.rgba,s,samples) for p in pieces] for s in SIDES}
    costs=[]
    for sa,sb in (("right","left"),("bottom","top")):
        ea,ga,la=map(np.asarray,zip(*feats[sa]))
        eb,gb,lb=map(np.asarray,zip(*feats[sb]))
        plain=cdist(ea,eb,"sqeuclidean")
        # Extrapolate across the seam from both sides. The two interior gradients
        # face opposite directions, hence e+g predicted against opposite edge.
        pred=(cdist(ea+ga,eb,"sqeuclidean")+cdist(ea,eb+gb,"sqeuclidean"))/2
        mat=(plain*(1-gradient_weight)+gradient_weight*pred)/(samples*3)
        mat=np.sqrt(mat).astype(np.float32)
        mat[la[:,None]!=lb[None,:]]=np.inf
        np.fill_diagonal(mat,np.inf)
        costs.append(mat)
    return costs


def _grid_edges(costs,top_k):
    n=len(costs[0])
    out=[]
    for axis,mat in enumerate(costs):
        k=min(top_k,max(1,n-1))
        idx=np.argsort(mat,axis=1,kind="stable")[:,:k]
        reverse=np.argsort(mat,axis=0,kind="stable")[:k,:]
        baseline=np.take_along_axis(mat,idx,axis=1)
        # Local scale normalized by alternative candidates; absolute seam cost
        # breaks otherwise equal rankings without inventing a confidence score.
        for i in range(n):
            for rank,j in enumerate(idx[i]):
                c=float(mat[i,j])
                if not np.isfinite(c):continue
                rr=np.flatnonzero(reverse[:,j]==i)
                if not len(rr):continue
                alt=float(baseline[i,min(1,baseline.shape[1]-1)])
                ralt=float(mat[reverse[min(1,k-1),j],j])
                denom=max(min(alt,ralt),1e-7)
                priority=(rank+int(rr[0]))*.55 + c/denom + .20*c
                out.append((priority,i,int(j),axis))
    return sorted(out)


def solve_grid(pieces: list[Fragment],shape:tuple[int,int],grid:tuple[int,int],
               samples:int=40,gradient_weight:float=.6,top_k:int=5,
               refine:int=2, *, costs=None, edge_list=None, contact_check=False) -> tuple[dict[str,Pose],dict]:
    rows,cols=grid;n=len(pieces);h,w=shape
    if rows*cols!=n:
        raise ValueError("Grid dimensions do not match count")
    ys=np.flatnonzero(np.r_[True,np.diff(np.arange(h)*rows//h)!=0,True])
    xs=np.flatnonzero(np.r_[True,np.diff(np.arange(w)*cols//w)!=0,True])
    rh=np.diff(ys);cw=np.diff(xs)
    ph=np.array([p.mask.shape[0] for p in pieces]);pw=np.array([p.mask.shape[1] for p in pieces])
    if n==1:
        return {pieces[0].id:Pose(0,0)},{"algorithm":"blind-grid","components":1}
    horizontal,vertical=costs=seam_costs(pieces,samples,gradient_weight) if costs is None else costs
    edges=_grid_edges(costs,top_k) if edge_list is None else edge_list
    comps={i:{i:(0,0)} for i in range(n)};owners=list(range(n))
    def shifts_possible(comp):
        minx=min(x for x,y in comp.values());maxx=max(x for x,y in comp.values())
        miny=min(y for x,y in comp.values());maxy=max(y for x,y in comp.values())
        if maxx-minx>=cols or maxy-miny>=rows:
            return []
        # Exact equal-sized grid slots are known from public geometry, not truth.
        sx=set(range(-minx,cols-maxx));sy=set(range(-miny,rows-maxy))
        for i,(x,y) in comp.items():
            sx={s for s in sx if cw[s+x]==pw[i]}
            sy={s for s in sy if rh[s+y]==ph[i]}
            if not sx or not sy:return []
        return [(x,y) for y in sorted(sy) for x in sorted(sx)]
    for _,i,j,axis in edges:
        ai,bi=owners[i],owners[j]
        if ai==bi:continue
        a,b=comps[ai],comps[bi]
        ax,ay=a[i];bx,by=b[j]
        dx=ax+(axis==0)-bx;dy=ay+(axis==1)-by
        if len(a)<len(b):
            ai,bi=bi,ai;a,b=b,a;dx,dy=-dx,-dy
        transformed={q:(x+dx,y+dy) for q,(x,y) in b.items()}
        if set(a.values())&set(transformed.values()):continue
        if contact_check:
            inv={xy:q for q,xy in a.items()}
            cross=[]
            for q,(x,y) in transformed.items():
                for xx,yy,k,rev in [(x-1,y,0,True),(x+1,y,0,False),(x,y-1,1,True),(x,y+1,1,False)]:
                    if (xx,yy) in inv:
                        j0=inv[xx,yy]
                        cross.append(costs[k][j0,q] if rev else costs[k][q,j0])
            if len(cross)>1 and (np.mean(cross)>1.6 or max(cross)>5.):continue
        joined={**a,**transformed}
        if not shifts_possible(joined):continue
        comps[ai]=joined
        for q in b:owners[q]=ai
        del comps[bi]
        if len(comps)==1:break
    # Pack components in decreasing order; failed pack leaves pieces to a
    # dimension-constrained assignment instead of forcing an overlapping join.
    arr=np.full((rows,cols),-1,int)
    def local(i,x,y):
        c=0.;num=0
        for dx,dy,mat,rev in [(-1,0,horizontal,True),(1,0,horizontal,False),(0,-1,vertical,True),(0,1,vertical,False)]:
            xx,yy=x+dx,y+dy
            if 0<=xx<cols and 0<=yy<rows:
                j=arr[yy,xx]
                if j>=0:
                    val=float(mat[j,i] if rev else mat[i,j]);c+=val;num+=1
        return c,num
    leftovers=[]
    for comp in sorted(comps.values(),key=len,reverse=True):
        best=None
        for dx,dy in shifts_possible(comp):
            coords={i:(x+dx,y+dy) for i,(x,y) in comp.items()}
            if any(arr[y,x]>=0 for x,y in coords.values()):continue
            vals=[local(i,x,y) for i,(x,y) in coords.items()]
            count=sum(v[1] for v in vals)
            c=sum(v[0] for v in vals)
            score=(c/max(count,1) if count else .20,-count,dy,dx)
            if best is None or score<best[0]:best=(score,coords)
        if best is None:
            leftovers.extend(comp)
        else:
            for i,(x,y) in best[1].items():arr[y,x]=i
    for i in leftovers:
        choices=[]
        for y,x in zip(*np.where(arr<0)):
            if rh[y]!=ph[i] or cw[x]!=pw[i]:continue
            c,num=local(i,x,y)
            choices.append((c/max(num,1) if num else .2,-num,int(y),int(x)))
        if choices:
            _,_,y,x=min(choices);arr[y,x]=i
    # Pairwise local swaps reduce whole-grid energy without using truth.
    positions={int(i):(int(y),int(x)) for y,x in zip(*np.where(arr>=0)) for i in [arr[y,x]]}
    def incident(y,x):
        out=[]
        for dx,dy,axis in [(-1,0,0),(1,0,0),(0,-1,1),(0,1,1)]:
            xx,yy=x+dx,y+dy
            if 0<=xx<cols and 0<=yy<rows:
                if dx<0 or dy<0:out.append((yy,xx,y,x,axis))
                else:out.append((y,x,yy,xx,axis))
        return out
    def energy(edges):
        return sum(float(costs[a][arr[y,x],arr[yy,xx]]) for y,x,yy,xx,a in edges if arr[y,x]>=0 and arr[yy,xx]>=0)
    rng=np.random.default_rng(0)
    swaps=0
    best_neighbors=[np.argsort(m,axis=1,kind="stable")[:,:min(4,n)] for m in costs]
    best_predecessors=[np.argsort(m,axis=0,kind="stable")[:min(4,n),:] for m in costs]
    for _ in range(refine):
        for i in rng.permutation(n):
            if i not in positions:continue
            y,x=positions[i]
            options=set()
            for dx,dy,a in [(-1,0,0),(1,0,0),(0,-1,1),(0,1,1)]:
                xx,yy=x+dx,y+dy
                if not (0<=xx<cols and 0<=yy<rows):continue
                q=arr[yy,xx]
                if q<0:continue
                if dx<0 or dy<0:options.update(best_neighbors[a][q])
                else:options.update(best_predecessors[a][:,q])
            for j in sorted(options):
                if i==j or j not in positions or (ph[i],pw[i])!=(ph[j],pw[j]):continue
                yy,xx=positions[j]
                affected=set(incident(y,x)+incident(yy,xx));before=energy(affected)
                arr[y,x],arr[yy,xx]=j,i
                if energy(affected)+1e-9<before:
                    positions[i]=(yy,xx);positions[j]=(y,x);y,x=yy,xx;swaps+=1
                else:arr[y,x],arr[yy,xx]=i,j
    poses={pieces[i].id:Pose(int(xs[x]),int(ys[y])) for i,(y,x) in positions.items()}
    return poses,{"algorithm":"blind-grid-graph","components":len(comps),"swaps":swaps,"candidates":len(edges),
                  "samples":samples,"gradient_weight":gradient_weight,"top_k":top_k}


def solve_rotated_grid(pieces: list[Fragment],shape:tuple[int,int],grid:tuple[int,int],
                       samples:int=24,gradient_weight:float=.6,top_k:int=3):
    """Orientation-aware sparse seam matching; collision-constrained component assembly.

    This slower fallback supports unknown 90 degree rotations but is a heuristic.
    """
    from .geometry import R,assemble
    # Treat each actual fragment as four image states, then keep only unique
    # piece pairs with compatible physical dimensions.
    states=[Fragment(p.id,p.rotated(k)) for p in pieces for k in range(4)]
    costs=seam_costs(states,samples,gradient_weight)
    for mat in costs:
        for i in range(len(pieces)):
            mat[4*i:4*i+4,4*i:4*i+4]=np.inf
    edges=_grid_edges(costs,top_k)
    seen={}
    def origin(piece,k):
        h,w=piece.mask.shape
        return -(np.array([[0,0],[w,0],[0,h],[w,h]])@R[k].T).min(axis=0)
    for pri,a,b,axis in edges:
        i,ka=divmod(a,4);j,kb=divmod(b,4)
        if i==j:continue
        h,w=states[a].mask.shape
        shift=np.array([w,0] if axis==0 else [0,h])
        r=(kb-ka)%4
        t=R[-ka%4]@(shift+origin(pieces[j],kb)-origin(pieces[i],ka))
        if i>j:
            i,j=j,i;r=-r%4;t=-R[r]@t
        key=(i,j,r,int(t[0]),int(t[1]))
        seen[key]=min(pri,seen.get(key,float("inf")))
    ranked=sorted(seen.items(),key=lambda kv:kv[1])
    votes=[(max(1,int(10000/(1+v))),*k) for k,v in ranked]
    poses,diag=assemble(pieces,votes,shape,min_votes=1)
    diag.update({"algorithm":"blind-rotated-grid-graph","candidates":len(votes)})
    return poses,diag
