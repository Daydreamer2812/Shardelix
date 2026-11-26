"""V12 bounded blind refinements: multi-gap strip path and explicit frame anchors.

The priors are public uniform cells and digital straight outer walls. Neither
a source raster, IDs encoding order, nor evaluation truth is consumed.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter1d
from .common import Pose, normalize_pose_set

def _run_length(mask):
    a=np.flatnonzero(np.diff(np.r_[False,mask,False].astype(np.int8)))
    return int(np.max(a[1::2]-a[::2],initial=0))

def anchor_irregular_frame(pieces,poses,shape):
    """Resolve a translation axis only with independent flat-wall evidence.

    The existing relative geometry is never altered. Multiple flat pieces on
    exactly one outer extreme support an outer wall; contradictory/weak evidence
    leaves the relative gauge explicitly unresolved.
    """
    if not poses:return poses,{"frame_anchor_status":"empty","anchor_translation_slack":list(shape)}
    p,extent=normalize_pose_set(pieces,poses)
    slack=[shape[i]-extent[i] for i in range(2)]
    if min(slack)<0:return poses,{"frame_anchor_status":"outside-canvas"}
    lookup={q.id:q for q in pieces}
    support=[[],[],[],[]] # top, bottom, left, right
    for ident,q in p.items():
        m=lookup[ident].rotated(q.k)[:,:,3]>0;h,w=m.shape
        for j,b,side in [(0,q.y==0,m[0]),(1,q.y+h==extent[0],m[-1]),(2,q.x==0,m[:,0]),(3,q.x+w==extent[1],m[:,-1])]:
            n=_run_length(side)
            # Full, straight edge, not a short tangent to a rounded curve.
            if b and n>=max(10,.86*len(side)):support[j].append(n)
    offset=[0,0];remaining=slack.copy();decisions=[]
    for a in range(2):
        if not slack[a]:decisions.append("full-extent");continue
        u,v=support[2*a:2*a+2]
        # Two independent fragments and convincing accumulated wall support.
        threshold=max(20,.18*extent[1-a])
        if len(u)>=2 and sum(u)>=threshold and not v:
            remaining[a]=0;decisions.append("leading-wall")
        elif len(v)>=2 and sum(v)>=threshold and not u:
            offset[a]=slack[a];remaining[a]=0;decisions.append("trailing-wall")
        else:decisions.append("unresolved")
    # Preserve the original translation along axes without evidence, rather
    # than silently pretending a relative origin is an absolute anchor.
    ymin=min(q.y for q in poses.values());xmin=min(q.x for q in poses.values())
    for a,base in enumerate((ymin,xmin)):
        if remaining[a]:offset[a]=max(0,min(slack[a],base))
    out={i:Pose(q.x+offset[1],q.y+offset[0],q.k) for i,q in p.items()}
    return out,{"frame_anchor_status":"resolved" if not any(remaining) else "ambiguous",
                "frame_anchor_decisions_yx":decisions,"frame_anchor_support":[len(x) for x in support],
                "frame_anchor_offset_yx":offset,"anchor_translation_slack":remaining}

def solve_multigap_strips(pieces,shape,grid,rotations):
    """Conservative missing strips with several contiguous holes.

    Long-axis endpoint signatures are compared with both sides' orientation.
    A calibrated lag curve estimates separation, then reciprocal unique
    joins produce a single acyclic path. Exact public missing budget required;
    endpoint missing/flat/weakly textured cases explicitly abstain.
    """
    n=len(pieces);missing=grid[0]*grid[1]-n;horizontal=grid[1]==1
    h,w=shape[0]//grid[0],shape[1]//grid[1];width=h if horizontal else w
    fail=lambda reason,**kw:({},{"multigap_status":reason,**kw})
    if not(6<=n<=768 and 1<=missing<=48 and 2<=width<=32):return fail("not-applicable")
    ims=[];turn=[]
    for p in pieces:
        k=0 if p.rgba.shape[:2]==(h,w) else 1
        if k and not rotations:return fail("orientation")
        a=p.rotated(k)[:,:,:3]
        if a.shape[:2]!=(h,w):return fail("shape")
        if horizontal:a=a.transpose(1,0,2)
        a=gaussian_filter1d(a.astype(np.float32)/255,.65,axis=0)
        if len(a)>768:a=a[::max(1,len(a)//768)]
        ims.append(a);turn.append(k)
    # Estimate the visible within-strip correlation decay at every lag.
    def corr(a,b):
        a=a-a.mean(axis=0);b=b-b.mean(axis=0)
        return float(np.sum(a*b)/max(1e-8,np.linalg.norm(a)*np.linalg.norm(b)))
    lag=[]
    for d in range(1,min(width,9)):
        vals=[corr(a[:,j],a[:,j+d]) for a in ims for j in range(0,width-d,max(1,width//5))]
        lag.append(float(np.median(vals)))
    if not lag or lag[0]<.72:return fail("low-continuity")
    # Estimate Gaussian covariance length, bounded to avoid unidentifiable gaps.
    fit=[-np.log(max(.005,min(.999,l)))/(d+1)**2 for d,l in enumerate(lag) if .10<l<.998]
    decay=float(np.median(fit)) if fit else .10
    if not .0005<decay<.5:return fail("unreliable-lag")
    ends=np.stack([v for a in ims for v in (a[:,0],a[:,-1])])
    means=ends.mean(axis=1);X=(ends-means[:,None,:]).reshape(2*n,-1)
    norm=np.linalg.norm(X,axis=1);X/=np.maximum(norm[:,None],1e-8)
    Y=X.reshape(2*n,-1,3)[:,::-1].reshape(2*n,-1)
    ix=np.arange(2*n);same=(ix[:,None]%2)==(ix[None,:]%2)
    sim=np.where(same,X@Y.T,X@X.T)-.35*np.abs(means[:,None]-means[None,:]).mean(axis=2)
    sim[ix[:,None]//2==ix[None,:]//2]=-np.inf
    sim[norm<.3,:]=-np.inf;sim[:,norm<.3]=-np.inf
    if not rotations:sim[same]=-np.inf
    direct=max(.60,lag[0]-.15)
    pred=[float(np.exp(-decay*(1+g*width)**2)) for g in range(1,min(missing,4)+1)]
    # No trustworthy gap-length information past the observable covariance.
    if max(pred)<.17:return fail("gap-outside-correlation")
    order=np.argsort(-sim,axis=1,kind="stable");edges={};parent=list(range(n))
    def root(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    def add(a,b,g):
        x,y=root(a//2),root(b//2)
        if x==y:return False
        parent[x]=y;edges[a]=(b,g);edges[b]=(a,g);return True
    for a in range(2*n):
        b=int(order[a,0]);s=float(sim[a,b])
        if a<b and order[b,0]==a and s>=direct and min(s-sim[a,order[a,1]],s-sim[b,order[b,1]])>=.035:add(a,b,0)
    direct_links=len(edges)//2;budget=0
    # Search only remaining endpoints; possible gap lengths by calibrated lag.
    for _ in range(n):
        free=[i for i in range(2*n) if i not in edges]
        if len(free)<=2:break
        sub=sim[np.ix_(free,free)].copy()
        roots=np.array([root(i//2) for i in free]);sub[roots[:,None]==roots[None,:]]=-np.inf
        rank=np.argsort(-sub,axis=1,kind="stable");choices=[]
        for a,i in enumerate(free):
            b=int(rank[a,0]);j=free[b];s=float(sub[a,b])
            if a>=b or rank[b,0]!=a or not .17<=s<direct:continue
            margin=min(s-sub[a,rank[a,1]],s-sub[b,rank[b,1]])
            if margin<.075:continue
            distances=sorted((abs(s-c),g+1) for g,c in enumerate(pred) if c>=.10)
            if not distances or distances[0][0]>.16:continue
            if len(distances)>1 and distances[1][0]-distances[0][0]<.075:continue
            choices.append((s,i,j,distances[0][1]))
        if not choices:break
        changed=0
        for s,i,j,g in sorted(choices,reverse=True):
            if i in edges or j in edges or budget+g>missing:continue
            if add(i,j,g):budget+=g;changed+=1
        if not changed:break
    if len(edges)!=2*(n-1) or budget!=missing:return fail("ambiguous-gap-budget",multigap_direct_links=direct_links,multigap_inferred_missing=budget)
    free=[i for i in range(2*n) if i not in edges]
    at=min(free) if rotations else next((i for i in free if i%2==0),-1)
    if at<0:return fail("orientation")
    out={};cell=0
    while pieces[at//2].id not in out:
        i,rev=divmod(at,2);k=(turn[i]+rev*2)%4
        out[pieces[i].id]=Pose(0,cell*h,k) if horizontal else Pose(cell*w,0,k)
        cell+=1
        if at^1 not in edges:break
        at,g=edges[at^1];cell+=g
    if len(out)!=n or cell!=grid[0]*grid[1]:return fail("path")
    return out,{"multigap_status":"resolved-conditional-on-endpoints","multigap_direct_links":direct_links,
                "multigap_inferred_missing":budget,"multigap_decay":decay,"multigap_expected_similarity":pred}

def verify_extra_grid(pieces,poses,diag,shape,grid,rotations,expected_count):
    """High-accuracy profile: one additional bounded mixed-index hypothesis.
    Select using common observed seam/coverage objective and collision checks.
    """
    from .sparse_grid import eligible,solve_indexed_grid
    from .gridmetrics10 import grid_objective,grid_state
    n=len(pieces)
    if not (grid and eligible(shape,grid) and min(grid)>1 and 8<=n<=4096):
        return poses,{**diag,"extra_verification":"not-applicable"}
    if len(poses)==n and diag.get("sparse_high_consensus") is True:
        return poses,{**diag,"extra_verification":"strong-consensus"}
    if diag.get("identical_piece_count",0):
        return poses,{**diag,"extra_verification":"equivalent-fragments"}
    q,d=solve_indexed_grid(pieces,shape,grid,rotations,shortlist=64,
        report_components=expected_count>n,index_mode="mixed",auto_tune=False)
    a=grid_objective(pieces,poses,shape,grid);b=grid_objective(pieces,q,shape,grid)
    _,_,coll,invalid=grid_state(pieces,q,shape,grid)
    accept=(not coll and not invalid and b>a+1e-8 and len(q)>=len(poses)
            and d.get("reciprocal_seam_fraction",0)>=max(.45,diag.get("reciprocal_seam_fraction",0)-.05))
    trial={"primary_objective":a,"mixed_objective":b,"primary_placed":len(poses),"mixed_placed":len(q),"accepted":bool(accept)}
    if accept:
        keep={k:v for k,v in diag.items() if k not in {"_unanchored_components","unanchored_component_count","algorithm"}}
        diag={**keep,**d,"algorithm":"extra-mixed-grid-v12",
              "sparse_high_consensus":bool(len(q)==n and d.get("shortlist_seam_fraction",0)>=.96 and d.get("reciprocal_seam_fraction",0)>=.85 and not d.get("identical_piece_count"))}
        poses=q
    return poses,{**diag,"extra_verification":trial}
