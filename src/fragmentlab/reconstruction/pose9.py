"""Classical continuous-angle raster normalization, missing-aware assembly.

Angle estimation uses central moments for rectangles and the fourth harmonic
of alpha gradients for digital irregular masks. The phase of lattice edges
estimates pixel alignment; only the public sampling factor is required.
No source, evaluation labels, orientation metadata or trained model is used.
"""
from __future__ import annotations
import math,time
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
from .common import Fragment,Pose
from .sparse_grid import solve_indexed_grid,eligible
from .geometry import solve_geometry
from .verified import solve_verified,exact_holes
from .solver import orient_to_canvas


def _inverse_binary_mask(alpha,aa,u,v,cx,cy,co,si,cubic_a=-1.):
    """Fit the original binary raster through a cubic resampling forward model.

    Only the observed alpha is used. Sparse 16-tap interpolation and a few
    projected least-squares iterations remove inverse-warp threshold errors.
    """
    from scipy.sparse import coo_matrix
    H,W=aa.shape
    # Select boundary measurements only; interior samples provide no refinement.
    gy,gx=np.gradient(alpha)
    from scipy.ndimage import binary_dilation
    select=binary_dilation((np.hypot(gx,gy)>.01)|((alpha>.001)&(alpha<.999)),iterations=2)
    yy,xx=np.nonzero(select)
    if not len(xx):return aa>=.5
    U=co*(xx-cx)+si*(yy-cy)-u[0];V=-si*(xx-cx)+co*(yy-cy)-v[0]
    fu=np.floor(U).astype(int);fv=np.floor(V).astype(int)
    def kernel(z):
        q=np.abs(z);a=cubic_a
        return np.where(q<=1,(a+2)*q**3-(a+3)*q**2+1,np.where(q<2,a*q**3-5*a*q**2+8*a*q-4*a,0))
    rows=[];cols=[];values=[]
    for dy in (-1,0,1,2):
        Y=fv+dy;wy=kernel(V-Y)
        for dx in (-1,0,1,2):
            X=fu+dx;weights=wy*kernel(U-X)
            valid=(X>=0)&(X<W)&(Y>=0)&(Y<H)&(np.abs(weights)>1e-7)
            rows.append(np.nonzero(valid)[0]);cols.append(Y[valid]*W+X[valid]);values.append(weights[valid])
    A=coo_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(cols))),shape=(len(xx),H*W)).tocsr()
    B=A.T.tocsr();target=alpha[yy,xx]
    x=np.clip(aa.ravel(),0,1)
    diag=np.asarray(A.power(2).sum(axis=0)).ravel();good=diag>.001
    for _ in range(12):
        pred=A@x;res=np.clip(pred,0,1)-target
        grad=B@res
        x[good]=np.clip(x[good]-.6*grad[good]/diag[good],0,1)
    z=(x>=.5).astype(float);pred=A@z
    # Binary coordinate descent over only uncertain, measurable edge pixels.
    # Compare clipped observations to account for cubic overshoot.
    C=A.tocsc();unc=np.flatnonzero(good&(x>.025)&(x<.975))
    # 4x4 coloring: candidates of a color have disjoint cubic support.
    # Batch independent flip scores with bincount instead of Python pixel loops.
    colors=(unc//W%4)*4+(unc%W%4)
    batches=[]
    for color in range(16):
        inds=unc[colors==color]
        if len(inds):
            M=C[:,inds].tocoo();batches.append((inds,M.row,M.col,M.data))
    for _ in range(2):
        for inds,j,k,data in batches:
            d=data*(1-2*z[inds[k]])
            before=np.clip(pred[j],0,1)-target[j];after=np.clip(pred[j]+d,0,1)-target[j]
            diff=np.bincount(k,weights=after*after-before*before,minlength=len(inds))
            flip=diff<-.00005
            z[inds[flip]]=1-z[inds[flip]];sel=flip[k]
            pred[j[sel]]+=d[sel]
    return z.reshape(H,W)>.5

def _rectify(a, grid_cell=None, sampling=1, return_affine=False,cubic_a=-1.):
    """Return canonical RGBA, clockwise angle of the observed raster axes.
    Correction is a counterclockwise continuous rotation followed by k CW turns.
    Coordinates are pixel centers; beyond crop support samples are transparent.
    """
    # Cropped input may end at nonzero alpha; restore an explicit transparent
    # margin for the forward model and subpixel edge fit.
    a=np.pad(a,((4,4),(4,4),(0,0)))
    alpha=a[:,:,3].astype(np.float64)/255
    y,x=np.nonzero(alpha>.005)
    if not len(x):raise ValueError("Empty alpha support")
    weight=alpha[y,x]; cx=float(np.average(x,weights=weight));cy=float(np.average(y,weights=weight))
    xx=x-cx;yy=y-cy
    if grid_cell:
        h,w=grid_cell
        z=xx+1j*yy
        if h==w:
            theta=np.angle(-np.sum(weight*z**4))/4
        else:
            theta=np.angle(np.sum(weight*z*z))/2
            if h>w:theta+=np.pi/2
        theta=(theta+np.pi/2)%np.pi-np.pi/2
    else:
        # Subpixel 50% alpha crossings (linear interpolation); fitting their
        # periodic lattice phase is more stable than local normal histograms.
        ap=np.pad(alpha,1)
        gy,gx=np.gradient(ap)
        points=[];gradients=[]
        for axis in (0,1):
            aa,bb=(ap[:-1],ap[1:]) if axis==0 else (ap[:,:-1],ap[:,1:])
            iy,ix=np.nonzero((aa>=.5)!=(bb>=.5))
            frac=(.5-aa[iy,ix])/(bb[iy,ix]-aa[iy,ix])
            y1=iy-1+(frac if axis==0 else 0);x1=ix-1+(frac if axis==1 else 0)
            xx2=np.rint(x1+1).astype(int);yy2=np.rint(y1+1).astype(int)
            points.append(np.column_stack((x1-cx,y1-cy)))
            gradients.append(np.column_stack((gx[yy2,xx2],gy[yy2,xx2])))
        pt=np.concatenate(points);gr=np.concatenate(gradients)
        # Bound work for very large contours; deterministic subsample.
        if len(pt)>2500:sel=np.linspace(0,len(pt)-1,2500,dtype=int);pt=pt[sel];gr=gr[sel]
        phase_scale=2*np.pi/sampling
        def lattice(th,parts=False):
            c,s=np.cos(th),np.sin(th)
            u=c*pt[:,0]+s*pt[:,1];v=-s*pt[:,0]+c*pt[:,1]
            gu=c*gr[:,0]+s*gr[:,1];gv=-s*gr[:,0]+c*gr[:,1]
            # Soft assignments suppress corner responses and off-axis runs.
            wu=np.abs(gu)**4/(gu*gu+gv*gv+1e-6)**1.5
            wv=np.abs(gv)**4/(gu*gu+gv*gv+1e-6)**1.5
            zu=np.sum(wu*np.exp(1j*phase_scale*u));zv=np.sum(wv*np.exp(1j*phase_scale*v))
            score=(abs(zu)+abs(zv))/(wu.sum()+wv.sum()+1e-9)
            return (score,np.angle(zu)/phase_scale,np.angle(zv)/phase_scale) if parts else score
        z=gr[:,0]+1j*gr[:,1]
        theta0=float(np.angle(np.sum(z**4/(np.abs(z)**3+1e-6)))/4)
        # Global angle search, then bounded continuous refinement; fine step
        # for large contours, with an upper bound on tested hypotheses.
        extent=max(np.ptp(pt[:,0]),np.ptp(pt[:,1]),1)
        step=float(np.clip(sampling/extent*.50,.008,.035))
        offsets=np.arange(-np.pi/4,np.pi/4,step)
        vals=np.array([lattice(theta0+q) for q in offsets])
        from scipy.optimize import minimize_scalar
        opt_theta=theta0+offsets[int(vals.argmax())]
        opt=minimize_scalar(lambda q:-lattice(q),bounds=(opt_theta-step,opt_theta+step),method="bounded",options={"xatol":1e-5})
        theta=float((opt.x+np.pi/4)%(np.pi/2)-np.pi/4)
        _,phase_u,phase_v=lattice(theta,True)
    co,si=np.cos(theta),np.sin(theta)
    if grid_cell:
        dims=grid_cell
        h,w=dims
        u=(np.arange(w)-(w-1)/2)*sampling;v=(np.arange(h)-(h-1)/2)*sampling
    else:
        def axis(ph,extent):
            origin=ph+sampling/2
            lo=int(np.floor((extent.min()-origin)/sampling))-1
            hi=int(np.ceil((extent.max()-origin)/sampling))+2
            return origin+np.arange(lo,hi)*sampling
        u=axis(phase_u,co*xx+si*yy);v=axis(phase_v,-si*xx+co*yy)
    U,V=np.meshgrid(u,v)
    X=cx+co*U-si*V;Y=cy+si*U+co*V
    # Interpolate premultiplied RGB with alpha and divide, avoiding black halos.
    coords=[Y,X]
    aa=map_coordinates(alpha,coords,order=1,mode="grid-constant",cval=0,prefilter=False)
    rgb=np.stack([map_coordinates(a[:,:,c].astype(np.float32)*alpha,coords,order=1,mode="grid-constant",cval=0,prefilter=False) for c in range(3)],axis=-1)
    mask=np.ones(aa.shape,bool) if grid_cell else _inverse_binary_mask(alpha,aa,u,v,cx,cy,co,si,cubic_a) if sampling==1 else aa>=.5
    if not mask.any():raise ValueError("Could not normalize fragment")
    # Border samples can land outside the crop; use nearest valid color only for
    # cells whose shape is known, and record the unsupported fraction separately.
    if grid_cell and (aa<.08).any() and (aa>=.08).any():
        from scipy.ndimage import distance_transform_edt
        ind=distance_transform_edt(aa<.08,return_distances=False,return_indices=True)
        invalid=aa<.08;rgb[invalid]=rgb[tuple(ind[:,invalid])];aa[invalid]=aa[tuple(ind[:,invalid])]
    b=np.zeros((*aa.shape,4),np.uint8)
    b[:,:,:3]=np.clip(np.rint(rgb/np.maximum(aa[:,:,None],.001)),0,255).astype(np.uint8)
    b[:,:,3]=mask*255;b[~mask]=0
    yy2,xx2=np.nonzero(mask);b=b[yy2.min():yy2.max()+1,xx2.min():xx2.max()+1].copy()
    if not return_affine:return b,float(np.degrees(theta))
    A=np.array([[co,si],[-si,co]])/sampling
    offset=A@np.array([4-cx,4-cy])-np.array([u[0],v[0]])/sampling-np.array([xx2.min(),yy2.min()])
    return b,float(np.degrees(theta)),np.column_stack((A,offset))

def rotate_raster(a,angle,sampling=1):
    """Physical raster rotation; no hidden inverse transform stored in input."""
    im=Image.fromarray(a)
    if sampling!=1:im=im.resize((im.width*sampling,im.height*sampling),Image.Resampling.NEAREST)
    # Transparent padding permits antialiasing at the cut boundary.
    pad=Image.new("RGBA",(im.width+4,im.height+4));pad.paste(im,(2,2))
    return np.asarray(pad.rotate(float(angle),Image.Resampling.BICUBIC,expand=True)).copy()

def normalize_fragments(pieces,shape,grid,angle_mode,sampling,return_affines=False,cubic_a=-1.):
    out=[];angles={};affines={}
    cell=(shape[0]//grid[0],shape[1]//grid[1]) if grid else None
    for p in pieces:
        if angle_mode=="continuous":
            a,t,A=_rectify(p.rgba,cell,sampling,True,cubic_a);out.append(Fragment(p.id,a));angles[p.id]=t
        else:
            out.append(p);angles[p.id]=0.;A=np.column_stack((np.eye(2),np.zeros(2)))
        if return_affines:
            A[:,2]-=A[:,:2]@np.asarray(p.input_origin)
            affines[p.id]=A.tolist()
    return (out,angles,affines) if return_affines else (out,angles)

def solve_pose9(pieces,shape,cut,grid,angle_mode,sampling,expected_count,rotations=True,params=None,auto_tune=True):
    from .limits import validate_constraints
    start=time.perf_counter()
    if angle_mode not in {"none","quarter","continuous"}:raise ValueError("Invalid angle_mode")
    if type(sampling) is not int or sampling not in (1,2,3,4):raise ValueError("sampling must be 1, 2, 3 or 4")
    if sampling!=1 and angle_mode!="continuous":raise ValueError("sampling applies only to continuous rotation")
    if type(expected_count) is not int or expected_count<len(pieces):raise ValueError("Invalid expected_count")
    validate_constraints(expected_count,shape,cut,grid,rotations)
    if grid and not eligible(shape,grid):raise ValueError("Continuous/missing rectangular cases require uniform cells")
    pp,angles,affines=normalize_fragments(pieces,shape,grid,angle_mode,sampling,True)
    normalize_seconds=time.perf_counter()-start
    selected_kernel=-1.
    if grid:
        p,d=solve_indexed_grid(pp,shape,grid,rotations,int((params or {}).get("shortlist",24 if len(pp)>32768 else 40)),report_components=expected_count>len(pp),profile_samples=16 if angle_mode=="continuous" and min(grid)==1 else None,index_mode=(params or {}).get("index_mode","auto"),auto_tune=auto_tune)
        if auto_tune and min(grid)>1 and 8<=len(pp)<=1024 and len(p)<.85*len(pp) and (params or {}).get("index_mode","auto")=="auto":
            from .dense11 import solve_dense_fallback
            from .gridmetrics10 import grid_objective
            q,e=solve_dense_fallback(pp,shape,grid,rotations,expected_count>len(pp))
            # More coverage alone is insufficient; require comparably strong
            # seam support and select by observable (not evaluated) energy.
            a=grid_objective(pp,p,shape,grid);b=grid_objective(pp,q,shape,grid)
            attempt={"placed":len(q),"objective":b,"reciprocal":e.get("reciprocal_seam_fraction",0)}
            if b>a and e.get("shortlist_seam_fraction",0)>=max(.72,d.get("shortlist_seam_fraction",0)-.10) and e.get("reciprocal_seam_fraction",0)>=max(.40,d.get("reciprocal_seam_fraction",0)-.15):
                d={**e,"identical_piece_count":d.get("identical_piece_count",0),"index_fallback_used":True,"prior_sparse_index":d.get("index_mode"),"prior_sparse_placed":len(p)}
                p=q
            d["dense_recovery_trial"]=attempt
        if auto_tune and min(grid)==1 and expected_count>len(pp):
            from .strip_gap11 import solve_thin_missing
            recovered,gap=solve_thin_missing(pp,shape,grid,rotations)
            d.update(gap)
            if len(recovered)==len(pp):
                d["thin_gap_added"]=len(recovered)-len(p);p=recovered
                d.pop("_unanchored_components",None);d["unanchored_component_count"]=0
                d.update(placed_fraction=1.,algorithm="thin-strip-gap-graph-v11")
        if auto_tune and min(grid)==1 and expected_count>len(pp) and d.get("thin_gap_status")!="resolved-single-gap-budget" and (len(p)<len(pp) or d.get("reciprocal_seam_fraction",0)<.85):
            from .recovery12 import solve_multigap_strips
            z,gap=solve_multigap_strips(pp,shape,grid,rotations)
            d.update(gap)
            if len(z)==len(pp):
                from .gridmetrics10 import grid_objective
                accepted=len(p)<len(pp) or grid_objective(pp,z,shape,grid)>grid_objective(pp,p,shape,grid)+1e-8
                d["multigap_accepted"]=bool(accepted)
            else:accepted=False
            if accepted:
                p=z;d.pop("_unanchored_components",None)
                d.update(algorithm="multi-gap-strip-graph-v12",unanchored_component_count=0)
        d["sparse_high_consensus"]=bool(len(p)>.95*len(pp) and d.get("shortlist_seam_fraction",0)>=.96
            and d.get("reciprocal_seam_fraction",0)>=.85 and not d.get("identical_piece_count"))
        low=sum(float(np.std(q.rgba[:,:,:3].astype(np.float32),axis=(0,1)).mean())<1.5 for q in pp)
        d.update(low_texture_piece_count=low,low_texture_piece_fraction=low/len(pp))
    else:
        if angle_mode=="continuous" and sampling==1:
            from .tolerant9 import solve_tolerant,repair_boundary_ownership,tolerant_candidates
            # Slight raster aliasing is not the same as physically broken edges.
            # Full-seam verification and bounded perimeter-only overlap prevent
            # a single damaged pixel from disconnecting an entire correct group.
            # Sparse exact-arc graph remains the cheapest high-confidence path.
            if cut in {"jigsaw","wavy"}:
                p,d=solve_verified(pp,shape,rotations,k=16,max_bucket=14,pack=False,return_groups=True)
            else:
                p,d=solve_geometry(pp,shape,rotations,k=18,min_votes=2,max_bucket=8,content=False)
                p=orient_to_canvas(pp,p,shape) if rotations else p
                p,_=exact_holes(pp,p,shape,rotations)
            groups=d.pop("_groups",[]);verified=d.pop("_verified_candidates",[])
            if auto_tune and len(p)<.985*len(pp):
                precomputed=tolerant_candidates(pp,rotations)
                q,e=solve_tolerant(pp,shape,rotations,precomputed,return_groups=True)
                tolerant_groups=e.pop("_groups",[])
                # Prefer the existing exact anchor, then verify all new contacts.
                from .solver import objective
                if cut in {"jigsaw","wavy"} and len(p)>=.3*len(pp):
                    from .anchor11 import anchored_complete,soft_pack
                    z,f=anchored_complete(pp,p,groups,list(verified)+list(precomputed[0]),shape,rotations)
                    if len(z)<.985*len(pp):
                        z,g=soft_pack(pp,z,groups,shape,rotations,rounds=2);f.update(g)
                    fixed,raw,core=repair_boundary_ownership(pp,z,shape)
                    if core==0 and raw<=.01*sum(int(v.mask.sum()) for v in pp) and objective(fixed,z,shape)>objective(pp,p,shape):
                        p,d=z,{**d,**f,"algorithm":"anchored-mask-completion-v11"}
                    else:d.update(anchor_attempt=f)
                if .35*len(pp)<=len(q)<.985*len(pp) and len(pp)<=1024 and objective(pp,q,shape)>=objective(pp,p,shape)-.025:
                    # V11 discarded every tolerant component except the largest.
                    # Preserve them and try independent verified multi-contacts.
                    from .anchor11 import anchored_complete,soft_pack
                    z,f=anchored_complete(pp,q,tolerant_groups,precomputed[0],shape,rotations,rounds=8)
                    if len(z)<.985*len(pp):z,g=soft_pack(pp,z,tolerant_groups,shape,rotations,rounds=2);f.update(g)
                    fixed,raw,core=repair_boundary_ownership(pp,z,shape)
                    if (core==0 and raw<=.01*sum(int(v.mask.sum()) for v in pp)
                            and objective(fixed,z,shape)>objective(pp,q,shape)+1e-8):
                        q,e=z,{**e,**f,"algorithm":"tolerant-component-recovery-v12"}
                    else:e["tolerant_component_trial"]=f
                if objective(pp,q,shape)>objective(pp,p,shape):p,d=q,e
            if auto_tune and cut in {"wavy","jigsaw"} and len(p)<.95*len(pp):
                # Different rotation pipelines have different cubic kernels.
                # Consider a second inverse raster model only on weak native
                # hypotheses, and select by observed canvas/seam energy.
                from .solver import objective
                t=time.perf_counter()
                alt,aa,af=normalize_fragments(pieces,shape,grid,angle_mode,sampling,True,-.5)
                normalize_seconds+=time.perf_counter()-t
                q,e=solve_tolerant(alt,shape,rotations)
                primary=objective(pp,p,shape);secondary=objective(alt,q,shape)
                trials=[{"cubic_a":-1.,"placed":len(p),"energy":primary},{"cubic_a":-.5,"placed":len(q),"energy":secondary}]
                if secondary>primary:pp,angles,affines,p,d,selected_kernel=alt,aa,af,q,e,-.5
                d["raster_hypotheses"]=trials
            if auto_tune and p and len(p)<len(pp):
                from .recovery12 import anchor_irregular_frame
                p,frame=anchor_irregular_frame(pp,p,shape)
                d.update(frame)
            pp,raw,core=repair_boundary_ownership(pp,p,shape)
            d.update(raw_boundary_conflicts=raw,core_conflicts=core)
        else:
            p,d=solve_geometry(pp,shape,rotations,k=18,min_votes=2,max_bucket=8)
            p=orient_to_canvas(pp,p,shape) if rotations else p
            if len(p)<.97*len(pp):
                q,e=solve_verified(pp,shape,rotations,k=14,max_bucket=12,pack=False)
                q=orient_to_canvas(pp,q,shape) if rotations else q
                if len(q)>len(p):p,d=q,e
    d.update(algorithm="pose-normalized-"+d.get("algorithm","mask-graph")+"-v12",angle_mode=angle_mode,
             sampling=sampling,expected_count=expected_count,observed_count=len(pp),missing_count=expected_count-len(pp),
             normalization_seconds=normalize_seconds,normalization_cubic_a=selected_kernel,mode="blind",source_reference_used=False,
             runtime_seconds=time.perf_counter()-start,pose_convention="rectify_ccw_deg then clockwise k*90; x/y are normalized crop top-left")
    return pp,p,d,angles,affines
