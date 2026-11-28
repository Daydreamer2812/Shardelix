"""Missing-aware and alias-tolerant irregular mask assembly."""
from __future__ import annotations
import time
from shardelix.reconstruction.irregular.lattice import solve_geometry
from shardelix.reconstruction.irregular.verified import solve_verified,exact_holes
from shardelix.reconstruction.quality import orient_to_canvas,objective
from shardelix.reconstruction.rectification import normalize_fragments

def solve_irregular_pose(pieces,pp,angles,affines,shape,cut,angle_mode,sampling,rotations,auto_tune,normalize_seconds):
    selected_kernel=-1.
    if angle_mode=="continuous" and sampling==1:
        from shardelix.reconstruction.irregular.tolerant import solve_tolerant,repair_boundary_ownership,tolerant_candidates
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
            from shardelix.reconstruction.quality import objective
            if cut in {"jigsaw","wavy"} and len(p)>=.3*len(pp):
                from shardelix.reconstruction.irregular.anchor import anchored_complete,soft_pack
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
                from shardelix.reconstruction.irregular.anchor import anchored_complete,soft_pack
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
            from shardelix.reconstruction.quality import objective
            t=time.perf_counter()
            alt,aa,af=normalize_fragments(pieces,shape,None,angle_mode,sampling,True,-.5)
            normalize_seconds+=time.perf_counter()-t
            q,e=solve_tolerant(alt,shape,rotations)
            primary=objective(pp,p,shape);secondary=objective(alt,q,shape)
            trials=[{"cubic_a":-1.,"placed":len(p),"energy":primary},{"cubic_a":-.5,"placed":len(q),"energy":secondary}]
            if secondary>primary:pp,angles,affines,p,d,selected_kernel=alt,aa,af,q,e,-.5
            d["raster_hypotheses"]=trials
        if auto_tune and p and len(p)<len(pp):
            from shardelix.reconstruction.irregular.frame_anchor import anchor_irregular_frame
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
    return pp,p,d,angles,affines,normalize_seconds,selected_kernel
