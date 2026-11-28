"""Missing-aware assembly of normalized rectangular tiles and strips."""
from __future__ import annotations
import numpy as np
from shardelix.reconstruction.tiles.sparse import solve_indexed_grid

def solve_grid_pose(pp,shape,grid,rotations,angle_mode,expected_count,params,auto_tune):
    p,d=solve_indexed_grid(pp,shape,grid,rotations,int((params or {}).get("shortlist",24 if len(pp)>32768 else 40)),report_components=expected_count>len(pp),profile_samples=16 if angle_mode=="continuous" and min(grid)==1 else None,index_mode=(params or {}).get("index_mode","auto"),auto_tune=auto_tune)
    if auto_tune and min(grid)>1 and 8<=len(pp)<=1024 and len(p)<.85*len(pp) and (params or {}).get("index_mode","auto")=="auto":
        from shardelix.reconstruction.tiles.dense_recovery import solve_dense_fallback
        from shardelix.reconstruction.tiles.metrics import grid_objective
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
        from shardelix.reconstruction.strips.single_gap import solve_thin_missing
        recovered,gap=solve_thin_missing(pp,shape,grid,rotations)
        d.update(gap)
        if len(recovered)==len(pp):
            d["thin_gap_added"]=len(recovered)-len(p);p=recovered
            d.pop("_unanchored_components",None);d["unanchored_component_count"]=0
            d.update(placed_fraction=1.,algorithm="thin-strip-gap-graph-v11")
    if auto_tune and min(grid)==1 and expected_count>len(pp) and d.get("thin_gap_status")!="resolved-single-gap-budget" and (len(p)<len(pp) or d.get("reciprocal_seam_fraction",0)<.85):
        from shardelix.reconstruction.strips.multi_gap import solve_multigap_strips
        z,gap=solve_multigap_strips(pp,shape,grid,rotations)
        d.update(gap)
        if len(z)==len(pp):
            from shardelix.reconstruction.tiles.metrics import grid_objective
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
    return p,d
