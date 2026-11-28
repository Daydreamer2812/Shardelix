"""Additional blind candidate verification for uncertain rectangular grids."""
from __future__ import annotations
import numpy as np

def verify_extra_grid(pieces,poses,diag,shape,grid,rotations,expected_count):
    """High-accuracy profile: one additional bounded mixed-index hypothesis.
    Select using common observed seam/coverage objective and collision checks.
    """
    from shardelix.reconstruction.tiles.sparse import eligible,solve_indexed_grid
    from shardelix.reconstruction.tiles.metrics import grid_objective,grid_state
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
