"""Normalize arbitrary-angle fragments, dispatch by geometry, report poses."""
from __future__ import annotations
import time
from shardelix.reconstruction.tiles.sparse import eligible
from shardelix.reconstruction.rectification import normalize_fragments
from shardelix.reconstruction.tiles.pose_recovery import solve_grid_pose
from shardelix.reconstruction.irregular.pose_recovery import solve_irregular_pose

def solve_pose(pieces,shape,cut,grid,angle_mode,sampling,expected_count,rotations=True,params=None,auto_tune=True):
    from shardelix.reconstruction.limits import validate_constraints
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
        p,d=solve_grid_pose(pp,shape,grid,rotations,angle_mode,expected_count,params,auto_tune)
    else:
        pp,p,d,angles,affines,normalize_seconds,selected_kernel=solve_irregular_pose(
            pieces,pp,angles,affines,shape,cut,angle_mode,sampling,rotations,auto_tune,normalize_seconds)
    d.update(algorithm="pose-normalized-"+d.get("algorithm","mask-graph")+"-v12",angle_mode=angle_mode,
             sampling=sampling,expected_count=expected_count,observed_count=len(pp),missing_count=expected_count-len(pp),
             normalization_seconds=normalize_seconds,normalization_cubic_a=selected_kernel,mode="blind",source_reference_used=False,
             runtime_seconds=time.perf_counter()-start,pose_convention="rectify_ccw_deg then clockwise k*90; x/y are normalized crop top-left")
    return pp,p,d,angles,affines
