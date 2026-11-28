"""Route by observable task constraints and tune using a blind image objective."""
from __future__ import annotations

import time
import numpy as np

from shardelix.reconstruction.common import Pose
from shardelix.reconstruction.quality import objective,orient_to_canvas
from shardelix.reconstruction.irregular.lattice import solve_geometry
from shardelix.reconstruction.tiles.legacy import solve_grid,solve_rotated_grid
from shardelix.reconstruction.tiles.straight import solve_straight
from shardelix.reconstruction.rotation import solve_rotated_strips,solve_rotated_tiles
from shardelix.reconstruction.irregular.verified import solve_verified,exact_holes
from shardelix.reconstruction.strips.ordering import is_documentlike,solve_document_strips


def _solve_baseline(pieces,shape,cut="random",grid=None,rotations=False,
          mode="blind",reference=None,auto_tune=True,params=None,template=None):
    start=time.perf_counter()
    params=params or {}
    if mode!="blind" or reference is not None or template is not None:
        raise ValueError("Shardelix V7 is blind only; source and cut template are prohibited")
    if not pieces:raise ValueError("Expected at least one piece")
    is_strip=grid is not None and min(grid)==1
    if is_strip and (params.get("strategy")=="document" or (not params and is_documentlike(pieces))):
        c=dict(params);c.pop("strategy",None)
        poses,diag=solve_document_strips(pieces,shape,grid,rotations,**c)
        return poses,{**diag,"mode":"blind","runtime_seconds":time.perf_counter()-start,
                      "source_reference_used":False,"auto_tune":auto_tune,
                      "blind_objective":objective(pieces,poses,shape),
                      "trials":[{"params":c,"placed":len(poses)}]}
    if grid:
        legacy=[{"strategy":"legacy","samples":36,"gradient_weight":.6,"top_k":5,"refine":2},
                {"strategy":"legacy","samples":56,"gradient_weight":.25,"top_k":8,"refine":2},
                {"strategy":"legacy","samples":48,"gradient_weight":.9,"top_k":4,"refine":3}]
        if rotations:
            strips=min(grid)==1
            if strips:
                configs=[params] if params else ([{"kind":"l1"},{"kind":"l2"}] if auto_tune else [{"kind":"l1"}])
                run=lambda c:solve_rotated_strips(pieces,shape,grid,**c)
            else:
                configs=[params] if params else [{"samples":24,"gradient_weight":.25,"top_k":3}]
                run=lambda c:solve_rotated_tiles(pieces,shape,grid,**c)
        else:
            strips=min(grid)==1
            defaults=[{"strategy":"straight","kind":"l1"}]
            alternatives=[{"strategy":"straight","kind":"l2"}] if strips else [
                {"strategy":"straight","kind":"l2"},
                {"strategy":"straight","kind":"l2","normalize":False,"cycle":False},
                {"strategy":"straight","kind":"mgc","gradient_weight":.35},
                {"strategy":"straight","kind":"l1","threshold":1.4},
                *legacy]
            configs=[params] if params else defaults+(alternatives if auto_tune else [])
            def run(c):
                c=dict(c)
                # Old explicit tuning JSONs continue to mean the V2 algorithm.
                old_keys={"samples","gradient_weight","top_k","refine"}
                strategy=c.pop("strategy", "legacy" if set(c)<=old_keys else "straight")
                if strategy=="legacy":return solve_grid(pieces,shape,grid,**c)
                if strategy=="straight":return solve_straight(pieces,shape,grid,**c)
                raise ValueError("Unknown straight strategy")
    else:
        configs=[params] if params else (
            [{"strategy":"geometry","k":24,"min_votes":2,"max_bucket":5},
             {"strategy":"geometry","k":18,"min_votes":3,"max_bucket":4},
             {"strategy":"geometry","k":32,"min_votes":2,"max_bucket":7},
             {"strategy":"geometry","k":14,"min_votes":4,"max_bucket":4},
             {"strategy":"verified","k":16,"max_bucket":18}] if auto_tune else
            [{}])
        def run(c):
            c=dict(c);strategy=c.pop("strategy","geometry")
            if strategy=="verified":return solve_verified(pieces,shape,rotations,**c)
            if strategy=="geometry":return solve_geometry(pieces,shape,rotations,**c)
            raise ValueError("Unknown irregular strategy")
    best=None;trials=[]
    for c in configs:
        poses,diag=run(c)
        if rotations:poses=orient_to_canvas(pieces,poses,shape)
        if not grid and len(poses)<len(pieces):
            poses,filled=exact_holes(pieces,poses,shape,rotations)
            diag["extra_exact_holes"]=filled
        score=objective(pieces,poses,shape)
        trials.append({"params":c,"blind_objective":score,"placed":len(poses)})
        if best is None or score>best[0]:best=(score,poses,diag)
        if not grid and cut in {"jagged","wavy","jigsaw","random"} and len(poses)==len(pieces) and score>.98:
            break
    score,poses,diag=best
    return poses,{**diag,"mode":mode,"runtime_seconds":time.perf_counter()-start,
                  "source_reference_used":False,"auto_tune":auto_tune,"trials":trials,
                  "blind_objective":score}


def solve_classic(pieces,shape,cut="random",grid=None,rotations=False,
          mode="blind",reference=None,auto_tune=True,params=None,template=None):
    """Small-case consensus and targeted strip, grid and contour refinement."""
    if mode!="blind" or reference is not None or template is not None:
        raise ValueError("Shardelix V7 is blind only; source and template are prohibited")
    if not pieces:raise ValueError("Expected at least one piece")
    if not grid or min(grid)==1 or (params and params.get("strategy")!="consensus"):
        start=time.perf_counter()
        poses,diag=_solve_baseline(pieces,shape,cut,grid,rotations,mode,reference,auto_tune,params,template)
        if auto_tune and not params:
            if grid and min(grid)==1 and len(pieces)>=64 and not is_documentlike(pieces):
                # Thin digital strips: contiguous block/order errors remain
                # even when nearly all immediate adjacencies are correct.
                from shardelix.reconstruction.tiles.straight import _grid_bounds
                ys,xs=_grid_bounds(*shape,*grid)
                widths=np.diff(xs) if grid[0]==1 else np.diff(ys)
                if widths.min()<=3:
                    from shardelix.reconstruction.strips.refinement import refine_strip_blocks
                    pp,dd=refine_strip_blocks(pieces,poses,shape,grid,rotations)
                    if dd.get("block_moves",0) or dd.get("adjacent_swaps",0):
                        poses=pp;diag={**diag,**dd,"algorithm":"strip-block-curvature-v7"}
            elif not grid and .60*len(pieces)<=len(poses)<len(pieces):
                from shardelix.reconstruction.irregular.frontier import complete_frontier,repair_small_gaps
                pp,dd=complete_frontier(pieces,poses,shape,rotations)
                pp,gap=repair_small_gaps(pieces,pp,shape,rotations)
                dd.update(gap)
                if objective(pieces,pp,shape)>objective(pieces,poses,shape):
                    poses=pp;diag={**diag,**dd,"algorithm":"exact-frontier-v7"}
        return poses,{**diag,"blind_objective":objective(pieces,poses,shape),
                      "runtime_seconds":time.perf_counter()-start}
    from shardelix.reconstruction.tiles.consensus import solve_consensus_grid,edge_costs,score_grid
    start=time.perf_counter()
    c=dict(params or {});c.pop("strategy",None)
    configs=[c]
    # Only uncertain cases need more hypotheses. Score all on the SAME
    # fixed MGC compatibility; cannot use source image or truth.
    raw=edge_costs(pieces,rotations,"mgc",.65)
    reuse=not c or (c.get("metric","mgc")=="mgc" and c.get("gradient",.65)==.65 and c.get("depth",1)==1 and not c.get("dimensions",False))
    poses,diag=solve_consensus_grid(pieces,shape,grid,rotations,raw=raw if reuse else None,**c)
    score=score_grid(pieces,poses,shape,grid,rotations,raw)
    if not np.isfinite(score):
        score=1e8
    best=(score,poses,diag)
    trials=[{"params":{"strategy":"consensus",**c},"blind_grid_energy":score,"placed":len(poses)}]
    uncertain=diag.get("anchor_pieces",len(pieces))<.96*len(pieces)
    if auto_tune and not params and uncertain:
        # Diversity in gradient weighting and confidence thresholds.
        for cc in [{"gradient":.85,"strict":.18},{"gradient":.35,"strict":.30}]:
            pp,dd=solve_consensus_grid(pieces,shape,grid,rotations,**cc)
            ss=score_grid(pieces,pp,shape,grid,rotations,raw)
            if not np.isfinite(ss):ss=1e8
            trials.append({"params":{"strategy":"consensus",**cc},"blind_grid_energy":ss,"placed":len(pp)})
            if ss<best[0]:best=(ss,pp,dd)
        pp,dd=_solve_baseline(pieces,shape,cut,grid,rotations,auto_tune=True)
        ss=score_grid(pieces,pp,shape,grid,rotations,raw)
        if not np.isfinite(ss):ss=1e8
        trials.append({"params":{"strategy":"v5-fallback"},"blind_grid_energy":ss,"placed":len(pp)})
        if ss<best[0]:best=(ss,pp,{**dd,"algorithm":"v5-fallback"})
    # V7 uncertainty refinement is targeted at large, difficult grids.
    # Keep V6 small-grid behavior and all high-confidence solutions.
    if auto_tune and not params and uncertain and len(pieces)>=512 and max(grid)<=64:
        from shardelix.reconstruction.tiles.refinement import refine_grid_lines,refine_grid_regions,refine_grid_phase
        def consider(pp,dd,tag,cc=None):
            nonlocal best
            pp,rr=refine_grid_lines(pieces,pp,shape,grid,rotations,raw)
            ss=score_grid(pieces,pp,shape,grid,rotations,raw)
            if not np.isfinite(ss):ss=1e8
            trials.append({"params":{"strategy":tag,**(cc or {})},"blind_grid_energy":ss,"placed":len(pp)})
            if ss<best[0]-1e-8:best=(ss,pp,{**dd,**rr,"algorithm":"grid-line-consensus-v7"})
        consider(best[1],best[2],"v6-line-refine")
        for cc in [{"gradient":.85,"strict":.35,"dimensions":True},
                   {"gradient":1.,"strict":.5,"dimensions":True}]:
            pp,dd=solve_consensus_grid(pieces,shape,grid,rotations,**cc)
            consider(pp,dd,"dimension-consensus-v7",cc)
        # Repair local contiguous high-residual areas, without changing
        # confident tiles unless the fixed blind seam energy improves.
        pp,rr=refine_grid_regions(pieces,best[1],shape,grid,rotations,raw,iterations=12)
        if rr.get("region_repairs",0):
            consider(pp,{**best[2],**rr},"region-repair-v7")
        pp,rr=refine_grid_phase(pieces,best[1],shape,grid,rotations,raw)
        ss=score_grid(pieces,pp,shape,grid,rotations,raw)
        if ss<best[0]-1e-8:
            trials.append({"params":{"strategy":"phase-grid-v7"},"blind_grid_energy":ss,"placed":len(pp)})
            best=(ss,pp,{**best[2],**rr})
    score,poses,diag=best
    return poses,{**diag,"mode":"blind","source_reference_used":False,
                  "runtime_seconds":time.perf_counter()-start,"blind_grid_energy":score,
                  "blind_objective":objective(pieces,poses,shape),"auto_tune":auto_tune,
                  "trials":trials}

def _solve_large_irregular(pieces,shape,cut,rotations,auto_tune,start):
    """Bounded exact arc graph and verified frontier recovery."""
    n=len(pieces)
    # Bounded large mask pipeline. Never enter the expensive exhaustive
    # component-packing fallback on thousands of unresolved fragments.
    from shardelix.reconstruction.irregular.lattice import solve_geometry
    from shardelix.reconstruction.irregular.frontier import complete_frontier
    best=None
    # Smooth jigsaw edges generate repeated arc codes; verify full seam
    # photometry and collision before committing a rigid merge.
    if cut in {"jigsaw","wavy"}:
        pp,dd=solve_verified(pieces,shape,rotations,k=16,max_bucket=14,pack=False)
        score=objective(pieces,pp,shape);best=(score,pp,dd)
    if best is None or (auto_tune and len(best[1])<.98*n):
        for kk in (18,24) if auto_tune and cut=="jagged" else (18,):
            pp,dd=solve_geometry(pieces,shape,rotations,k=kk,min_votes=2,max_bucket=8)
            if rotations:pp=orient_to_canvas(pieces,pp,shape)
            pp,filled=exact_holes(pieces,pp,shape,rotations)
            score=objective(pieces,pp,shape)
            if best is None or score>best[0]:best=(score,pp,dd)
            if len(pp)>=.98*n:break
    if cut in {"random","custom"} and auto_tune and len(best[1])<.97*n:
        pp,dd=solve_verified(pieces,shape,rotations,k=16,max_bucket=14,pack=False)
        score=objective(pieces,pp,shape)
        if score>best[0]:best=(score,pp,dd)
    score,pp,dd=best
    if .85*n<=len(pp)<n:
        pp0,ff=complete_frontier(pieces,pp,shape,rotations,rounds=6)
        ss=objective(pieces,pp0,shape)
        if ss>score:pp,score,dd=pp0,ss,{**dd,**ff}
    return pp,{**dd,"algorithm":"bounded-mask-graph-v8","mode":"blind",
               "source_reference_used":False,"auto_tune":auto_tune,
               "runtime_seconds":time.perf_counter()-start,"blind_objective":score}


def _solve_sparse_grid(pieces,shape,cut,grid,rotations,auto_tune,params,forced,start):
    """Sparse shortlist with classic fallback only when consensus is weak."""
    from shardelix.reconstruction.limits import LEGACY_MAX
    from shardelix.reconstruction.tiles.sparse import solve_indexed_grid
    n=len(pieces)
    pp,dd=solve_indexed_grid(pieces,shape,grid,rotations,int((params or {}).get("shortlist",24 if n>32768 else 40)),index_mode=(params or {}).get("index_mode","auto"),auto_tune=auto_tune)
    confident=(len(pp)==n and dd.get("reciprocal_seam_fraction",0)>=.94 and
               dd.get("shortlist_seam_fraction",0)>=.96 and dd.get("median_reciprocal_margin",0)>=.12 and
               not dd["identical_piece_count"])
    if forced or n>LEGACY_MAX or confident:
        from shardelix.reconstruction.tiles.metrics import grid_objective
        score=grid_objective(pieces,pp,shape,grid)
        return pp,{**dd,"mode":"blind","source_reference_used":False,"auto_tune":auto_tune,
                   "sparse_high_consensus":confident,"runtime_seconds":time.perf_counter()-start,
                   "blind_objective":score}
    p,d=solve_classic(pieces,shape,cut,grid,rotations,auto_tune=auto_tune,params=params)
    return p,{**d,"sparse_attempt_seconds":dd["retrieval_seconds"]+dd["assembly_seconds"],
              "runtime_seconds":time.perf_counter()-start}


def solve(pieces,shape,cut="random",grid=None,rotations=False,
          mode="blind",reference=None,auto_tune=True,params=None,template=None):
    """Dispatch known geometry to exact, sparse or classic reconstruction."""
    from shardelix.reconstruction.limits import validate_constraints,LEGACY_MAX
    from shardelix.reconstruction.tiles.sparse import eligible
    if mode!="blind" or reference is not None or template is not None:
        raise ValueError("Blind only: source/reference/template prohibited")
    start=time.perf_counter()
    n=len(pieces)
    # Historical callers supply grid but omit cut; preserve their public geometry.
    if grid and cut=="random":
        cut="strips" if grid[0]==1 else "horizontal-strips" if grid[1]==1 else "grid"
    validate_constraints(n,shape,cut,grid,rotations)
    forced=bool(params and params.get("strategy")=="sparse")
    fast=bool(grid and eligible(shape,grid) and n>=512 and
              (min(grid)>1 or n>LEGACY_MAX) and not params)
    if n==1:
        p=pieces[0]
        ks=[k for k in range(4 if rotations else 1) if p.rotated(k).shape[:2]==tuple(shape)]
        if ks:return {p.id:Pose(0,0,ks[0])},{"algorithm":"single","mode":"blind","runtime_seconds":time.perf_counter()-start,"source_reference_used":False}
    if not grid and n>LEGACY_MAX:
        return _solve_large_irregular(pieces,shape,cut,rotations,auto_tune,start)
    if forced or fast or (grid and n>LEGACY_MAX):
        return _solve_sparse_grid(pieces,shape,cut,grid,rotations,auto_tune,params,forced,start)
    return solve_classic(pieces,shape,cut,grid,rotations,mode,reference,auto_tune,params,template)

# Historical benchmark hooks. New code uses semantic stage names.
_solve_v5 = _solve_baseline
_solve_v7 = solve_classic
