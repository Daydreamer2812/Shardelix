"""Shardelix V7: one public mode, blind reconstruction."""
from __future__ import annotations
import argparse
import json
import numpy as np
from pathlib import Path
from .common import Pose,read_source,load_fragments,save_rgba,render,json_write
from .generator import CUTS,generate
from .solver import solve
from .evaluation import evaluate
from .overlay import write_cut_outputs

def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))

def solve_case(folder:Path,output:Path,mode="blind",reference=None,tune=True,params=None,template=None,profile=None):
    """Solve using only case.json (public constraints) and pieces/.

    The old Python signature rejects any non-blind input. CLI exposes no modes.
    """
    if mode!="blind" or reference is not None or template is not None:
        raise ValueError("Blind only: source/reference/template are prohibited")
    from shardelix.api import Solver
    meta=read_json(folder/"case.json")
    if not isinstance(meta,dict):raise ValueError("case.json must contain an object")
    for k in ("height","width","piece_count","cut","rotations"):
        if k not in meta:raise ValueError(f"Missing public metadata: {k}")
    from .limits import validate_constraints,max_count
    shape=(meta["height"],meta["width"])
    grid=tuple(meta["grid"]) if meta.get("grid") is not None else None
    observed=meta["piece_count"];expected=meta.get("expected_count",observed)
    if type(observed) is not int or type(expected) is not int or not 1<=observed<=expected:
        raise ValueError("Invalid public piece counts")
    validate_constraints(expected,shape,meta["cut"],grid,meta["rotations"])
    pieces=load_fragments(folder/"pieces",max_fragments=min(observed,max_count(meta["cut"])))
    if observed!=len(pieces):
        raise ValueError("Declared piece_count differs from input files")
    solver=Solver(auto_tune=tune,profile=profile)
    result=solver.solve(pieces,shape=shape,cut=meta["cut"],grid=grid,rotations=meta["rotations"],params=params,angle_mode=meta.get("angle_mode"),sampling=meta.get("sampling",1),expected_count=meta.get("expected_count"))
    return result.save(output)

def evaluate_case(folder,output):
    meta=read_json(folder/"case.json")
    result=read_json(output/"placements.json")
    # Only evaluator can open truth, AFTER solver and cut overlay completed.
    truth=read_json(folder/"evaluation"/"ground_truth.json")
    pieces=load_fragments(folder/"pieces")
    poses={i:Pose(**p) for i,p in result["poses"].items()}
    if "expected_count" in meta:
        from .pose9_eval import evaluate_pose9
        metrics=evaluate_pose9(pieces,poses,truth,meta,result,np.load(folder/"evaluation"/"truth_labels.npz")["labels"])
    else:
        gt={i:Pose(**p) for i,p in truth["poses"].items()}
        metrics=evaluate(pieces,poses,gt,(meta["height"],meta["width"]),
                         allow_global_rotation=meta["rotations"],allow_translation=False)
    metrics.update(mode="blind",cut=meta["cut"],seed=truth["seed"],rotations=meta["rotations"],
                   runtime_seconds=result["diagnostics"]["runtime_seconds"],
                   source_reference_used=False,cut_template_used=False)
    json_write(output/"metrics.json",metrics)
    return metrics

def _source_opts(p):
    p.add_argument("source",type=Path)
    p.add_argument("--max-side",type=int,default=2048,help="0 preserves resolution")
    p.add_argument("--page",type=int,default=1,help="PDF page (1-based)")
def _gen_opts(p):
    _source_opts(p)
    p.add_argument("--pieces",type=int,default=64)
    p.add_argument("--cut",choices=CUTS,default="random")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--rotations",action="store_true",help="Independent random quarter-turns (0/90/180/270 degrees)")
    p.add_argument("--angle-mode",choices=["none","quarter","continuous"],help="Continuous angles in [-180,180)");p.add_argument("--missing",type=float,default=0.,help="Random missing input fraction [0,1)");p.add_argument("--sampling",type=int,choices=[1,2,3,4],default=1,help="Sample factor before raster rotation (1=native; >1=oversampled)")
    p.add_argument("--fit-grid",action="store_true",help="Resize source by a few pixels to uniform rectangular cells; generation only")
    p.add_argument("--label-map",type=Path,help="Custom unlabeled raster partition (.npy), generator only")
def _solve_opts(p):
    p.add_argument("--no-tune",action="store_true",help="Single blind hypothesis")
    p.add_argument("--profile",choices=["fast","balanced","accurate","custom"],help="Performance policy")
    p.add_argument("--params",type=json.loads,help="Explicit classical solver parameter JSON")

def solve_external_document(folder:Path,output:Path,rotations=False,params=None,profile=None,angle_mode=None,sampling=1,max_canvas_pixels=None,color_aware=False):
    """Blind external document strips, unequal widths permitted. Input must contain only pieces."""
    from .document import solve_document_strips
    import time
    pieces=load_fragments(folder,max_fragments=1024)
    if len(pieces)>1024:raise ValueError("External document solver supports up to 1024 strips")
    start=time.perf_counter()
    angles=affines=None;extra={}
    if angle_mode=="continuous":
        from .document12 import normalize_document_strips
        pieces,angles,affines,extra=normalize_document_strips(pieces,sampling)
        rotations=True
    elif angle_mode not in (None,"none","quarter"):
        raise ValueError("Invalid external document rotation mode")
    elif angle_mode is not None:rotations=angle_mode=="quarter"
    if angle_mode!="continuous" and sampling!=1:raise ValueError("Sampling only applies to continuous mode")
    from .document import is_documentlike
    if color_aware and not is_documentlike(pieces):
        from .rotation import solve_unknown_color_strips
        poses,diag=solve_unknown_color_strips(pieces)
    elif profile=="fast":
        from .document import _solve_document_v5
        poses,diag=_solve_document_v5(pieces,rotations=rotations,**(params or {}))
    else:
        poses,diag=solve_document_strips(pieces,rotations=rotations,**(params or {}))
    diag["performance_profile"]=profile or "legacy"
    shape=(max(p.rotated(poses[p.id].k).shape[0] for p in pieces),
           sum(p.rotated(poses[p.id].k).shape[1] for p in pieces))
    from .limits import MAX_PIXELS
    if shape[0]*shape[1]>min(MAX_PIXELS,max_canvas_pixels or MAX_PIXELS):
        raise ValueError("Estimated document canvas exceeds pixel budget")
    if angles is not None:
        from shardelix.api import ReconstructionResult
        diag.update(extra,angle_mode="continuous",sampling=sampling,expected_count=len(pieces),missing_count=0,
                    algorithm="estimated-document-pose-v12-"+diag.get("algorithm","document"),
                    mode="blind",runtime_seconds=time.perf_counter()-start,orientation_ambiguous=True,document_missing_unknown=True)
        return ReconstructionResult(pieces,poses,shape,diag,None,True,angles,affines).save(output)
    rgba,labels,collision=render(pieces,poses,shape)
    save_rgba(output/"assembly.png",rgba)
    write_cut_outputs(output,rgba,labels)
    for alt in diag.get("equally_scored_alternatives",[]):
        pp={k:Pose(**v) for k,v in alt["poses"].items()}
        aa,ll,_=render(pieces,pp,shape)
        dest=output/("alternative_"+alt["hypothesis"])
        save_rgba(dest/"assembly.png",aa);write_cut_outputs(dest,aa,ll)
    diag.update(mode="blind",runtime_seconds=time.perf_counter()-start,
                render_collision_pixels=collision,coverage=float((labels>=0).mean()),
                canvas_height=shape[0],canvas_width=shape[1],placed=len(poses),piece_count=len(pieces),
                status="assembled" if len(poses)==len(pieces) and collision==0 else "partial",accuracy_verified=False)
    result={"mode":"blind","poses":{k:v.as_dict() for k,v in poses.items()},"diagnostics":diag}
    json_write(output/"placements.json",result);json_write(output/"diagnostics.json",diag)
    return result

def main(argv=None):
    parser=argparse.ArgumentParser(prog="shardelix",description="Classical CPU fragment reconstruction (blind by default)")
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("studio",help="Launch local browser UI");p.add_argument("--port",type=int,default=8765);p.add_argument("--no-browser",action="store_true")
    p=sub.add_parser("generate");_gen_opts(p);p.add_argument("--out",type=Path,required=True)
    p=sub.add_parser("reconstruct",help="Automatically reconstruct files or a ZIP; original image is optional")
    p.add_argument("fragments",type=Path,nargs="+");p.add_argument("--out",type=Path,required=True)
    p.add_argument("--reference",type=Path,help="Explicit original image for matching, not blind reconstruction")
    p.add_argument("--ground-truth",type=Path,help="Independent evaluation only, not matching")
    p.add_argument("--profile",choices=["fast","balanced","accurate"],default="balanced")
    p.add_argument("--input-mode",choices=["auto","digital","scan"],default="auto")
    p=sub.add_parser("solve-document");p.add_argument("--pieces",type=Path,required=True);p.add_argument("--out",type=Path,required=True);p.add_argument("--rotations",action="store_true");p.add_argument("--params",type=json.loads);p.add_argument("--angle-mode",choices=["none","quarter","continuous"]);p.add_argument("--sampling",type=int,default=1);p.add_argument("--profile",choices=["fast","balanced","accurate","custom"])
    p=sub.add_parser("solve");_solve_opts(p);p.add_argument("--case",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p=sub.add_parser("evaluate");p.add_argument("--case",type=Path,required=True);p.add_argument("--out",type=Path,required=True);p.add_argument("--require-95",action="store_true")
    p=sub.add_parser("run");_gen_opts(p);_solve_opts(p);p.add_argument("--out",type=Path,required=True);p.add_argument("--require-95",action="store_true")
    p=sub.add_parser("benchmark");_source_opts(p);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--counts",type=int,nargs="+",default=[64,256,1024]);p.add_argument("--cuts",choices=CUTS,nargs="+",default=list(CUTS))
    p.add_argument("--seeds",type=int,nargs="+",default=[42,142]);p.add_argument("--rotations",action="store_true")
    p.add_argument("--no-tune",action="store_true");p.add_argument("--keep-cases",action="store_true");p.add_argument("--require-95",action="store_true")
    p=sub.add_parser("tune");_source_opts(p);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--cut",choices=CUTS,default="random");p.add_argument("--pieces",type=int,default=64)
    p.add_argument("--train-seeds",type=int,nargs="+",default=[10,20]);p.add_argument("--holdout-seeds",type=int,nargs="+",default=[30,40])
    p.add_argument("--holdout-source",type=Path);p.add_argument("--rotations",action="store_true")
    p.add_argument("--max-trials",type=int,default=12);p.add_argument("--require-95",action="store_true")
    a=parser.parse_args(argv)
    if a.command=="studio":
        from shardelix.studio import serve
        serve(a.port,not a.no_browser);return 0
    if a.command=="reconstruct":
        from shardelix import reconstruct_files
        inputs=a.fragments[0] if len(a.fragments)==1 else a.fragments
        state=reconstruct_files(inputs,a.out,matching_reference=a.reference,ground_truth=a.ground_truth,
                                profile=a.profile,input_mode=a.input_mode)
        print(json.dumps(state,indent=2));return 0
    if a.command in {"generate","run"}:
        src=read_source(a.source,a.max_side,a.page)
        fitted_grid=None
        if a.fit_grid:
            from .limits import RECTANGULAR,MAX_PIXELS
            from .generator import fit_generation_canvas
            from PIL import Image
            if a.cut in RECTANGULAR and not a.label_map:
                hh,ww,fitted_grid=fit_generation_canvas(src.shape[:2],a.pieces,a.cut,MAX_PIXELS,True)
                src=np.asarray(Image.fromarray(src).resize((ww,hh),Image.Resampling.LANCZOS))
        case=a.out if a.command=="generate" else a.out/"case"
        m=generate(src,case,a.pieces,a.cut,a.seed,a.rotations,np.load(a.label_map,allow_pickle=False) if a.label_map else None,angle_mode=a.angle_mode,missing=a.missing,sampling=a.sampling,grid=fitted_grid)
        if a.command=="generate":print(json.dumps(m,indent=2));return 0
        solve_case(case,a.out/"result",tune=not a.no_tune,params=a.params,profile=a.profile)
        r=evaluate_case(case,a.out/"result");print(json.dumps(r,indent=2));return int(a.require_95 and not r["pass_95"])
    if a.command=="solve-document":
        r=solve_external_document(a.pieces,a.out,a.rotations,a.params,profile=a.profile,angle_mode=a.angle_mode,sampling=a.sampling);print(json.dumps(r["diagnostics"],indent=2));return 0
    if a.command=="solve":
        r=solve_case(a.case,a.out,tune=not a.no_tune,params=a.params,profile=a.profile);print(json.dumps(r["diagnostics"],indent=2));return 0
    if a.command=="evaluate":
        r=evaluate_case(a.case,a.out);print(json.dumps(r,indent=2));return int(a.require_95 and not r["pass_95"])
    if a.command=="benchmark":
        from .benchmark import benchmark
        rows=benchmark(read_source(a.source,a.max_side,a.page),a.out,a.counts,a.cuts,a.seeds,a.rotations,not a.no_tune,a.keep_cases)
    elif a.command=="tune":
        from .benchmark import tune
        rows=tune(read_source(a.source,a.max_side,a.page),a.out,a.pieces,a.cut,a.train_seeds,a.holdout_seeds,a.rotations,
                  read_source(a.holdout_source,a.max_side,a.page) if a.holdout_source else None,a.max_trials)
    else:return 2
    return int(a.require_95 and not all(r.get("pass_95",False) for r in rows))

if __name__=="__main__":raise SystemExit(main())
