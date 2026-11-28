"""Reproducible scaling benchmark and guarded train/holdout parameter search."""
from __future__ import annotations

import csv
import itertools
import json
import platform
import shutil

import numpy as np
import scipy
import PIL

from shardelix.reconstruction.common import json_write
from shardelix.reconstruction.generator import generate
from shardelix.reconstruction.case_io import solve_case,evaluate_case

GRID_CONFIGS=[
    {"strategy":"straight","kind":"l1"},
    {"strategy":"straight","kind":"l2"},
    {"strategy":"straight","kind":"l2","normalize":False,"cycle":False},
    {"strategy":"straight","kind":"mgc","gradient_weight":.35},
    {"strategy":"straight","kind":"l1","threshold":1.4},
    {"strategy":"straight","kind":"l1","raw_weight":1.},
    *[{"strategy":"legacy","samples":s,"gradient_weight":g,"top_k":k,"refine":2}
      for s,g,k in itertools.product([32,56],[.25,.65],[4,8])]]
STRIP_CONFIGS=[{"strategy":"straight","kind":k,"gradient_weight":g}
               for k,g in itertools.product(["l1","l2"],[0.,.35])]
GEOM_CONFIGS=[{"k":k,"min_votes":v,"max_bucket":b}
              for k,v,b in itertools.product([16,24,32],[2,4],[4,8])]


def _write_summary(path,rows):
    path.mkdir(parents=True,exist_ok=True)
    json_write(path/"summary.json",{"environment":{"python":platform.python_version(),"numpy":np.__version__,
         "scipy":scipy.__version__,"pillow":PIL.__version__},"cases":rows,
         "all_pass_95":bool(rows) and all(r.get("pass_95",False) for r in rows)})
    if rows:
        keys=list(dict.fromkeys(k for r in rows for k in r))
        with (path/"summary.csv").open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)


def benchmark(src,output,counts,cuts,seeds,rotations=False,tune=True,keep=False):
    rows=[]
    for n,cut,seed in itertools.product(counts,cuts,seeds):
        case=output/"cases"/f"{cut}_{n}_{seed}";out=output/"runs"/f"{cut}_{n}_{seed}"
        try:
            generate(src,case,n,cut,seed,rotations)
            # Remove truth while solving, then restore solely for evaluation.
            hidden=case/"held_truth";(case/"evaluation").rename(hidden)
            try:solve_case(case,out,tune=tune)
            finally:hidden.rename(case/"evaluation")
            row=evaluate_case(case,out)
        except (ValueError,RuntimeError) as e:
            row={"piece_count":n,"cut":cut,"seed":seed,"mode":"blind","error":str(e),"pass_95":False}
        rows.append(row);print(json.dumps(row),flush=True);_write_summary(output,rows)
        if not keep and (case/"pieces").exists():shutil.rmtree(case/"pieces")
    return rows

def tune(src,output,count,cut,train_seeds,holdout_seeds,rotations=False,holdout_source=None,max_trials=12):
    if set(train_seeds)&set(holdout_seeds):
        raise ValueError("Training and holdout seeds must be disjoint")
    if not train_seeds or not holdout_seeds:raise ValueError("Both seed sets required")
    for seed in train_seeds+holdout_seeds:
        generate(holdout_source if holdout_source is not None and seed in holdout_seeds else src,output/"cases"/str(seed),count,cut,seed,rotations)
    if rotations and cut in {"strips","horizontal-strips"}:configs=[{"kind":"l1"},{"kind":"l2"}]
    elif rotations and cut=="grid":configs=[{"top_k":3},{"top_k":5}]
    else:configs=STRIP_CONFIGS if cut in {"strips","horizontal-strips"} else GRID_CONFIGS if cut=="grid" else GEOM_CONFIGS
    if cut not in {"grid","strips","horizontal-strips"}:
        configs=[{"strategy":"verified","k":16,"max_bucket":18},*configs]
    scores=[]
    if max_trials<1:raise ValueError("max-trials must be positive")
    for i,p in enumerate(configs[:max_trials]):
        rows=[]
        for seed in train_seeds:
            case=output/"cases"/str(seed);out=output/"train"/f"{i}_{seed}"
            solve_case(case,out,"blind",None,False,p)
            rows.append(evaluate_case(case,out))
        score=min(min(r["direct_position_accuracy"],r["adjacency_f1"]) for r in rows)
        scores.append({"config":p,"train_min":score,"rows":rows})
        print(f"train config {i}: minimum={score:.4f}",flush=True)
        if score>.95 and all(r["pass_95"] for r in rows):break
    winner=max(scores,key=lambda x: x["train_min"])
    # Freeze config before the first holdout solve. Never select using holdout.
    json_write(output/"frozen_config.json",winner["config"])
    held=[]
    for seed in holdout_seeds:
        case=output/"cases"/str(seed);out=output/"holdout"/str(seed)
        solve_case(case,out,"blind",None,False,winner["config"])
        held.append(evaluate_case(case,out))
    json_write(output/"tuning_report.json",{"train_seeds":train_seeds,"holdout_seeds":holdout_seeds,
        "train_results":scores,"frozen_config":winner["config"],"holdout":held,
        "different_holdout_source":holdout_source is not None,"max_trials":max_trials,
        "all_holdout_pass_95":all(r["pass_95"] for r in held)})
    _write_summary(output,held)
    return held
