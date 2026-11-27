"""Portable native/WebAssembly reconstruction workflow. No server dependencies."""
from __future__ import annotations
import csv,itertools,json,shutil,time,zipfile,hashlib
from pathlib import Path
from .reconstruction.limits import GRID_MAX,MAX_PIXELS,RECTANGULAR,max_count,validate_constraints

IMAGE_EXT={".png",".jpg",".jpeg",".tif",".tiff",".webp",".bmp"}
MAX_UPLOAD=200*1024**2
WEB_UPLOAD=32*1024**2
WEB_COUNT=4096
WEB_PIXELS=8_000_000
WEB_INPUT_PIXELS=24_000_000
WEB_BATCH=24
WEB_BATCH_OUTPUT_BYTES=192*1024**2
NATIVE_BATCH=64
DOWNLOADS={"assembly.png","assembly_with_cuts.png","assembly_reference_aligned.png","assembly_reference_aligned_with_cuts.png","cut_lines.png","placements.json","unanchored_components.json","ambiguity.json","diagnostics.json","metrics.json","source.png","result.zip","fragments.zip","benchmark.csv","benchmark.json","fragments_preview.png","ground_truth.zip","comparison.json"}

def safe_zip(source:Path,case:Path,*,browser=False,extract=True):
    """Inspect/extract public metadata and image leaves; never extract truth."""
    target=case/"pieces"
    if extract:target.mkdir(parents=True,exist_ok=True)
    size=0;num=0;meta=None;names=set()
    limit=WEB_COUNT if browser else GRID_MAX
    byte_limit=WEB_UPLOAD*3 if browser else MAX_UPLOAD*2
    with zipfile.ZipFile(source) as z:
        entries=z.infolist()
        if len(entries)>limit+100:raise ValueError("Too many archive entries")
        for e in entries:
            parts=e.filename.replace("\\","/").split("/")
            if e.is_dir():continue
            if ".." in parts or e.filename.startswith(("/","\\")) or ":" in parts[0]:raise ValueError("Unsafe archive path")
            leaf=parts[-1];suffix=Path(leaf).suffix.lower()
            if "evaluation" in parts or "result" in parts:continue
            if leaf=="case.json":
                if meta is not None or e.file_size>64*1024:raise ValueError("Ambiguous or oversized metadata")
                meta=json.loads(z.read(e))
                if not isinstance(meta,dict):raise ValueError("Invalid public metadata")
                continue
            if suffix not in IMAGE_EXT or not (len(parts)==1 or "pieces" in parts[:-1]):continue
            num+=1;size+=e.file_size
            if num>limit or size>byte_limit:raise ValueError("Archive exceeds fragment/byte budget")
            if leaf in names:raise ValueError("Duplicate fragment filename")
            names.add(leaf)
            if extract:
                with z.open(e) as f,(target/leaf).open("wb") as out:shutil.copyfileobj(f,out,1024*1024)
    if not num:raise ValueError("No supported fragments found in ZIP")
    return meta,num

def public_metadata(meta):
    if meta is None:return None
    fields=("height","width","piece_count","cut","rotations","grid","expected_count","angle_mode","sampling")
    return {k:meta[k] for k in fields if k in meta}

def profiles(conf):
    profile=conf.get("profile")
    if profile not in (None,"fast","balanced","accurate","custom"):
        raise ValueError("Unknown performance profile")
    tune=bool(conf.get("tune",True)) if profile in (None,"custom") else profile!="fast"
    idx=conf.get("index_mode","auto") if profile in (None,"custom") else "auto"
    strategy=conf.get("strategy","auto") if profile in (None,"custom") else "auto"
    if idx not in {"auto","legacy","midpoint","mixed"} or strategy not in {"auto","sparse","classic"}:
        raise ValueError("Invalid custom solver parameters")
    params={}
    if strategy=="classic":raise ValueError("Classic strategy is available through the Python API, not this workflow")
    if strategy=="sparse" or idx!="auto":params["strategy"]="sparse"
    if idx!="auto":params["index_mode"]=idx
    return profile,tune,params

def _pixel_budget(files,limit):
    from PIL import Image
    total=0
    for f in files:
        with Image.open(f) as im:total+=im.width*im.height
        if total>limit:raise ValueError("Decoded input exceeds pixel budget")
    return total

def _check_preview(files,browser):
    if not files:raise ValueError("Select input files first")
    if sum(p.stat().st_size for p in files)>(WEB_UPLOAD if browser else MAX_UPLOAD):raise ValueError("Upload exceeds byte budget")
    if len(files)>(WEB_COUNT if browser else GRID_MAX):raise ValueError("Too many fragments")

def inspect_uploads(files,browser=False):
    _check_preview(files,browser)
    if len(files)==1 and files[0].suffix.lower()==".zip":
        meta,n=safe_zip(files[0],files[0].parent,browser=browser,extract=False)
        return {"metadata":public_metadata(meta),"observed_count":n}
    return {"metadata":None,"observed_count":len(files)}

def _generate_case(root,files,conf,update,browser=False):
    from .reconstruction.common import read_source,save_rgba
    from .reconstruction.generator import generate,fit_generation_canvas
    from PIL import Image
    import numpy as np
    case=root/"case";out=root/"result";out.mkdir(parents=True,exist_ok=True)
    angle=conf.get("angle_mode","quarter" if conf.get("rotations",True) else "none")
    rotations=angle!="none";sampling=int(conf.get("sampling",1));missing=float(conf.get("missing",0))
    if angle not in ("none","quarter","continuous"):raise ValueError("Invalid rotation mode")
    if sampling not in (1,2,3,4) or (sampling!=1 and angle!="continuous"):raise ValueError("Invalid sampling")
    count_cap=WEB_COUNT if browser else GRID_MAX
    pixel_cap=WEB_PIXELS if browser else MAX_PIXELS
    if len(files)!=1:raise ValueError("Select one source image or PDF")
    cut=conf["cut"];n=int(conf["count"])
    if not 1<=n<=min(max_count(cut),count_cap):raise ValueError("Piece count exceeds backend limit")
    page=int(conf.get("page",1));max_side=int(conf.get("max_side",2048))
    if not (max_side==0 or 128<=max_side<=8192):raise ValueError("Resolution must be original or 128–8192")
    if browser and files[0].suffix.lower()==".pdf":raise ValueError("Use an image or convert PDF page to PNG for browser mode")
    if files[0].suffix.lower()!=".pdf":_pixel_budget(files,WEB_INPUT_PIXELS if browser else MAX_PIXELS)
    src=read_source(files[0],max_side,page)
    initial=src.shape[:2]
    if browser and angle=="continuous":
        pixel_cap=min(pixel_cap,WEB_INPUT_PIXELS//(2*sampling*sampling))
    h,w,grid=fit_generation_canvas(initial,n,cut,pixel_cap,n>1024 or angle=="continuous" or bool(missing))
    resized=(h,w)!=initial
    if resized:
        update(f"Auto-fitting canvas: {initial[1]} × {initial[0]} → {w} × {h}")
        src=np.asarray(Image.fromarray(src).resize((w,h),Image.Resampling.LANCZOS))
    if grid and min(h//grid[0],w//grid[1])<2:
        update("Generating thin cells (limited texture evidence)")
    validate_constraints(n,(h,w),cut,grid,rotations)
    save_rgba(out/"source.png",src)
    update("Generating shuffled fragments")
    generate(src,case,n,cut,int(conf.get("seed",42)),rotations,angle_mode=angle,missing=missing,sampling=sampling,grid=grid)
    del src
    meta=json.loads((case/"case.json").read_text())
    return meta,case,out,{"preferred_size":[initial[1],initial[0]],"auto_fitted":resized,"grid":list(grid) if grid else None,"thin_cells":bool(grid and min(h//grid[0],w//grid[1])<2)}

def _public_archive(case,out):
    with zipfile.ZipFile(out/"fragments.zip","w",zipfile.ZIP_STORED) as z:
        if (case/"case.json").exists():z.write(case/"case.json","case.json")
        for p in sorted((case/"pieces").iterdir()):z.write(p,"pieces/"+p.name)

def _generate_only(root,files,conf,notify,browser=False):
    from PIL import Image,ImageDraw
    from .reconstruction.common import json_write
    started=time.perf_counter()
    update=lambda stage:notify({"state":"running","stage":stage,"case":1,"total_cases":1,"elapsed_seconds":time.perf_counter()-started})
    meta,case,out,fit_info=_generate_case(root,files,conf,update,browser)
    update("Packaging fragments and separate ground truth")
    _public_archive(case,out)
    with zipfile.ZipFile(out/"ground_truth.zip","w",zipfile.ZIP_DEFLATED) as z:
        z.write(out/"source.png","source.png")
        z.writestr("evaluation/fragment_manifest.json",json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((case/"pieces").iterdir())},separators=(",",":")))
        for f in sorted((case/"evaluation").iterdir()):
            if f.name in ("ground_truth.json","truth_labels.npz"):z.write(f,"evaluation/"+f.name)
    # Contact sheet uses generated shuffled fragments only, bounded preview size.
    paths=sorted((case/"pieces").iterdir());n=len(paths)
    show=min(96,n);cols=min(8,show);rows=(show+cols-1)//cols
    tile=128;sheet=Image.new("RGB",(cols*tile,rows*tile),(240,242,239));draw=ImageDraw.Draw(sheet)
    for j,f in enumerate(paths[:show]):
        with Image.open(f) as im:
            im=im.convert("RGBA");im.thumbnail((tile-14,tile-30))
            x=(j%cols)*tile+(tile-im.width)//2;y=(j//cols)*tile+(tile-25-im.height)//2
            sheet.paste(im,(x,y),im);draw.text(((j%cols)*tile+8,(j//cols)*tile+tile-21),f"{j+1:03d}",fill=(75,85,80))
    sheet.save(out/"fragments_preview.png",optimize=True)
    shutil.rmtree(case)
    return {"state":"done","stage":"Fragments generated","generation":{"observed":n,"expected":meta.get("expected_count",n),"width":meta["width"],"height":meta["height"],"cut":meta["cut"],"angle_mode":meta.get("angle_mode","quarter" if meta["rotations"] else "none"),"preview_count":show,"seconds":time.perf_counter()-started,**fit_info},"files":sorted(p.name for p in out.iterdir() if p.name in DOWNLOADS)}

def _evaluate_external(case,out,meta,truth_dir,browser,allow_rotation):
    """Called only AFTER solve and render; truth never enters solver inputs."""
    from PIL import Image
    import numpy as np
    from .reconstruction.common import json_write
    from .reconstruction.cli import evaluate_case
    files=list(truth_dir.iterdir()) if truth_dir.exists() else []
    if not files:return
    if len(files)!=1:raise ValueError("Select one reference image or ground-truth ZIP")
    f=files[0];reference=None;warning=None;pose_evaluated=False;pose_rotation=None
    if f.suffix.lower()==".zip":
        allowed={"source.png","evaluation/ground_truth.json","evaluation/truth_labels.npz","evaluation/fragment_manifest.json"}
        limit=WEB_PIXELS if browser else MAX_PIXELS
        with zipfile.ZipFile(f) as z:
            seen=set()
            for e in z.infolist():
                if e.is_dir():continue
                if e.filename not in allowed or e.filename in seen:raise ValueError("Ground-truth ZIP must contain only the generated evaluation files")
                seen.add(e.filename)
                if e.file_size>limit*8+1024*1024:raise ValueError("Ground-truth entry too large")
                p=(out/"source.png") if e.filename=="source.png" else case/e.filename
                p.parent.mkdir(parents=True,exist_ok=True)
                with z.open(e) as src,p.open("wb") as dst:shutil.copyfileobj(src,dst,1024*1024)
        reference=out/"source.png" if (out/"source.png").exists() else None
        # Verify the pose archive belongs to these fragment IDs and canvas; no silent partial scoring.
        if meta is not None and (case/"evaluation/ground_truth.json").exists():
            t=json.loads((case/"evaluation/ground_truth.json").read_text())
            poses=t.get("poses",{})
            ids={p.stem for p in (case/"pieces").iterdir()}
            manifest_path=case/"evaluation/fragment_manifest.json"
            manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else None
            actual={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((case/"pieces").iterdir())}
            if (not isinstance(poses,dict) or set(poses)!=ids or t.get("case",{}).get("height")!=meta["height"]
                    or t.get("case",{}).get("width")!=meta["width"] or manifest!=actual):
                warning="Ground-truth manifest does not match the imported fragment bytes and dimensions."
            else:
                try:
                    if "expected_count" in meta:
                        labels=case/"evaluation/truth_labels.npz"
                        with zipfile.ZipFile(labels) as z:
                            if len(z.infolist())!=1 or z.infolist()[0].file_size>limit*8+65536:raise ValueError("Invalid label archive")
                        with np.load(labels,allow_pickle=False) as z:
                            if z["labels"].shape!=(meta["height"],meta["width"]):raise ValueError("Truth canvas mismatch")
                    metrics=evaluate_case(case,out);pose_evaluated=True;pose_rotation=metrics.get("evaluation_rotation_cw_deg")
                except (ValueError,KeyError,TypeError,OSError) as exc:warning="Pose evaluation unavailable: "+str(exc)[:180]
        else:warning="No compatible per-piece ground truth; image comparison only."
    elif f.suffix.lower() in IMAGE_EXT:
        reference=f
    else:raise ValueError("Ground truth must be an image or generated truth ZIP")
    comparison={"pose_evaluated":pose_evaluated,"warning":warning}
    if reference:
        from .reconstruction.reference_alignment import compare_reference
        measured=compare_reference(out,reference,allow_rotation=allow_rotation,
                                   pose_rotation=pose_rotation,pixel_cap=WEB_INPUT_PIXELS if browser else MAX_PIXELS)
        comparison.update(measured)
        if warning:comparison["warning"]=" ".join(filter(None,[warning,measured.get("warning")]))
    json_write(out/"comparison.json",comparison)

def _single(root,files,conf,update,browser=False,truth_dir=None):
    from .reconstruction.common import read_source,save_rgba,json_write
    from .reconstruction.generator import generate,dimensions
    from .reconstruction.cli import solve_case,evaluate_case,solve_external_document
    from PIL import Image
    import numpy as np
    case=root/"case";out=root/"result";out.mkdir(parents=True,exist_ok=True)
    task=conf.get("task","demo");rotations=bool(conf.get("rotations",True))
    angle=conf.get("angle_mode","quarter" if rotations else "none")
    sampling=int(conf.get("sampling",1));missing=float(conf.get("missing",0))
    if angle not in ("none","quarter","continuous"):raise ValueError("Invalid rotation mode")
    if sampling not in (1,2,3,4) or (sampling!=1 and angle!="continuous"):raise ValueError("Invalid sampling")
    rotations=angle!="none"
    count_cap=WEB_COUNT if browser else GRID_MAX
    pixel_cap=WEB_PIXELS if browser else MAX_PIXELS
    profile,tune,params=profiles(conf)
    started=time.perf_counter()
    if task=="demo":
        meta,case,out,fit_info=_generate_case(root,files,conf,update,browser)
    elif task=="blind":
        meta=None
        if len(files)==1 and files[0].suffix.lower()==".zip":
            meta,n=safe_zip(files[0],case,browser=browser)
        else:
            (case/"pieces").mkdir(parents=True)
            for f in files:
                if f.suffix.lower() not in IMAGE_EXT:raise ValueError("Blind mode accepts images or a ZIP")
                shutil.copy2(f,case/"pieces"/f.name)
            n=len(files)
        if meta is None or not conf.get("use_metadata",True):
            cut=conf.get("cut","grid")
            if cut=="document":meta=None
            else:
                h,w=int(conf["height"]),int(conf["width"])
                expected=int(conf.get("expected_count",0)) or n
                grid=[1,expected] if cut=="strips" else [expected,1] if cut=="horizontal-strips" else [int(conf["rows"]),int(conf["cols"])] if cut=="grid" else None
                if cut=="grid":expected=grid[0]*grid[1]
                meta=dict(height=h,width=w,piece_count=n,cut=cut,rotations=rotations,grid=grid)
                if expected!=n or angle=="continuous":meta.update(expected_count=expected,angle_mode=angle,sampling=sampling)
        if meta is not None:
            meta=public_metadata(meta)
            # The metadata is authoritative when the import toggle is enabled.
            angle=meta.get("angle_mode","quarter" if meta["rotations"] else "none")
            rotations=angle!="none"
            json_write(case/"case.json",meta)

    else:raise ValueError("Invalid workflow")
    if meta is not None:
        if meta["height"]*meta["width"]>pixel_cap:raise ValueError("Canvas exceeds backend pixel budget")
        if meta.get("expected_count",meta["piece_count"])>count_cap:raise ValueError("Count exceeds backend limit")
        if params.get("strategy")=="sparse" and (not meta.get("grid") or meta["height"]%meta["grid"][0] or meta["width"]%meta["grid"][1]):
            raise ValueError("Explicit sparse/index settings require a uniform rectangular grid")
    elif params:
        raise ValueError("Custom grid settings do not apply to document strips")
    if browser:_pixel_budget(list((case/"pieces").iterdir()),WEB_INPUT_PIXELS if meta is not None else WEB_PIXELS)
    update("Matching and assembling — CPU")
    held=None
    # Enforce independent evaluation by moving truth OUTSIDE the case before solve.
    if task=="demo":
        held=root/"truth-held-out";(case/"evaluation").rename(held)
    try:
        if meta is None:
            result=solve_external_document(case/"pieces",out,rotations,profile=profile,angle_mode=angle,sampling=sampling,max_canvas_pixels=pixel_cap)
        else:
            result=solve_case(case,out,tune=tune,params=params or None,profile=profile)
    finally:
        if held is not None:held.rename(case/"evaluation")
    update("Rendering and evaluating")
    if task=="demo":
        from .reconstruction.reference_alignment import compare_reference
        metrics=evaluate_case(case,out)
        comparison=compare_reference(out,out/"source.png",allow_rotation=rotations,
                                     pose_rotation=metrics.get("evaluation_rotation_cw_deg"),pixel_cap=pixel_cap)
        json_write(out/"comparison.json",{**comparison,"pose_evaluated":True})
    elif truth_dir is not None:
        try:_evaluate_external(case,out,meta,truth_dir,browser,rotations)
        except Exception as exc:
            json_write(out/"comparison.json",{"warning":"Reference evaluation failed: "+str(exc)[:300],"pose_evaluated":False})
    d=json.loads((out/"diagnostics.json").read_text())
    m=json.loads((out/"metrics.json").read_text()) if (out/"metrics.json").exists() else None
    _public_archive(case,out)
    d["workflow_elapsed_seconds"]=time.perf_counter()-started
    d["performance_profile"]=profile or "legacy"
    from .native_core import backend
    d["compute_backend"]=backend()
    # Peak RSS is the process high-water mark (not an incremental per-case value).
    try:
        import resource,sys
        d["process_peak_rss_mib"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1048576 if sys.platform=="darwin" else 1024)
    except (ImportError,AttributeError,OSError):d["process_peak_rss_mib"]=None
    json_write(out/"diagnostics.json",d)
    return d,m,out

def case_configs(conf,browser=False):
    if not conf.get("batch"):return [dict(conf)]
    if conf.get("task")!="demo":raise ValueError("Batch benchmark requires a source image")
    b=conf["batch"]
    if not isinstance(b,dict):raise ValueError("Invalid batch settings")
    choices=[b.get("counts",[conf["count"]]),b.get("seeds",[conf.get("seed",42)]),b.get("cuts",[conf["cut"]]),b.get("angles",[conf.get("angle_mode","quarter")]),b.get("profiles",[conf.get("profile","balanced")]),b.get("missing_rates",[conf.get("missing",0)])]
    cap=WEB_BATCH if browser else NATIVE_BATCH
    total=1
    for c in choices:
        if not isinstance(c,list) or not c:raise ValueError("Every benchmark axis must be nonempty")
        total*=len(c)
        if total>cap:raise ValueError(f"Batch limit is {cap} cases")
    out=[]
    for n,seed,cut,angle,profile,missing in itertools.product(*choices):
        out.append({**conf,"batch":None,"count":int(n),"seed":int(seed),"cut":cut,"angle_mode":angle,"sampling":conf.get("sampling",1) if angle=="continuous" else 1,"profile":profile,"missing":float(missing)})
    return out

def run_job(root:Path,conf,notify=lambda s:None,*,browser=False):
    """Single source or batch; all failures retained in the benchmark report."""
    from .reconstruction.common import json_write
    files=sorted((root/"upload").iterdir());_check_preview(files,browser)
    if conf.get("task")=="generate":return _generate_only(root,files,conf,notify,browser)
    result=root/"result";result.mkdir(exist_ok=True)
    configs=case_configs(conf,browser);batch=bool(conf.get("batch"))
    rows=[];last=None;started=time.perf_counter();retained=0
    for i,c in enumerate(configs):
        dest=root/"runs"/f"case-{i+1:03d}";dest.mkdir(parents=True)
        def update(stage):
            notify({"state":"running","stage":stage,"case":i+1,"total_cases":len(configs),"elapsed_seconds":time.perf_counter()-started})
        update("Validating inputs")
        row={k:c.get(k) for k in ("count","seed","cut","angle_mode","profile","missing","sampling")}
        row["case"]=i+1
        try:
            if browser and batch and retained>=WEB_BATCH_OUTPUT_BYTES:raise ValueError("Browser benchmark retained-output budget exhausted; use native Studio")
            d,m,o=_single(dest,files,c,update,browser,root/"truth")
            row.update(status="done",algorithm=d.get("algorithm"),runtime_seconds=d.get("runtime_seconds"),
                elapsed_seconds=d.get("workflow_elapsed_seconds"),peak_process_rss_mib=d.get("process_peak_rss_mib"),placed=d.get("placed"),coverage=d.get("coverage"),**{k:m.get(k) for k in ("direct_position_accuracy","raw_position_accuracy","evaluation_rotation_cw_deg","adjacency_f1","pass_95","collision_pixels")} if m else {})
            if last is not None and batch:
                for name in ("fragments.zip","source.png"):
                    (last[2]/name).unlink(missing_ok=True)
            last=(d,m,o)
            retained=sum(p.stat().st_size for p in (root/"runs").rglob("*") if p.is_file()) if browser and batch else 0
        except Exception as exc:
            row.update(status="error",error=str(exc)[:500])
            if not batch:raise
        rows.append(row)
        # Keep only benchmark outputs, not all generated input/GT on disk.
        shutil.rmtree(dest/"case",ignore_errors=True)
    if last:
        d,m,o=last
        shutil.copytree(o,result,dirs_exist_ok=True)
    else:d,m={},None
    if batch:
        fields=list(dict.fromkeys(k for row in rows for k in row))
        with (result/"benchmark.csv").open("w",encoding="utf-8",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        json_write(result/"benchmark.json",{"cases":rows,"total":len(rows),"completed":sum(r["status"]=="done" for r in rows),"passed":sum(r.get("pass_95") is True for r in rows),"elapsed_seconds":time.perf_counter()-started,"native_peak_rss_note":"Process high-water mark, cumulative across cases; unavailable in WASM."})
    if not batch and last is None:raise ValueError("No completed result")
    with zipfile.ZipFile(result/"result.zip","w",zipfile.ZIP_DEFLATED) as z:
        for p in result.rglob("*"):
            if p.is_file() and p.name not in ("result.zip","fragments.zip","source.png"):z.write(p,p.relative_to(result))
        if batch:
            for directory in (root/"runs").iterdir():
                for p in (directory/"result").rglob("*"):
                    if p.is_file() and p.name not in ("fragments.zip","source.png"):
                        z.write(p,str(Path("cases")/directory.name/p.relative_to(directory/"result")))
    shutil.rmtree(root/"runs",ignore_errors=True)
    return {"state":"done","stage":"Benchmark complete" if batch else "Reconstruction exported",
            "diagnostics":{k:v for k,v in d.items() if k in ("algorithm","runtime_seconds","status","placed","piece_count","coverage","collision_pixels","render_collision_pixels","identical_piece_count","sparse_high_consensus","reciprocal_seam_fraction","expected_count","missing_count","angle_mode","sampling","available_piece_recovery","normalization_seconds","index_mode","index_fallback_used","index_query_seconds","retrieval_seconds","assembly_seconds","performance_profile","process_peak_rss_mib","workflow_elapsed_seconds","frame_anchor_status","multigap_status","extra_verification","compute_backend","fast_rescue_v13")},
            "metrics":m,"comparison":json.loads((result/"comparison.json").read_text()) if (result/"comparison.json").exists() else None,"benchmark":rows if batch else None,"files":sorted(p.name for p in result.iterdir() if p.name in DOWNLOADS)}
