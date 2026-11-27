"""Portable native/WebAssembly reconstruction workflow. No server dependencies."""
from __future__ import annotations
import csv,itertools,json,shutil,time,zipfile,hashlib,math
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
DOWNLOADS={"assembly.png","assembly_with_cuts.png","assembly_reference_aligned.png","assembly_reference_aligned_with_cuts.png","cut_lines.png","placements.json","unanchored_components.json","ambiguity.json","diagnostics.json","metrics.json","source.png","result.zip","fragments.zip","benchmark.csv","benchmark.json","fragments_preview.png","ground_truth.zip","comparison.json","scan_report.json"}

# Auxiliary views and truth are never counted as fragments. Unknown nested
# layouts are accepted only when one primary image directory is unambiguous.
_AUX_DIRS={"evaluation","result","results","ground_truth","ground-truth","truth","__macosx",
           "bg_mask","bg_mask_vis","masks","mask","mask_vis","preview","previews","thumbnails",
           "reference","references","original","originals","labels","gt","output","outputs"}
_AUX_FILES={"source.png","reference.png","ground_truth.png","assembly.png","assembly_with_cuts.png",
            "fragments_preview.png","cut_lines.png"}

def _strip_hint(openers,count,browser):
    from PIL import Image
    if not 2<=count<=1024:return None
    dims=[];total=0
    for opener in openers:
        with opener() as f,Image.open(f) as im:
            w,h=im.size;total+=w*h
            if total>(WEB_INPUT_PIXELS if browser else MAX_PIXELS*3):
                raise ValueError("Decoded input exceeds pixel budget")
            if max(w,h)<4*min(w,h):return None
            dims.append((min(w,h),max(w,h)))
    long=[d[1] for d in dims]
    med=sorted(long)[len(long)//2]
    if max(abs(v-med) for v in long)>max(2,.015*med):return None
    return {"height":med,"width_range":[min(d[0] for d in dims),max(d[0] for d in dims)]}

def safe_zip(source:Path,case:Path,*,browser=False,extract=True,inspect_geometry=False,scan=False,
             details=False,extract_masks=False):
    """Inspect/extract primary raster images. Never extract truth or mask previews as pieces."""
    from PIL import Image
    target=case/"pieces"
    if extract:target.mkdir(parents=True,exist_ok=True)
    meta=None;names=set();groups={};mask_entries={};size=0
    limit=WEB_COUNT if browser else GRID_MAX
    byte_limit=WEB_UPLOAD*3 if browser else MAX_UPLOAD*2
    with zipfile.ZipFile(source) as z:
        entries=z.infolist()
        if len(entries)>3*limit+100:raise ValueError("Too many archive entries")
        for e in entries:
            parts=e.filename.replace("\\","/").split("/")
            if e.is_dir():continue
            if ".." in parts or e.filename.startswith(("/","\\")) or ":" in parts[0]:raise ValueError("Unsafe archive path")
            dirs=tuple(parts[:-1]);low=tuple(p.lower() for p in dirs);leaf=parts[-1];suffix=Path(leaf).suffix.lower()
            if any(x in _AUX_DIRS for x in low):
                # Only the documented binary foreground-mask PNG convention is
                # recognized, not arbitrary .npy arrays or result images.
                if suffix==".png" and low and low[-1]=="bg_mask_vis" and not any(x in _AUX_DIRS for x in low[:-1]):
                    key=(dirs[:-1],Path(leaf).stem)
                    if key in mask_entries:raise ValueError("Duplicate foreground mask")
                    mask_entries[key]=e
                continue
            if leaf=="case.json":
                if meta is not None or e.file_size>64*1024:raise ValueError("Ambiguous or oversized metadata")
                meta=json.loads(z.read(e))
                if not isinstance(meta,dict):raise ValueError("Invalid public metadata")
                continue
            explicit_dir="pieces" in low or (scan and "scans" in low)
            if suffix not in IMAGE_EXT or (leaf.lower() in _AUX_FILES and not explicit_dir):continue
            groups.setdefault(dirs,[]).append(e)
        explicit=[d for d in groups if "pieces" in tuple(x.lower() for x in d) or (scan and "scans" in tuple(x.lower() for x in d))]
        if explicit:chosen=explicit
        elif () in groups:chosen=[()]
        else:
            # Standard folder-wrapped ZIPs produced by OS archive tools.
            # Do not guess when multiple unrelated raster directories exist.
            chosen=list(groups)
            if len(chosen)>1:raise ValueError("Multiple image directories in ZIP; put primary fragments in pieces/ or scans/")
        images=[(d,e) for d in chosen for e in groups[d]]
        num=len(images)
        if not num:raise ValueError("No supported fragments found in ZIP")
        if num>limit:raise ValueError("Archive exceeds fragment count budget")
        mask_count=sum((d,Path(e.filename.replace("\\","/").split("/")[-1]).stem) in mask_entries for d,e in images)
        if extract_masks and mask_count and mask_count!=num:
            raise ValueError("Foreground-mask archive must supply bg_mask_vis PNG for every fragment")
        decoded=0
        for d,e in images:
            leaf=e.filename.replace("\\","/").split("/")[-1]
            if leaf in names:raise ValueError("Duplicate fragment filename")
            names.add(leaf)
            mask=mask_entries.get((d,Path(leaf).stem))
            size+=e.file_size+(mask.file_size if mask is not None else 0)
            if size>byte_limit:raise ValueError("Archive exceeds fragment/byte budget")
            if extract:
                if extract_masks and mask is not None:
                    # Decode only verified same-size binary PNG masks. Alpha
                    # explicitly describes foreground (white=paper, black=background).
                    import numpy as np
                    with z.open(e) as f,Image.open(f) as im,z.open(mask) as mf,Image.open(mf) as mm:
                        if im.size!=mm.size:raise ValueError("Foreground mask dimensions do not match image")
                        decoded+=im.width*im.height
                        if decoded>(WEB_INPUT_PIXELS if browser else MAX_PIXELS*3):raise ValueError("Decoded input exceeds pixel budget")
                        if mm.mode not in ("1","L","RGB","RGBA"):raise ValueError("Foreground mask must be binary raster")
                        m=np.asarray(mm.convert("L"))
                        if not np.all((m==0)|(m==255)) or not np.any(m):raise ValueError("Foreground mask must be nonempty and binary (0/255)")
                        rgba=im.convert("RGBA");rgba.putalpha(Image.fromarray(m));dest=target/(Path(leaf).stem+".png")
                        if dest.exists():raise ValueError("Duplicate fragment filename stem")
                        rgba.save(dest)
                else:
                    with z.open(e) as f,(target/leaf).open("wb") as out:shutil.copyfileobj(f,out,1024*1024)
        if inspect_geometry:
            openers=[lambda e=e:z.open(e) for _,e in images]
            explicit_scans=any("scans" in tuple(x.lower() for x in d) for d,_ in images)
            suggestion=_suggest_grid(openers,num,browser) if meta is None and not mask_count and not explicit_scans else None
            hint=_strip_hint(openers,num,browser) if meta is None and (suggestion is None or mask_count) else None
            extra={"suggested_document":hint,"foreground_mask_count":mask_count}
            return (meta,num,suggestion,extra) if details else (meta,num,suggestion)
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

def _suggest_grid(openers, count, browser):
    """Detect same-sized opaque raster tiles; shape is a suggestion, not truth."""
    from PIL import Image
    from .reconstruction.generator import _factor_pairs
    total=0;tile=None
    pixel_limit=WEB_INPUT_PIXELS if browser else MAX_PIXELS*3
    for opener in openers:
        with opener() as f,Image.open(f) as im:
            total+=im.width*im.height
            if total>pixel_limit:raise ValueError("Decoded input exceeds pixel budget")
            if tile is None:tile=im.size
            if im.size!=tile:return None
            if ("A" in im.getbands() or "transparency" in im.info) and im.convert("RGBA").getchannel("A").getextrema()[0]==0:
                return None
    if not tile or total>(WEB_PIXELS if browser else MAX_PIXELS):return None
    tw,th=tile
    # Rank factor pairs by common landscape aspect. Pixels/count alone do not
    # identify the original shape; the UI must expose editable alternatives.
    pairs=sorted(_factor_pairs(count),key=lambda rc:(abs(math.log((rc[1]*tw/(rc[0]*th))/(4/3))),rc))
    layouts=[{"rows":r,"cols":c,"height":r*th,"width":c*tw} for r,c in pairs]
    return {"tile_width":tw,"tile_height":th,"expected_count":count,
            "basis":"equal-size opaque tiles; assumes no missing fragments",
            "layouts":layouts}


def inspect_uploads(files,browser=False):
    _check_preview(files,browser)
    if len(files)==1 and files[0].suffix.lower()==".zip":
        meta,n,suggestion,extra=safe_zip(files[0],files[0].parent,browser=browser,extract=False,inspect_geometry=True,scan=True,details=True)
        return {"metadata":public_metadata(meta),"observed_count":n,"suggested_grid":suggestion,**extra}
    if any(p.suffix.lower() not in IMAGE_EXT for p in files):
        raise ValueError("Select supported fragment images or one ZIP")
    openers=[lambda p=p:p.open("rb") for p in files]
    suggestion=_suggest_grid(openers,len(files),browser)
    hint=_strip_hint(openers,len(files),browser) if suggestion is None else None
    return {"metadata":None,"observed_count":len(files),"suggested_grid":suggestion,
            "suggested_document":hint,"foreground_mask_count":0}

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

def _scan_single(root,files,conf,update,browser=False,truth_dir=None):
    from .reconstruction.scan import preprocess,reconstruct
    from .reconstruction.common import json_write
    from .native_core import backend
    started=time.perf_counter()
    case=root/"case";out=root/"result";out.mkdir(parents=True,exist_ok=True)
    if len(files)==1 and files[0].suffix.lower()==".zip":
        safe_zip(files[0],case,browser=browser,scan=True,extract_masks=True)
        (case/"pieces").rename(case/"scans")
        paths=sorted((case/"scans").iterdir())
    else:
        if any(p.suffix.lower() not in IMAGE_EXT for p in files):raise ValueError("Scan mode accepts images or one ZIP")
        paths=files
    hint=_strip_hint([lambda p=p:p.open("rb") for p in paths],len(paths),browser)
    if hint:
        from .reconstruction.physical_document import reconstruct_strips,NotDocumentStrips
        update("Rectifying scanned paper strips")
        try:
            d=reconstruct_strips(paths,case/"pieces",out,update,pixel_cap=WEB_PIXELS if browser else MAX_PIXELS)
        except NotDocumentStrips:
            update("Strip silhouette inconsistent; using irregular scan processing")
        else:
            _public_archive(case,out)
            d.update(compute_backend=backend(),workflow_elapsed_seconds=time.perf_counter()-started,performance_profile="scan-document-auto")
            json_write(out/"diagnostics.json",d)
            if not browser and conf.get("ocr_refine",True):
                from .reconstruction.physical_ocr import native_refine
                refined=native_refine(out,root/"ocr-work",update)
                if refined is not None:d=refined
            return d,None,out
    update("Segmenting paper from scan backgrounds")
    pieces,report=preprocess(paths,case/"pieces",pixel_limit=WEB_INPUT_PIXELS if browser else MAX_PIXELS*3)
    d=reconstruct(pieces,out,report,update,pixel_cap=WEB_PIXELS if browser else MAX_PIXELS)
    _public_archive(case,out)
    d.update(compute_backend=backend(),workflow_elapsed_seconds=time.perf_counter()-started,performance_profile="scan-auto")
    json_write(out/"diagnostics.json",d)
    if truth_dir is not None and truth_dir.exists() and any(truth_dir.iterdir()):
        json_write(out/"comparison.json",{"pose_evaluated":False,"warning":"Scan mode does not evaluate digital ground truth: arbitrary rotation, scale and torn-edge loss require a separate physical benchmark."})
    return d,None,out

def _single(root,files,conf,update,browser=False,truth_dir=None):
    from .reconstruction.common import read_source,save_rgba,json_write
    if conf.get("task")=="blind":
        mode=conf.get("input_mode","digital")
        if mode not in ("digital","auto","scan"):raise ValueError("Unknown reconstruction input mode")
        if mode=="scan":
            return _scan_single(root,files,conf,update,browser,truth_dir)
        if mode=="auto" and conf.get("auto_layout",False):
            info=inspect_uploads(files,browser)
            if not info["metadata"] and (not info["suggested_grid"] or info["observed_count"]==1):
                return _scan_single(root,files,conf,update,browser,truth_dir)
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
    started=time.perf_counter();import_geometry=None
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
        if conf.get("auto_layout",False):
            # Resolve again from the actual extracted fragments, not the UI's
            # cached inspection or old manual fields. Never inspect ground truth.
            if meta is not None:
                import_geometry="archive-metadata"
            else:
                update("Auto-detecting fragment geometry")
                paths=sorted((case/"pieces").iterdir())
                suggestion=_suggest_grid((lambda p=p:p.open("rb") for p in paths),n,browser)
                if not suggestion:
                    raise ValueError("Automatic setup cannot determine the canvas for irregular or unequal-size fragments without metadata. Turn off Automatic setup to enter the known canvas/cut (or use Document strips).")
                layout=suggestion["layouts"][0]
                # Unknown square-tile orientation is resolved by quarter-turn
                # matching; there is no need to specify a rotation mode.
                meta=dict(height=layout["height"],width=layout["width"],piece_count=n,
                          cut="grid",rotations=True,grid=[layout["rows"],layout["cols"]])
                import_geometry="inferred-grid"
        elif meta is None or not conf.get("use_metadata",True):
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
    if import_geometry:
        d["import_geometry_mode"]=import_geometry
        d["import_geometry"]={k:meta[k] for k in ("width","height","grid","cut","piece_count")}
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
            "diagnostics":{k:v for k,v in d.items() if k in ("algorithm","runtime_seconds","status","placed","piece_count","coverage","collision_pixels","render_collision_pixels","identical_piece_count","sparse_high_consensus","reciprocal_seam_fraction","expected_count","missing_count","angle_mode","sampling","available_piece_recovery","normalization_seconds","index_mode","index_fallback_used","index_query_seconds","retrieval_seconds","assembly_seconds","performance_profile","process_peak_rss_mib","workflow_elapsed_seconds","frame_anchor_status","multigap_status","extra_verification","compute_backend","fast_rescue_v13","import_geometry_mode","import_geometry","scan")},
            "metrics":m,"comparison":json.loads((result/"comparison.json").read_text()) if (result/"comparison.json").exists() else None,"benchmark":rows if batch else None,"files":sorted(p.name for p in result.iterdir() if p.name in DOWNLOADS)}
