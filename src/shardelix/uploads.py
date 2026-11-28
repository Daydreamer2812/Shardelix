"""Upload validation, safe archive extraction and observable input inspection."""
from __future__ import annotations
import json,shutil,zipfile
from pathlib import Path
from shardelix.reconstruction.limits import GRID_MAX,MAX_PIXELS,RECTANGULAR,max_count,validate_constraints

IMAGE_EXT={".png",".jpg",".jpeg",".tif",".tiff",".webp",".bmp"}
MAX_UPLOAD=200*1024**2
WEB_UPLOAD=32*1024**2
WEB_COUNT=4096
WEB_PIXELS=8_000_000
WEB_INPUT_PIXELS=24_000_000

def native_upload_limit(path=None, *, multipart=False):
    """Disk-headroom limit; additionally reserve RAM for Studio's MIME parser."""
    import tempfile
    from shardelix.reconstruction.resource_budget import available_memory
    try:limit=shutil.disk_usage(path or tempfile.gettempdir()).free//8
    except OSError:limit=MAX_UPLOAD
    if multipart:
        available=available_memory()
        limit=min(limit,available//12 if available is not None else MAX_UPLOAD)
    return max(0,int(limit))


# Auxiliary views and truth are never counted as fragments. Unknown nested
# layouts are accepted only when one primary image directory is unambiguous.
_MASK_DIRS={"bg_mask","bg_mask_vis","masks","mask","mask_vis","foreground_masks","alpha_masks","segmentation_masks"}
_PRIMARY_DIRS={"pieces","fragments","tiles","patches","scans","frag_eroded"}
_AUX_DIRS={"evaluation","result","results","ground_truth","ground-truth","groundtruth","truth","__macosx",
           "preview","previews","thumbnails","reference","references","original","originals",
           "labels","gt","annotations","output","outputs"}|_MASK_DIRS
_AUX_FILES={"source.png","reference.png","ground_truth.png","assembly.png","assembly_with_cuts.png",
            "fragments_preview.png","cut_lines.png"}

def _strip_hint(openers,count,browser):
    from PIL import Image
    if not 2<=count<=128:return None
    dims=[];total=0
    for opener in openers:
        with opener() as f,Image.open(f) as im:
            w,h=im.size;total+=w*h
            if total>(WEB_INPUT_PIXELS if browser else MAX_PIXELS*3):
                raise ValueError("Decoded input exceeds pixel budget")
            if "A" in im.getbands() or "transparency" in im.info:
                a=im.convert("RGBA").getchannel("A").point(lambda v:255 if v>=128 else 0)
                box=a.getbbox()
                if box is None:return None
                w,h=box[2]-box[0],box[3]-box[1]
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
    meta=None;meta_seen=False;metadata_warning=None;names=set();groups={};mask_entries={};size=0
    limit=WEB_COUNT if browser else GRID_MAX
    byte_limit=WEB_UPLOAD*3 if browser else native_upload_limit(source.parent)*2
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
                if suffix==".png" and low and low[-1] in _MASK_DIRS and not any(x in _AUX_DIRS for x in low[:-1]):
                    key=(dirs[:-1],Path(leaf).stem)
                    if key in mask_entries:raise ValueError("Duplicate foreground mask")
                    mask_entries[key]=e
                continue
            if leaf.lower()=="case.json":
                if meta_seen or e.file_size>64*1024:raise ValueError("Ambiguous or oversized metadata")
                meta_seen=True
                try:
                    meta=json.loads(z.read(e))
                    if not isinstance(meta,dict):raise ValueError("expected a JSON object")
                except (ValueError,UnicodeError) as exc:
                    metadata_warning="Ignored incompatible case.json: "+str(exc)[:160]
                    meta=None
                continue
            explicit_dir=bool(_PRIMARY_DIRS.intersection(low))
            if suffix not in IMAGE_EXT or (leaf.lower() in _AUX_FILES and not explicit_dir):continue
            groups.setdefault(dirs,[]).append(e)
        explicit=[d for d in groups if _PRIMARY_DIRS.intersection(x.lower() for x in d)]
        dafne=[d for d in explicit if d and d[-1].lower()=="frag_eroded"]
        if dafne:
            if len(dafne)!=1:raise ValueError("DAFNE archive contains multiple cases; upload one frag_eroded directory or use the batch benchmark")
            if len(explicit)!=1:raise ValueError("Ambiguous DAFNE and other primary fragment directories")
            chosen=dafne
        elif explicit:chosen=explicit
        elif () in groups:chosen=[()]
        else:
            # Standard folder-wrapped ZIPs produced by OS archive tools.
            # Do not guess when multiple unrelated raster directories exist.
            chosen=list(groups)
            if len(chosen)>1:raise ValueError("Multiple image directories in ZIP; put primary fragments in pieces/ or scans/")
        images=[(d,e) for d in chosen for e in groups[d]]
        piece_rasters=bool(chosen) and all(any(x.lower() in {"pieces","fragments","tiles","patches","frag_eroded"} for x in d) and not any(x.lower()=="scans" for x in d) for d in chosen)
        num=len(images)
        if not num:raise ValueError("No supported fragments found in ZIP")
        if num>limit:raise ValueError("Archive exceeds fragment count budget")
        def find_mask(d,stem):
            folders=[d]
            if d and d[-1].lower() in _PRIMARY_DIRS:folders.append(d[:-1])
            matches=[mask_entries[(folder,name)] for folder in folders for name in (stem,stem+"_mask",stem+"-mask","mask_"+stem)
                     if (folder,name) in mask_entries]
            if len(matches)>1:raise ValueError("Ambiguous foreground masks for "+stem)
            return matches[0] if matches else None
        resolved_masks={(d,e.filename):find_mask(d,Path(e.filename.replace("\\","/").split("/")[-1]).stem) for d,e in images}
        mask_count=sum(m is not None for m in resolved_masks.values())
        # Only the canonical public schema is authoritative. Foreign/stale
        # metadata is not an instruction to a solver; re-infer from pixels.
        if meta is not None:
            try:
                required={"height","width","piece_count","cut","rotations"}
                if not required.issubset(meta):raise ValueError("not the Shardelix public geometry schema")
                observed=meta["piece_count"];expected=meta.get("expected_count",observed)
                if type(observed) is not int or type(expected) is not int or observed!=num or expected<observed:
                    raise ValueError("declared count differs from actual fragments")
                validate_constraints(expected,(meta["height"],meta["width"]),meta["cut"],meta.get("grid"),meta["rotations"])
                angle=meta.get("angle_mode","quarter" if meta["rotations"] else "none")
                if angle not in ("none","quarter","continuous") or (angle!="none")!=meta["rotations"]:raise ValueError("invalid angle mode")
                sampling=meta.get("sampling",1)
                if type(sampling) is not int or sampling not in (1,2,3,4) or (sampling!=1 and angle!="continuous"):raise ValueError("invalid sampling")
                if meta["cut"] in RECTANGULAR and angle!="continuous":
                    r,c=meta["grid"];h,w=meta["height"],meta["width"]
                    rh,remh=divmod(h,r);cw,remw=divmod(w,c)
                    allowed={(a,b) for a in (rh,rh+1)[:1+bool(remh)]
                                   for b in (cw,cw+1)[:1+bool(remw)]}
                    if angle=="quarter":allowed|={(b,a) for a,b in allowed}
                    # Header-only sample prevents a same-count stale grid from
                    # blocking the automatic workflow, without repeatedly
                    # decoding large archives.
                    for i in sorted({round(k*(num-1)/max(1,min(num,32)-1)) for k in range(min(num,32))}):
                        with z.open(images[i][1]) as f,Image.open(f) as im:
                            w0,h0=im.size
                            if im.getexif().get(274,1) in (5,6,7,8):w0,h0=h0,w0
                            if (h0,w0) not in allowed:raise ValueError("fragment dimensions differ from declared grid")
            except (ValueError,TypeError,KeyError) as exc:
                metadata_warning="Ignored incompatible case.json: "+str(exc)[:160]
                meta=None
        decoded=0
        for d,e in images:
            leaf=e.filename.replace("\\","/").split("/")[-1]
            if leaf in names:raise ValueError("Duplicate fragment filename")
            names.add(leaf)
            mask=resolved_masks[(d,e.filename)]
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
                        if not np.all((m==0)|(m==1)|(m==255)) or not np.any(m):raise ValueError("Foreground mask must be nonempty and binary (0/1 or 0/255)")
                        from PIL import ImageOps
                        rgba=im.convert("RGBA");rgba.putalpha(Image.fromarray((m>0).astype(np.uint8)*255));rgba=ImageOps.exif_transpose(rgba);dest=target/(Path(leaf).stem+".png")
                        if dest.exists():raise ValueError("Duplicate fragment filename stem")
                        rgba.save(dest)
                else:
                    with z.open(e) as f,(target/leaf).open("wb") as out:shutil.copyfileobj(f,out,1024*1024)
        if inspect_geometry:
            openers=[lambda e=e:z.open(e) for _,e in images]
            explicit_scans=any({"scans","frag_eroded"}.intersection(x.lower() for x in d) for d,_ in images)
            if meta is None:
                extra=_inspect_rasters(openers,num,browser,mask_count=mask_count,explicit_scans=explicit_scans)
            else:
                extra={"suggested_grid":None,"suggested_document":None,"foreground_mask_count":mask_count,
                       "route":"metadata","input_evidence":{}}
            extra["piece_rasters"]=piece_rasters
            extra["metadata_warning"]=metadata_warning
            extra["partial_masks"]=bool(0<mask_count<num)
            suggestion=extra["suggested_grid"]
            return (meta,num,suggestion,extra) if details else (meta,num,suggestion)
    if details:return meta,num,None,{"piece_rasters":piece_rasters}
    return meta,num

def public_metadata(meta):
    if meta is None:return None
    fields=("height","width","piece_count","cut","rotations","grid","expected_count","angle_mode","sampling")
    return {k:meta[k] for k in fields if k in meta}

def _pixel_budget(files,limit):
    from PIL import Image
    total=0
    for f in files:
        with Image.open(f) as im:total+=im.width*im.height
        if total>limit:raise ValueError("Decoded input exceeds pixel budget")
    return total

def _check_preview(files,browser):
    if not files:raise ValueError("Select input files first")
    if sum(p.stat().st_size for p in files)>(WEB_UPLOAD if browser else native_upload_limit(files[0].parent)):
        raise ValueError("Upload exceeds browser byte budget" if browser else "Upload exceeds available disk byte budget")
    if len(files)>(WEB_COUNT if browser else GRID_MAX):raise ValueError("Too many fragments")

def _inspect_rasters(openers,count,browser,**kwargs):
    from shardelix.reconstruction.input_detection import inspect_rasters
    return inspect_rasters(openers,count,WEB_INPUT_PIXELS if browser else MAX_PIXELS*3,
                           WEB_PIXELS if browser else MAX_PIXELS,**kwargs)

def _suggest_grid(openers,count,browser):
    return _inspect_rasters(openers,count,browser)["suggested_grid"]

def inspect_uploads(files,browser=False):
    _check_preview(files,browser)
    if len(files)==1 and files[0].suffix.lower()==".zip":
        meta,n,suggestion,extra=safe_zip(files[0],files[0].parent,browser=browser,extract=False,inspect_geometry=True,scan=True,details=True)
        return {"metadata":public_metadata(meta),"observed_count":n,"suggested_grid":suggestion,**extra}
    if any(p.suffix.lower() not in IMAGE_EXT for p in files):
        raise ValueError("Select supported fragment images or one ZIP")
    return {"metadata":None,"observed_count":len(files),
            **_inspect_rasters([lambda p=p:p.open("rb") for p in files],len(files),browser)}
