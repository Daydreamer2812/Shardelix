"""Shared, conservative geometry and foreground inference for external rasters.

No filename order, reference image or ground-truth metadata enters detection.
Dimension histograms constrain complete rectangular grids; otherwise use the
observed foreground or an unknown-canvas solver without inventing geometry.
"""
from __future__ import annotations

from collections import Counter
import math
import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage as ndi


def _coherent_foreground(mask):
    # Distinguish a few substantial silhouettes from disconnected printed
    # characters/texture on a white full-page digital image.
    labels,n=ndi.label(mask)
    if not n:return False
    sizes=np.bincount(labels.ravel())[1:]
    large=sizes[sizes>=max(20,.012*mask.size)]
    return bool(len(large) and large.sum()>=.70*sizes.sum() and large.max()>=.12*sizes.sum())


def solid_background(rgb, *, exact=False):
    """Reliable near-solid outer rim: return foreground and the border color.

    Unlike a specific chroma key, remove only background connected to the rim;
    a same-colored region inside the piece is not automatically deleted.
    """
    a=np.asarray(rgb)[...,:3]
    if min(a.shape[:2])<8:return None
    rim=np.concatenate((a[0],a[-1],a[:,0],a[:,-1]))
    bg=np.median(rim,axis=0)
    delta=np.max(abs(rim.astype(np.float32)-bg),axis=1)
    if exact and (delta<=1).mean()<.98:return None
    near=delta<=5
    if near.mean()<.96:return None
    # Preserve full white/solid-color tiles and avoid interpreting a handful
    # of isolated speckles as a physical fragment.
    same=np.max(abs(a.astype(np.float32)-bg),axis=2)<=5
    seed=np.zeros_like(same);seed[[0,-1]]=same[[0,-1]];seed[:,[0,-1]]=same[:,[0,-1]]
    foreground=~ndi.binary_propagation(seed,mask=same)
    if not .025<=foreground.mean()<=.90:return None
    if not _coherent_foreground(foreground):return None
    return foreground,bg


def scan_evidence(im):
    """A bounded preview checks for a matte rim and substantial interior.

    Exact alpha/key decisions are made on the real raster; this merely routes
    ambiguous opaque images. An ordinary textured border is not a matte.
    """
    if ("A" in im.getbands() or "transparency" in im.info):
        if im.convert("RGBA").getchannel("A").getextrema()[0]<128:return "alpha"
    # Preserve explicit saturated magenta sentinels, including clipped pieces.
    rgb=im.convert("RGB")
    w,h=rgb.size
    if min(w,h)<8:return None
    corners=np.array([rgb.getpixel((x,y)) for x,y in ((0,0),(w-1,0),(0,h-1),(w-1,h-1))])
    if ((corners[:,0]>=250)&(corners[:,1]<=5)&(corners[:,2]>=250)).sum()>=3:
        from .scan import chroma_mask
        if chroma_mask(np.array(rgb)) is not None:return "chroma-key"
    # Use original sampling for small inputs and a bounded thumbnail for larger ones.
    if max(w,h)>96:
        rgb.thumbnail((96,96),Image.Resampling.BILINEAR)
    a=np.asarray(rgb)
    if solid_background(a) is not None:return "solid-background"
    # Test a simple illumination plane on border samples, then ask whether
    # the predicted background is consistent and the contrast is internal.
    h,w=a.shape[:2]
    if min(h,w)<8:return None
    y,x=np.indices((h,w));rim=(y==0)|(y==h-1)|(x==0)|(x==w-1)
    X=np.stack([np.ones_like(x),x/max(1,w-1),y/max(1,h-1)],axis=-1).astype(float)
    C=a.astype(float)
    coeff=np.linalg.lstsq(X[rim],C[rim],rcond=None)[0]
    if abs(coeff[1:]).max()>100:return None
    predicted=X@coeff
    d=np.linalg.norm(C-predicted,axis=2)/math.sqrt(3)
    if np.percentile(d[rim],95)>12:return None
    mask=d>max(16.,4*float(np.median(d[rim])))
    if not .025<=mask.mean()<=.85 or mask[rim].mean()>.08:return None
    if _coherent_foreground(mask):return "matte-background"
    return None


def _factor_pairs(n):
    for a in range(1,math.isqrt(n)+1):
        if n%a==0:
            yield a,n//a
            if a!=n//a:yield n//a,a


def _expected_shapes(h,w,r,c):
    nh,eh=divmod(h,r);nw,ew=divmod(w,c)
    result=Counter()
    for a,ca in ((nh,r-eh),(nh+1,eh)):
        for b,cb in ((nw,c-ew),(nw+1,ew)):
            if ca and cb:result[tuple(sorted((a,b)))]+=ca*cb
    return result


def infer_grid(dims,canvas_cap):
    """Test exact cell-size multiplicities, permitting quarter-turns and ±1px
    cells from an integer raster partition. Never resize input to force a grid."""
    n=len(dims);total=sum(w*h for w,h in dims)
    if not n or total>canvas_cap:return None
    hist=Counter(tuple(sorted(x)) for x in dims)
    layouts=[]
    if len(hist)==1:
        a,b=next(iter(hist))
        # Preserve the conventional orientation for uniform unrotated tiles.
        same=len(set(dims))==1
        choices=[dims[0]] if same else [(a,b),(b,a)] if a!=b else [(a,b)]
        for w,h in choices:
            for r,c in _factor_pairs(n):
                layouts.append({"rows":r,"cols":c,"height":r*h,"width":c*w})
        basis="equal-size opaque tiles" if same else "quarter-turned equal-size rectangular tiles"
        tw,th=dims[0]
    else:
        # Nonuniform regular cells are supported by the dense solver up to 1024.
        if n>1024 or len(hist)>4:return None
        grids=[(r,c) for r,c in _factor_pairs(n) if min(r,c)>1]
        sizes=set(v for s in hist for v in s)
        # The complete raster's area is the sum of its cell areas.
        for h,w in _factor_pairs(total):
            for r,c in grids:
                if h<r or w<c or h//r not in sizes or w//c not in sizes:continue
                if _expected_shapes(h,w,r,c)==hist:
                    layouts.append({"rows":r,"cols":c,"height":h,"width":w})
        if not layouts:return None
        basis="exact compatible cell-size histogram (nonuniform, quarter-turns allowed)"
        tw,th=map(int,np.median(np.asarray(dims),axis=0))
    layouts.sort(key=lambda l:(abs(math.log((l["width"]/l["height"])/(4/3))),l["rows"],l["cols"],l["height"],l["width"]))
    return {"tile_width":tw,"tile_height":th,"expected_count":n,
            "basis":basis+"; assumes no missing fragments","layouts":layouts}


def _alpha_strip_moment(im):
    """Orientation-invariant geometry of a nearly rectangular alpha mask."""
    alpha=im.convert("RGBA").getchannel("A")
    if alpha.getextrema()[0]>=128:return None
    a=np.asarray(alpha).astype(np.float64)/255
    y,x=np.nonzero(a>.005);v=a[y,x];mass=v.sum()
    if mass<12:return None
    x=x-np.average(x,weights=v);y=y-np.average(y,weights=v)
    cov=np.array([[np.sum(v*x*x),np.sum(v*x*y)],[np.sum(v*x*y),np.sum(v*y*y)]])/mass
    short,long=np.sqrt(np.maximum(1,12*np.linalg.eigvalsh(cov)+1))
    if long<max(12,3*short) or abs(mass/long-short)>max(1.5,.08*short):return None
    return float(long),float(short)


def inspect_rasters(openers,count,pixel_limit,canvas_cap,*,mask_count=0,explicit_scans=False):
    """Inventory all image headers and route a representative set of pixels."""
    dims=[];evidence=Counter();total=0;alpha_lengths=[]
    # For large opaque sets, don't decode every image just to classify the rim.
    sample=set(np.linspace(0,max(0,count-1),min(count,32),dtype=int).tolist())
    for i,opener in enumerate(openers):
        with opener() as f,Image.open(f) as original:
            im=ImageOps.exif_transpose(original) if original.getexif().get(274,1)!=1 else original
            w,h=im.size;dims.append((w,h));total+=w*h
            if total>pixel_limit:raise ValueError("Decoded input exceeds pixel budget")
            if i in sample:
                signal=scan_evidence(im)
                if signal:evidence[signal]+=1
            elif ("A" in im.getbands() or "transparency" in im.info) and im.convert("RGBA").getchannel("A").getextrema()[0]<128:
                evidence["alpha"]+=1
            if ("A" in im.getbands() or "transparency" in im.info) and count<=1024 and not mask_count:
                geom=_alpha_strip_moment(im)
                if geom:alpha_lengths.append(geom)
    has_mask=bool(mask_count or evidence["alpha"] or evidence["chroma-key"])
    background=(evidence["solid-background"]+evidence["matte-background"]>=max(1,math.ceil(.6*len(sample))))
    strip=None
    if 2<=count<=1024 and all(max(w,h)>=4*min(w,h) for w,h in dims):
        long=[max(s) for s in dims];short=[min(s) for s in dims];h=int(np.median(long))
        if max(abs(v-h) for v in long)<=max(2,.015*h):
            strip={"height":h,"width_range":[min(short),max(short)]}
    continuous=None
    if count>=2 and len(alpha_lengths)==count and not explicit_scans:
        long=[p[0] for p in alpha_lengths];h=int(round(np.median(long)))
        if max(abs(v-h) for v in long)<=max(2,.012*h):
            continuous={"height":h,"width_range":[int(round(min(p[1] for p in alpha_lengths))),int(round(max(p[1] for p in alpha_lengths)))]}
    if continuous and not strip:
        route="continuous-strips";strip=continuous;grid=None
    elif has_mask or background or explicit_scans:
        route="scan-strips" if strip and count<=128 else "irregular"
        grid=None
    elif strip and len(set(max(s) for s in dims))==1 and total<=canvas_cap:
        # Unlike a scan, an opaque digital strip needs no background removal.
        route="opaque-strips";grid=None
    else:
        grid=infer_grid(dims,canvas_cap)
        route="grid" if grid else "irregular"
    return {"suggested_grid":grid,"suggested_document":strip if route in ("scan-strips","opaque-strips","continuous-strips") else None,
            "route":route,"input_evidence":{"sampled_images":len(sample),"alpha_or_key":int(evidence["alpha"]+evidence["chroma-key"]),
            "solid_or_matte_background":int(evidence["solid-background"]+evidence["matte-background"]),
            "foreground_masks":mask_count,"decoded_pixels":total},"foreground_mask_count":mask_count}


def select_visual_grid(layouts,pieces,profile=None):
    """Small bounded blind layout search. A weak/tied result retains the prior.

    Different aspect hypotheses are observationally ambiguous from dimensions
    alone. Compare plausible 2-D alternatives using the same fast reconstruction
    and normalized seam/coverage objective, not a source image.
    """
    from shardelix.api import Solver
    from .solver import objective
    prior=layouts[0]
    n=len(pieces)
    if not 4<=n<=64 or prior["width"]*prior["height"]>500_000 or profile=="fast":
        return prior,[]
    candidates=[];seen=set()
    for layout in layouts:
        # A single global quarter-turn is an equivalent hypothesis.
        key=(tuple(sorted((layout["rows"],layout["cols"]))),tuple(sorted((layout["height"],layout["width"]))))
        if key in seen:continue
        if min(layout["rows"],layout["cols"])==1 and candidates:continue
        seen.add(key);candidates.append(layout)
        if len(candidates)>=4:break
    if len(candidates)<2:return prior,[]
    trials=[];best=prior;prior_score=None;best_score=-np.inf
    for layout in candidates:
        shape=(layout["height"],layout["width"]);grid=(layout["rows"],layout["cols"])
        try:
            result=Solver(profile="fast").solve(pieces,shape=shape,cut="grid",grid=grid,rotations=True)
            score=objective(pieces,result.poses,shape)
            trials.append({"layout":layout,"blind_objective":score,"placed":len(result.poses)})
            if layout==prior:prior_score=score
            if len(result.poses)==n and score>best_score:best_score=score;best=layout
        except (ValueError,RuntimeError) as exc:
            trials.append({"layout":layout,"error":str(exc)[:100]})
    # Objective = coverage - collision penalty - 0.1 * normalized seam.
    # Require a meaningful improvement, rather than arbitrarily choosing from
    # textureless or nearly equal layouts.
    if prior_score is None or best_score>prior_score+.003:return best,trials
    return prior,trials
