"""Blind scanned strip reconstruction with foreground-mask row rectification.

No filename order, OCR, original image or supplied order.txt is used.
The rectified pieces are renderable raster strips, not rigid SE(2) poses.
"""
from __future__ import annotations

import time
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from scipy.ndimage import map_coordinates, gaussian_filter1d
from scipy.spatial.distance import cdist

from .common import Fragment, Pose, render, save_rgba, json_write
from .document import chain, _projection_cost
from .rotation import _signed_tree
from .overlay import write_cut_outputs


class NotDocumentStrips(ValueError):
    """Elongated image shape alone is insufficient to identify a single strip."""


def _rectify(path):
    with Image.open(path) as im:
        a=np.array(ImageOps.exif_transpose(im).convert("RGBA"))
    original_size=[a.shape[1],a.shape[0]]
    if not np.any(a[:,:,3]<128):
        from .input_detection import solid_background,scan_evidence
        from .scan import chroma_mask
        keyed=chroma_mask(a)
        if keyed is None:
            solid=solid_background(a,exact=True)
            keyed=solid[0] if solid is not None else None
        if keyed is not None:
            a[:,:,3]=keyed.astype(np.uint8)*255
            y,x=np.nonzero(keyed);a=a[y.min():y.max()+1,x.min():x.max()+1].copy()
        elif scan_evidence(Image.fromarray(a))=="matte-background":
            raise NotDocumentStrips("Unmasked scan with variable illumination requires foreground segmentation")
    turn=int(a.shape[0]<a.shape[1])
    if turn:a=np.rot90(a,-1)
    mask=a[:,:,3]>=128
    valid=mask.any(axis=1)
    if valid.sum()<.96*len(valid):
        raise NotDocumentStrips("Strip silhouette must span nearly all scan rows")
    y=np.arange(len(mask));left=mask.argmax(1);right=mask.shape[1]-1-mask[:,::-1].argmax(1)
    left=np.interp(y,y[valid],left[valid]);right=np.interp(y,y[valid],right[valid])
    width=right-left+1
    W=int(round(np.median(width)))
    if W<5 or np.percentile(abs(width-W),95)>max(3,.22*W):
        raise NotDocumentStrips("Scan silhouette is not a near-uniform strip")
    if mask.sum()/max(1,width.sum())<.96:
        raise NotDocumentStrips("Strip mask has substantial interior holes or multiple pieces")
    h=len(mask)
    yy=np.broadcast_to(y[:,None],(h,W))
    xx=left[:,None]+(right-left)[:,None]*np.linspace(0,1,W)
    rgb=np.stack([map_coordinates(a[:,:,k],(yy,xx),order=1,prefilter=False) for k in range(3)],axis=2)
    # Preserve invalid rows as transparent rather than fabricating pixels.
    observed=map_coordinates(mask.astype(np.uint8),(yy,xx),order=0,prefilter=False)>0
    rgba=np.dstack([rgb,observed.astype(np.uint8)*255]);rgba[~observed]=0
    return Fragment(path.stem,rgba),{"file":path.name,"id":path.stem,
        "original_wh":original_size,"rectified_wh":[W,h],"initial_clockwise_turns":turn,
        "masked":bool(np.any(~mask)),"median_width":W,"row_width_range":[int(width.min()),int(width.max())]}


def _ink(p):
    a=p.rgba
    v=1-(.299*a[:,:,0].astype(np.float32)+.587*a[:,:,1]+.114*a[:,:,2])/255
    v[a[:,:,3]==0]=0
    return v


def _profile(im,right,samples):
    if right:im=im[:,::-1]
    W=im.shape[1];depth=min(6,W);inset=min(1,depth-1)
    center=im[:,max(0,W//4):max(1,3*W//4)]
    blank=center.mean(1)<.03
    # In rows where the page interior is blank, estimate scanner edge shadow
    # independently for every sampled edge column.
    baseline=np.percentile(im[blank,:depth],65,axis=0) if blank.sum()>20 else np.percentile(im[:,:depth],40,axis=0)
    a=np.maximum(0,(im[:,:depth]-baseline)/(1-baseline+.001))
    curves=[gaussian_filter1d(a[:,inset:min(depth,inset+2)].mean(1),.9),
            gaussian_filter1d(a[:,inset:].mean(1),1.5),
            gaussian_filter1d(a[:,:min(2,depth)].mean(1),.9)]
    if len(im)!=samples:
        curves=[np.interp(np.linspace(0,len(im)-1,samples),np.arange(len(im)),q) for q in curves]
    return curves


def _cost(A,B):
    vals=[]
    for a,b in zip(A,B):
        # A small common floor handles blank margins without cosine NaNs.
        vals.append(.7*cdist(a+.0001,b+.0001,"cosine")+.3*cdist(a,b,"cityblock")/a.shape[1])
    return np.average(vals,axis=0,weights=[.45,.35,.2])


def reconstruct_strips(paths,target:Path,out:Path,update=lambda _:None,*,pixel_cap=8_000_000):
    if not 2<=len(paths)<=128:raise ValueError("Physical document mode supports 2–128 scanned strips")
    started=time.perf_counter();pieces=[];records=[]
    for path in paths:
        p,r=_rectify(path);pieces.append(p);records.append(r)
    if len({p.id for p in pieces})!=len(pieces):raise ValueError("Strip filenames must have unique stems")
    h=max(p.rgba.shape[0] for p in pieces);w=sum(p.rgba.shape[1] for p in pieces)
    if h*w>pixel_cap:raise ValueError("Rectified document exceeds canvas pixel budget")
    target.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    for p in pieces:save_rgba(target/(p.id+".png"),p.rgba)
    images=[_ink(p) for p in pieces]
    # Common text-line projection determines relative 180° orientation.
    # Global upside-down ambiguity remains without a reference or OCR.
    inner=[im[:,im.shape[1]//6:max(im.shape[1]//6+1,5*im.shape[1]//6)] for im in images]
    same,flipped=_projection_cost(inner)
    signs=_signed_tree(same,flipped,neighbors=12)
    ims=[np.rot90(im,2) if signs[i] else im for i,im in enumerate(images)]
    samples=min(h,1800)
    L,R=[np.array([_profile(im,right,samples) for im in ims]).transpose(1,0,2) for right in (False,True)]
    update("Matching ink continuity along scanned strip edges")
    costs=_cost(R,L)
    # Small vertical scan misalignment is tolerated in matching only; the
    # rendered rows are not translated or synthesized based on a noisy seam.
    for d in range(1,5):
        for a,b in ((R[:,:,d:],L[:,:,:-d]),(R[:,:,:-d],L[:,:,d:])):
            costs=np.minimum(costs,_cost(a,b)+d*.012)
    np.fill_diagonal(costs,np.inf)
    l=L[1].mean(1);r=R[1].mean(1);s=max(.02,float(np.median(np.r_[l,r])))
    order=chain(costs,.5,borders=(3*l/s,3*r/s))
    poses={};x=0
    for i in order:
        p=pieces[i];poses[p.id]=Pose(x,0,2*int(signs[i]));x+=p.rgba.shape[1]
    update("Rendering rectified scanned document")
    rgba,labels,collision=render(pieces,poses,(h,w))
    save_rgba(out/"assembly.png",rgba);write_cut_outputs(out,rgba,labels)
    warnings=["Physical strip order is experimental. Full placement does not establish correct adjacency.",
              "Global orientation is ambiguous without OCR/reference; minor scan alignment and edge damage may remain."]
    if any(not r["masked"] for r in records):warnings.append("Some inputs had no foreground mask; their entire raster was treated as paper.")
    diag={"algorithm":"physical-document-foreground-multiscale","status":"unverified",
          "placed":len(poses),"piece_count":len(pieces),"coverage":float((labels>=0).mean()),
          "render_collision_pixels":collision,"runtime_seconds":time.perf_counter()-started,
          "accuracy_verified":False,"orientation_ambiguous":True,"source_reference_used":False,
          "canvas_height":h,"canvas_width":w,
          "scan":{"kind":"document-strips","input_images":len(paths),"detected_pieces":len(pieces),
                  "components":1,"warnings":warnings,"common_scale":1.0,"accuracy_verified":False}}
    json_write(out/"placements.json",{"mode":"blind-rectified-strips","poses":{k:v.as_dict() for k,v in poses.items()},
              "diagnostics":diag})
    json_write(out/"scan_report.json",{"input_images":len(paths),"detected_pieces":len(pieces),
        "mode":"document-strips","rectification":"per-row linear sampling between foreground mask edges",
        "pieces":records,"orientation_ambiguous":True,"accuracy_verified":False,"warnings":warnings})
    tile=140;cols=min(8,len(pieces));rows=math.ceil(len(pieces)/cols)
    preview=Image.new("RGB",(cols*tile,rows*tile),(225,231,228));draw=ImageDraw.Draw(preview)
    for i,p in enumerate(pieces):
        im=Image.fromarray(p.rgba);im.thumbnail((tile-12,tile-24))
        dx=(i%cols)*tile+(tile-im.width)//2;dy=(i//cols)*tile+(tile-20-im.height)//2
        preview.paste(im,(dx,dy),im);draw.text(((i%cols)*tile+6,(i//cols+1)*tile-17),p.id,fill=(40,55,45))
    preview.save(out/"fragments_preview.png")
    json_write(out/"diagnostics.json",diag)
    return diag
