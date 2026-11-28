"""Blind scanned strip reconstruction using pixel continuity and patch recurrence.

No filename order, OCR, original image or supplied order.txt is used.
The rectified pieces are renderable raster strips, not rigid SE(2) poses.
"""
from __future__ import annotations

import time
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from scipy.ndimage import map_coordinates, gaussian_filter1d, gaussian_filter
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from shardelix.reconstruction.common import Fragment, Pose, render, save_rgba, json_write
from shardelix.reconstruction.strips.ordering import chain, _projection_cost
from shardelix.reconstruction.rotation import _signed_tree, solve_unknown_color_strips
from shardelix.reconstruction.overlay import write_cut_outputs


class NotDocumentStrips(ValueError):
    """Elongated image shape alone is insufficient to identify a single strip."""


def _rectify(path):
    with Image.open(path) as im:
        a=np.array(ImageOps.exif_transpose(im).convert("RGBA"))
    original_size=[a.shape[1],a.shape[0]]
    if not np.any(a[:,:,3]<128):
        from shardelix.reconstruction.input_detection import solid_background,scan_evidence
        from shardelix.reconstruction.irregular.scan import chroma_mask
        keyed=chroma_mask(a)
        if keyed is None:
            solid=solid_background(a,exact=True)
            keyed=solid[0] if solid is not None else None
        if keyed is not None:
            a[:,:,3]=keyed.astype(np.uint8)*255
        elif scan_evidence(Image.fromarray(a))=="matte-background":
            raise NotDocumentStrips("Unmasked scan with variable illumination requires foreground segmentation")
    had_mask=bool(np.any(a[:,:,3]<128))
    # Trim empty scan margins before deciding the strip axis. A padded
    # photograph with a valid mask should not be rejected as a short strip.
    yy,xx=np.nonzero(a[:,:,3]>=128)
    if not len(xx):raise NotDocumentStrips("Empty foreground mask")
    x0,y0=int(xx.min()),int(yy.min())
    a=a[y0:int(yy.max())+1,x0:int(xx.max())+1].copy()
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
    if len(mask)<4*W or W<5 or np.percentile(abs(width-W),95)>max(3,.22*W):
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
        "masked":had_mask,"crop_xy":[x0,y0],"median_width":W,"row_width_range":[int(width.min()),int(width.max())]}


def _ink(p):
    a=p.rgba
    v=1-(.299*a[:,:,0].astype(np.float32)+.587*a[:,:,1]+.114*a[:,:,2])/255
    v[a[:,:,3]==0]=0
    return v


def _unit_cost(c):
    """Scale positive score matrices without creating negative edge costs."""
    v=c[~np.eye(len(c),dtype=bool)]
    return c/max(1.e-6,float(np.median(v)))


def _clean_ink(im):
    # Estimate paper/shadow intensity separately in every scan column.
    # No pixels are changed in the exported image.
    base=np.percentile(im,20,axis=0)
    return np.clip((im-base)/.65,0,1).astype(np.float32)


def _textlike(pieces,images):
    a=np.concatenate([p.rgba[p.mask][::max(1,int(p.mask.sum())//800),:3] for p in pieces])
    dark=np.concatenate([im.ravel()[::max(1,im.size//800)] for im in images])
    return bool(len(a) and (np.ptp(a.astype(np.int16),axis=1)<35).mean()>.85
                and np.median(dark)<.30 and .012<(dark>.40).mean()<.40)


def _line_cost(left,right):
    """Remove common row layout; compare local ink rather than blank paper."""
    def compare(l,r):
        l=l-l.mean(0);r=r-r.mean(0)
        cosine=cdist(r+1.e-5,l+1.e-5,"cosine")
        absolute=cdist(r,l,"cityblock")/(np.abs(r).sum(1)[:,None]+np.abs(l).sum(1)[None,:]+1)
        return cosine,absolute
    a,b=compare(left,right)
    for d in (1,2):
        for l,r in ((left[:,d:],right[:,:-d]),(left[:,:-d],right[:,d:])):
            u,v=compare(l,r)
            a=np.minimum(a,u+.007*d);b=np.minimum(b,v+.007*d)
    return .5*_unit_cost(a)+.5*_unit_cost(b)


def _patch_recurrence(images,half,gap):
    """Classical nonparametric matching of half patches from observed pixels.

    Use a bounded index of interior patches of the same input page. For every
    seam-side patch, find geometrically similar half patches in intact local
    neighborhoods and compare their observed continuations with the opposing
    side. No glyph labels, OCR, training, language model, or generated pixels.
    """
    n=len(images);h=len(images[0]);step=2
    yoffset=np.arange(-6,7,3)
    offset=np.arange(half-step//2,0,-step)-.5+gap
    xoffset=np.r_[-offset,offset[::-1]]
    # Bound both query count and corpus size; same coordinates for every strip.
    stride=max(3,math.ceil(n*h/35_000))
    ycenter=np.arange(6,h-6,stride)
    if not len(ycenter) or any(im.shape[1]<half+3 for im in images):return None
    corpus=[];edges=[]
    stride_x=max(3,math.ceil(len(ycenter)*sum(max(0,im.shape[1]-2*(half+gap+3)) for im in images)/300_000))
    for im in images:
        sm=gaussian_filter(im,.6)
        xs=np.arange(half+gap+3,im.shape[1]-half-gap-3,stride_x)
        if len(xs):
            Y,X=np.meshgrid(ycenter,xs,indexing="ij")
            v=map_coordinates(sm,[Y[...,None,None]+yoffset[:,None]+np.zeros((1,len(xoffset))),
                                  X[...,None,None]+xoffset[None,:]+np.zeros((len(yoffset),1))],
                              order=1,mode="nearest").reshape(-1,len(yoffset),len(xoffset))
            corpus.append(v[(v>.18).sum((1,2))>=2])
        xL=im.shape[1]-3-np.arange(len(offset))[::-1]*step
        xR=2+np.arange(len(offset))*step
        Y=ycenter[:,None,None]+yoffset[None,:,None]+np.zeros((1,1,len(offset)))
        edges.append([map_coordinates(sm,[Y,x[None,None,:]+np.zeros_like(Y)],order=1,mode="nearest")
                      .reshape(len(ycenter),-1) for x in (xL,xR)])
    if not corpus:return None
    c=np.concatenate(corpus)
    if len(c)<32:return None
    if len(c)>50_000:c=c[np.linspace(0,len(c)-1,50_000,dtype=int)]
    a,b=np.split(c,2,axis=2);a=a.reshape(len(c),-1);b=b.reshape(len(c),-1)
    e=np.asarray(edges);L,R=e[:,1],e[:,0]
    def predict(source,target,query):
        tree=cKDTree(source)
        q=query.reshape(-1,query.shape[-1])
        result=np.empty_like(q)
        # Bound working memory, including the neighbor tensor.
        for start in range(0,len(q),2048):
            stop=min(start+2048,len(q));d,nn=tree.query(q[start:stop],k=8,eps=.08)
            weight=np.exp(-d*d/(.06+d[:,0:1]**2))
            result[start:stop]=(weight[...,None]*target[nn]).sum(1)/weight.sum(1)[:,None]
        return result.reshape(query.shape)
    PL,PR=predict(a,b,R),predict(b,a,L)
    def pair(a,b):
        a=a.reshape(n,-1);b=b.reshape(n,-1)
        return cdist(a,b,"sqeuclidean")/(np.square(a).sum(1)[:,None]+np.square(b).sum(1)[None,:]+1)
    return _unit_cost(.5*(pair(PL,L)+pair(R,PR)))


def _stroke_order(images,profile,update):
    # Common-height normalized evidence; only estimation, no resizing of output.
    h=min(1800,max(map(len,images)))
    cleaned=[_clean_ink(im) for im in images]
    if any(len(im)!=h for im in cleaned):
        cleaned=[map_coordinates(im,[np.broadcast_to(np.linspace(0,len(im)-1,h)[:,None],(h,im.shape[1])),
                                    np.broadcast_to(np.arange(im.shape[1]),(h,im.shape[1]))],order=1)
                 if len(im)!=h else im for im in cleaned]
    background=float(np.median([np.percentile(im,20) for im in images]))
    # The artifact band on gray/shadowed scans is less reliable than deeper
    # ink. A near-white scan retains more high-frequency boundary detail.
    inset,depth=(1,2) if background<.035 else (3,4)
    def side(im,right):
        if right:im=im[:,::-1]
        a=min(inset,max(0,im.shape[1]-2))
        return gaussian_filter1d(im[:,a:a+depth].mean(1),.9)
    L,R=[np.asarray([side(im,right) for im in cleaned]) for right in (False,True)]
    c=_line_cost(L,R)
    patch=False
    if profile!="fast" and len(images)>=8 and h>=48:
        update("Checking stroke continuation against observed interior patches")
        half,gap=(6,0) if inset==1 else (8,2)
        q=_patch_recurrence(cleaned,half,gap)
        if q is not None:
            c=.40*c+.60*q;patch=True
    np.fill_diagonal(c,np.inf)
    l,r=L.mean(1),R.mean(1);s=max(.02,float(np.median(np.r_[l,r])))
    order=chain(c,.5,borders=(3*l/s,3*r/s))
    # Reciprocal ranking is a qualitative warning, not a probability or GT.
    rowrank=np.argsort(np.argsort(c,axis=1),axis=1)
    colrank=np.argsort(np.argsort(c,axis=0),axis=0)
    a,b=order[:-1],order[1:]
    return order,{"patch_recurrence":patch,"edge_inset_px":inset,"profile_depth_px":depth,
                  "reciprocal_top3_joins":int(np.count_nonzero((rowrank[a,b]<3)&(colrank[a,b]<3))),
                  "total_joins":len(a)}


def reconstruct_strips(paths,target:Path,out:Path,update=lambda _:None,*,pixel_cap=8_000_000,profile="balanced"):
    if not 2<=len(paths)<=128:raise ValueError("Physical document mode supports 2–128 scanned strips")
    started=time.perf_counter();pieces=[];records=[]
    for path in paths:
        p,r=_rectify(path);pieces.append(p);records.append(r)
    if len({p.id for p in pieces})!=len(pieces):raise ValueError("Strip filenames must have unique stems")
    heights=[p.rgba.shape[0] for p in pieces]
    h=max(heights);w=sum(p.rgba.shape[1] for p in pieces)
    if h-min(heights)>max(3,.035*h):raise NotDocumentStrips("Rectified strips lack comparable long dimensions")
    if h*w>pixel_cap:raise ValueError("Rectified document exceeds canvas pixel budget")
    target.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    for p in pieces:save_rgba(target/(p.id+".png"),p.rgba)
    images=[_ink(p) for p in pieces]
    update("Matching continuity along scanned strip edges")
    if _textlike(pieces,images):
        # Retain the full cleaned row projection. Cropping to the middle of
        # each narrow strip discarded the few signals linking page columns and
        # sometimes reversed an entire region independently.
        same,flipped=_projection_cost([_clean_ink(im) for im in images])
        signs=_signed_tree(same,flipped,neighbors=12)
        ims=[np.rot90(im,2) if signs[i] else im for i,im in enumerate(images)]
        order,matching=_stroke_order(ims,profile,update)
        poses={};x=0
        for i in order:
            p=pieces[i];poses[p.id]=Pose(x,0,2*int(signs[i]));x+=p.rgba.shape[1]
        matching["relative_orientation_flips"]=int(np.count_nonzero(signs))
        algorithm="physical-document-foreground-multiscale"
    else:
        # A long photo strip is not an ink document. Retain chromatic evidence.
        poses,color=solve_unknown_color_strips(pieces)
        matching={"patch_recurrence":False,"color_aware":True}
        algorithm="physical-color-strip-chain"
    update("Rendering rectified scanned document")
    rgba,labels,collision=render(pieces,poses,(h,w))
    save_rgba(out/"assembly.png",rgba);write_cut_outputs(out,rgba,labels)
    warnings=["Physical strip order is experimental. Full placement does not establish correct adjacency.",
              "Global orientation is ambiguous in blind pixel-only matching; minor scan alignment and edge damage may remain."]
    if any(not r["masked"] for r in records):warnings.append("Some inputs had no foreground mask; their entire raster was treated as paper.")
    diag={"algorithm":algorithm,"status":"unverified",
          "placed":len(poses),"piece_count":len(pieces),"coverage":float((labels>=0).mean()),
          "render_collision_pixels":collision,"runtime_seconds":time.perf_counter()-started,
          "accuracy_verified":False,"orientation_ambiguous":True,"source_reference_used":False,
          "canvas_height":h,"canvas_width":w,
          "scan":{"kind":"document-strips","input_images":len(paths),"detected_pieces":len(pieces),
                  "components":1,"matching":matching,"warnings":warnings,"common_scale":1.0,"accuracy_verified":False}}
    json_write(out/"placements.json",{"mode":"blind-rectified-strips","poses":{k:v.as_dict() for k,v in poses.items()},
              "diagnostics":diag})
    json_write(out/"scan_report.json",{"input_images":len(paths),"detected_pieces":len(pieces),
        "mode":"document-strips","rectification":"per-row linear sampling between foreground mask edges",
        "pieces":records,"matching":matching,"orientation_ambiguous":True,"accuracy_verified":False,"warnings":warnings})
    tile=140;cols=min(8,len(pieces));rows=math.ceil(len(pieces)/cols)
    preview=Image.new("RGB",(cols*tile,rows*tile),(225,231,228));draw=ImageDraw.Draw(preview)
    for i,p in enumerate(pieces):
        im=Image.fromarray(p.rgba);im.thumbnail((tile-12,tile-24))
        dx=(i%cols)*tile+(tile-im.width)//2;dy=(i//cols)*tile+(tile-20-im.height)//2
        preview.paste(im,(dx,dy),im);draw.text(((i%cols)*tile+6,(i//cols+1)*tile-17),p.id,fill=(40,55,45))
    preview.save(out/"fragments_preview.png")
    json_write(out/"diagnostics.json",diag)
    return diag
