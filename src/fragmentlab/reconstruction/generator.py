"""Seeded partition generator. All cuts are shared by their two neighbors.

The generator writes random names, cropped lossless RGBA pieces, public size/
cut metadata, and evaluation-only ground truth outside the solver input folder.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from .common import json_write, save_rgba
from .limits import max_count,validate_constraints

CUTS = ("grid", "strips", "horizontal-strips", "jagged", "wavy", "jigsaw", "random")


def dimensions(count: int, h: int, w: int) -> tuple[int, int]:
    # A rectangular factorization close to the image aspect ratio.
    pairs={(r,count//r) for r in range(1,math.isqrt(count)+1) if count%r==0}
    pairs|={(c,r) for r,c in pairs}
    return min(sorted(pairs),key=lambda rc:abs(math.log((rc[1]/rc[0])/(w/h))))


def _split(image_shape: tuple[int, int], n: int, rng: np.random.Generator, cut: str) -> np.ndarray:
    """Exact arbitrary piece counts via recursive partitioning for non-grid cuts."""
    h, w = image_shape
    if n == 1:
        return np.zeros((h, w), np.int32)
    if cut in {"grid", "strips", "horizontal-strips"}:
        rows, cols = (1, n) if cut == "strips" else (n, 1) if cut == "horizontal-strips" else dimensions(n, h, w)
        if rows > h or cols > w:
            raise ValueError("Image is too small for the requested grid/strip count")
        ys = np.minimum(rows-1, np.arange(h) * rows // h)
        xs = np.minimum(cols-1, np.arange(w) * cols // w)
        return (ys[:, None] * cols + xs[None, :]).astype(np.int32)
    # A recursive split uses a single boundary function, preserving every pixel.
    # For the irregular cases the profiles are distinct, not watermarks.
    labels = np.full((h,w), -1, np.int32)
    def recurse(yy: np.ndarray, xx: np.ndarray, k: int, start: int) -> None:
        if k == 1:
            labels[yy,xx] = start
            return
        a = k // 2
        b = k-a
        spanx, spany = int(np.ptp(xx))+1, int(np.ptp(yy))+1
        vertical = spanx >= spany
        u,v = (xx,yy) if vertical else (yy,xx)
        su,sv = int(np.ptp(u))+1, int(np.ptp(v))+1
        vmin = int(v.min())
        t = (np.arange(sv)+.5)/sv
        style = cut if cut != "random" else rng.choice(["jagged","wavy","jigsaw"])
        # Small distortions near later splits; avoid partition starvation.
        amp = min(su*.19, max(1., sv*.16))
        phase = rng.uniform(0,2*np.pi)
        if style == "wavy":
            shift = amp*(.7*np.sin(rng.uniform(1,3)*2*np.pi*t+phase)+.3*np.sin(5*2*np.pi*t+phase))
        elif style == "jigsaw":
            shift = amp*np.sin(2*np.pi*t+phase)*np.sin(np.pi*t)**2
        else:
            shift = gaussian_filter(rng.normal(size=sv), sigma=max(.45,sv/75))
            shift = amp*shift/max(float(np.max(np.abs(shift))),1e-9)
            # Natural high-frequency torn edges create discriminative shapes.
            shift += rng.integers(-2,3,sv)
        score = u.astype(float) - shift[v-vmin]
        target = int(np.clip(round(len(yy)*a/k), a, len(yy)-b))
        # Tie-breaking is deterministic; partitions are disjoint and exhaustive.
        order = np.lexsort((v,score))
        left,right = order[:target],order[target:]
        recurse(yy[left],xx[left],a,start)
        recurse(yy[right],xx[right],b,start+a)
    yy,xx = np.indices((h,w))
    recurse(yy.ravel(),xx.ravel(),n,0)
    return labels


def generate(source: np.ndarray, output: Path, count: int, cut: str, seed: int = 42,
             rotations: bool = False, label_map: np.ndarray | None = None, *, angle_mode: str | None = None, missing: float = 0., sampling: int = 1) -> dict:
    if (cut not in CUTS and not (cut=="custom" and label_map is not None)) or not 1 <= count <= max_count(cut):
        raise ValueError("Choose a supported cut and a count within its size limit")
    if type(rotations) is not bool:raise ValueError("rotations must be a boolean")
    angle_mode=angle_mode or ("quarter" if rotations else "none")
    if angle_mode not in {"none","quarter","continuous"}:raise ValueError("Invalid angle mode")
    if not 0<=missing<1:raise ValueError("Missing fraction must be in [0,1)")
    if type(sampling) is not int or sampling not in (1,2,3,4) or sampling!=1 and angle_mode!="continuous":raise ValueError("Invalid sampling")
    rotations=angle_mode!="none"
    h,w = source.shape[:2]
    grid=(1,count) if cut=="strips" else (count,1) if cut=="horizontal-strips" else dimensions(count,h,w) if cut=="grid" else None
    if label_map is not None: grid=None
    validate_constraints(count,(h,w),"custom" if label_map is not None else cut,grid,rotations)
    if h*w < count or min(h,w) < 2:
        raise ValueError("Image too small")
    if (angle_mode=="continuous" or missing) and grid and (h%grid[0] or w%grid[1]):
        raise ValueError("Continuous/missing rectangular cases require uniform cells")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be new or empty: {output}")
    folder = output/"pieces"
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    if label_map is not None:
        if label_map.shape!=(h,w) or not np.issubdtype(label_map.dtype,np.integer):
            raise ValueError("Custom label map must be a 2-D integer array matching the source image")
        unique,indices=np.unique(label_map,return_inverse=True)
        if len(unique)!=count:raise ValueError("Number of unique label-map values must equal --pieces")
        labels=indices.reshape(h,w).astype(np.int32);cut="custom"
    else:
        # Rectangular partitions are separable: never sort H*W pixels.
        # The ceil boundary is identical to floor(pixel*count/extent).
        uniform=grid is not None
        labels=None if uniform else _split((h,w),count,rng,cut)
    if labels is not None:
        flat=labels.ravel()
        sort=np.argsort(flat,kind="stable")
        counts=np.bincount(flat,minlength=count)
        if len(counts)!=count or counts.min()==0:raise RuntimeError("Partition contains empty pieces")
        groups=np.split(sort,np.cumsum(counts)[:-1])
    else:
        r,c=grid
        ys=(np.arange(r+1,dtype=np.int64)*h+r-1)//r
        xs=(np.arange(c+1,dtype=np.int64)*w+c-1)//c
    truth = {}
    observed_indices=[]
    missing_indices=set(np.random.default_rng(seed+190273).choice(count,min(count-1,int(round(count*missing))),replace=False).tolist()) if missing else set()
    for index in rng.permutation(count):
        if labels is None:
            row,col=divmod(int(index),grid[1])
            y,y1=int(ys[row]),int(ys[row+1]);x,x1=int(xs[col]),int(xs[col+1])
            ph,pw=y1-y,x1-x
            a=np.empty((ph,pw,4),np.uint8);a[:,:,:3]=source[y:y1,x:x1];a[:,:,3]=255
        else:
            yy,xx=np.divmod(groups[index],w)
            x,y=int(xx.min()),int(yy.min())
            pw,ph=int(xx.max())-x+1,int(yy.max())-y+1
            a=np.zeros((ph,pw,4),np.uint8)
            a[yy-y,xx-x,:3]=source[yy,xx]
            a[yy-y,xx-x,3]=255
        rotation=int(rng.integers(4)) if rotations else 0
        angle=float(rng.uniform(-180,180)) if angle_mode=="continuous" else float(-90*rotation)
        ident=f"p_{rng.bytes(12).hex()}"
        if index in missing_indices:continue
        if angle_mode=="continuous":
            from .pose9 import rotate_raster
            save_rgba(folder/f"{ident}.png",rotate_raster(a,angle,sampling))
        else:save_rgba(folder/f"{ident}.png",np.rot90(a,-rotation))
        observed_indices.append((ident,int(index)))
        truth[ident]={"x":x,"y":y,"k":(-rotation)%4}
        if angle_mode=="continuous":
            truth[ident].update(input_ccw_deg=angle,height=ph,width=pw)

    meta={"width":w,"height":h,"piece_count":count,"cut":cut,
          "rotations":rotations, "grid": list((1,count) if cut=="strips" else (count,1) if cut=="horizontal-strips" else dimensions(count,h,w)) if cut in {"grid","strips","horizontal-strips"} else None}
    if angle_mode=="continuous" or missing:
        meta.update(piece_count=len(truth),expected_count=count,angle_mode=angle_mode,sampling=sampling)
    json_write(output/"case.json", meta)
    # Nothing in pieces/ exposes position, RNG seed, or source path.
    payload={"poses":truth,"seed":seed,"case":meta}
    if angle_mode=="continuous" or missing:
        payload["label_ids"]=[i for i,_ in observed_indices]
        mapping=np.full(count,-1,np.int32)
        for j,(_,index) in enumerate(observed_indices):mapping[index]=j
        (output/"evaluation").mkdir(exist_ok=True)
        if labels is None:
            ids=mapping.reshape(grid)
            truth_labels=np.repeat(np.repeat(ids,np.diff(ys),axis=0),np.diff(xs),axis=1)
        else:truth_labels=mapping[labels]
        np.savez_compressed(output/"evaluation"/"truth_labels.npz",labels=truth_labels)
    json_write(output/"evaluation"/"ground_truth.json",payload)
    return meta
