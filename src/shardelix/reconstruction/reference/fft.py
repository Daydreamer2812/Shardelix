"""Explicitly reference-assisted classical registration of damaged fragments.

The independent matching reference is required by this mode. Ground-truth
coordinates, labels and solution images are not inputs. Blind routes do not
import or use this module. Only observed fragment pixels are rendered.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from pathlib import Path

import numpy as np
from scipy import fft, ndimage as ndi
from scipy.optimize import least_squares
from PIL import Image, ImageOps

from shardelix.reconstruction.irregular.scan import rotation, _overlap, _shape
from shardelix.reconstruction.common import json_write,save_rgba


@dataclass(frozen=True)
class Match:
    theta: float
    center: np.ndarray
    rmse: float


def _samples(im,points):
    coords=[points[...,1]-.5,points[...,0]-.5]
    return np.stack([ndi.map_coordinates(im[...,k],coords,order=1,mode="nearest",prefilter=False)
                     for k in range(3)],axis=-1)


def _mask_samples(mask,points):
    return ndi.map_coordinates(mask,[points[...,1]-.5,points[...,0]-.5],order=1,
                                mode="constant",prefilter=False)


def _smooth(rgb,mask,sigma):
    weight=ndi.gaussian_filter(mask.astype(np.float32),sigma)
    return ndi.gaussian_filter(rgb*mask[...,None],(sigma,sigma,0))/np.maximum(weight[...,None],1.e-5)


class PyramidSearch:
    """Masked RGB SSD for every translation at sampled rotations."""
    def __init__(self,reference,max_radius,step,angle_step):
        self.step=int(step);self.angle_step=angle_step
        h,w=reference.shape[:2]
        self.h,self.w=h//step,w//step
        yy,xx=np.indices((self.h,self.w),dtype=np.float32)
        xy=np.stack([(xx+.5)*step,(yy+.5)*step],axis=-1)
        self.ref=_samples(ndi.gaussian_filter(reference,(.45*step,.45*step,0)),xy)
        self.rad=int(math.ceil(max_radius/step))+1
        N=2*self.rad+1
        self.size=(fft.next_fast_len(self.h+N-1,real=True),fft.next_fast_len(self.w+N-1,real=True))
        r=np.concatenate([self.ref.transpose(2,0,1),
                          np.sum(self.ref**2,axis=-1)[None],
                          np.ones((1,self.h,self.w),np.float32)],axis=0)
        self.freq=fft.rfft2(r,s=self.size,axes=(-2,-1))

    def search(self,piece,count=16):
        rgb=piece.rgba[:,:,:3].astype(np.float32)/255.
        mask=piece.mask;h,w=mask.shape
        core=ndi.binary_erosion(mask,iterations=1)
        if core.sum()<20:core=mask
        smooth=_smooth(rgb,mask,.45*self.step)
        radius=int(math.ceil(math.hypot(w,h)/(2*self.step)))+1
        y,x=np.indices((2*radius+1,2*radius+1),dtype=np.float32)
        off=np.stack([(x-radius)*self.step,(y-radius)*self.step],axis=-1)
        center=np.array([w/2,h/2])
        best=[]
        for deg in np.arange(-180,180,self.angle_step):
            theta=float(np.deg2rad(deg));R=rotation(theta)
            xy=off@R+center
            m=_mask_samples(core.astype(np.float32),xy)
            m=np.where(m>.8,m,0.).astype(np.float32)
            n=float(m.sum())
            if n<5:continue
            p=_samples(smooth,xy)
            tm=fft.rfft2(np.concatenate([m[None],(m[...,None]*p).transpose(2,0,1)],axis=0),
                         s=self.size,axes=(-2,-1)).conj()
            cross=np.sum(self.freq[:3]*tm[1:],axis=0)
            energy=self.freq[3]*tm[0]-2*cross
            freq=np.stack([energy,self.freq[4]*tm[0]])
            corr=fft.irfft2(freq,s=self.size,axes=(-2,-1))
            corr=np.roll(corr,(radius,radius),axis=(-2,-1))[:,:self.h,:self.w]
            err=(corr[0]+float(np.sum(m[...,None]*p*p)))/(3*n)
            err=np.maximum(err,0.)
            err[corr[1]<.985*n]=np.inf
            for _ in range(3):
                q=int(np.argmin(err));a,b=np.unravel_index(q,err.shape);cost=float(err[a,b])
                if not np.isfinite(cost):break
                pos=np.array([(b+.5)*self.step,(a+.5)*self.step])
                best.append((cost,theta,pos))
                err[max(0,a-3):a+4,max(0,b-3):b+4]=np.inf
        out=[]
        for cost,theta,pos in sorted(best,key=lambda q:q[0]):
            if any(np.linalg.norm(pos-p)<max(5.,self.step*2) and
                   abs(math.atan2(math.sin(theta-a),math.cos(theta-a)))<math.radians(self.angle_step*1.2)
                   for _,a,p in out):continue
            out.append((cost,theta,pos))
            if len(out)>=count:break
        return out


def _refine(piece,reference,hypotheses,step,angle_step):
    h,w=piece.mask.shape;center=np.array([w/2,h/2])
    core=ndi.binary_erosion(piece.mask,iterations=1)
    if core.sum()<20:core=piece.mask
    yy,xx=np.nonzero(core)
    stride=max(1,math.ceil(len(xx)/1000))
    xy=np.column_stack([xx[::stride]+.5,yy[::stride]+.5])-center
    rgb=_smooth(piece.rgba[:,:,:3].astype(np.float32)/255.,piece.mask,.65)
    target=rgb[yy[::stride],xx[::stride]]
    radius=max(6.,math.sqrt(float(piece.mask.sum())/np.pi))
    hh,ww=reference.shape[:2]
    Gx=ndi.gaussian_filter1d(reference,.7,axis=1,order=1)
    Gy=ndi.gaussian_filter1d(reference,.7,axis=0,order=1)
    # For optimization only; final error includes every sampled observed pixel.
    def residual(v):
        theta=v[2]/radius
        q=xy@rotation(theta).T+v[:2]
        obs=_samples(reference,q)
        valid=(q[:,0]>=.5)&(q[:,0]<=ww-.5)&(q[:,1]>=.5)&(q[:,1]<=hh-.5)
        err=(obs-target)
        err[~valid]=1.
        return err.ravel()
    def jac(v):
        theta=v[2]/radius
        p=xy@rotation(theta).T;q=p+v[:2]
        dx=_samples(Gx,q);dy=_samples(Gy,q)
        rot=(-dx*p[:,1,None]+dy*p[:,0,None])/radius
        J=np.stack([dx,dy,rot],axis=-1)
        valid=(q[:,0]>=.5)&(q[:,0]<=ww-.5)&(q[:,1]>=.5)&(q[:,1]<=hh-.5)
        J[~valid]=0.
        return J.reshape(-1,3)
    out=[]
    for _,theta,pos in hypotheses:
        v=np.r_[pos,theta*radius]
        rad=math.radians(angle_step*1.25)*radius
        bound=np.array([max(5.,step*2.),max(5.,step*2.),rad])
        try:fit=least_squares(residual,v,jac=jac,bounds=(v-bound,v+bound),
                              loss="soft_l1",f_scale=.045,max_nfev=25,ftol=.001,xtol=.001,gtol=.001)
        except ValueError:continue
        z=fit.x;err=residual(z)
        rmse=float(np.sqrt(np.mean(err*err))*255.)
        result=Match(float(z[2]/radius),z[:2].copy(),rmse)
        if any(np.linalg.norm(result.center-q.center)<3. and abs(math.atan2(math.sin(result.theta-q.theta),math.cos(result.theta-q.theta)))<.12
               for q in out):
            old=next(q for q in out if np.linalg.norm(result.center-q.center)<3. and abs(math.atan2(math.sin(result.theta-q.theta),math.cos(result.theta-q.theta)))<.12)
            if result.rmse<old.rmse:out=[q for q in out if q is not old]+[result]
        else:out.append(result)
    return sorted(out,key=lambda q:q.rmse)


def match_fragments(pieces,reference,update=lambda _:None,*,profile="balanced"):
    """Return independent pose candidates from pixels only."""
    # Smaller pieces benefit from a finer pyramid: coarse grids discard
    # small, distinctive strokes. Keep the full-size reference for refinement.
    ref=np.asarray(reference,dtype=np.float32)/255.
    fine=ndi.gaussian_filter(ref,(.65,.65,0))
    radius=max(math.hypot(*p.mask.shape)/2 for p in pieces)
    coarse=PyramidSearch(ref,radius,3,10)
    finer=None
    results=[]
    for i,p in enumerate(pieces):
        update(f"Reference-assisted masked search {i+1}/{len(pieces)}")
        found=coarse.search(p,count=14)
        refined=_refine(p,fine,found,coarse.step,coarse.angle_step)
        # Refine uncertain/small pieces with more angular and spatial samples;
        # no truth labels decide when to use the expensive search.
        if not refined or refined[0].rmse>9. or min(p.mask.shape)<30:
            if finer is None:finer=PyramidSearch(ref,radius,2,6)
            second=_refine(p,fine,finer.search(p,count=16),2,6)
            refined=sorted(refined+second,key=lambda q:q.rmse)
        results.append(refined)
    return results


def reconstruct_reference(pieces,reference,out:Path,report,update=lambda _:None,*,pixel_cap=8_000_000,profile="balanced"):
    started=time.perf_counter()
    if not pieces:raise ValueError("No visible fragments")
    if len(pieces)>256:raise ValueError("Reference-assisted recovery supports up to 256 fragments")
    with Image.open(reference) as im:
        im=ImageOps.exif_transpose(im).convert("RGB")
        w,h=im.size
        if w*h>pixel_cap:raise ValueError("Matching reference exceeds the canvas pixel budget")
        image=np.array(im)
    results=match_fragments(pieces,image,update,profile=profile)
    shapes=[_shape(p,2.)for p in pieces]
    # Absolute error, uniqueness and collision reject distractors. We do not
    # infer a required number of genuine fragments, or use filenames.
    proposals=[];unplaced=[]
    for i,cs in enumerate(results):
        if not cs:unplaced.append({"id":pieces[i].id,"reason":"no candidate"});continue
        a=cs[0]
        different=[q for q in cs[1:] if np.linalg.norm(q.center-a.center)>8. or abs(math.atan2(math.sin(q.theta-a.theta),math.cos(q.theta-a.theta)))>.23]
        margin=(different[0].rmse-a.rmse) if different else 100.
        proposals.append((a.rmse,i,a,margin))
    placed={};details=[]
    canvas=np.zeros((h,w,4),np.uint8);labels=np.full((h,w),-1,np.int32)
    swap=np.array([[0,1],[1,0]])
    for err,i,q,margin in sorted(proposals,key=lambda t:t[0]):
        p=pieces[i];ph,pw=p.mask.shape
        R=rotation(q.theta);t=q.center-R@np.array([pw/2,ph/2])
        if err>13. or (err>6. and margin<.55):
            unplaced.append({"id":p.id,"reason":"photometric mismatch or ambiguity","rmse":err,"margin":margin});continue
        if any(_overlap(shapes[k],shapes[i],A.T@R,A.T@(t-u))>.025 for k,(A,u) in placed.items()):
            unplaced.append({"id":p.id,"reason":"foreground collision","rmse":err,"margin":margin});continue
        corners=np.array([[0,0],[pw,0],[0,ph],[pw,ph]])@R.T+t
        x0,y0=np.maximum(0,np.floor(corners.min(axis=0)).astype(int))
        x1,y1=np.minimum([w,h],np.ceil(corners.max(axis=0)).astype(int))
        dim=(y1-y0,x1-x0)
        if min(dim)<=0:continue
        offset=swap@(R.T@(np.array([x0+.5,y0+.5])-t)-.5);matrix=swap@R.T@swap
        warp=np.stack([ndi.affine_transform(p.rgba[:,:,k],matrix,offset,output_shape=dim,order=0,prefilter=False)for k in range(4)],axis=2)
        m=warp[:,:,3]>0;region=labels[y0:y1,x0:x1]
        if np.count_nonzero(m&(region>=0))>max(3,.035*np.count_nonzero(m)):
            unplaced.append({"id":p.id,"reason":"raster collision","rmse":err,"margin":margin});continue
        keep=m&(region<0);canvas[y0:y1,x0:x1][keep]=warp[keep];region[keep]=i
        placed[i]=(R,t);details.append({"id":p.id,"rmse":err,"margin":margin})
    out.mkdir(parents=True,exist_ok=True)
    save_rgba(out/"assembly.png",canvas)
    overlay=canvas.copy();cut=np.zeros_like(canvas)
    for i in placed:
        area=labels==i;edge=area&~ndi.binary_erosion(area);overlay[edge]=[220,30,40,255];cut[edge]=[220,30,40,255]
    save_rgba(out/"assembly_with_cuts.png",overlay);save_rgba(out/"cut_lines.png",cut)
    poses={pieces[i].id:{"input_to_canvas_affine":np.column_stack([R,t]).tolist(),"crop_xy":list(pieces[i].input_origin)}for i,(R,t)in placed.items()}
    json_write(out/"placements.json",poses)
    report.update({"matching_reference_used":True,"reference_wh":[w,h],"accuracy_verified":False,
                   "accepted_reference_matches":details,"unplaced":unplaced,
                   "reference_candidates":[{"id":p.id,"candidates":[{"center":q.center.tolist(),"angle_deg":math.degrees(q.theta),"rmse":q.rmse}for q in cs[:5]]}for p,cs in zip(pieces,results)]})
    json_write(out/"scan_report.json",report)
    d={"algorithm":"reference-assisted-masked-se2","status":"unverified","matching_reference_used":True,
       "placed":len(placed),"piece_count":len(pieces),"coverage":float((labels>=0).mean()),
       "runtime_seconds":time.perf_counter()-started,"accuracy_verified":False,
       "scan":{"kind":"reference-assisted","input_images":report["input_images"],"detected_pieces":len(pieces),
               "warnings":["Reference-assisted registration; not a blind result."],"common_scale":report["common_scale"]}}
    json_write(out/"diagnostics.json",d)
    return d
