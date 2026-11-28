"""Indexed, non-learned reference registration at native pixel scale.

RootSIFT pose votes (optional native OpenCV) and dense rotation-invariant
colour/scale descriptors propose rigid poses. Only sampled observed RGB
pixels verify them. Local SE(2) refinement uses bilinear image derivatives.
The dense route is NumPy/SciPy-only and also works in WebAssembly.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from shardelix.reconstruction.irregular.scan import rotation

try:
    import cv2
except ImportError:  # portable browser engine
    cv2 = None


@dataclass(frozen=True)
class IndexedMatch:
    theta: float
    center: np.ndarray
    rmse: float
    roughness: float
    method: str


def _blur(im, sigma):
    if cv2 is not None:
        return cv2.GaussianBlur(im, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT)
    return ndi.gaussian_filter(im, (sigma, sigma, 0) if im.ndim == 3 else sigma)


def _masked_blur(rgb, mask, sigma):
    m = mask.astype(np.float32)
    return _blur(rgb*m[..., None], sigma)/np.maximum(_blur(m, sigma)[..., None], 1.e-5)


def _root_sift(d):
    return np.sqrt(d/np.maximum(d.sum(axis=1, keepdims=True), 1.e-12)).astype(np.float32)


def _wrap(theta):
    return (theta+np.pi)%(2*np.pi)-np.pi


def _grad(gray):
    return np.gradient(gray.astype(np.float32))


def _features(a, b):
    # Absolute colour and rotation-invariant differences of Gaussian responses.
    return np.concatenate([a, (a-b)*2.4], axis=-1).astype(np.float32)


def _bilinear(im, xy, gradients=False):
    h,w=im.shape[:2]
    q=np.asarray(xy, dtype=np.float32)-.5
    x=q[...,0];y=q[...,1]
    valid=(x>=0)&(y>=0)&(x<=w-1)&(y<=h-1)
    x=np.clip(x,0,max(0,w-1.001));y=np.clip(y,0,max(0,h-1.001))
    ix=x.astype(np.intp);iy=y.astype(np.intp)
    wx=(x-ix).astype(np.float32)[...,None];wy=(y-iy).astype(np.float32)[...,None]
    jx=np.minimum(ix+1,w-1);jy=np.minimum(iy+1,h-1)
    a=im[iy,ix];b=im[iy,jx];c=im[jy,ix];d=im[jy,jx]
    upper=a+(b-a)*wx;lower=c+(d-c)*wx
    value=upper+(lower-upper)*wy
    if gradients:
        return value,(b-a)*(1-wy)+(d-c)*wy,lower-upper,valid
    return value,valid


def _score_many(image, xy, colors, centers, angles, chunk=1024):
    """Batched masked SSD; no full-canvas search per hypothesis."""
    cost=np.empty(len(centers),np.float32)
    px=xy[:,0][None,:];py=xy[:,1][None,:]
    for a in range(0,len(centers),chunk):
        t=angles[a:a+chunk];p=centers[a:a+chunk]
        c=np.cos(t)[:,None];s=np.sin(t)[:,None]
        q=np.stack([p[:,0,None]+px*c-py*s,p[:,1,None]+px*s+py*c],axis=-1).astype(np.float32)
        if cv2 is not None:
            h,w=image.shape[:2]
            valid=(q[...,0]>=.5)&(q[...,0]<=w-.5)&(q[...,1]>=.5)&(q[...,1]<=h-.5)
            obs=cv2.remap(image,q[...,0]-.5,q[...,1]-.5,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        else:
            obs,valid=_bilinear(image,q)
        e=np.mean((obs-colors[None,:,:])**2,axis=-1)
        e[~valid]=30000.
        cost[a:a+len(t)]=np.sqrt(np.mean(e,axis=1))
    return cost


def _nms(cost,centers,angles,n=8,dist=5.,angle=.20):
    order=np.argsort(cost,kind="stable")
    out=[]
    for i in order:
        if out:
            j=np.asarray(out)
            if np.any((np.linalg.norm(centers[j]-centers[i],axis=1)<dist)&(abs(_wrap(angles[j]-angles[i]))<angle)):
                continue
        out.append(int(i))
        if len(out)>=n:break
    return np.asarray(out,int)


def _local_fit(im,xy,colors,center,theta,radius,iters=12):
    """Damped Gauss-Newton on three SE(2) variables, with robust weights."""
    v=np.array([center[0],center[1],theta*radius],np.float64)
    def calc(v,jac=False):
        angle=v[2]/radius
        p=xy@rotation(angle).T
        q=p+v[:2]
        if jac:
            obs,dx,dy,valid=_bilinear(im,q,True)
        else:
            obs,valid=_bilinear(im,q)
        r=obs-colors;r[~valid]=100.
        score=float(np.sqrt(np.mean(r*r)))
        if not jac:return score
        J=np.stack([dx,dy,(-dx*p[:,1,None]+dy*p[:,0,None])/radius],axis=-1)
        J[~valid]=0
        return score,r.ravel(),J.reshape(-1,3)
    score=calc(v)
    for _ in range(iters):
        _,r,J=calc(v,True)
        wt=1./np.sqrt(1.+(r/7.)**2)
        A=J.T@(J*wt[:,None]);b=J.T@(r*wt)
        A+=np.eye(3)*max(.1,float(np.trace(A))*2.e-5)
        try:d=-np.linalg.solve(A,b)
        except np.linalg.LinAlgError:break
        norm=float(np.linalg.norm(d))
        if norm<.035:break
        d*=min(1.,4./max(norm,1.e-9))
        improved=False
        for k in (1.,.5,.25):
            cand=v+d*k;val=calc(cand)
            if val+1.e-4<score:
                v=cand;score=val;improved=True;break
        if not improved:break
    return score,v[:2],float(_wrap(v[2]/radius))


class ReferenceIndex:
    def __init__(self, image, update=lambda _:None, *, max_points=2_200_000, use_sift=True):
        self.image=np.asarray(image,dtype=np.float32)
        self.h,self.w=self.image.shape[:2]
        self.fine=_blur(self.image,.65)
        self.coarse=_blur(self.image,1.6)
        self.step=max(1,int(math.ceil(math.sqrt(self.h*self.w/max_points))))
        # Build only compact sampled features; never store a full descriptor tensor.
        step=self.step
        a=self.coarse[::step,::step]
        b=_blur(self.image,3.8)[::step,::step].copy()
        f=_features(a,b).reshape(-1,6)
        self.grid_shape=a.shape[:2]
        update("Building rotation-invariant dense colour index")
        # A single exact six-dimensional KD-tree has a smaller memory
        # footprint than a forest of millions of FLANN nodes.
        self.flann=None
        self.tree=cKDTree(f,leafsize=32,compact_nodes=True,balanced_tree=True)
        del f,b
        gray=self.coarse.mean(axis=2)
        gy,gx=_grad(gray)
        self.gx=gx[::step,::step].copy().ravel();self.gy=gy[::step,::step].copy().ravel()
        del gray,gy,gx,a,self.image
        self.micro_tree=None;self.edge_tree=None;self.edge_xy=None
        self.sift=None;self.sift_index=None
        if cv2 is not None and use_sift:
            cv2.setNumThreads(1)
            cv2.setRNGSeed(2026)
            update("Indexing classical SIFT keypoints in overlapping tiles")
            self.sift=cv2.SIFT_create(nfeatures=2200,contrastThreshold=.009,edgeThreshold=18)
            g=cv2.cvtColor(np.asarray(image,dtype=np.uint8),cv2.COLOR_RGB2GRAY)
            coords=[];angles=[];sizes=[];descs=[]
            tile=768;pad=64
            for y in range(0,self.h,tile):
                for x in range(0,self.w,tile):
                    x0=max(0,x-pad);y0=max(0,y-pad)
                    x1=min(self.w,x+tile+pad);y1=min(self.h,y+tile+pad)
                    kp,d=self.sift.detectAndCompute(g[y0:y1,x0:x1],None)
                    if d is None:continue
                    pos=np.array([p.pt for p in kp],np.float32)+[x0+.5,y0+.5]
                    keep=(pos[:,0]>=x)&(pos[:,0]<min(x+tile,self.w))&(pos[:,1]>=y)&(pos[:,1]<min(y+tile,self.h))
                    if not keep.any():continue
                    coords.append(pos[keep]);angles.append(np.radians([p.angle for p in kp])[keep])
                    sizes.append(np.array([p.size for p in kp])[keep]);descs.append(d[keep])
            if descs:
                self.sift_xy=np.concatenate(coords).astype(np.float32)
                self.sift_angle=np.concatenate(angles).astype(np.float32)
                self.sift_size=np.concatenate(sizes).astype(np.float32)
                self.sift_desc=_root_sift(np.concatenate(descs))
                if len(self.sift_desc)>4:
                    self.sift_index=cv2.flann_Index(self.sift_desc,dict(algorithm=1,trees=5))
    def _query(self,f,k):
        k=min(k,int(np.prod(self.grid_shape)))
        if self.flann is not None:
            ix,_=self.flann.knnSearch(np.ascontiguousarray(f,dtype=np.float32),k,params=dict(checks=max(128,k*2)))
            return ix
        _,ix=self.tree.query(f,k=k,workers=1)
        return np.asarray(ix).reshape(len(f),-1)
    def _sift_candidates(self,rgb,mask,center):
        if self.sift_index is None:return np.empty((0,2)),np.empty(0)
        # Mask edges are not scene features: use only true interiors.
        g=cv2.cvtColor(rgb.astype(np.uint8),cv2.COLOR_RGB2GRAY)
        core=ndi.binary_erosion(mask,iterations=2)
        if core.sum()<30:core=mask
        kp,d=self.sift.detectAndCompute(g,core.astype(np.uint8)*255)
        if d is None:return np.empty((0,2)),np.empty(0)
        ix,_=self.sift_index.knnSearch(_root_sift(d),min(4,len(self.sift_xy)),params=dict(checks=80))
        a=np.radians([p.angle for p in kp]).astype(np.float32)
        xy=np.array([p.pt for p in kp],np.float32)+.5-center
        sizes=np.array([p.size for p in kp])
        match=ix.ravel();j=np.repeat(np.arange(len(kp)),ix.shape[1])
        good=(self.sift_size[match]/sizes[j]>.68)&(self.sift_size[match]/sizes[j]<1.5)
        match=match[good];j=j[good]
        theta=_wrap(self.sift_angle[match]-a[j])
        c=np.cos(theta);s=np.sin(theta);f=xy[j]
        centers=self.sift_xy[match]-np.column_stack([f[:,0]*c-f[:,1]*s,f[:,0]*s+f[:,1]*c])
        return centers,theta
    def _dense_candidates(self,rgb,mask,center,smooth,wide=False):
        b=_masked_blur(rgb,mask,3.8);feat=_features(smooth,b)
        gy,gx=_grad(smooth.mean(axis=2))
        d=ndi.distance_transform_edt(mask)
        # Prefer reliable, distinctive, spatially distinct interior patches.
        good=d>=min(6.,max(1.5,float(d.max())*.72))
        score=np.hypot(gx,gy)+np.linalg.norm(smooth-b,axis=-1)+.15*d
        score[~good]=-1.e9
        coords=[]
        for _ in range(9 if wide else 5):
            iy,ix=np.unravel_index(int(np.argmax(score)),score.shape)
            if score[iy,ix]<-1.e8:break
            coords.append((iy,ix))
            rad=max(4,min(mask.shape)//5)
            score[max(0,iy-rad):iy+rad+1,max(0,ix-rad):ix+rad+1]=-1.e9
        if not coords:return np.empty((0,2)),np.empty(0)
        yy,xx=np.asarray(coords).T
        k=192 if wide else 64
        ids=self._query(feat[yy,xx],k)
        qy,qx=np.unravel_index(ids,self.grid_shape)
        locations=np.stack([qx*self.step+.5,qy*self.step+.5],axis=-1).astype(np.float32)
        rel=np.stack([xx+.5,yy+.5],axis=-1).astype(np.float32)-center
        # Gradient orientation generates precise candidates when meaningful.
        # Uniform rotations retain hypotheses for flat or symmetric pieces.
        orient=np.arctan2(self.gy[ids],self.gx[ids])-np.arctan2(gy[yy,xx],gx[yy,xx])[:,None]
        offsets=np.radians([-24,-12,0,12,24] if wide else [-12,0,12])
        ang1=orient[...,None]+offsets
        n=24 if wide else 12
        ang2=np.broadcast_to(np.arange(n)*2*np.pi/n,(*orient.shape,n))
        angles=np.concatenate([ang1,ang2],axis=-1)
        c=np.cos(angles);s=np.sin(angles)
        dx=rel[:,0,None,None];dy=rel[:,1,None,None]
        cent=locations[:,:,None,:]-np.stack([c*dx-s*dy,s*dx+c*dy],axis=-1)
        return cent.reshape(-1,2),_wrap(angles.ravel())
    def _micro_candidates(self,rgb,mask,center):
        """Fine-scale descriptors and a full-resolution border-band index.

        Boundary-clipped and very small pieces lose context under broad
        Gaussian smoothing. A second handcrafted scale recovers these cases.
        """
        fine=_masked_blur(rgb,mask,.65);coarse=_masked_blur(rgb,mask,1.6)
        feat=_features(fine,coarse)
        if self.micro_tree is None:
            f=_features(self.fine[::self.step,::self.step],self.coarse[::self.step,::self.step]).reshape(-1,6)
            self.micro_tree=cKDTree(f,leafsize=32)
        if self.edge_tree is None:
            # Border samples are indexed at native pixel resolution.
            band=min(36,max(2,min(self.h,self.w)//2))
            yy,xx=np.indices((self.h,self.w),sparse=True)
            edge=(yy<band)|(yy>=self.h-band)|(xx<band)|(xx>=self.w-band)
            ey,ex=np.nonzero(edge)
            self.edge_xy=np.column_stack([ex,ey]).astype(np.int32)
            self.edge_tree=cKDTree(_features(self.fine[ey,ex],self.coarse[ey,ex]),leafsize=24)
        gy,gx=_grad(fine.mean(axis=2));dist=ndi.distance_transform_edt(mask)
        score=np.hypot(gx,gy)+np.linalg.norm(fine-coarse,axis=-1)+.25*dist
        score[dist<min(3.,max(1.,float(dist.max())*.6))]=-1.e9
        coords=[]
        for _ in range(6):
            iy,ix=np.unravel_index(int(np.argmax(score)),score.shape)
            if score[iy,ix]<-1.e8:break
            coords.append((iy,ix));rad=max(2,min(mask.shape)//5)
            score[max(0,iy-rad):iy+rad+1,max(0,ix-rad):ix+rad+1]=-1.e9
        if not coords:return np.empty((0,2)),np.empty(0)
        yy,xx=np.asarray(coords).T;f=feat[yy,xx]
        _,ids=self.micro_tree.query(f,k=min(192,self.micro_tree.n),workers=1)
        ids=np.asarray(ids).reshape(len(f),-1);iy,ix=np.unravel_index(ids,self.grid_shape)
        loc=np.stack([ix*self.step,iy*self.step],axis=-1)
        _,edge_ids=self.edge_tree.query(f,k=min(160,self.edge_tree.n),workers=1)
        loc=np.concatenate([loc,self.edge_xy[np.asarray(edge_ids).reshape(len(f),-1)]],axis=1)
        ix=loc[...,0];iy=loc[...,1]
        gx_ref=(self.fine[iy,np.minimum(ix+1,self.w-1)]-self.fine[iy,np.maximum(ix-1,0)]).mean(axis=-1)
        gy_ref=(self.fine[np.minimum(iy+1,self.h-1),ix]-self.fine[np.maximum(iy-1,0),ix]).mean(axis=-1)
        orient=np.arctan2(gy_ref,gx_ref)-np.arctan2(gy[yy,xx],gx[yy,xx])[:,None]
        angles=np.concatenate([orient[...,None]+np.radians([-18,-9,0,9,18]),np.broadcast_to(np.arange(24)*np.pi/12,(*orient.shape,24))],axis=-1)
        rel=np.stack([xx+.5,yy+.5],axis=-1).astype(np.float32)-center
        dx=rel[:,0,None,None];dy=rel[:,1,None,None];c=np.cos(angles);ss=np.sin(angles)
        centers=loc[:,:,None,:]+.5-np.stack([c*dx-ss*dy,ss*dx+c*dy],axis=-1)
        return centers.reshape(-1,2),_wrap(angles.ravel())

    def micro_match(self,piece):
        rgb=piece.rgba[:,:,:3].astype(np.float32);mask=piece.mask;h,w=mask.shape
        center=np.array([w/2,h/2],np.float32)
        core=ndi.binary_erosion(mask)
        if core.sum()<20:core=mask
        y,x=np.nonzero(core)
        if len(y)<8:return []
        f=_masked_blur(rgb,mask,.65);coords=np.column_stack([x+.5,y+.5]).astype(np.float32)-center
        rough=float(np.sqrt(np.mean((rgb[y,x]-f[y,x])**2)))
        color=f[y,x];ids=np.linspace(0,len(y)-1,min(48,len(y)),dtype=int)
        c,a=self._micro_candidates(rgb,mask,center)
        if not len(c):return []
        cost=_score_many(self.fine,coords[ids],color[ids],c,a)
        radius=max(6.,math.sqrt(float(mask.sum())/np.pi));out=[]
        ids=np.linspace(0,len(y)-1,min(420,len(y)),dtype=int)
        for i in _nms(cost,c,a,14,dist=3.,angle=.13):
            if cost[i]>17:continue
            score,pos,theta=_local_fit(self.fine,coords[ids],color[ids],c[i],a[i],radius,18)
            out.append(IndexedMatch(theta,pos,score,rough,"micro-fine"))
        return sorted(out,key=lambda m:m.rmse)

    def match(self,piece,*,wide=False,deep=False):
        rgb=piece.rgba[:,:,:3].astype(np.float32);mask=piece.mask;h,w=mask.shape
        center=np.array([w/2,h/2],np.float32)
        core=ndi.binary_erosion(mask,iterations=1)
        if core.sum()<20:core=mask
        y,x=np.nonzero(core)
        if len(y)<8:return []
        coarse=_masked_blur(rgb,mask,1.6);fine=_masked_blur(rgb,mask,.65)
        # Deterministic spatially distributed samples.
        def samples(n,im):
            ids=np.linspace(0,len(y)-1,min(n,len(y)),dtype=int)
            return np.column_stack([x[ids]+.5,y[ids]+.5]).astype(np.float32)-center,im[y[ids],x[ids]]
        xy,c=samples(64,coarse);xf,cf=samples(420,fine);xc,cc=samples(220,coarse)
        rough=float(np.sqrt(np.mean((rgb[y,x]-fine[y,x])**2)))
        radius=max(6.,math.sqrt(float(mask.sum())/np.pi))
        matches=[]
        def search(cent,theta,method,n):
            if not len(cent):return
            good=(cent[:,0]>-w/2)&(cent[:,0]<self.w+w/2)&(cent[:,1]>-h/2)&(cent[:,1]<self.h+h/2)
            cent=cent[good];theta=theta[good]
            if not len(cent):return
            costs=_score_many(self.coarse,xy,c,cent,theta)
            for j in _nms(costs,cent,theta,n):
                score,pc,pa=_local_fit(self.coarse,xc,cc,cent[j],theta[j],radius,9)
                # Avoid expensive full-resolution fit on manifestly poor hypotheses.
                if score>22:continue
                score,pc,pa=_local_fit(self.fine,xf,cf,pc,pa,radius,14)
                m=IndexedMatch(pa,pc,score,rough,method)
                matches.append(m)
        cen,ang=self._sift_candidates(rgb,mask,center)
        search(cen,ang,"sift",5)
        best=min((m.rmse for m in matches),default=1.e6)
        # A close, high-information SIFT match needs no dense sweep.
        if best>(3.8 if deep else 5.2) or len(cen)<2 or wide:
            cen,ang=self._dense_candidates(rgb,mask,center,coarse,wide=wide)
            search(cen,ang,"dense-wide" if wide else "dense",7 if wide else 5)
        matches.sort(key=lambda q:q.rmse)
        out=[]
        for m in matches:
            if any(np.linalg.norm(m.center-q.center)<3 and abs(_wrap(m.theta-q.theta))<.13 for q in out):continue
            out.append(m)
        return out


MAX_REFERENCE_PIECES = 8192
MAX_REFERENCE_PIXELS = 64_000_000
MAX_REFERENCE_INPUT_PIXELS = 330_000_000


def _distinct_margin(candidates):
    if not candidates:return 0.
    a=candidates[0]
    for b in candidates[1:]:
        if np.linalg.norm(a.center-b.center)>8. or abs(_wrap(a.theta-b.theta))>.23:
            return b.rmse-a.rmse
    return 100.


def _acceptable(m,margin,area):
    """Absolute photometric error plus ambiguity check (no expected piece count)."""
    threshold=min(7.2,max(3.8,2.7+.75*m.roughness))
    if m.rmse>threshold:return False
    if m.rmse>2.5 and margin<max(.35,.10*m.rmse):return False
    if m.roughness<.35 and margin<.35:return False
    return True


def reconstruct_indexed(pieces,reference,out,report,update=lambda _:None,*,pixel_cap=MAX_REFERENCE_PIXELS,
                        piece_cap=MAX_REFERENCE_PIECES,use_sift=True,deep=False,adaptive_micro=False):
    """Register independently, reject uncertain/colliding poses, render only observed pixels."""
    from PIL import Image,ImageOps
    from shardelix.reconstruction.common import save_rgba,json_write
    from pathlib import Path
    started=time.perf_counter()
    if not pieces:raise ValueError("No visible fragments")
    if len(pieces)>piece_cap:raise ValueError(f"Reference-assisted recovery supports up to {piece_cap} fragments")
    with Image.open(reference) as img:
        img=ImageOps.exif_transpose(img).convert("RGB")
        w,h=img.size
        if w*h>pixel_cap:raise ValueError("Matching reference exceeds the canvas pixel budget")
        image=np.array(img)
    idx=ReferenceIndex(image,update,use_sift=use_sift)
    index_seconds=time.perf_counter()-started
    del image
    results=[]
    for i,p in enumerate(pieces):
        update(f"Indexed reference registration {i+1}/{len(pieces)}")
        cs=idx.match(p,deep=deep)
        area=int(p.mask.sum())
        if not cs or cs[0].rmse>(3.8 if deep else 5.2) or not _acceptable(cs[0],_distinct_margin(cs),area):
            more=idx.match(p,wide=True,deep=deep)
            cs=sorted(cs+more,key=lambda m:m.rmse)
            unique=[]
            for m in cs:
                if any(np.linalg.norm(m.center-q.center)<3 and abs(_wrap(m.theta-q.theta))<.13 for q in unique):continue
                unique.append(m)
            cs=unique
        if deep and area<=900 and (not cs or cs[0].rmse>3.8):
            extra=idx.micro_match(p)
            cs=sorted(cs+extra,key=lambda m:m.rmse)
            unique=[]
            for m in cs:
                if any(np.linalg.norm(m.center-q.center)<3 and abs(_wrap(m.theta-q.theta))<.13 for q in unique):continue
                unique.append(m)
            cs=unique
        results.append(cs)
    matching_backend="opencv-sift-flann+scipy-colour-kdtree" if idx.sift_index is not None else "dense-colour-index"
    adaptive=adaptive_micro and not deep and w*h<=8_000_000
    if not adaptive:del idx
    retrieval_seconds=time.perf_counter()-started-index_seconds
    rescue={"attempted":0,"placed":0,"eligible":0}
    # The full-canvas label map is a constant-time spatial occupancy index.
    # Only each warped bounding box is validated; no pairwise all-fragment loop.
    canvas=np.zeros((h,w,4),np.uint8);labels=np.zeros((h,w),np.uint32)
    cut=np.zeros_like(canvas)
    proposals=[];unplaced=[];placed={};details=[]
    for i,cs in enumerate(results):
        if not cs:
            unplaced.append({"id":pieces[i].id,"reason":"no candidate"});continue
        m=cs[0];margin=_distinct_margin(cs)
        if not _acceptable(m,margin,int(pieces[i].mask.sum())):
            unplaced.append({"id":pieces[i].id,"reason":"photometric mismatch or ambiguity","rmse":m.rmse,"margin":margin});continue
        proposals.append((m.rmse,i,m,margin))
    swap=np.array([[0,1],[1,0]])
    collision_pixels=0
    def place(i,m,margin):
        nonlocal collision_pixels
        err=m.rmse;p=pieces[i];ph,pw=p.mask.shape;R=rotation(m.theta)
        t=m.center-R@np.array([pw/2,ph/2])
        corners=np.array([[0,0],[pw,0],[0,ph],[pw,ph]])@R.T+t
        x0,y0=np.maximum(0,np.floor(corners.min(axis=0)).astype(int))
        x1,y1=np.minimum([w,h],np.ceil(corners.max(axis=0)).astype(int))
        dim=(y1-y0,x1-x0)
        if min(dim)<=0:return {"id":p.id,"reason":"outside canvas"}
        offset=swap@(R.T@(np.array([x0+.5,y0+.5])-t)-.5);matrix=swap@R.T@swap
        warp=np.stack([ndi.affine_transform(p.rgba[:,:,k],matrix,offset,output_shape=dim,order=0,prefilter=False)for k in range(4)],axis=2)
        mask=warp[:,:,3]>0;region=labels[y0:y1,x0:x1]
        n=int(mask.sum());overlap=int(np.count_nonzero(mask&(region>0)))
        if n==0 or overlap>max(3,.035*n):
            return {"id":p.id,"reason":"foreground collision","rmse":err,"margin":margin,"conflict_pixels":overlap}
        keep=mask&(region==0);canvas[y0:y1,x0:x1][keep]=warp[keep];region[keep]=i+1
        collision_pixels+=overlap
        edge=keep&~ndi.binary_erosion(keep);cut[y0:y1,x0:x1][edge]=[220,30,40,255]
        placed[i]=(R,t);details.append({"id":p.id,"rmse":err,"margin":margin,"candidate_method":m.method})
        return None
    for _,i,m,margin in sorted(proposals,key=lambda t:t[0]):
        rejected=place(i,m,margin)
        if rejected is not None:unplaced.append(rejected)
    # Retry only small fragments rejected by photometric/ambiguity checks OR
    # occupancy. Baseline placements are immutable: the rescue cannot evict a
    # previously placed fragment even if its own candidate has lower RMSE.
    if adaptive:
        eligible=[]
        for i,(p,cs) in enumerate(zip(pieces,results)):
            area=int(p.mask.sum())
            if i not in placed and 8<=area<=900:
                err=cs[0].rmse if cs else 20.
                if err<=20:eligible.append((err,i,area))
        rescue["eligible"]=len(eligible)
        rescue_started=time.perf_counter()
        for _,i,area in sorted(eligible)[:32]:
            rescue["attempted"]+=1
            update(f"Fine-scale tiny fragment rescue {rescue['attempted']}/{min(32,len(eligible))}")
            cs=sorted(results[i]+idx.micro_match(pieces[i]),key=lambda m:m.rmse)
            unique=[]
            for m in cs:
                if any(np.linalg.norm(m.center-q.center)<3 and abs(_wrap(m.theta-q.theta))<.13 for q in unique):continue
                unique.append(m)
            if not unique or not _acceptable(unique[0],_distinct_margin(unique),area):continue
            if place(i,unique[0],_distinct_margin(unique)) is None:
                results[i]=unique;rescue["placed"]+=1
                unplaced=[row for row in unplaced if row["id"]!=pieces[i].id]
        retrieval_seconds+=time.perf_counter()-rescue_started
        del idx
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    save_rgba(out/"assembly.png",canvas)
    overlay=canvas.copy();overlay[cut[:,:,3]>0]=[220,30,40,255]
    save_rgba(out/"assembly_with_cuts.png",overlay);save_rgba(out/"cut_lines.png",cut)
    poses={pieces[i].id:{"input_to_canvas_affine":np.column_stack([R,t]).tolist(),"crop_xy":list(pieces[i].input_origin)} for i,(R,t) in placed.items()}
    json_write(out/"placements.json",poses)
    report.update({"matching_reference_used":True,"reference_wh":[w,h],"accuracy_verified":False,
                   "accepted_reference_matches":details,"unplaced":unplaced,"matching_backend":matching_backend,
                   "micro_rescue":rescue,"reference_candidates":[{"id":p.id,"candidates":[{"center":q.center.tolist(),"angle_deg":math.degrees(q.theta),"rmse":q.rmse,"roughness":q.roughness,"method":q.method} for q in cs[:5]]}for p,cs in zip(pieces,results)]})
    json_write(out/"scan_report.json",report)
    d={"algorithm":"reference-assisted-indexed-se2-deep" if deep else "reference-assisted-indexed-se2","status":"unverified","matching_reference_used":True,
       "placed":len(placed),"piece_count":len(pieces),"coverage":float((labels>0).mean()),
       "runtime_seconds":time.perf_counter()-started,"index_query_seconds":index_seconds,"retrieval_seconds":retrieval_seconds,
       "matching_backend":matching_backend,"render_collision_pixels":collision_pixels,"accuracy_verified":False,"micro_rescue":rescue,
       "scan":{"kind":"reference-assisted","input_images":report["input_images"],"detected_pieces":len(pieces),
               "warnings":["Reference-assisted registration; not a blind result."],"common_scale":report["common_scale"]}}
    json_write(out/"diagnostics.json",d)
    return d
