"""Anchor-preserving, tolerant multi-contact placement of disconnected masks.

All proposed transforms come from observed pair matches. The main component is
never moved. A candidate must fit the public canvas, have no interior conflict,
and expose sufficient consistent geometry/colour contact. No template or truth.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import numpy as np
from scipy.ndimage import binary_erosion
from shardelix.reconstruction.common import Pose, render, normalize_pose_set
from shardelix.reconstruction.irregular.lattice import R, _vertex_to_pose

D=((1,0),(-1,0),(0,1),(0,-1))

def vertex(piece,pose):
    h,w=piece.rgba.shape[:2]
    return np.array([pose.x,pose.y])-np.min(np.array([[0,0],[w,0],[0,h],[w,h]])@R[pose.k].T,axis=0)

def _group_cache(pieces, group, rotations):
    select=[pieces[i] for i in group]
    pp={pieces[i].id:p for i,p in group.items()}
    pp,extent=normalize_pose_set(select,pp)
    local={i:pp[pieces[i].id] for i in group}
    rgba,lab,coll=render(select,pp,extent)
    if coll:return None
    result={}
    for k in range(4 if rotations else 1):
        a=np.rot90(rgba,-k)
        m=a[:,:,3]>0;core=np.zeros(m.shape,bool)
        # Erode each constituent separately so their internal seams do not
        # become accidental outer cores.
        h,w=extent
        origin=np.min(np.array([[0,0],[w,0],[0,h],[w,h]])@R[k].T,axis=0)
        gp={}
        for i,p in local.items():
            q=_vertex_to_pose(pieces[i],(p.k+k)%4,R[k]@vertex(pieces[i],p)-origin)
            gp[i]=q
            mask=pieces[i].rotated(q.k)[:,:,3]>0
            hh,ww=mask.shape
            core[q.y:q.y+hh,q.x:q.x+ww]|=binary_erosion(mask)
        pad=np.pad(m,1)
        borders=[]
        for (dx,dy),nb in zip(D,(pad[1:-1,2:],pad[1:-1,:-2],pad[2:,1:-1],pad[:-2,1:-1])):
            yy,xx=np.nonzero(m&~nb);borders.append((yy,xx,dx,dy))
        result[k]=(a,m,core,borders,gp)
    return result

def _fit(canvas,owner,core,entry,x,y,max_cost=.135):
    a,m,c,borders,gp=entry;h,w=m.shape;H,W=owner.shape
    if min(x,y)<0 or x+w>W or y+h>H:return None
    occ=owner[y:y+h,x:x+w]>=0
    overlap=int((occ&m).sum())
    if overlap>max(3,.012*int(m.sum())) or (core[y:y+h,x:x+w]&c).any():return None
    n=wall=perim=0;cost=0.;touch=set()
    for yy,xx,dx,dy in borders:
        Y=yy+y+dy;X=xx+x+dx;perim+=len(yy)
        ok=(Y>=0)&(X>=0)&(Y<H)&(X<W);wall+=int((~ok).sum())
        Y,X,ay,ax=Y[ok],X[ok],yy[ok],xx[ok]
        good=owner[Y,X]>=0
        if good.any():
            n+=int(good.sum())
            cost+=np.abs(a[ay[good],ax[good],:3].astype(np.int16)-canvas[Y[good],X[good],:3].astype(np.int16)).sum()
            touch.update(np.unique(owner[Y[good],X[good]]).tolist())
    if n<max(12,.055*perim) or overlap>max(3,.12*n):return None
    cost=float(cost)/(n*3*255);frac=(n+wall)/max(1,perim)
    if cost>max_cost:return None
    # Contacts with multiple independent neighbours make a global fit much
    # less ambiguous than one short repeated arc.
    score=(n**.45)*frac**.65*(1+.11*min(len(touch),5))/(.018+cost)
    score*=np.exp(-5*overlap/max(n,1))
    return float(score),frac,cost,len(touch),n,overlap

def _seam_limit(canvas,owner):
    """Calibrate colour tolerance using only already assembled visible seams."""
    values=[]
    for a,b,pa,pb in ((owner[:,:-1],owner[:,1:],canvas[:,:-1],canvas[:,1:]),
                       (owner[:-1],owner[1:],canvas[:-1],canvas[1:])):
        sel=(a>=0)&(b>=0)&(a!=b)
        if sel.any():
            cost=np.abs(pa[sel,:3].astype(np.int16)-pb[sel,:3].astype(np.int16)).mean(axis=1)/255
            values.append(cost)
    if not values:return .12
    return float(np.clip(np.mean(np.concatenate(values))*1.65+.015,.12,.245))

def anchored_complete(pieces,poses,groups,candidates,shape,rotations=True,rounds=20):
    """Try rigid groups, then remaining singleton pieces. Never move anchors."""
    if not poses or len(poses)==len(pieces):return poses,{"anchor_added":0}
    relative,extent=normalize_pose_set(pieces,poses)
    H,W=shape;eh,ew=extent
    if eh>H or ew>W:return poses,{"anchor_added":0,"anchor_skipped":"extent"}
    # A fragment cluster can be correct in relative coordinates but not yet
    # anchored to the image frame. Reserve the unknown translation slack on
    # both sides, allowing it to grow in either direction. Never let the
    # occupied extent exceed the PUBLIC canvas dimensions.
    slack_y,slack_x=H-eh,W-ew
    workspace=(H+slack_y,W+slack_x)
    if workspace[0]*workspace[1]>20_000_000 and (slack_x or slack_y):
        return poses,{"anchor_added":0,"anchor_skipped":"workspace-budget"}
    poses={i:Pose(p.x+slack_x,p.y+slack_y,p.k) for i,p in relative.items()}
    bounds=[slack_x,slack_y,slack_x+ew,slack_y+eh]
    def within_extent(entry,x,y):
        h,w=entry[1].shape
        return max(bounds[2],x+w)-min(bounds[0],x)<=W and max(bounds[3],y+h)-min(bounds[1],y)<=H
    index={p.id:i for i,p in enumerate(pieces)}
    anchor={index[i]:p for i,p in poses.items()}
    canvas,owner,_=render(pieces,poses,workspace)
    core=np.zeros(workspace,bool)
    for i,p in anchor.items():
        m=pieces[i].rotated(p.k)[:,:,3]>0;h,w=m.shape
        core[p.y:p.y+h,p.x:p.x+w]|=binary_erosion(m)
    edges=defaultdict(list)
    for strength,i,j,r,tx,ty in candidates:
        t=np.array([tx,ty])
        edges[j].append((i,r,t,float(strength)))
        rr=(-r)%4;edges[i].append((j,rr,-R[rr]@t,float(strength)))
    initial=len(anchor);history=[];limit=_seam_limit(canvas,owner)
    raw_groups=[{index[i]:p for i,p in g.items()} for g in groups if not set(g)&set(poses)]
    # A rejected large component may have a wrong internal relation. Second
    # phase splits it, allowing only independently verified singletons.
    for phase in range(2):
        remaining=raw_groups if phase==0 else [{i:Pose(0,0)} for i in range(len(pieces)) if i not in anchor]
        remaining=[g for g in remaining if not set(g)&anchor.keys()]
        caches=[_group_cache(pieces,g,rotations) for g in remaining]
        relaxed=False
        for _ in range(rounds):
            proposals=[]
            for gi,(g,cache) in enumerate(zip(remaining,caches)):
                if cache is None or set(g)&anchor.keys():continue
                votes=Counter();evidence=defaultdict(set)
                # Candidate global position from every observed relative edge.
                for j in g:
                    for i,r,t,strength in edges[j]:
                        p=anchor.get(i)
                        if p is None:continue
                        k=(p.k+r)%4
                        q=_vertex_to_pose(pieces[j],k,R[p.k]@t+vertex(pieces[i],p))
                        for z,entry in cache.items():
                            jp=entry[-1][j]
                            if jp.k!=k:continue
                            key=(q.x-jp.x,q.y-jp.y,z)
                            votes[key]+=1;evidence[key].add(i)
                fits=[]
                if relaxed:
                    # At a plateau, a one-pixel raster offset can invalidate
                    # an otherwise consistent contour. Aggregate supporting
                    # contacts across nearby hypotheses, but require strong
                    # independent evidence and whole-mask verification.
                    for (x,y,k),v in list(votes.most_common(12)):
                        origin=evidence[x,y,k].copy()
                        for dx,dy in ((-1,0),(1,0),(0,-1),(0,1)):
                            key=(x+dx,y+dy,k)
                            if key not in votes:votes[key]=v
                            evidence[key].update(origin)
                for (x,y,k),v in votes.most_common(56 if relaxed else 40):
                    if not within_extent(cache[k],x,y):continue
                    fit=_fit(canvas,owner,core,cache[k],x,y,limit)
                    if fit is None:continue
                    score,frac,cost,touches,n,over=fit
                    if frac<(.10 if relaxed else .14) or (touches<2 and (frac<.26 or cost>.85*limit)):continue
                    score*=1+.10*min(5,len(evidence[x,y,k]))
                    fits.append((score,x,y,k,frac,cost,touches,n,v))
                if not fits:continue
                fits.sort(reverse=True);best=fits[0]
                score,x,y,k,frac,cost,touches,n,v=best
                margin=score/max(1e-9,fits[1][0]) if len(fits)>1 else 2.
                normal=(touches>=2 and frac>=.22 and cost<=limit and margin>=1.08) or (frac>=.50 and cost<=limit and margin>=1.05) or (frac>=.22 and cost<=.75*limit and margin>=1.25)
                extra=relaxed and touches>=2 and len(evidence[x,y,k])>=2 and frac>=.13 and cost<=.95*limit and margin>=1.20
                if normal or extra:proposals.append((score*min(1.5,margin),gi,x,y,k))
            changed=0
            for _,gi,x,y,k in sorted(proposals,reverse=True):
                g,cache=remaining[gi],caches[gi]
                if set(g)&anchor.keys():continue
                if not within_extent(cache[k],x,y):continue
                fit=_fit(canvas,owner,core,cache[k],x,y,limit)
                if fit is None:continue
                _,frac,cost,touch,n,over=fit
                if frac<(.10 if relaxed else .14) or cost>limit:continue
                for i,p in cache[k][-1].items():
                    q=Pose(x+p.x,y+p.y,p.k);a=pieces[i].rotated(q.k)
                    mask=a[:,:,3]>0;h,w=mask.shape
                    reg=owner[q.y:q.y+h,q.x:q.x+w];free=mask&(reg<0)
                    reg[free]=i;canvas[q.y:q.y+h,q.x:q.x+w][free]=a[free]
                    core[q.y:q.y+h,q.x:q.x+w]|=binary_erosion(mask)
                    anchor[i]=q;poses[pieces[i].id]=q;changed+=1
                h,w=cache[k][1].shape
                bounds[:]=[min(bounds[0],x),min(bounds[1],y),max(bounds[2],x+w),max(bounds[3],y+h)]
            history.append(changed)
            if len(anchor)==len(pieces):break
            if not changed:
                if phase==1 and not relaxed:relaxed=True;continue
                break
    poses,final_extent=normalize_pose_set(pieces,poses)
    added=len(anchor)-initial
    return poses,{"anchor_added":added,"anchor_batches":history,"anchor_extent":list(final_extent),
                   "anchor_translation_slack":[H-final_extent[0],W-final_extent[1]]}

def _regions(owner):
    from scipy.ndimage import label,find_objects,binary_closing
    occ=owner>=0
    # Close thin (one-pixel) aliasing cracks before identifying empty regions.
    solid=binary_closing(np.pad(occ,2,constant_values=True),iterations=1)[2:-2,2:-2]
    lab,num=label(~solid)
    areas=np.bincount(lab.ravel(),minlength=num+1)
    regions=[];H,W=occ.shape
    for i,sl in enumerate(find_objects(lab),1):
        if sl is None or areas[i]<20:continue
        y,x=sl;y0=max(0,y.start-2);x0=max(0,x.start-2);y1=min(H,y.stop+2);x1=min(W,x.stop+2)
        regions.append((int(areas[i]),x0,y0,x1,y1))
    return regions

def _soft_options(canvas,owner,core,cache,regions,occ,contact_field,limit):
    from scipy.signal import fftconvolve
    hits={}
    for k,entry in cache.items():
        a,m,c,borders,gp=entry;h,w=m.shape;size=int(m.sum());perim=sum(len(z[0]) for z in borders)
        ker=m[::-1,::-1].astype(np.float32)
        # Most holes are small. Region-based correlations avoid a full-image
        # FFT per candidate, and skip impossible bounding boxes/areas.
        for area,x0,y0,x1,y1 in regions:
            if area<.85*size or x1-x0<w or y1-y0<h:continue
            if (x1-x0)*(y1-y0)>max(20000,70*size):continue
            ob=occ[y0:y1,x0:x1].astype(np.float32)
            err=fftconvolve(ob,ker,mode="valid")
            valid=err<=max(3,.012*size)+.3
            if not valid.any():continue
            count=fftconvolve(contact_field[y0:y1,x0:x1],ker,mode="valid")
            # Subtract interior occupied contribution for tolerant boundaries.
            metric=count-8*np.maximum(0,err);metric[~valid]=-1
            maximum=float(metric.max())
            if maximum<.20*perim:continue
            sub=np.flatnonzero((metric>=max(.20*perim,maximum-.12*perim)).ravel())
            if len(sub)>8:sub=sub[np.argpartition(metric.ravel()[sub],-8)[-8:]]
            for ix in sub:
                y,x=np.unravel_index(ix,metric.shape);key=(int(x+x0),int(y+y0),k)
                hits[key]=max(float(metric[y,x]),hits.get(key,0))
    # Only full contact verification decides; geometric score merely proposes.
    fits=[]
    for (x,y,k),_ in sorted(hits.items(),key=lambda f:-f[1])[:32]:
        f=_fit(canvas,owner,core,cache[k],x,y,limit)
        if f is None:continue
        score,frac,cost,touches,n,over=f
        if frac<.24 or cost>limit or (touches<2 and frac<.50):continue
        fits.append((score,x,y,k,frac,cost,touches,n))
    return sorted(fits,reverse=True)

def soft_pack(pieces,poses,groups,shape,rotations=True,rounds=3):
    """Soft empty-region correlation & whole-contact verification.
    
    This bounded fallback is restricted to unresolved masks on a fully
    anchored canvas. The anchor never moves. Region FFT is independent of
    the original partition generator and ignores hidden coordinates.
    """
    if len(poses)<.4*len(pieces) or len(poses)==len(pieces) or len(pieces)>2048 or shape[0]*shape[1]>16_000_000:return poses,{"soft_pack_added":0}
    _,extent=normalize_pose_set(pieces,poses)
    if tuple(extent)!=tuple(shape):return poses,{"soft_pack_added":0}
    poses=dict(poses);initial=len(poses);index={p.id:i for i,p in enumerate(pieces)}
    anchor={index[i]:p for i,p in poses.items()};canvas,owner,_=render(pieces,poses,shape)
    core=np.zeros(shape,bool)
    for i,p in anchor.items():
        m=pieces[i].rotated(p.k)[:,:,3]>0;h,w=m.shape
        core[p.y:p.y+h,p.x:p.x+w]|=binary_erosion(m)
    history=[];limit=_seam_limit(canvas,owner)
    for phase in (0,1):
        work=[{index[i]:p for i,p in g.items()} for g in groups if not set(g)&set(poses) and len(g)>1] if phase==0 else [{i:Pose(0,0)} for i in range(len(pieces)) if i not in anchor]
        work=[g for g in work if not set(g)&anchor.keys()]
        caches=[_group_cache(pieces,g,rotations) for g in work]
        for _ in range(rounds):
            regions=_regions(owner);proposals=[]
            occ=owner>=0
            pad=np.pad(occ.astype(np.float32),1,constant_values=1)
            contact_field=pad[1:-1,:-2]+pad[1:-1,2:]+pad[:-2,1:-1]+pad[2:,1:-1]
            for gi,cache in enumerate(caches):
                if cache is None or set(work[gi])&anchor.keys():continue
                f=_soft_options(canvas,owner,core,cache,regions,occ,contact_field,limit)
                if not f:continue
                score,x,y,k,frac,cost,touch,n=f[0]
                margin=score/max(f[1][0],1e-9) if len(f)>1 else 2.
                if (frac>=.52 and cost<=limit and margin>=1.08) or (frac>=.32 and touch>=2 and cost<=.86*limit and margin>=1.14):
                    proposals.append((score*min(1.5,margin),gi,x,y,k))
            changed=0
            for _,gi,x,y,k in sorted(proposals,reverse=True):
                cache=caches[gi]
                if set(work[gi])&anchor.keys():continue
                f=_fit(canvas,owner,core,cache[k],x,y,limit)
                if f is None:continue
                _,frac,cost,touch,n,over=f
                if frac<.24 or cost>limit:continue
                for i,p in cache[k][-1].items():
                    q=Pose(x+p.x,y+p.y,p.k);a=pieces[i].rotated(q.k);m=a[:,:,3]>0;h,w=m.shape
                    reg=owner[q.y:q.y+h,q.x:q.x+w];free=m&(reg<0);reg[free]=i
                    canvas[q.y:q.y+h,q.x:q.x+w][free]=a[free];core[q.y:q.y+h,q.x:q.x+w]|=binary_erosion(m)
                    anchor[i]=q;poses[pieces[i].id]=q;changed+=1
            history.append(changed)
            if not changed or len(poses)==len(pieces):break
    return poses,{"soft_pack_added":len(poses)-initial,"soft_pack_batches":history}
