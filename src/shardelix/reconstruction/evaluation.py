"""Ground truth belongs ONLY to evaluation. Metrics penalize missing fragments."""
from __future__ import annotations

import numpy as np

from shardelix.reconstruction.common import Fragment, Pose, render
from shardelix.reconstruction.irregular.lattice import R


def adjacency(labels):
    edges=set()
    for a,b in ((labels[:,:-1],labels[:,1:]),(labels[:-1,:],labels[1:,:])):
        m=(a!=b)&(a>=0)&(b>=0)
        if m.any():
            pairs=np.sort(np.column_stack((a[m],b[m])),axis=1)
            edges.update(map(tuple,np.unique(pairs,axis=0)))
    return edges


def gauge_align(pieces,poses,truth,allow_global_rotation):
    """Align only a SINGLE global rigid gauge, using truth in evaluator only."""
    if not poses:return poses
    best=(-1,None)
    pd={p.id:p for p in pieces}
    from collections import Counter
    for r in range(4 if allow_global_rotation else 1):
        rotated={}
        for id,p in poses.items():
            a=pd[id].rotated(p.k);h,w=a.shape[:2]
            corners=np.array([[p.x,p.y],[p.x+w,p.y],[p.x,p.y+h],[p.x+w,p.y+h]])
            xy=(corners@R[r].T).min(axis=0)
            rotated[id]=Pose(int(xy[0]),int(xy[1]),(p.k+r)%4)
        votes=Counter((truth[id].x-p.x,truth[id].y-p.y) for id,p in rotated.items() if id in truth and truth[id].k==p.k)
        if not votes:continue
        (dx,dy),score=votes.most_common(1)[0]
        if score>best[0]:
            best=(score,{id:Pose(p.x+dx,p.y+dy,p.k) for id,p in rotated.items()})
    return best[1] if best[1] is not None else poses


def canvas_align(pieces,poses,truth,shape,allow_global_rotation,*,return_rotation=False):
    """Align only rotations of the known full canvas, never a free translation."""
    if not allow_global_rotation:return (poses,0) if return_rotation else poses
    lookup={p.id:p for p in pieces};H,W=shape
    best=(-1,poses,0)
    for r in range(4 if H==W else 2):
        k=r if H==W else r*2
        rot=R[k]
        origin=(np.array([[0,0],[W,0],[0,H],[W,H]])@rot.T).min(axis=0)
        out={}
        for id,p in poses.items():
            h,w=lookup[id].rotated(p.k).shape[:2]
            corners=np.array([[p.x,p.y],[p.x+w,p.y],[p.x,p.y+h],[p.x+w,p.y+h]])
            xy=(corners@rot.T).min(axis=0)-origin
            out[id]=Pose(int(xy[0]),int(xy[1]),(p.k+k)%4)
        score=sum(truth.get(i)==p for i,p in out.items())
        if score>best[0]:best=(score,out,k)
    return (best[1],best[2]) if return_rotation else best[1]


def evaluate(pieces: list[Fragment],poses:dict[str,Pose],truth:dict[str,Pose],
             shape:tuple[int,int],allow_global_rotation:bool=False,allow_translation:bool=False) -> dict:
    raw,Lraw,raw_collision=render(pieces,poses,shape)
    gauge=gauge_align(pieces,poses,truth,allow_global_rotation)
    aligned,k=canvas_align(pieces,poses,truth,shape,allow_global_rotation,return_rotation=True)
    if allow_translation:aligned=gauge
    pred,Lp,coll=render(pieces,aligned,shape)
    orig,Lt,_=render(pieces,truth,shape)
    ap,at=adjacency(Lp),adjacency(Lt)
    tp=len(ap&at);precision=tp/max(1,len(ap));recall=tp/max(1,len(at))
    f1=2*precision*recall/(precision+recall) if precision+recall else float(len(at)==0 and len(ap)==0)
    ids={p.id:i for i,p in enumerate(pieces)}
    correct={ids[i] for i,p in aligned.items() if truth.get(i)==p}
    graph={i:set() for i in correct}
    for a,b in ap&at:
        if a in correct and b in correct:
            graph[a].add(b);graph[b].add(a)
    largest=0;seen=set()
    for i in correct:
        if i in seen:continue
        todo=[i];seen.add(i);size=0
        while todo:
            j=todo.pop();size+=1
            for q in graph[j]-seen:
                seen.add(q);todo.append(q)
        largest=max(largest,size)
    n=len(truth)
    raw_dpa=sum(truth.get(i)==p for i,p in poses.items())/n
    gauge_dpa=sum(truth.get(i)==p for i,p in gauge.items())/n
    return {"piece_count":n,"placed":len(poses),"direct_position_accuracy":len(correct)/n,
            "raw_position_accuracy":raw_dpa,"gauge_aligned_position_accuracy":gauge_dpa,
            "evaluation_rotation_cw_deg":None if allow_translation else k*90,
            "adjacency_precision":precision,"adjacency_recall":recall,"adjacency_f1":f1,
             "largest_correct_component":largest/n,"coverage":float(np.mean(Lraw>=0)),
            "aligned_coverage":float(np.mean(Lp>=0)),
            "raw_pixel_accuracy":float(np.mean(np.all(raw==orig,axis=-1))),
            "pixel_accuracy":float(np.mean(np.all(pred==orig,axis=-1))),
            "collision_pixels":raw_collision,
            "pass_95":len(correct)/n>.95 and f1>.95 and raw_collision==0 and float(np.mean(Lraw>=0))>.95}


