"""Shared blind canvas orientation and observed-image objective."""
from __future__ import annotations
import numpy as np
from shardelix.reconstruction.common import Pose,render

def orient_to_canvas(pieces,poses,shape):
    """Choose a canvas-fitting global orientation without ground truth."""
    if not poses:return poses
    from shardelix.reconstruction.irregular.lattice import R
    lookup={p.id:p for p in pieces}
    candidates=[]
    for r in range(4):
        rotated={}
        for key,pose in poses.items():
            h,w=lookup[key].rotated(pose.k).shape[:2]
            corners=np.array([[pose.x,pose.y],[pose.x+w,pose.y],[pose.x,pose.y+h],[pose.x+w,pose.y+h]])
            xy=(corners@R[r].T).min(axis=0)
            rotated[key]=Pose(int(xy[0]),int(xy[1]),(pose.k+r)%4)
        from shardelix.reconstruction.common import normalize_pose_set
        normalized,extent=normalize_pose_set(pieces,rotated)
        if extent[0]<=shape[0] and extent[1]<=shape[1]:
            candidates.append((abs(shape[0]-extent[0])+abs(shape[1]-extent[1]),r,normalized))
    return min(candidates,key=lambda a:a[:2])[2] if candidates else poses


def objective(pieces,poses,shape):
    """No reference or truth: reward coverage, punish collisions and rough seams."""
    image,labels,collision=render(pieces,poses,shape)
    coverage=np.mean(labels>=0)
    m1=(labels[:,:-1]!=labels[:,1:])&(labels[:,:-1]>=0)&(labels[:,1:]>=0)
    m2=(labels[:-1,:]!=labels[1:,:])&(labels[:-1,:]>=0)&(labels[1:,:]>=0)
    im=image[:,:,:3].astype(np.int16)
    a=np.abs(im[:,:-1]-im[:,1:])[m1]
    b=np.abs(im[:-1,:]-im[1:,:])[m2]
    seam=(a.sum()+b.sum())/(255*max(1,a.size+b.size))
    return float(coverage-2*collision/labels.size-.1*seam)
