"""Blind anchoring of irregular components to the visible canvas frame."""
from __future__ import annotations
import numpy as np
from shardelix.reconstruction.common import Pose, normalize_pose_set

def _run_length(mask):
    a=np.flatnonzero(np.diff(np.r_[False,mask,False].astype(np.int8)))
    return int(np.max(a[1::2]-a[::2],initial=0))

def anchor_irregular_frame(pieces,poses,shape):
    """Resolve a translation axis only with independent flat-wall evidence.

    The existing relative geometry is never altered. Multiple flat pieces on
    exactly one outer extreme support an outer wall; contradictory/weak evidence
    leaves the relative gauge explicitly unresolved.
    """
    if not poses:return poses,{"frame_anchor_status":"empty","anchor_translation_slack":list(shape)}
    p,extent=normalize_pose_set(pieces,poses)
    slack=[shape[i]-extent[i] for i in range(2)]
    if min(slack)<0:return poses,{"frame_anchor_status":"outside-canvas"}
    lookup={q.id:q for q in pieces}
    support=[[],[],[],[]] # top, bottom, left, right
    for ident,q in p.items():
        m=lookup[ident].rotated(q.k)[:,:,3]>0;h,w=m.shape
        for j,b,side in [(0,q.y==0,m[0]),(1,q.y+h==extent[0],m[-1]),(2,q.x==0,m[:,0]),(3,q.x+w==extent[1],m[:,-1])]:
            n=_run_length(side)
            # Full, straight edge, not a short tangent to a rounded curve.
            if b and n>=max(10,.86*len(side)):support[j].append(n)
    offset=[0,0];remaining=slack.copy();decisions=[]
    for a in range(2):
        if not slack[a]:decisions.append("full-extent");continue
        u,v=support[2*a:2*a+2]
        # Two independent fragments and convincing accumulated wall support.
        threshold=max(20,.18*extent[1-a])
        if len(u)>=2 and sum(u)>=threshold and not v:
            remaining[a]=0;decisions.append("leading-wall")
        elif len(v)>=2 and sum(v)>=threshold and not u:
            offset[a]=slack[a];remaining[a]=0;decisions.append("trailing-wall")
        else:decisions.append("unresolved")
    # Preserve the original translation along axes without evidence, rather
    # than silently pretending a relative origin is an absolute anchor.
    ymin=min(q.y for q in poses.values());xmin=min(q.x for q in poses.values())
    for a,base in enumerate((ymin,xmin)):
        if remaining[a]:offset[a]=max(0,min(slack[a],base))
    out={i:Pose(q.x+offset[1],q.y+offset[0],q.k) for i,q in p.items()}
    return out,{"frame_anchor_status":"resolved" if not any(remaining) else "ambiguous",
                "frame_anchor_decisions_yx":decisions,"frame_anchor_support":[len(x) for x in support],
                "frame_anchor_offset_yx":offset,"anchor_translation_slack":remaining}
