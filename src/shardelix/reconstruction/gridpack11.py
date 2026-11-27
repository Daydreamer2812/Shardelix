"""Conservative component placement with multiple independent contacts.

Sparse edge graph only; no dense pair matrix or inferred missing locations.
"""
from collections import Counter
import numpy as np

def pack_grid_components(pieces,shape,grid,rotations,group,occupied,components,rows,margin,rounds=3):
    from .sparse_grid import rotate,D
    rmax,cmax=grid;ph,pw=shape[0]//rmax,shape[1]//cmax
    dims=[p.rgba.shape[:2] for p in pieces]
    used=set(group);initial=len(used);history=[]
    # Limit work on extremely ambiguous large puzzles. Singletons have
    # already passed the frontiers; require a multi-piece rigid component.
    left=[{q:v for q,v in comp.items() if q not in used} for comp in components]
    left=[g for g in left if len(g)>=2]
    if len(left)>4096:return 0,[]
    def transform(g,rr,tx,ty):
        moved={}
        for q,(x,y,k) in g.items():
            a,b=rotate(x,y,rr);a+=tx;b+=ty;z=(k+rr)%4
            if not(0<=a<cmax and 0<=b<rmax) or (a,b) in occupied:return None
            if (dims[q][::-1] if z%2 else dims[q])!=(ph,pw):return None
            moved[q]=(a,b,z)
        return moved
    def quality(moved):
        cross=good=strong=rank_sum=0;others=set();from_p=set()
        for q,(x,y,k) in moved.items():
            for d,(dx,dy) in enumerate(D):
                j=occupied.get((x+dx,y+dy))
                if j is None:continue
                cross+=1;others.add(j);from_p.add(q)
                a=4*q+(d-k)%4;b=4*j+(d+2-group[j][2])%4
                ra=rows[a].get(b);rb=rows[b].get(a)
                if ra is None and rb is None:continue
                good+=1
                if ra is not None and rb is not None:
                    if ra[0]==rb[0]==0 and min(margin[a],margin[b])>.05:strong+=1
                    rank_sum+=ra[0]+rb[0]
                else:rank_sum+=5
        if cross<2 or len(others)<2 or len(from_p)<2 or good<.85*cross or strong<1:return None
        # Score rewards independent contacts and punishes unsupported ones.
        return good+strong*.8-(cross-good)*4-rank_sum*.10
    for _ in range(rounds):
        candidates=[]
        for g in left:
            if any(q in used for q in g):continue
            votes=Counter()
            for q,(x,y,k) in g.items():
                for side in range(4):
                    for b,(rank,cost) in rows[4*q+side].items():
                        j,t=divmod(b,4)
                        if j not in used or rank>2:continue
                        xx,yy,jk=group[j];d=(t+jk+2)%4;z=(d-side)%4;rr=(z-k)%4
                        if rr and not rotations:continue
                        a0,b0=rotate(x,y,rr);dx,dy=D[d]
                        votes[(rr,xx-dx-a0,yy-dy-b0)]+=1
            fits=[]
            for (rr,x,y),v in votes.most_common(16):
                if v<2:continue
                m=transform(g,rr,x,y)
                if m is None:continue
                q=quality(m)
                if q is not None:fits.append((q,rr,x,y))
            if not fits:continue
            fits.sort(reverse=True);best=fits[0]
            if len(fits)>1 and best[0]<fits[1][0]+.9:continue
            candidates.append((best[0],len(g),best[1:],g))
        added=0
        for _,_,(rr,x,y),g in sorted(candidates,key=lambda a:(-a[0],-a[1])):
            if any(q in used for q in g):continue
            m=transform(g,rr,x,y)
            if m is None or quality(m) is None:continue
            for q,p in m.items():group[q]=p;occupied[p[:2]]=q;used.add(q)
            added+=len(m)
        history.append(added)
        if not added:break
    return len(used)-initial,history
