"""Reference-free gap-tolerant matching of incomplete planar fragment sets.

Independent implementation inspired by multiscale contour registration, seam
extrapolation and loop consistency (see docs/REFERENCES.md).  Extrapolated
colour is *evidence only*: the renderer never paints inferred missing pixels.
All geometry is in the input's common pixel coordinate system.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

# Bounded on both desktop and WebAssembly. Exact contact remains the fast path.
MAX_GAP_PIECES = 128
MAX_SPARSE_GAP_PIECES = 512
MAX_PAIR_HYPOTHESES = 12
MAX_GLOBAL_MATCHES = 120_000


@dataclass
class Profile:
    points: np.ndarray
    tangents: np.ndarray
    inward: np.ndarray
    color: np.ndarray
    slope: np.ndarray
    texture: np.ndarray
    samples: np.ndarray
    gradient: np.ndarray
    step: float


def _profiles(shapes):
    """Masked, inward-facing samples; extrapolate RGB, never the silhouette."""
    profiles = []
    for sh in shapes:
        v = sh.points
        t = sh.tangent
        inward = np.column_stack([-t[:, 1], t[:, 0]])
        mask = sh.fragment.mask
        # Nearest valid foreground for off-mask interpolation at sharp corners.
        # The samples below still have a depth-validity weight.
        _, nearest = ndi.distance_transform_edt(~mask, return_indices=True)
        rgb = sh.fragment.rgba[:, :, :3][nearest[0], nearest[1]].astype(np.float32)
        depths = np.array([1.5, 4., 8., 12.], np.float32)
        coord = v[:, None, :] + inward[:, None, :] * depths[None, :, None]
        xy = [coord[:, :, 1] - .5, coord[:, :, 0] - .5]
        samples = np.stack([ndi.map_coordinates(rgb[..., k], xy, order=1, mode="nearest")
                            for k in range(3)], axis=-1)
        validity = ndi.map_coordinates(mask.astype(np.float32), xy, order=1, mode="constant") > .75
        # A robust short normal-line fit. At a narrow tip only the near sample
        # is trusted. Smooth the slope along the arc to curb extrapolation noise.
        slope = (samples[:, 2] - samples[:, 0]) / (depths[2]-depths[0])
        slope[~(validity[:, 0] & validity[:, 2])] = 0.
        slope = np.clip(ndi.gaussian_filter1d(slope, .8, axis=0, mode="wrap"), -6., 6.)
        base = np.clip(samples[:, 0] - depths[0]*slope, 0., 255.)
        base = ndi.gaussian_filter1d(base, .65, axis=0, mode="wrap")
        # Local contrast is important: matching two homogeneous pieces is weak
        # evidence irrespective of how close their colours happen to be.
        avg = ndi.uniform_filter1d(base, size=9, axis=0, mode="wrap")
        var = np.maximum(0., ndi.uniform_filter1d(base*base, 9, axis=0, mode="wrap")-avg*avg)
        texture = np.sqrt(var.mean(axis=1))
        # Gradient in the fragment coordinate system, smoothed over the
        # boundary so staircase noise does not dominate the continuation cue.
        along=(np.roll(base,-1,axis=0)-np.roll(base,1,axis=0))/(2*sh.step)
        grad=along[:,:,None]*t[:,None,:]+slope[:,:,None]*inward[:,None,:]
        profiles.append(Profile(v, t, inward, base, slope, texture, samples, grad, sh.step))
    return profiles


def _windows(profiles, half, reverse=False, extension=5.):
    """A multichannel, arc-length-normalized local boundary representation."""
    feats, ids, centers, directions, indices = [], [], [], [], []
    offsets = np.rint(np.linspace(-half, half, 9)).astype(int)
    if reverse:
        offsets = -offsets
    for i, p in enumerate(profiles):
        n = len(p.points)
        if n <= 2*half+6:
            continue
        # All anchors are covered; alternate samples at long perimeters.
        anchor = np.arange(0, n, 1 if n < 160 else 2)
        ix = (anchor[:, None]+offsets) % n
        v = p.points[ix]
        span = 2*half*p.step
        u = v[:, -1]-v[:, 0]
        d = np.linalg.norm(u, axis=1)
        good = d >= span*.46
        if not np.any(good):
            continue
        anchor, ix, v, u, d = anchor[good], ix[good], v[good], u[good], d[good]
        u /= np.maximum(d[:, None], 1.e-6)
        ortho = np.column_stack([-u[:, 1], u[:, 0]])
        rel = v-v[:, 4:5, :]
        x = np.sum(rel*u[:, None, :], axis=2)/span
        y = np.sum(rel*ortho[:, None, :], axis=2)/span
        col = np.clip(p.color[ix]-extension*p.slope[ix], 0., 255.)/255.
        demean = col-col.mean(axis=1, keepdims=True)
        # Geometry receives a lower weight than a distinctive colour profile:
        # after abrasion the two observed contours are not exact complements.
        feature = np.column_stack([.8*x[:, ::2], 1.8*y[:, ::2], .8*d/span,
                                   .8*col[:, ::2].reshape(len(v), -1),
                                   1.2*demean[:, ::2].reshape(len(v), -1)])
        feats.append(feature.astype(np.float32))
        ids.append(np.full(len(v), i, np.int32))
        centers.append(v[:, 4])
        directions.append(u)
        indices.append(anchor)
    if not feats:
        return None
    return tuple(np.concatenate(q) for q in (feats, ids, centers, directions, indices))


def _hypotheses(profiles, max_gap, update, *, sparse_degree=None):
    """Global nearest-neighbour retrieval followed by quantized SE(2) voting.

    A pose maps fragment j to fragment i.  No all-pairs raster correlation
    occurs; candidates are generated from distinctive short/long arcs.
    """
    chunks = []
    for half in (4, 8, 13):
        A = _windows(profiles, half, extension=max_gap*.18)
        B = _windows(profiles, half, True, extension=max_gap*.18)
        if A is None or B is None:
            continue
        fa, ia, pa, ua, sa = A
        fb, ib, pb, ub, sb = B
        k = min(32, len(fb))
        if k < 2:
            continue
        if sparse_degree is None:
            distance, nb = cKDTree(fb).query(fa, k=k)
            valid = (ia[:, None] != ib[nb]) & (distance < .70)
            a, slot = np.nonzero(valid)
            b = nb[a, slot]
            dis = distance[a, slot]
        else:
            # Coarse descriptor subset followed by full-distance verification.
            # Query batches keep temporary arrays independent of the total N.
            cols=[1,3,6,8,10,11,12,13,17,18,19,23,24,25,29,32,38]
            tree=cKDTree(fb[:,cols]);aa=[];bb=[];dd=[]
            for start in range(0,len(fa),512):
                f=fa[start:start+512]
                _,nb=tree.query(f[:,cols],k=k,eps=.2)
                distance=np.linalg.norm(f[:,None,:]-fb[nb],axis=2)
                row,slot=np.nonzero((ia[start:start+len(f),None]!=ib[nb])&(distance<.70))
                aa.append(row+start);bb.append(nb[row,slot]);dd.append(distance[row,slot])
            a,b,dis=map(np.concatenate,(aa,bb,dd))
        if not len(a):
            continue
        # At most one neighbour from the same piece at a particular anchor;
        # suppress identical votes rather than favoring a large perimeter.
        order = np.lexsort((dis, ib[b], a))
        a, b, dis = a[order], b[order], dis[order]
        first = np.r_[True, (a[1:] != a[:-1]) | (ib[b[1:]] != ib[b[:-1]])]
        a, b, dis = a[first], b[first], dis[first]
        if sparse_degree is not None:
            # Divvy the global vote budget among source fragments so large
            # perimeters and repetitive textures cannot starve small pieces.
            limit=max(24,MAX_GLOBAL_MATCHES//(3*len(profiles)))
            order=np.lexsort((dis,ia[a]))
            starts=np.r_[0,np.flatnonzero(ia[a[order]][1:]!=ia[a[order]][:-1])+1,len(order)]
            take=np.concatenate([order[l:min(r,l+limit)] for l,r in zip(starts[:-1],starts[1:])])
            a,b,dis=a[take],b[take],dis[take]
        elif len(a) > MAX_GLOBAL_MATCHES//3:
            # Deterministic confidence cut; complexity remains bounded.
            take = np.argsort(dis, kind="stable")[:MAX_GLOBAL_MATCHES//3]
            a, b, dis = a[take], b[take], dis[take]
        ta = np.arctan2(ua[a, 1], ua[a, 0])
        tb = np.arctan2(ub[b, 1], ub[b, 0])
        theta = ta-tb
        cs, sn = np.cos(theta), np.sin(theta)
        bp = pb[b]
        rotated = np.column_stack([cs*bp[:, 0]-sn*bp[:, 1], sn*bp[:, 0]+cs*bp[:, 1]])
        outward = np.column_stack([ua[a, 1], -ua[a, 0]])
        swap = ia[a] > ib[b]
        i, j = np.minimum(ia[a], ib[b]), np.maximum(ia[a], ib[b])
        for gap in np.unique(np.round(np.array([0., .21, .43, .68, 1.])*max_gap, 1)):
            t = pa[a]-rotated+outward*gap
            # Reverse direction if needed to canonicalize the pair.
            t[swap] = np.column_stack([-cs[swap]*t[swap, 0]-sn[swap]*t[swap, 1],
                                       sn[swap]*t[swap, 0]-cs[swap]*t[swap, 1]])
            angle = np.where(swap, -theta, theta)
            angle = (angle+np.pi)%(2*np.pi)-np.pi
            keys = np.column_stack([i, j, np.rint(angle/.14),
                                     np.rint(t[:, 0]/6.), np.rint(t[:, 1]/6.)]).astype(np.int32)
            # The values carry original, unquantized poses.
            chunks.append((keys, np.column_stack([angle, t, dis, np.full(len(a), half)])))
        update(f"Gap-aware contour retrieval (arc radius {half})")
    if not chunks:
        return []
    keys = np.concatenate([q[0] for q in chunks])
    vals = np.concatenate([q[1] for q in chunks])
    # Collapse votes by SE(2) bin; use the best representative for verification.
    order = np.lexsort((vals[:, 3], keys[:, 4], keys[:, 3], keys[:, 2], keys[:, 1], keys[:, 0]))
    keys, vals = keys[order], vals[order]
    first = np.flatnonzero(np.r_[True, np.any(keys[1:] != keys[:-1], axis=1)])
    count = np.diff(np.r_[first, len(keys)])
    key, val = keys[first], vals[first]
    # Reward repeated votes, while keeping some strong single arc matches.
    rank = (1.+np.log1p(count))*np.exp(-2.6*val[:, 3])*(1.+.018*val[:, 4])
    order = np.lexsort((-rank, key[:, 1], key[:, 0]))
    if sparse_degree is not None:
        # Keep strongest incident pair nominations in either direction.
        pairs=defaultdict(list);best={}
        for k in order:
            i,j=map(int,key[k,:2])
            if (i,j) not in best:best[i,j]=float(rank[k])
        for (i,j),v in best.items():
            pairs[i].append((v,j));pairs[j].append((v,i))
        allowed={(min(i,j),max(i,j)) for i,near in pairs.items()
                 for _,j in sorted(near,reverse=True)[:sparse_degree]}
        order=np.array([k for k in order if tuple(key[k,:2]) in allowed],dtype=int)
    out = []
    current = None
    kept = []
    # Allow alternative gap/angle candidates but deduplicate closely related bins.
    for k in order:
        pair = (int(key[k, 0]), int(key[k, 1]))
        if pair != current:
            current, kept = pair, []
        if len(kept) >= (6 if sparse_degree is not None else MAX_PAIR_HYPOTHESES):
            continue
        angle, tx, ty, dis, half = val[k]
        if any(abs(math.atan2(math.sin(angle-r), math.cos(angle-r))) < .10 and
               (tx-x)**2+(ty-y)**2 < 25. for r, x, y in kept):
            continue
        kept.append((angle, tx, ty))
        out.append((*pair, float(angle), np.array([tx, ty]), int(count[k]), float(dis)))
    return out


def _longest_run(good):
    n = len(good)
    if not n or not np.any(good):
        return np.empty(0, dtype=int)
    if np.all(good):
        return np.arange(n)
    # Bridge one missed point, not a large invisible portion of the seam.
    good = good | (np.roll(good, 1)&np.roll(good, -1))
    a = np.r_[good, good]
    br = np.flatnonzero(np.r_[True, a[1:] != a[:-1], True])
    start, length = max(((int(l), min(n, int(r-l))) for l, r in zip(br[:-1], br[1:]) if a[l]),
                        key=lambda q:q[1], default=(0,0))
    return (start+np.arange(length))%n


def _batch_contacts(a, b, pa, pb, hypotheses, max_gap):
    """Vectorized preliminary contact and colour scoring of a pair's poses."""
    from .scan import rotation, _overlap
    angles = np.array([h[0] for h in hypotheses])
    tt = np.stack([h[1] for h in hypotheses])
    rr = np.stack([rotation(angle) for angle in angles])
    v = np.einsum("kij,nj->kni", rr, pb.points)+tt[:, None, :]
    tangent = np.einsum("kij,nj->kni", rr, pb.tangents)
    dist, idx = a.tree.query(v.reshape(-1, 2))
    dist, idx = dist.reshape(v.shape[:2]), idx.reshape(v.shape[:2])
    diff = v-pa.points[idx]
    opposed = -np.einsum("kni,kni->kn", pa.tangents[idx], tangent)
    normal = -np.einsum("kni,kni->kn", pa.inward[idx], diff)
    # b must lie outside a; a small amount of local roughness is acceptable.
    good = (dist <= max_gap+3.) & (opposed > .68) & (normal >= -2.) & (normal >= .72*dist-2.)
    results = []
    for k, (angle, t, votes, dd) in enumerate(hypotheses):
        run = _longest_run(good[k])
        run = run[good[k, run]]
        length = len(run)*pb.step
        minimum = max(15., min(len(pa.points)*pa.step, len(pb.points)*pb.step)*.10)
        if length < minimum:
            continue
        ia = idx[k, run]
        # Nearest contour samples should actually span a seam, not one vertex.
        if len(np.unique(ia))*pa.step < length*.56:
            continue
        g = dist[k, run]
        median = float(np.median(g))
        dispersion = float(np.median(abs(g-median)))
        if dispersion > max(3.5, max_gap*.20):
            continue
        R = rr[k]
        overlap = _overlap(a, b, R, t)
        if overlap > .025:
            continue
        extension = g[:, None]*.5
        ca = np.clip(pa.color[ia]-extension*pa.slope[ia], 0, 255)
        cb = np.clip(pb.color[run]-extension*pb.slope[run], 0, 255)
        # Evaluate original and extrapolated profiles; neither alone can
        # establish a join. A robust penalty tolerates fading and a few defects.
        extrap = float(np.mean(np.minimum(np.mean(abs(ca-cb), axis=1), 60)))
        raw = float(np.mean(np.minimum(np.mean(abs(pa.color[ia]-pb.color[run]), axis=1), 60)))
        color = min(.75*extrap+.25*raw, raw+4.)
        # Bidirectional prediction into the *observed* interior, rather
        # than matching two hypothetical colours in the middle of the gap.
        # The normal estimate is deliberately local, avoiding a hallucinated
        # texture completion. Compare several depths, robust to color fading.
        def continuation(src, dst, dst_ids, T, u, tree):
            pts=dst.points[dst_ids,None,:]+dst.inward[dst_ids,None,:]*np.array([1.5,4.,8.])[None,:,None]
            xy=pts@T.T+u
            dd,nn=tree.query(xy.reshape(-1,2));dd=dd.reshape(xy.shape[:2]);nn=nn.reshape(xy.shape[:2])
            delta=xy-src.points[nn]
            # Use signed outward distance; this also tolerates nonparallel
            # rounded edges with locally varying erosion.
            depth=-np.sum(delta*src.inward[nn],axis=2)
            valid=(depth>=-1.)&(dd<=max_gap+10.)&(depth>=.55*dd-2.)
            if not np.any(valid):return 45.
            expected=np.clip(src.color[nn]-depth[...,None]*src.slope[nn],0,255)
            actual=dst.samples[dst_ids,:3,:]
            err=np.mean(abs(expected-actual),axis=2)
            return float(np.mean(np.minimum(err[valid],60)))
        patch=.5*(continuation(pa,pb,run,R,t,a.tree)+
                  continuation(pb,pa,ia,R.T,-R.T@t,b.tree))
        ga=pa.gradient[ia];gb=pb.gradient[run]@R.T
        an=np.sqrt(np.sum(ga*ga,axis=(1,2)));bn=np.sqrt(np.sum(gb*gb,axis=(1,2)))
        weight=np.minimum(an,bn)
        cosine=np.sum(ga*gb,axis=(1,2))/np.maximum(an*bn,1e-6)
        grad=float(np.sum(weight*cosine)/max(1.,weight.sum()))
        grad_strength=float(np.mean(weight))

        # Compare the variation along the matched arc, not just absolute hue.
        ac, bc = ca-ca.mean(axis=0), cb-cb.mean(axis=0)
        texture = float(min(np.sqrt(np.mean(ac*ac)), np.sqrt(np.mean(bc*bc))))
        variation = float(np.mean(abs(ac-bc)))
        # Very short homogeneous arcs are non-identifying. A long arc must
        # have either pictorial variation or a distinctive shape.
        arc = pb.points[run]
        chord = arc[-1]-arc[0]
        bend = float(np.sqrt(np.mean(((arc-arc[0])@np.array([-chord[1], chord[0]])/
                                      max(np.linalg.norm(chord), 1.e-6))**2)))
        if color > 30. or patch>34. or (texture < 4. and bend < 3.):
            continue
        # High contrast must be *explained* by the other fragment. Low
        # contrast round arcs are poor identifiers even with a long contact.
        reliability = (.10+.90*min(1.,texture/25.))*(.65+.35*min(1.,bend/8.+texture/25.))
        # Contrast-normalized continuation. Real gradients across the gap
        # should align as vectors after rotating the fragment.
        fit = math.exp(-color/24.-variation/24.-patch/24.-dispersion/7.
                       -median/max(18.,max_gap*3.) - max(0., .55-grad)*min(1.2,grad_strength/3.))
        score = length*fit*reliability*float(np.mean(opposed[k, run])) + min(votes,8)*.23
        if score < 4.5:
            continue
        info = dict(contact_length=float(length), median_gap=median, gap_dispersion=dispersion,
                    interior_overlap=float(overlap), color_difference=color,
                    texture=texture, profile_residual=variation, seam_bend_rms=bend,
                    continuation_error=patch, gradient_agreement=grad, gradient_strength=grad_strength,
                    votes=int(votes), descriptor_distance=float(dd),
                    arc_a=int(ia[len(ia)//2]), arc_b=int(run[len(run)//2]),
                    arc_fraction_a=float(length/(len(pa.points)*pa.step)),
                    arc_fraction_b=float(length/(len(pb.points)*pb.step)))
        results.append((score, R, t, info))
    return results


def _consistent_candidates(shapes, candidates):
    """Competing-seam confidence plus three-edge SE(2) cycle consistency."""
    from .scan import _overlap
    pairs = defaultdict(list)
    incident = defaultdict(list)
    for z, (score, i, j, R, t, info) in enumerate(candidates):
        pairs[i, j].append(z)
        incident[i].append((z, info["arc_a"], info["arc_fraction_a"]))
        incident[j].append((z, info["arc_b"], info["arc_fraction_b"]))
    # Limit alternative hypotheses per edge and score their SE(2) agreement.
    for key in pairs:
        pairs[key] = sorted(pairs[key], key=lambda z:-candidates[z][0])[:3]
    # The expensive loop check is performed on a bounded sparse graph.
    # All retained scores still enter the competing-seam confidence measure.
    nominations=defaultdict(list)
    for (i,j),ids in pairs.items():
        score=candidates[ids[0]][0]
        nominations[i].append((score,j));nominations[j].append((score,i))
    active_pairs={(min(i,j),max(i,j)) for i,near in nominations.items()
                  for _,j in sorted(near,reverse=True)[:12]}
    adjacency = defaultdict(set)
    for i, j in active_pairs:
        adjacency[i].add(j);adjacency[j].add(i)
    def transforms(i, j):
        for z in pairs.get((min(i,j),max(i,j)), ()):
            _, a, b, R, t, _ = candidates[z]
            yield (z, R, t) if a == i else (z, R.T, -R.T@t)
    cycles = np.zeros(len(candidates), dtype=np.int32)
    for i, j in sorted(active_pairs):
        for k in adjacency[i]&adjacency[j]:
            if not i < j < k:
                continue
            for za, A, a in transforms(i, j):
                for zb, B, b in transforms(j, k):
                    C, c = A@B, A@b+a
                    for zc, D, d in transforms(i, k):
                        # Use a center, not the crop origin, to measure disagreement.
                        center = shapes[k].points.mean(axis=0)
                        delta = np.linalg.norm((C-D)@center+c-d)
                        angle = math.acos(float(np.clip(np.trace(C.T@D)/2, -1, 1)))
                        if delta <= max(5., .05*np.sqrt(shapes[k].fragment.mask.sum())) and angle < .13:
                            # A nominally closed triangle must not overlap internally.
                            if _overlap(shapes[i],shapes[k],C,c) <= .025:
                                cycles[[za,zb,zc]] += 1
    output = []
    for z, (score, i,j,R,t, info) in enumerate(candidates):
        if z not in pairs[i,j]:
            continue
        ratios = []
        for u, arc, fraction in ((i,info["arc_a"],info["arc_fraction_a"]),(j,info["arc_b"],info["arc_fraction_b"])):
            n = len(shapes[u].points)
            competitors = []
            for other, pos, frac in incident[u]:
                if other == z:
                    continue
                _, oi, oj, oR, ot, _ = candidates[other]
                # Treat adjacent bins for the same mating as the same proposal.
                if (oi,oj)==(i,j) and np.linalg.norm(oR-R)<.19 and np.linalg.norm(ot-t)<10:
                    continue
                delta = abs(pos-arc); delta = min(delta,n-delta)
                if delta <= n*max(.05,min(.25,(fraction+frac)*.36)):
                    competitors.append(candidates[other][0])
            ratios.append(score/max(1., max(competitors,default=1.)))
        uniqueness = float(min(ratios))
        support = int(cycles[z])
        # A textured, distinctive seam can bootstrap a component. Ambiguous
        # seams require independent loop support. Do not force all fragments.
        if support == 0 and (uniqueness < 1.10 or score < 6. or info["texture"] < 9. or info["gradient_agreement"] < .25):
            continue
        weight = score*(1.+.6*min(3,support))*min(1.5,max(.5,uniqueness))
        output.append((weight,i,j,R,t,{**info,"gap_score":float(score),
                                      "seam_uniqueness":uniqueness,"loop_support":support}))
    return output


def gap_candidates(shapes, update=lambda _:None, *, sparse_degree=None):
    """Return independently corroborated hypotheses and search diagnostics."""
    if len(shapes) < 2 or len(shapes) > (MAX_SPARSE_GAP_PIECES if sparse_degree is not None else MAX_GAP_PIECES):
        return [], {"reason":"piece budget", "evaluated":0}
    profiles = _profiles(shapes)
    median_side = float(np.median([min(sh.fragment.mask.shape) for sh in shapes]))
    if sum(float(np.percentile(p.texture,80))>=4. for p in profiles)<2:
        return [], {"reason":"insufficient pictorial texture","evaluated":0}
    max_gap = min(36., max(5., median_side*.35))
    hyps = _hypotheses(profiles,max_gap,update,sparse_degree=sparse_degree)
    pairs = defaultdict(list)
    for i,j,angle,t,votes,dist in hyps:
        pairs[i,j].append((angle,t,votes,dist))
    candidates = []
    for count, ((i,j), hypothesis) in enumerate(pairs.items()):
        if count % max(1,len(pairs)//12)==0:
            update(f"Verifying gap-aware matches {count+1}/{len(pairs)}")
        matches = _batch_contacts(shapes[i],shapes[j],profiles[i],profiles[j],hypothesis,max_gap)
        for c in sorted(matches,key=lambda q:-q[0])[:3]:
            candidates.append((c[0],i,j,*c[1:]))
    # A short search around the best discovered transforms refines the
    # quantized proposal. Rotate about the moving fragment's center.
    from .scan import rotation
    refined=[]
    refine_count=64 if sparse_degree is not None else 160
    for z in sorted(candidates,key=lambda q:-q[0])[:refine_count]:
        score,i,j,R,t,info=z
        center=profiles[j].points.mean(0)
        angle=float(np.arctan2(R[1,0],R[0,0]))
        norm=-profiles[i].inward[info["arc_a"]]
        tangent=profiles[i].tangents[info["arc_a"]]
        offsets=[np.zeros(2),2*norm,-2*norm,4*norm,-4*norm,2*tangent,-2*tangent]
        proposals=[]
        for da in (0.,-.035,.035):
            Q=rotation(angle+da)
            u=t+(R-Q)@center
            for offset in offsets if da==0. else offsets[:3]:
                proposals.append((angle+da,u+offset,info["votes"],info["descriptor_distance"]))
        matches=_batch_contacts(shapes[i],shapes[j],profiles[i],profiles[j],proposals,max_gap)
        if matches:
            best=max(matches,key=lambda q:q[0])
            if best[0] > score*1.02:
                refined.append((best[0],i,j,*best[1:]))
                continue
        refined.append(z)
    # Keep the lower-ranked geometrically valid alternatives for confidence
    # calibration, without counting a refinement twice.
    top={id(z) for z in sorted(candidates,key=lambda q:-q[0])[:refine_count]}
    candidates=[z for z in candidates if id(z) not in top]+refined
    accepted = _consistent_candidates(shapes,candidates)
    return accepted, {"sparse_degree":sparse_degree,"gap_limit_px":float(max_gap),"retrieved_pairs":len(pairs),
                      "evaluated":len(hyps),"verified_hypotheses":len(candidates),
                      "corroborated_hypotheses":len(accepted),
                      "cycle_supported":sum(c[-1]["loop_support"]>0 for c in accepted)}


def rank_components(shapes, groups, joins):
    """Rank intact components by confidence, not arbitrary global placement.

    A weak bridge must not earn credit merely for connecting more fragments.
    """
    def rank(g):
        members={shapes[i].fragment.id for i in g}
        edges=[q for q in joins if q["a"] in members and q["b"] in members]
        if not edges:
            return (0.,0.,len(g))
        # The geometric mean penalizes weak bridges; a corroborated loop
        # provides a small bonus. Larger high-quality groups remain preferred.
        scores=np.array([max(1.e-6,e["score"]) for e in edges])
        confidence=float(np.exp(np.mean(np.log(scores))))
        return (confidence*(1.+.08*math.log2(len(g))),float(np.min(scores)),len(g))
    return sorted(groups,key=rank,reverse=True)


def refine_components(shapes, groups, candidates):
    """Refine locally consistent loops with robust SE(2) least squares.

    Tree-only and disconnected groups have no redundant observations and
    cannot be improved by pose-graph optimization; leave them unchanged.
    """
    from scipy.optimize import least_squares
    from .scan import rotation, _overlap
    result=[]
    for g in groups:
        if len(g)<3 or len(g)>64:
            result.append(g);continue
        keys=sorted(g);root=keys[0];free=keys[1:];ix={k:j for j,k in enumerate(free)}
        constraints={}
        for score,i,j,R,t,info in candidates:
            if i not in g or j not in g:continue
            A,a=g[i];B,b=g[j]
            center=shapes[j].points.mean(0)
            err=np.linalg.norm((A@R-B)@center+A@t+a-b)
            diff=math.acos(float(np.clip(np.trace((A@R).T@B)/2,-1,1)))
            if err>8. or diff>.15:continue
            key=(i,j)
            if key not in constraints or score>constraints[key][0]:
                constraints[key]=(score,R,t)
        if len(constraints)<len(g):
            result.append(g);continue
        def angle(R):return math.atan2(R[1,0],R[0,0])
        x0=np.array([q for k in free for q in (angle(g[k][0]),*g[k][1])])
        def poses(x):
            return {root:g[root],**{k:(rotation(x[3*v]),x[3*v+1:3*v+3]) for k,v in ix.items()}}
        def residual(x):
            q=poses(x);r=[]
            for (i,j),(score,R,t) in constraints.items():
                A,a=q[i];B,b=q[j]
                center=shapes[j].points.mean(0)
                d=(A@R-B)@center+A@t+a-b
                da=angle((A@R).T@B)
                weight=math.sqrt(min(2.,score/10.))
                r.extend([weight*d[0],weight*d[1],weight*30.*da])
            return np.array(r)
        fit=least_squares(residual,x0,loss="soft_l1",f_scale=3.,max_nfev=25)
        if not fit.success or np.linalg.norm(residual(fit.x))>=np.linalg.norm(residual(x0)):
            result.append(g);continue
        q=poses(fit.x)
        collision=any(_overlap(shapes[i],shapes[j],q[i][0].T@q[j][0],q[i][0].T@(q[j][1]-q[i][1]))>.035
                      for a,i in enumerate(keys) for j in keys[a+1:])
        result.append(g if collision else q)
    return result
