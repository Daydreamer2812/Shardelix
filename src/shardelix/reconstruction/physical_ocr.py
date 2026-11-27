"""Optional local English OCR for scanned strip seams, without labels/reference.

Only confidence-weighted words crossing a proposed seam contribute, never
filename order or text in order.txt. OCR is evidence, not verified accuracy.
"""
from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from .common import Fragment, Pose, render, save_rgba, json_write
from .document import chain
from .overlay import write_cut_outputs
from .physical_document import _ink, _profile, _cost


def _clean(im):
    """Remove scan-column shading using nearly blank interior rows."""
    center = im[:, im.shape[1]//4:max(1, 3*im.shape[1]//4)]
    blank = center.mean(1) < .03
    baseline = (np.percentile(im[blank], 85, axis=0) if blank.sum() > 20
                else np.percentile(im, 30, axis=0))
    return ((1-np.clip((im-baseline)/(1-baseline+.01), 0, 1))*255).astype(np.uint8)


def seam_scores(tsv, spans):
    """Return one scalar per pair; do not keep recognized text in reports."""
    scores = np.zeros(len(spans))
    for line in tsv.splitlines():
        f = line.split("\t", 11)
        if len(f) != 12 or f[0] != "5":
            continue
        try:
            x, w, conf = int(f[6]), int(f[8]), float(f[10])
        except ValueError:
            continue
        word = f[11].strip()
        if not np.isfinite(conf) or conf < 40 or sum(c.isalpha() for c in word) < 2:
            continue
        for i, (start, seam, end) in enumerate(spans):
            if start <= x and x+w <= end and x+2 < seam < x+w-2:
                scores[i] += (min(100, conf)-40)**1.5/(60**1.5)*min(7, len(word))
                break
    return scores.tolist()


def _read(out):
    p = json.loads((out/"placements.json").read_text())
    with zipfile.ZipFile(out/"fragments.zip") as z:
        pieces = [Fragment(Path(name).stem, np.array(Image.open(io.BytesIO(z.read(name))).convert("RGBA")))
                  for name in sorted(z.namelist()) if name.startswith("pieces/") and name.endswith(".png")]
    signs = np.array([p["poses"][q.id]["k"] == 2 for q in pieces])
    return p, pieces, signs


def _matrix(images):
    h = min(1800, max(map(len, images)))
    L, R = [np.array([_profile(im, right, h) for im in images]).transpose(1, 0, 2)
            for right in (False, True)]
    costs = _cost(R, L)
    for d in range(1, 5):
        for a, b in ((R[:, :, d:], L[:, :, :-d]), (R[:, :, :-d], L[:, :, d:])):
            costs = np.minimum(costs, _cost(a, b)+d*.012)
    np.fill_diagonal(costs, np.inf)
    l, r = L[1].mean(1), R[1].mean(1)
    s = max(.02, float(np.median(np.r_[l, r])))
    return costs, 3*l/s, 3*r/s


def _shortlist(costs, cap=900):
    # Rank both directions so ambiguous successors do not starve a strip.
    n = len(costs)
    ranks = np.minimum(np.argsort(np.argsort(costs, axis=0), axis=0),
                       np.argsort(np.argsort(costs, axis=1), axis=1))
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j and ranks[i, j] < 12]
    pairs.sort(key=lambda p: (ranks[p], costs[p], p))
    return sorted(pairs[:cap])


def _tile(images, pairs, path):
    h = max(map(len, images))
    width = sum(images[i].shape[1]+images[j].shape[1] for i, j in pairs)+40*(len(pairs)+1)
    canvas = np.full((h+20, width), 255, np.uint8)
    spans = []
    x = 20
    for i, j in pairs:
        a, b = images[i], images[j]
        seam = x+a.shape[1]
        end = seam+b.shape[1]
        canvas[10:10+len(a), x:seam] = a
        canvas[10:10+len(b), seam:end] = b
        spans.append([x, seam, end])
        x = end+40
    Image.fromarray(canvas).save(path)
    return spans


def prepare(out, work):
    """Build bounded OCR candidate batches from an already-rendered blind result."""
    out, work = Path(out), Path(work)
    diag = json.loads((out/"diagnostics.json").read_text())
    if diag.get("scan", {}).get("kind") != "document-strips":
        return None
    _, pieces, signs = _read(out)
    n = len(pieces)
    if n < 5 or min(p.rgba.shape[0] for p in pieces) < 250:
        return None
    images = [_ink(p) for p in pieces]
    if not .008 <= np.median([im.mean() for im in images]) <= .45:
        return None
    work.mkdir(parents=True, exist_ok=True)
    images = [np.rot90(im, 2) if signs[i] else im for i, im in enumerate(images)]
    base, left, right = _matrix(images)
    clean = [_clean(im) for im in images]
    scale = min(1., 1800/max(map(len, clean)))
    if scale < 1:
        clean = [np.array(Image.fromarray(im).resize((max(1, round(im.shape[1]*scale)),
                   max(1, round(im.shape[0]*scale))), Image.Resampling.LANCZOS)) for im in clean]
    # If relative line projections split the document into two sizeable sign
    # groups, check each group independently: different page-line phases can
    # otherwise make one upright page look upside down relative to another.
    groups = [np.flatnonzero(signs == v).tolist() for v in (False, True)]
    if min(map(len, groups)) < 5:
        groups = [list(range(n))]
    orientation = []
    for indices in groups:
        sub = np.ix_(indices, indices)
        order = np.array(indices)[chain(base[sub], .5, borders=(left[indices], right[indices]))]
        pairs = [(int(i), int(j)) for i, j in zip(order, order[1:])]
        chosen = sorted(pairs, key=lambda ij: base[ij])[:min(8, len(pairs))]
        # Cap orientation canvas just as with candidate tiles.
        while len(chosen)>2 and (max(map(len, clean))+20)*(
                sum(clean[i].shape[1]+clean[j].shape[1]+40 for i,j in chosen)+40)>4_000_000:
            chosen.pop()
        for flip in (False, True):
            ims = [np.rot90(im, 2) for im in clean] if flip else clean
            pp = [(j, i) for i, j in chosen] if flip else chosen
            path = work/f"orientation-{len(orientation)}.png"
            spans = _tile(ims, pp, path)
            orientation.append({"path": str(path), "spans": spans})
    json_write(work/"orientation-groups.json", groups)
    np.savez(work/"base.npz", base=base, left=left, right=right, signs=signs)
    # Individual arrays avoid a pickled object array.
    for i, im in enumerate(clean):
        Image.fromarray(im).save(work/f"strip-{i}.png")
    return {"orientation": orientation, "pieces": n}


def batches(work, orientation_scores):
    """Orient globally only when OCR has decisive asymmetric evidence."""
    work = Path(work)
    with np.load(work/"base.npz", allow_pickle=False) as z:
        base, left, right, signs = (z[k] for k in ("base", "left", "right", "signs"))
    groups = json.loads((work/"orientation-groups.json").read_text())
    if len(orientation_scores) != 2*len(groups):
        raise ValueError("Incomplete orientation evidence")
    flips = np.zeros(len(base), dtype=bool)
    decisive = True
    for k, indices in enumerate(groups):
        a, b = (float(sum(v)) for v in orientation_scores[2*k:2*k+2])
        strong = max(a, b) > 10 and max(a, b) > 1.35*(min(a, b)+1)
        decisive &= strong
        if strong and b > a:
            flips[indices] = True
    flip = bool(np.all(flips))
    if flip:
        base, left, right, signs = base.T.copy(), right, left, ~signs
    elif flips.any():
        signs = signs ^ flips
        _, pieces, _ = _read(work.parent/"result")
        raw = [_ink(p) for p in pieces]
        base, left, right = _matrix([np.rot90(im, 2) if signs[i] else im for i,im in enumerate(raw)])
    clean = [np.array(Image.open(work/f"strip-{i}.png")) for i in range(len(base))]
    clean = [np.rot90(im, 2) if flips[i] else im for i,im in enumerate(clean)]
    np.savez(work/"base.npz", base=base, left=left, right=right, signs=signs)
    pairs = _shortlist(base)
    result, group = [], []
    width = 40
    h = max(map(len, clean))+20
    def flush():
        nonlocal group, width
        if not group:
            return
        path = work/f"batch-{len(result):03d}.png"
        spans = _tile(clean, group, path)
        result.append({"path": str(path), "spans": spans, "pairs": group})
        group, width = [], 40
    for i, j in pairs:
        w = clean[i].shape[1]+clean[j].shape[1]+40
        if group and (len(group)>=10 or (width+w)*h>4_000_000):
            flush()
        group.append([i, j])
        width += w
    flush()
    manifest = {"batches": result, "candidate_pairs": len(pairs),
                "orientation_decisive": decisive, "global_flip": bool(flip),
                "orientation_groups": len(groups), "flipped_strips": int(flips.sum())}
    json_write(work/"manifest.json", manifest)
    return manifest


def _update_reports(out, ocr, *, poses=None, elapsed=0, collision=None, coverage=None):
    d = json.loads((out/"diagnostics.json").read_text())
    p = json.loads((out/"placements.json").read_text())
    r = json.loads((out/"scan_report.json").read_text())
    d["scan"]["ocr"] = r["ocr"] = ocr
    applied = ocr["status"] == "applied"
    warning = ("English OCR seam evidence applied locally; joins and orientation remain unverified."
               if applied else "OCR refinement: "+ocr["status"].replace("_", " ")+". Pixel-based result retained.")
    if applied and ocr["orientation_decisive"]:
        for record in (d["scan"], r):
            record["warnings"] = [w for w in record["warnings"] if not w.startswith("Global orientation is ambiguous")]
    d["scan"]["warnings"].append(warning)
    r["warnings"].append(warning)
    d["runtime_seconds"] += elapsed
    d["workflow_elapsed_seconds"] = d.get("workflow_elapsed_seconds", 0)+elapsed
    if applied:
        d["algorithm"] = "physical-document-ocr-wordpath"
        d["orientation_ambiguous"] = not ocr["orientation_decisive"]
        r["orientation_ambiguous"] = d["orientation_ambiguous"]
        p["poses"] = {k:v.as_dict() for k, v in poses.items()}
        d["coverage"], d["render_collision_pixels"] = coverage, collision
    p["diagnostics"] = d
    for name, value in (("diagnostics.json", d), ("placements.json", p), ("scan_report.json", r)):
        json_write(out/name, value)
    # Browser calls this after initial workflow packaging; native also uses it.
    with zipfile.ZipFile(out/"result.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for path in out.iterdir():
            if path.is_file() and path.name not in ("result.zip", "fragments.zip", "source.png"):
                z.write(path, path.name)
    return d


def finish(out, work, values, elapsed=0):
    out, work = Path(out), Path(work)
    manifest = json.loads((work/"manifest.json").read_text())
    with np.load(work/"base.npz", allow_pickle=False) as z:
        base, left, right, signs = (z[k] for k in ("base", "left", "right", "signs"))
    n = len(base)
    scores = np.zeros_like(base)
    if len(values) != len(manifest["batches"]):
        raise ValueError("Incomplete OCR batches")
    for batch, v in zip(manifest["batches"], values):
        if len(v) != len(batch["pairs"]):
            raise ValueError("Incomplete OCR scores")
        for (i, j), value in zip(batch["pairs"], v):
            scores[i, j] = max(0, float(value))
    ocr = {k: manifest[k] for k in ("candidate_pairs", "orientation_decisive", "global_flip", "orientation_groups", "flipped_strips")}
    ocr.update(language="eng", elapsed_seconds=float(elapsed), status="insufficient_evidence")
    if np.count_nonzero(scores > 2) < n or np.median(np.r_[scores.max(0),scores.max(1)]) < 4:
        return _update_reports(out, ocr, elapsed=elapsed)
    finite = base[np.isfinite(base)]
    norm = (base-finite.min())/max(1e-6, np.median(finite)-finite.min())
    # Positive word evidence; absence of a recognized word is not proof of mismatch.
    combined = norm-.02*scores
    combined -= combined[np.isfinite(combined)].min()
    order = chain(combined, .5, borders=(left, right))
    ocr["supported_joins"] = int(np.count_nonzero(scores[order[:-1], order[1:]] > 2))
    if ocr["supported_joins"] < max(2, (n-1)//3):
        return _update_reports(out, ocr, elapsed=elapsed)
    _, pieces, _ = _read(out)
    poses = {}
    x = 0
    for i in order:
        p = pieces[i]
        poses[p.id] = Pose(x, 0, 2*int(signs[i]))
        x += p.rgba.shape[1]
    h = max(p.rgba.shape[0] for p in pieces)
    rgba, labels, collision = render(pieces, poses, (h, x))
    render_out=work/"rendered"
    render_out.mkdir(exist_ok=True)
    save_rgba(render_out/"assembly.png", rgba)
    write_cut_outputs(render_out, rgba, labels)
    for path in render_out.iterdir():
        path.replace(out/path.name)
    ocr["status"] = "applied"
    return _update_reports(out, ocr, poses=poses, elapsed=elapsed, collision=collision,
                           coverage=float((labels>=0).mean()))


def unavailable(out, elapsed=0):
    return _update_reports(Path(out), {"status": "unavailable", "language": "eng",
                                      "elapsed_seconds": float(elapsed)}, elapsed=elapsed)


def native_refine(out, work, update):
    import concurrent.futures
    import os
    import shutil
    import subprocess
    started = time.perf_counter()
    try:
        plan = prepare(out, work)
        if plan is None:
            return None
        executable = shutil.which("tesseract")
        if not executable:
            return unavailable(out, time.perf_counter()-started)
        def recognize(batch):
            result = subprocess.run([executable, batch["path"], "stdout", "-l", "eng",
                                     "--psm", "11", "--dpi", "150", "tsv"],
                                    capture_output=True, text=True, check=True, timeout=90,
                                    env={**os.environ, "OMP_THREAD_LIMIT": "1"})
            return seam_scores(result.stdout, batch["spans"])
        update("Checking upright direction with English OCR")
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            orient = list(pool.map(recognize, plan["orientation"]))
            manifest = batches(work, orient)
            values = []
            for i, value in enumerate(pool.map(recognize, manifest["batches"]), 1):
                values.append(value)
                update(f"OCR seam verification {i}/{len(manifest['batches'])}")
        return finish(out, work, values, time.perf_counter()-started)
    except Exception:
        return unavailable(out, time.perf_counter()-started)
    finally:
        shutil.rmtree(work, ignore_errors=True)
