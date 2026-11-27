"""Evaluate public DocDataset physical strips using anonymized images only.

Download the archive from the original research page:
https://people.tamu.edu/~xinli/DocReassembly/data/DocDataset.zip
Run PYTHONPATH=src python scripts/benchmark_scanned_strips.py /path/DocDataset.zip
Requires optional native Tesseract + English traineddata. Images and truth are
not distributed with this repository. The ordering file is read only after
both solutions are computed.
"""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np
from PIL import Image
from shardelix.reconstruction.physical_document import reconstruct_strips
from shardelix.reconstruction.physical_ocr import native_refine


def metrics(order, truth):
    edges = set(zip(truth, truth[1:]))
    joins = list(zip(order, order[1:]))
    return {"directed_correct": sum(e in edges for e in joins),
            "undirected_correct": sum(e in edges or e[::-1] in edges for e in joins),
            "positions_correct": sum(a == b for a, b in zip(order, truth)),
            "total_joins": len(truth)-1, "total_pieces": len(truth)}


def evaluate(z, name, case):
    prefix = f"DocDataset/stripes/{name}/"
    scans, pieces, out = (case/p for p in ("scans", "pieces", "result"))
    scans.mkdir(parents=True)
    rng = np.random.default_rng(624)
    aliases = {}
    # Deterministic anonymization in archive entry order, independent of truth.
    for member in z.namelist():
        if not member.startswith(prefix):
            continue
        leaf = member[len(prefix):]
        if "/" in leaf or not leaf.endswith(".png"):
            continue
        stem = Path(leaf).stem
        alias = rng.bytes(8).hex()
        aliases[stem] = alias
        with Image.open(io.BytesIO(z.read(member))) as source:
            rgba = source.convert("RGBA")
        with Image.open(io.BytesIO(z.read(prefix+"bg_mask_vis/"+leaf))) as mask:
            rgba.putalpha(mask.convert("L"))
        rgba.save(scans/(alias+".png"))
    reconstruct_strips(sorted(scans.glob("*.png")), pieces, out)
    def order():
        poses = json.loads((out/"placements.json").read_text())["poses"]
        return sorted(poses, key=lambda i: poses[i]["x"])
    before = order()
    with zipfile.ZipFile(out/"fragments.zip", "w") as archive:
        for p in sorted(pieces.glob("*.png")):
            archive.write(p, "pieces/"+p.name)
    diag = native_refine(out, case/"ocr-work", lambda msg: print(name,msg,flush=True))
    after = order()
    # Evaluation labels enter the process only once all inference has finished.
    truth = [aliases[v] for v in z.read(prefix+"order.txt").decode().split()]
    return {"case": name, "baseline": metrics(before,truth), "ocr": metrics(after,truth),
            "ocr_diagnostics": diag.get("scan",{}).get("ocr") if diag else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(args.archive) as z:
        results = [evaluate(z, name, Path(tmp)/name)
                   for name in ("real1_27", "real2_56", "real3_39")]
    data = {"dataset":"DocDataset, physical stripes", "seed":624,
            "protocol":"Anonymized scan pixels + foreground masks only; truth read after both solves.",
            "results":results}
    text = json.dumps(data,indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(text+"\n")

if __name__ == "__main__":
    main()
