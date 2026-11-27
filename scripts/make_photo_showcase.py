"""Generate an honest photographic demo from the actual blind solver."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw
from skimage import data
from make_showcase import font

def main():
    assets = Path("docs/assets")
    assets.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = Image.fromarray(data.astronaut()).convert("RGB")
        source.save(root/"source.png")
        subprocess.run([sys.executable,"-m","shardelix","run",str(root/"source.png"),
                        "--cut","jigsaw","--pieces","36","--rotations","--seed","42",
                        "--profile","balanced","--out",str(root/"run")],check=True)
        result = root/"run/result"
        metrics = json.loads((result/"metrics.json").read_text())
        pieces = sorted((root/"run/case/pieces").glob("*.png"))
        if not pieces:
            raise RuntimeError("No generated fragments")
        n = len(pieces)
        preview = Image.new("RGB",(1500,570),"#101b2c")
        d=ImageDraw.Draw(preview)
        d.text((36,24),"SHARDELIX  /  REAL SOLVER DEMONSTRATION",font=font(22,True),fill="#91d9c3")
        d.text((36,65),"From shuffled fragments to a reconstructed photograph",font=font(32,True),fill="#f6f9ff")
        cards=[("01  Reference","Evaluation only"),("02  Shuffled input",f"All {n} rotated fragments"),
               ("03  Predicted image","Blind solver output"),("04  Predicted seams","Actual reconstructed boundaries")]
        for i,(title,subtitle) in enumerate(cards):
            x=36+i*360
            d.rounded_rectangle((x,125,x+348,504),radius=16,fill="#1c2b41",outline="#344861",width=2)
            d.text((x+13,141),title,font=font(21,True),fill="#f3f6fc")
            d.text((x+13,170),subtitle,font=font(16),fill="#b5c6de")
        def place(im,i):
            im=im.convert("RGBA")
            im.thumbnail((310,298),Image.Resampling.LANCZOS)
            bg=Image.new("RGBA",(310,298),"#24364e")
            bg.alpha_composite(im,((310-im.width)//2,(298-im.height)//2))
            preview.paste(bg.convert("RGB"),(55+360*i,194))
        place(source,0)
        panel=Image.new("RGBA",(310,298),"#24364e")
        cols=int(np.ceil(np.sqrt(n)))
        rows=int(np.ceil(n/cols))
        cell=min(310//cols,298//rows)
        order=np.random.default_rng(7).permutation(n)
        for j,idx in enumerate(order):
            p=Image.open(pieces[int(idx)]).convert("RGBA")
            p.thumbnail((cell-5,cell-5),Image.Resampling.LANCZOS)
            panel.alpha_composite(p,((j%cols)*cell+(cell-p.width)//2,(j//cols)*cell+(cell-p.height)//2))
        place(panel,1)
        place(Image.open(result/"assembly.png"),2)
        place(Image.open(result/"assembly_with_cuts.png"),3)
        d.text((36,529),f"{n} PIECES  |  JIGSAW  |  RANDOM QUARTER-TURNS  |  SEED 42",font=font(16,True),fill="#b5c6de")
        summary=f"DPA {metrics['direct_position_accuracy']:.1%}  |  F1 {metrics['adjacency_f1']:.1%}  |  collisions {metrics['collision_pixels']}"
        d.text((1464,529),summary,font=font(17,True),fill="#91d9c3",anchor="ra")
        preview.save(assets/"shardelix-photo-showcase.webp",quality=93)
        public={k:metrics[k] for k in ("piece_count","placed","direct_position_accuracy","adjacency_f1","collision_pixels","runtime_seconds")}
        public.update(seed=42,cut="jigsaw",profile="balanced",angle_mode="quarter",source="scikit-image astronaut (NASA public-domain photograph)")
        report=Path("reports/v13/photo-showcase.json")
        report.parent.mkdir(parents=True,exist_ok=True)
        report.write_text(json.dumps(public,indent=2)+"\n")
        caption=f"![Shardelix actual photographic reconstruction: reference, all shuffled fragments, blind assembly, and predicted seams](docs/assets/shardelix-photo-showcase.webp)\n\n*Actual Shardelix output: {n} rotated jigsaw fragments, seed 42; DPA = {metrics['direct_position_accuracy']:.1%}, adjacency F1 = {metrics['adjacency_f1']:.1%}, collision pixels = {metrics['collision_pixels']}. Reference photograph: NASA via scikit-image. [Measured results](reports/v13/photo-showcase.json) · [Reproduce](scripts/make_photo_showcase.py).*"
        path=Path("README.md")
        before=path.read_text()
        start="<!-- SHOWCASE_START -->"
        end="<!-- SHOWCASE_END -->"
        left,rest=before.split(start,1)
        _,right=rest.split(end,1)
        path.write_text(left+start+"\n"+caption+"\n"+end+right)

if __name__=="__main__":
    main()
