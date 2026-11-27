"""Build the README demonstration from a deterministic synthetic image and real solver output."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default(size=size)


def source_image():
    rng = np.random.default_rng(42)
    y, x = np.mgrid[:512, :512].astype(float)
    z = np.sin(x*.022+y*.015)+np.cos(y*.031-x*.012)+.5*np.sin((x*x+y*y)*.000085)
    t = np.sin(x*.041+y*.053)+np.cos(x*.077-y*.039)
    rgb = np.stack([112+61*z+23*t, 124+45*np.sin(z*1.4+.9)+37*np.cos(t+1)+19*np.sin(x*.11-y*.005),
                    161+58*np.cos(z+.4)+26*np.sin(t*.7+y*.012)], -1)
    rgb = np.clip(rgb,0,255)
    noise = gaussian_filter(rng.normal(0,1,rgb.shape),(.8,.8,0))
    return Image.fromarray(np.clip(rgb+24*noise,0,255).astype("uint8"))


def main():
    output = Path("docs/assets")
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = source_image()
        source.save(root/"source.png")
        subprocess.run([sys.executable, "-m", "shardelix", "run", str(root/"source.png"),
                        "--cut", "jigsaw", "--pieces", "64", "--rotations", "--seed", "42",
                        "--profile", "balanced", "--out", str(root/"run")], check=True,
                       stdout=subprocess.DEVNULL)
        metrics = json.loads((root/"run/result/metrics.json").read_text())
        pieces = sorted((root/"run/case/pieces").glob("*.png"))
        preview = Image.new("RGB", (1440, 600), "#101b2c")
        d = ImageDraw.Draw(preview)
        d.text((45,26), "SHARDELIX  /  RECONSTRUCTION IN PRACTICE", font=font(22,True), fill="#a9c9ff")
        d.text((45,68), "From scrambled pieces to a reconstructed image",font=font(34,True),fill="#f6f9ff")
        cards=[("01  Original input", "Used only to generate and evaluate"),
               ("02  Scrambled fragments", "25 of 64 rotated pieces shown"),
               ("03  Blind reconstruction", "Predicted pose + cut boundaries")]
        for i,(title,subtitle) in enumerate(cards):
            x=45+i*456
            d.rounded_rectangle((x,130,x+438,538),radius=18,fill="#1c2b41",outline="#344861",width=2)
            d.text((x+20,146),title,font=font(22,True),fill="#f3f6fc")
            d.text((x+20,178),subtitle,font=font(16),fill="#aebed3")
        # Keep the genuine source and actual reconstructed pixels, not drawn substitutes.
        preview.paste(source.resize((320,320),Image.Resampling.LANCZOS),(104,212))
        panel=Image.new("RGBA",(320,320),"#24364e")
        chosen=np.random.default_rng(7).choice(len(pieces),25,replace=False)
        for j,idx in enumerate(chosen):
            p=Image.open(pieces[int(idx)]).convert("RGBA")
            p.thumbnail((55,55),Image.Resampling.LANCZOS)
            left=(j%5)*64+(64-p.width)//2
            top=(j//5)*64+(64-p.height)//2
            panel.alpha_composite(p,(left,top))
        preview.paste(panel.convert("RGB"),(560,212))
        reconstruction=Image.open(root/"run/result/assembly_with_cuts.png").convert("RGBA")
        bg=Image.new("RGBA",reconstruction.size,"#edf1f5")
        bg.alpha_composite(reconstruction)
        preview.paste(bg.convert("RGB").resize((320,320),Image.Resampling.LANCZOS),(1016,212))
        d.text((45,557),"CASE:  64 JIGSAW PIECES  •  RANDOM QUARTER-TURNS  •  SEED 42",font=font(17,True),fill="#b5c6de")
        summary=f"DPA {metrics['direct_position_accuracy']:.0%}   |   F1 {metrics['adjacency_f1']:.0%}   |   collisions {metrics['collision_pixels']}"
        d.text((1395,557),summary,font=font(18,True),fill="#9ce5c0",anchor="ra")
        preview.save(output/"reconstruction-showcase.webp",quality=90)
        public={k:metrics[k] for k in ("piece_count","placed","direct_position_accuracy","adjacency_f1","collision_pixels","runtime_seconds","pass_95")}
        public.update(seed=42,cut="jigsaw",profile="balanced",angle_mode="quarter",source="synthetic")
        (Path("reports/v12")/"showcase.json").write_text(json.dumps(public,indent=2)+"\n")

if __name__=="__main__":
    main()
