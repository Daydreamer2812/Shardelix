"""Generate a deterministic text-document fixture; no downloaded data/models."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random
rng=random.Random(1701)
p=Path("data/examples/text_document.png")
p.parent.mkdir(parents=True,exist_ok=True)
W,H=1536,2048
im=Image.new("RGB",(W,H),"white");d=ImageDraw.Draw(im)
def font(size,bold=False):
    try: return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",size)
    except OSError: return ImageFont.load_default(size=size)
title=font(54,True); body=font(25); sub=font(33,True)
d.text((90,70),"Classical fragment reconstruction",fill=(25,37,54),font=title)
d.line((90,148,W-90,148),fill=(27,70,115),width=5)
paragraphs=[
"The reconstruction problem estimates the pose of each fragment using the complementary geometry and appearance along its boundary. No trained network is required.",
"Image formation and sampling determine the observed pixel values. Gradients carry information about local structure, whereas smooth regions may not contain a unique correspondence.",
"An admissible assembly uses every observed fragment once and prevents overlaps. Ground truth is reserved for evaluation rather than solver input.",
"Accuracy must distinguish correct neighboring pieces from exact positions. Repeated blank regions make several layouts indistinguishable without additional evidence.",
"Candidates are ranked by shape agreement, boundary continuity, and consistency of the global transformation. The method also reports failed and ambiguous cases.",
"Parameter selection is performed on a fixed training partition. The chosen configuration is frozen before any held out runs are evaluated."]
def wrap(txt,f,maxw):
    out=[];line=""
    for word in txt.split():
        proposal=(line+" "+word).strip()
        if d.textlength(proposal,font=f)>maxw:out.append(line);line=word
        else:line=proposal
    if line:out.append(line)
    return out
for col in range(2):
    x=90+col*700; y=180
    for section in range(6):
        d.text((x,y),f"{col*6+section+1}. "+["Introduction","Geometry","Appearance","Optimization","Evaluation","Limitations"][section],font=sub,fill=(25,37,54));y+=45
        for line in wrap(rng.choice(paragraphs),body,650):
            d.text((x,y),line,font=body,fill=(27,27,27));y+=33
        y+=26
d.rectangle((90,1770,1446,1930),outline=(40,70,100),width=3)
d.text((115,1795),"Experiment: shuffled 1–1024 fragments",font=sub,fill=(25,37,54))
d.text((115,1850),"DPA  |  adjacency F1  |  coverage  |  collision  |  runtime",font=body,fill=(25,37,54))
d.text((710,1980),"Page 1",font=body,fill="black")
im.save(p)
