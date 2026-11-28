"""Render cut lines solely from the reconstructed labels."""
import numpy as np
from pathlib import Path
from shardelix.reconstruction.common import save_rgba

def cut_overlay(image,labels,scale=1):
    """Internal predicted seams, exactly aligned with the reconstructed pixels.

    A scaled preview preserves visual detail for 1-pixel-thick strips.
    """
    scale=int(scale)
    if scale<1:raise ValueError("scale must be positive")
    right=(labels[:,:-1]>=0)&(labels[:,1:]>=0)&(labels[:,:-1]!=labels[:,1:])
    down=(labels[:-1,:]>=0)&(labels[1:,:]>=0)&(labels[:-1,:]!=labels[1:,:])
    H,W=labels.shape
    if scale==1:
        seam=np.zeros(labels.shape,bool);seam[:,:-1]|=right;seam[:-1,:]|=down
        out=image.copy();out[seam]=[255,48,35,255]
        cutmap=np.zeros_like(out);cutmap[seam]=[255,48,35,255]
        return out,cutmap
    seam=np.zeros((H*scale,W*scale),bool)
    y,x=np.where(right)
    for t in range(scale):seam[y*scale+t,(x+1)*scale-1]=True
    y,x=np.where(down)
    for t in range(scale):seam[(y+1)*scale-1,x*scale+t]=True
    out=np.repeat(np.repeat(image,scale,axis=0),scale,axis=1)
    out[seam]=[255,48,35,255]
    cutmap=np.zeros_like(out);cutmap[seam]=[255,48,35,255]
    return out,cutmap

def write_cut_outputs(output:Path,image,labels,scale=1):
    if scale==1:
        # Export sequentially: avoid holding two extra RGBA canvases in RAM.
        seam=np.zeros(labels.shape,bool)
        left,right=labels[:,:-1],labels[:,1:]
        seam[:,:-1]=(left>=0)&(right>=0)&(left!=right)
        top,bottom=labels[:-1,:],labels[1:,:]
        seam[:-1,:]|=(top>=0)&(bottom>=0)&(top!=bottom)
        show=image.copy();show[seam]=[255,48,35,255]
        save_rgba(output/"assembly_with_cuts.png",show)
        del show
        mask=np.zeros_like(image);mask[seam]=[255,48,35,255]
        save_rgba(output/"cut_lines.png",mask)
    else:
        show,mask=cut_overlay(image,labels,scale)
        save_rgba(output/"assembly_with_cuts.png",show)
        save_rgba(output/"cut_lines.png",mask)
