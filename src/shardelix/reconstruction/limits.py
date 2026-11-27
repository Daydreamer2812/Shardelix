"""Validated public constraints. Caps are resource guards, not accuracy claims."""
from __future__ import annotations
from numbers import Integral
from dataclasses import dataclass
GRID_MAX=262144
IRREGULAR_MAX=4096
STRIP_MAX=8192
LEGACY_MAX=1024
MAX_PIXELS=110_000_000
RECTANGULAR={"grid","strips","horizontal-strips"}
IRREGULAR={"jagged","wavy","jigsaw","random","custom"}

def max_count(cut):
    return GRID_MAX if cut=="grid" else STRIP_MAX if cut in RECTANGULAR else IRREGULAR_MAX

def validate_constraints(n,shape,cut,grid,rotations):
    if cut not in RECTANGULAR|IRREGULAR:raise ValueError(f"Unsupported cut: {cut!r}")
    if type(rotations) is not bool:raise ValueError("rotations must be a JSON boolean")
    if not isinstance(n,Integral) or isinstance(n,bool) or not 1<=n<=max_count(cut):
        raise ValueError(f"{cut} accepts 1–{max_count(cut)} fragments")
    if len(shape)!=2 or any(not isinstance(x,Integral) or isinstance(x,bool) or x<1 for x in shape):
        raise ValueError("Canvas width and height must be positive integers")
    h,w=shape
    if h*w>MAX_PIXELS or h*w<n:raise ValueError("Canvas exceeds pixel budget or has fewer pixels than fragments")
    if cut in RECTANGULAR:
        if not grid or len(grid)!=2 or any(not isinstance(x,Integral) or isinstance(x,bool) or x<1 for x in grid):
            raise ValueError("Rectangular cuts require positive grid rows/columns")
        r,c=grid
        if r*c!=n or r>h or c>w:raise ValueError("Grid dimensions do not match count/canvas")
        if cut=="strips" and r!=1 or cut=="horizontal-strips" and c!=1:
            raise ValueError("Grid axis does not match strip type")
        if n>LEGACY_MAX and (h%r or w%c):
            raise ValueError("Above 1024 rectangular pieces, canvas dimensions must be divisible by grid dimensions")
    elif grid is not None:raise ValueError("Irregular cuts must not specify a rectangular grid")


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    """Optional per-solver lower resource bounds for application integration.

    Counts and decoded pixel budgets are preflight checks, not a promise of
    exact RAM or runtime. Use process isolation for hard memory/time limits.
    """
    max_fragments: int = GRID_MAX
    max_canvas_pixels: int = MAX_PIXELS
    max_input_pixels: int = MAX_PIXELS * 3

    def __post_init__(self):
        for name, cap in (("max_fragments", GRID_MAX), ("max_canvas_pixels", MAX_PIXELS),
                          ("max_input_pixels", MAX_PIXELS * 3)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= cap:
                raise ValueError(f"{name} must be between 1 and {cap}")

    def check(self, count, shape):
        if count is not None and count > self.max_fragments:
            raise ValueError("Expected piece count exceeds configured resource limit")
        if len(shape) != 2 or any(not isinstance(x, Integral) or isinstance(x, bool) or x < 1 for x in shape):
            raise ValueError("Canvas width and height must be positive integers")
        if shape[0] * shape[1] > self.max_canvas_pixels:
            raise ValueError("Canvas exceeds configured resource limit")
