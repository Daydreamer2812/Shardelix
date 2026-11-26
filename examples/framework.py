"""Minimal framework API example using an in-memory, model-free digital puzzle.

No network, training or files required. Run with PYTHONPATH=src python
examples/framework.py. To integrate external pieces, pass a directory
of images to the same solve method.
"""
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from fragmentlab import Fragment, ResourceLimits, Solver

rng = np.random.default_rng(815)
source = np.clip(127 + 420 * gaussian_filter(
    rng.normal(size=(128, 128, 3)), (1.4, 1.4, 0)), 0, 255).astype(np.uint8)
pieces = []
for i in rng.permutation(64):
    y, x = divmod(int(i), 8)
    rgba = np.full((16, 16, 4), 255, dtype=np.uint8)
    rgba[:, :, :3] = source[y*16:(y+1)*16, x*16:(x+1)*16]
    pieces.append(Fragment(rng.bytes(12).hex(), np.rot90(rgba, int(rng.integers(4))).copy()))
# Only anonymous fragments and publicly known geometry are supplied.
del source
solver = Solver(limits=ResourceLimits(max_fragments=8192, max_canvas_pixels=16_000_000,
                                      max_input_pixels=48_000_000))
result = solver.solve(pieces, shape=(128, 128), grid=(8, 8), cut="grid", rotations=True)
print(result.summary())
result.save(Path("outputs/framework_example"))
