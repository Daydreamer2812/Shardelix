"""Post-solve raster comparison. Never imported by blind solver."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def compare_reference(output: Path, reference: Path, *, allow_rotation: bool,
                      pose_rotation: int | None = None, pixel_cap: int) -> dict:
    """Score only whole-canvas quarter turns, not reflections or local edits.

    Retain the original blind PNGs/poses. When a nonzero rotation improves
    alignment (or was selected by the verified pose evaluator), export a
    separately labeled comparison image and overlay.
    """
    with Image.open(reference) as im:
        if im.width * im.height > pixel_cap:
            raise ValueError("Reference exceeds pixel budget")
        ref = ImageOps.exif_transpose(im).convert("RGBA")
    ref.save(output / "source.png")
    with Image.open(output / "assembly.png") as im:
        pred = im.convert("RGBA")
    a, b = np.asarray(pred), np.asarray(ref)
    comparison = dict(reference_size=list(ref.size), reconstruction_size=list(pred.size),
                      metric_scope="Post-solve whole-canvas rotation; observed overlapping pixels only.")

    def score(k):
        rotated = np.rot90(a, -k)
        if rotated.shape != b.shape:
            return None
        observed = (rotated[:, :, 3] > 0) & (b[:, :, 3] > 0)
        if not observed.any():
            return None
        err = rotated[observed, :3].astype(np.float32) - b[observed, :3].astype(np.float32)
        return dict(mae=float(np.mean(np.abs(err))), rmse=float(np.sqrt(np.mean(err * err))),
                    exact_rgb_fraction=float(np.mean(np.all(err == 0, axis=1))),
                    compared_pixel_fraction=float(observed.mean()))

    raw = score(0)
    if raw:
        comparison.update(raw_mae=raw["mae"], raw_rmse=raw["rmse"],
                          exact_rgb_fraction=raw["exact_rgb_fraction"],
                          compared_pixel_fraction=raw["compared_pixel_fraction"])
    candidates = {}
    for k in range(4 if allow_rotation else 1):
        m = raw if k == 0 else score(k)
        if m is not None:
            candidates[k] = m
    if not candidates:
        comparison["warning"] = "Reference canvas differs in size or has no overlapping pixels; visual comparison only."
        return comparison
    if pose_rotation is not None and pose_rotation // 90 in candidates:
        k = pose_rotation // 90
        method = "verified pose evaluation"
    else:
        # Penalize absent overlap against the reference; don't prefer a rotation
        # simply because its score used fewer pixels. Stable ties prefer zero.
        k = min(candidates, key=lambda j: candidates[j]["mae"] * candidates[j]["compared_pixel_fraction"]
                + 255 * (1 - candidates[j]["compared_pixel_fraction"]))
        method = "best pixel agreement (not verified pose accuracy)"
    aligned = candidates[k]
    comparison.update(evaluation_rotation_cw_deg=k * 90, orientation_method=method,
                      aligned_mae=aligned["mae"], aligned_rmse=aligned["rmse"],
                      aligned_exact_rgb_fraction=aligned["exact_rgb_fraction"],
                      aligned_compared_pixel_fraction=aligned["compared_pixel_fraction"])
    if k:
        for src, dst in (("assembly.png", "assembly_reference_aligned.png"),
                         ("assembly_with_cuts.png", "assembly_reference_aligned_with_cuts.png")):
            if (output / src).exists():
                with Image.open(output / src) as im:
                    im.transpose({1: Image.Transpose.ROTATE_270, 2: Image.Transpose.ROTATE_180,
                                  3: Image.Transpose.ROTATE_90}[k]).save(output / dst)
    return comparison
