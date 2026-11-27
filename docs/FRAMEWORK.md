# Unified framework

Shardelix provides one input pipeline for generation, reconstruction and evaluation. The web worker, native Studio, Python file API and CLI all run the same workflow. The specialized modules are retained internally so that grid, document strip and torn-edge methods use the geometry appropriate to the input. They are not separate apps.

## Everyday workflow

1. In **Reconstruction**, upload a ZIP containing one case, a collection of image fragments, or scan(s).
2. Leave **Automatic detection** on and **Balanced** selected; click Run. Override input type or geometry only if automatic detection is wrong.
3. Optional: enable **Use original image for matching** and choose the intact artwork in a separate slot. This changes the information available to the solver, so it is explicitly labeled reference-assisted, never blind.
4. Optional: in blind digital mode upload **Ground truth · evaluation only**. It is used after solving, not for matching. Other protocols require independent evaluation tools.
5. Download Results ZIP, image, poses and diagnostics. Unknown pixels and unanchored components remain explicit.

## Automatic dispatch

| Observed input | Internal route |
|---|---|
| Rectangular uniform tiles / public grid metadata | Sparse grid or dense grid matching, consensus and pose verification |
| Document strips | Pixel-only strip ordering; foreground rectification, shadow-normalized row profiles, and observed-patch recurrence for scanned strips |
| Irregular masks / physical scans | Segmentation, complementary contour and color matching; bounded eroded-edge continuation and SE(2) cycle verification |
| Explicit original image enabled | Indexed rotation-invariant classical registration, continuous SE(2) refinement and distractor rejection |

| Shared profile | Blind | Reference-assisted |
|---|---|---|
| Fast | Reduced search, no patch recurrence or deep physical continuation | Validated indexed |
| Balanced (default) | Adaptive matching, bounded physical fallback | Indexed with bounded tiny-fragment rescue |
| High accuracy | Extra verification and deeper bounded physical search | Indexed-deep micro-fragment/border search (experimental) |

The original-image option is off by default. Ground-truth files and original matching images have distinct paths. Generator-to-reconstruction transfer never opts into matching. The old explicit `reference_engine: "fft"` config is available only for reproducing historical experiments. All matching uses classical CV, without OCR or trained models.

## One command or Python function

```bash
python -m shardelix reconstruct fragments.zip --out results
python -m shardelix reconstruct fragments.zip --reference original.jpg --profile accurate --out assisted
python -m shardelix reconstruct fragments.zip --ground-truth truth.zip --out evaluated
```

```python
from shardelix import reconstruct_files

state = reconstruct_files("fragments.zip", "results", profile="balanced")
state = reconstruct_files("fragments.zip", "assisted", matching_reference="original.jpg")
```

Accepts a ZIP, image directory or image paths. A new or empty output directory is required. Existing low-level `Solver.solve`, generate/solve/evaluate commands and benchmark scripts remain available for research and backward compatibility.

## Evidence and limits

- [Blind eroded recovery](ERODED_RECOVERY.md): reported failures and partial poses as well as successes. The severe 102-piece blind DAFNE development case remains unsolved.
- [DAFNE indexed reference-assisted results](DAFNE_INDEXED.md): 6027/6029 genuine poses and 1154/1154 distractors rejected over 28 development cases. These are historical reference-assisted measurements, not the blind accuracy or a full dataset/SOTA claim.
- [Optimization evidence](OPTIMIZATION.md): targeted native regression, physical strips, blind damaged fragments and reference-assisted tiny-fragment rescue.
- [Web resources and hosting](WEB.md): browser limits differ from native; large DB2 cases require native Studio.

Malformed or conflicting public metadata is ignored with a warning where geometry can be inferred; unsupported or ambiguous inputs may still need an override.

A resource guard is not an accuracy guarantee. Some missing-piece configurations are fundamentally ambiguous without the original image. Deep searches are bounded and can be slower, without guaranteed improvement over Balanced.
