# Experimental gap-aware recovery for degraded fragments

## Scope

The physical scan solver now has an independent reference-free model for
eroded boundaries, missing material, unknown continuous rotations and
distractor fragments. It does not synthesize missing image pixels. Neither
source images nor ground-truth poses are loaded by the matching algorithm.
This is an experimental extension, not a claim of a solved DAFNE benchmark.

## Pipeline

1. Foreground normalization uses alpha, sidecar masks or reliable background
   removal. In ZIPs explicitly labeled `pieces/`, `fragments/`, `tiles/` or
   `patches/`, one raster denotes one fragment: separated islands of surviving
   material are kept in its common coordinates. Overview scans still segment
   multiple physical objects. The largest contour drives pose proposals.
2. Run the existing complementary-contour matcher first.
3. If the largest exact-contact component contains less than 60% of the input
   pieces, or a small irregular set (up to 16 pieces) has weak local contact
   evidence, attempt bounded gap-aware recovery (2–128 detected pieces).
   The workflow configuration accepts `gap_recovery: false` to disable this
   stage, or `gap_recovery: "always"` to search it regardless of the trigger.
   The latter does not guarantee selection over the exact-contact model.
4. Sample color and gradients inward from masked contours. Index multiscale
   arc descriptors and propose continuous SE(2) transformations with
   nonzero seam gaps (bounded by 36 working pixels and 35% of the typical
   fragment side). Score contiguous opposing tangent support, absence of
   overlap, extrapolated color consistency, observed interior continuation,
   gradient agreement and texture. Reject ambiguous competing poses.
5. Check local triangle-cycle agreement and refine redundant pose constraints
   with robust soft-L1 residuals. Reject a refinement if it creates collisions
   or increases residuals. Rank independent components by seam confidence,
   not only their size. Keep the alternative model if applicable.
6. Output the primary component and transparently represent the lost material.
   No spatial relationship is assumed between disconnected components.

## Output

- `assembly.png`: observed pixels of the selected connected group only.
- `component_placements.json`: every component (including singletons),
  its accepted joins and each fragment's 2×3 working crop → component
  affine matrix, crop origin, model name, and optional alternative model.
  Each component uses its own independent coordinate gauge. The file
  is not a globally registered reconstruction; do not combine unrelated
  groups by their raw coordinate values.
- `scan_report.json`: exact-contact and gap-search diagnostics, detected
  counts, component sizes, search statistics and limitations.
- `placements.json`: primary component's output-canvas poses as before.

Global orientation remains ambiguous without additional information. Counts
and coverage are not accuracy. The gap solver is bounded to 128 pieces; the
existing physical solver still accepts up to 256. There is no claim of
65,000-piece support for irregular eroded data.

## Evaluation

Run the reproducible synthetic benchmark (randomly rotated textured curved
pieces, 3/5/8 px erosion, missing pieces and a distractor):
```sh
PYTHONPATH=src python scripts/benchmark_eroded.py
```
The script scores relative pose only after assembly (10 px center tolerance,
10 degree angle tolerance), with no credit for singleton anchors. The raw
results are in [eroded_benchmark.json](../reports/eroded_benchmark.json).

[Real/development results](../reports/eroded_real_development.json) compare
the previous commit with the prototype on a DAFNE example, two damaged scans
and the MIT example, using the existing evaluation harness and same input
ZIPs. The DAFNE case was used to develop the method and is not an independent
test split. The evaluation aligns one global rigid gauge for the rendered
group; recovered pose count is not simply number of placements. We report
weak results, including unchanged accuracy on one severely damaged scan.

Neither matches on synthetic textures nor a few correct DAFNE poses justify
a broad accuracy claim. DAFNE includes 87 genuine fragments and 15 distractors
in this test, and its official challenge permits a source-image reference;
the blind setting here is different.

## Next research steps

- Improve broad-gap proposals with learned masked patch descriptors trained
  on separately split source images and simulated erosion/distractors; include
  calibrated hard negative mining and compare with the CPU fallback.
- Learn compatibility from boundary extrapolation (e.g. Bridger et al.) and
  test against partial contour retrieval; generated pixels should remain
  evidence, not be presented as recovered historical content.
- Improve cycle-supported global selection with explicit outlier variables,
  multiple-object hypotheses and uncertainty-weighted pose constraints.
- Evaluate on additional licensed archaeological and real shredded-scan data
  (e.g. RePAIR-style physical fragments); separate reference-aided and blind
  protocols, and report both component precision and recovered coverage.
