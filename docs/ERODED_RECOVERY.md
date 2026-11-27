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
   evidence, attempt bounded gap-aware recovery (2–128 legacy full search;
   129–512 sparse bounded recovery in Balanced/High accuracy).
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

The optimization update also caches invariant masks and interior distance
transforms across spectral angles, chunks Fourier correlations, and checks
triangles with distinct boundary arcs before accepting borderline multi-evidence
seams. The [targeted rerun](OPTIMIZATION.md) preserved 6/16 correct relative
joins and 3/87 primary poses on the severe DAFNE case; it did not solve it.

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
and coverage are not accuracy. The deep spectral solver remains bounded to
128 pieces; sparse gap recovery supports up to 512. The physical import cap
is now 4,096 native / 1,024 browser. See [scaling evidence](IRREGULAR_SCALE.md).
There is no claim of 65,000-piece support for irregular eroded data.

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

## Deep classical search (no machine learning)

The newer high-quality stage is implemented in `continuation.py` and
`spectral_gap.py`. Balanced automatically tries it after sparse or weak
contact recovery. The High accuracy profile forces the additional search
within the same resource budgets; Fast disables this additional stage.
Pass `deep_recovery: "always"` or `false` to override the policy in the API.

- A multi-channel **structure tensor** estimates the local direction of
  isophotes. Outside the mask, rays project back into valid interior pixels;
  unreliable rays fall back to nearest observed color. The inferred band is
  evidence only and is never rendered.
- Distinctive **line exits** (pictorial lines crossing the fragment boundary)
  supply candidate orientations and translations. Two nonparallel exits can
  determine translation by a 2x2 linear system instead of assuming contact.
- A **masked spectral search** proposes additional transformations:
  for each nominated pair, sample 24 angles and all translations on a coarse
  grid using FFT correlations of observed and extrapolated bands. A weighted
  RGB squared error, support, texture and true-mask collision are evaluated
  over the translation grid, then retained hypotheses are checked at common
  working resolution. This is a bounded proposal search, not exhaustive
  continuous global optimization.
- **Pose-mode consensus** combines evidence from point/line retrieval,
  spectral retrieval, and previous gap verification. A robust, soft curvature
  prior downweights angular pieces only when a sufficiently large majority is
  smooth and compact; no input fragments are deleted. Contextual forest growth
  rejects candidates that collide with existing members and requires multiple
  line crossings or consistent independent pose evidence. Unanchored groups
  and alternate model poses remain available in the JSON. Primary ranking
  uses combined evidence with a weak-link penalty, not ground truth.
- The stage is limited to 2–128 pieces and five million padded field pixels.
  It can take minutes on large degraded data, and performance is hardware
  dependent. It uses only NumPy/SciPy and does not train or load any model.

Reproducible synthetic comparisons with the previous solver:
```sh
PYTHONPATH=src python scripts/benchmark_classical.py
```
Raw results: [classical_benchmark.json](../reports/classical_benchmark.json).
The selected main result is unchanged on these 11 fixtures; alternate
component predictions may have better precision at the cost of coverage.
The 8px erosion cases remain unresolved despite the larger search.

## Further research (classical only)

- Improve local structure preservation with classical exemplar/patch search
  or variational continuation, with uncertainty estimates.
- Develop multiple-hypothesis component search with exact or bounded
  combinatorial optimization, and validate longer cycles without forcing
  unrelated fragments to align.
- Introduce texture-frequency, contour curvature-scale-space and calibrated
  local likelihood evidence. Benchmark each cue by ablation.
- Evaluate on additional licensed artwork and real shredded-scan data, with
  independent test splits and separate reference-aided vs blind protocols.
  No trained networks or learned descriptors are required or planned.

### Development-set result: DAFNE Resurrection

With the same 102 input images (87 genuine, 15 distractors), comparing the
previous commit with deep classical continuation:

| Metric | Previous gap-aware | Deep classical |
|---|---:|---:|
| Correct primary poses (one global rigid gauge) | 2/87 | 3/87 |
| Correct relative joins across all independent components | 3/11 | 6/16 |
| Relative join precision | 27.3% | 37.5% |
| Non-singleton components | 7 | 14 |
| Median primary center error | 2.42 px | 1.60 px |
| Wall time, this machine | 81.5 s | 292.3 s |

The MIT regression remains 9/9 poses. The deep proposal set includes 24
sampled rotations, 898 nominated Fourier pairs and 12,572 evaluated spectral
proposals. A coherent rounded-population prior downweighted 15 pieces.
All decisions used observed rasters only; labels were read afterward by
the evaluator. Some of the secondary components are wrong, and the DAFNE
global recovery remains poor (3/87); it is not a solved challenge.

See [complete measurements](../reports/classical_real_development.json).
To reproduce the external evaluation with the separate dataset-pack truth
archive, unzip the solver's result ZIP and run:

```sh
python scripts/evaluate_dafne_components.py INPUT.zip TRUTH.zip RESULT_DIR
```

The evaluator accepts the DAFNE dataset-pack annotation JSON convention,
not arbitrary external truth formats. No external dataset files are bundled.
