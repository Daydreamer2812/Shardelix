# V13 Studio — GitHub Pages and local runs

## UI and solver routing

Fragment generator → generate anonymous pieces → export fragments ZIP and independent ground truth ZIP.
Reconstruction → direct transfer or import RGBA PNGs / ZIP + public metadata → Solver(profile) → optional independent evaluator.
The source/reference image never enters matching.

The benchmark switch repeats the same generated-case workflow across selected
counts, cuts, seeds and profiles. The independent evaluator reads known truth
only after solve; it is not a third solver. The browser and native Studio share
`src/shardelix/workflow12.py` and the reconstruction package.
All profiles use the same cut-specific algorithms and geometric validation.
Fast reduces search; a bounded rescue runs only on severe irregular under-coverage; Balanced is the default;
High accuracy adds bounded search and verifies candidate hypotheses against
observable data. It is not a guarantee of better accuracy. Native optional MILP
is not attempted in WebAssembly; browser and native results need not be identical.

## Hosted Studio and self-hosting

**[Open the official Web Studio](https://daydreamer2812.github.io/Shardelix/).**

For a separately hosted copy (subject to the repository license):

1. Put the release source at the repository root, including .github, src, web
   and scripts. Review rights/license first.
2. Under Settings → Pages, select GitHub Actions as the build and deployment source.
3. Commit/push to main or master. Actions → Deploy Shardelix Studio to GitHub Pages
   builds the static folder and deploys it. You can dispatch the workflow manually.
4. Open the Pages URL listed by GitHub, usually
   `https://OWNER.github.io/REPO/`. Assets use relative URLs.
5. Check Actions → Browser engine smoke. It installs Chromium/Playwright on
   the runner and tests real WebAssembly, single/batch solving, blind replay,
   exports, and responsive layout. Review the retained screenshot and logs.

V13 is a hybrid upgrade of the curated V12 code.
Enable Pages in repository settings; the included GitHub Actions workflow
builds and deploys the static site from main. For branch-based hosting, build
the static assets and deploy exactly the four output files to a dedicated
hosting branch. A Pages site exposes the distributed engine ZIP as well as
the rendered UI. No account credentials or API keys are needed to run it.

## Local static preview

```bash
python scripts/build_web.py --out build/pages --wasm
python -m http.server 8767 --bind 127.0.0.1 --directory build/pages
```
Visit http://127.0.0.1:8767. Opening index.html directly via file:// is not supported.
The accelerated build requires Emscripten (em++). Omit --wasm for the Python reference build.
The build script creates the static artifact locally; generated build files are not committed.
The deployment workflow rebuilds it from source, so do not manually edit engine.zip.
The C++ worker module uses SIMD and portable single-threaded WebAssembly. The pinned Pyodide 0.27.7 runtime and its NumPy, SciPy, Pillow packages download
from cdn.jsdelivr.net. Initial loading can take time and requires internet.
There is no application/backend upload endpoint in browser mode. The input files,
working state and downloads stay in the browser worker; CDN requests load code only.

## Limits and unsupported inputs

| | Browser | Native Studio |
|---|---|---|
| Input | Raster image or zipped/image fragments | Raster, optional PDF (PyMuPDF), fragments |
| ZIP/input bytes | 32 MiB | 200 MiB |
| Fragment cap | 4,096 (lower per family where applicable) | Per-family native ResourceLimits |
| Canvas | 8 million pixels | Up to 110 million pixels |
| Decoded input | 24 million pixels | Up to 330 million pixels |
| Batch cases | 24 | 64 |
| Batch retained export | 192 MiB | Native guards |

These are protective caps, not verified WebAssembly capacity or performance
guarantees. Use modest cases first. Mobile memory may be tighter. Cancel
terminates a browser worker and the next run reloads it. The native app starts
its own cancellable subprocess. Download before starting another run.
External fragment ZIPs do not need case.json. Automatic setup is enabled on every new import: upload and click Run. The engine uses public archive metadata when available; otherwise, for equal-sized opaque tiles, it recomputes the geometry at solve time and chooses the layout closest to a conventional 4:3 landscape aspect. Stale manual values never override this automatic mode. Disable Automatic setup to choose another layout or override the geometry. The original shape and rotation cannot be determined from the count alone: choose a different factor pair or edit the dimensions manually, especially for portrait images or missing fragments. For the Digital fragments override, irregular/unequal-sized pieces still require public canvas constraints. The default Input type also tries the physical scan route for metadata-free unequal-size or alpha-masked pieces and single images. The inspection never uses a reference or the filenames as pose labels.

The generator treats Resolution as a preferred maximum side. It can resize the working canvas to fit the requested exact count and resource limits (including sampling), then reports the actual canvas and grid. Uniform rectangular cuts may require snapping each dimension to a cell multiple. Thin cells have limited texture; infeasible counts return a descriptive error rather than silently changing the count.

DPA/F1 require a matching independent ground-truth archive. Raw DPA measures absolute poses in the original blind canvas; rotation-aligned DPA allows a single global quarter-turn of the known canvas (0/180° for nonsquare canvases). A correctly assembled image can therefore have 100% aligned DPA while being upside down. With a reference, the Studio shows raw and rotation-aligned pixel MAE on overlapping observed pixels, and can preview/export a separately labeled reference-aligned image and overlay. Reference-only alignment uses pixel agreement, not pose labels; it cannot establish DPA/F1. No reflection, local edit, or free translation is applied. The original blind PNG, cut overlay and poses are retained unchanged. If no reference is supplied, absolute upright orientation cannot generally be established.
Document strips require same-height digital rectangles, not photographs of torn
paper or arbitrary perspective. Unknown-width inference cannot know the total
missing area of strips not present in the input.

## Physical paper scans and photos (experimental)

In Reconstruction, leave **Input type → Automatic detection** for irregular
scans, one overview image, or a ZIP with images under `scans/`. Use **Physical
paper scans / photos** explicitly for several same-size opaque scan sheets,
which otherwise resemble digital tiles. Choose Digital fragments to force
the existing grid/canvas solver. No dimensions, rows/columns, or angles are
required in physical mode.

- Scan or photograph the same planar side of the paper with a uniform DPI or
  camera distance. Use a matte background that contrasts with the entire paper,
  include margins and space apart the pieces. Prefer a flatbed scanner.
- The system applies EXIF orientation, one common resolution scale, robust
  border color / optional planar illumination estimation, adaptive mask
  thresholding (including JPEG ringing suppression), morphology, connected
  components and speckle filtering. Transparent PNG alpha is supported.
- It matches complementary, resampled contour arcs at multiple lengths,
  estimates continuous SE(2) transforms, checks opposed tangents and seam
  color, retains alternative poses, then joins components with interior
  collision checks. Unanchored groups are not arbitrarily placed on the canvas.
- The preview shows detected masks; download the normalized mask ZIP, scan
  processing report, reconstruction PNG, boundary overlay and affine poses.
  Poses are local cropped working-pixel vertex coordinates → output canvas;
  crop offsets and common scale are in the report.
- Physical mode has an experimental cap of 64 detected pieces, working scans
  downscaled uniformly to at most 1,600 px per side and ~4 million total input
  pixels. Pairwise matching scales quadratically. All processing is local.
- Paper touching the scan edge, overlapping/stacked pieces, near-identical
  foreground/background, strong shadows, nonplanarity, perspective and
  inconsistent DPI can cause segmentation or alignment failures. Missing
  material can cause seam translation drift. No OCR or upright orientation
  guarantee. Global rotation is ambiguous. Coverage is occupancy of an
  estimated bounding box, not original-page coverage, and DPA/F1 are not
  computed on physical scans without independently registered truth.

The reproducible simulation `PYTHONPATH=src python scripts/benchmark_scan.py`
includes rotated noisy JPEGs, a 2-D wavy partition and one-pixel erosion.
`reports/scan_simulation.json` reports per-piece center and rotation errors
after a common rigid gauge (the first placed piece defines the gauge).
`reports/scan_real.json` records two public hand-torn magazine scan sets from
[AutoMosaic / OpenSoft14](https://github.com/raltgz/OpenSoft14), not redistributed
here. Both were fully connected (6/6 and 5/5), with visually plausible text and
art alignment, but no independent ground-truth poses or pixel accuracy claim.
The real examples have residual visible seam gaps and small raster overlaps.

### Folder-wrapped scanned strip archives

ZIP files may contain a single outer collection folder (for example
`collection/0.png`, `collection/1.png`). The importer selects the primary
raster directory, not auxiliary views or ground truth. If the archive contains
multiple unrelated image directories, use `pieces/` or `scans/` to identify the input.

The physical scan route supports binary, same-size foreground masks stored
under a sibling `bg_mask_vis/<image-stem>.png` folder (white/255 = paper,
black/0 = background). The mask folder is never counted as fragments.
Companion NumPy arrays and ordering/answer text files are not read by the solver.
All images in a masked collection must have corresponding valid masks.

If the images are elongated and share a common long dimension, the engine
attempts physical document-strip rectification before general contour matching.
It rectifies the foreground silhouette per row, estimates relative 180° text
orientation from row projections, compares robust multiscale ink-edge profiles
and optimizes a directed chain. It supports 2–128 scanned strips. For other
shapes, the irregular scan route remains available. Physical strip ordering
is experimental: aligned geometry and full placement do not prove that the
text is in correct order. The normalized strips and scan report are exportable;
the supplied ground-truth ordering is never used for inference.

## Verification status

The static artifact, JS syntax, archive contents, Python tests and local
native HTTP smoke were verified during release. GitHub Actions also ran real
Chromium WebAssembly successfully after two mobile-layout fixes. The regression
covers single reconstruction, batch, ZIP metadata replay, mobile viewport and
no JS errors. The public repository runs the same browser checks in its own CI.
See the repository Actions tab for current status.

Official references:
- https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages
- https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages
- https://pyodide.org/en/0.27.7/usage/index.html
