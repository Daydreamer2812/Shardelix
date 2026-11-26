# External document data and evaluation protocol (V5)

- LSU GVC Lab official source: https://www.ece.lsu.edu/xinli/DocReassembly/index.html
- Official download: https://www.ece.lsu.edu/xinli/DocReassembly/data/DocDataset.zip
- Research: Yongqing Liang and Xin Li, *Reassembling Shredded Document Stripes Using Word-path Metric and Greedy Composition Optimal Matching Solver*, IEEE TMM, 2019. https://ieeexplore.ieee.org/document/8844125
- User-supplied DocDataset.zip matches official bytes. SHA-256:
  d00ca3aabff4d099f716cc5f98ac9bef9e841bd2a891f8303e44cd27271ee364.
- Contents: 60 digital puzzles (20,30,40,60 strips); 3 physical scans (27,39,56);
  one mixed-orientation 36-strip puzzle. Raw benchmark is not included: use the provided ZIP.
- Blind loader accepts numeric individual PNG files and randomizes names/order.
  It never loads puzzle_img.png, order.txt or gt before solving. Public orientation
  case flag is metadata; original orientations aren't supplied. Physical scans have
  unequal strip widths and are treated as approximate order reconstruction.
- Train: 32 cases on even source document IDs. Holdout: 28 on odd source IDs.
  Group by document ID, not count, to avoid training on the same page as holdout.
  These are V5 held-out for current tuning, not entirely unseen historically.
  Frozen parameter selection is by development adjacency, tie-break order.
- Baseline is unweighted boundary L1 plus the same path machinery, with document
  margin and blank-split priors disabled; **not** a claimed full V4 comparison.
- Directed neighbor accuracy = correct ordered adjacencies / (n-1).
  Since all strips placed exactly once, equals directed adjacency F1.
  Direct order accuracy = indices correct / n; no circular shift or translation.
  Orientation case permits global reversal of the complete chain; individual
  rotation accuracy not scored (no matching orientation truth in order.txt).
- For digital cases only, exact pixel fraction and MAE use gt after prediction,
  if shape matches. Coverage of real scans is relative to inferred canvas only.
- --previews outputs assembly, cut overlay and predicted layout for each case.
  Metrics are in summary.csv/json. The archive includes summary and selected
  previews, not the original external dataset. Do not redistribute LSU raw data.


## User-supplied Cho 432
- 20 directories, each 432 individual 28x28 PNGs, with row/column GT.
- Public grid 18x24, canvas 504x672. Loader independently permutes and anonymizes IDs, quarter-turns optional.
- For the 432 benchmark, inference receives pieces and public dimensions only.
  GT is read after prediction. For 1024 recuts, original image is reconstructed
  from GT in generation ONLY, then truth is moved outside solver input.
- V6 single-consensus parameters chosen on even-numbered cases (10 images),
  frozen then tested on odd-numbered images (10). Multi-hypothesis integrated
  results are a subsequent full-suite evaluation (odd results previously observed).
- Baseline V5 measured on identical anonymous inputs and rotations. Duplicate
  metrics are diagnostic; strict ID DPA/F1 never adjusted for identical pixels.
- User-supplied archive not redistributed; all outputs reproducible locally.
