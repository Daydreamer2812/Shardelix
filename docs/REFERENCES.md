# Research references

Shardelix implements its own classical matching and assembly pipeline. The
following published work motivates comparisons and future research; it does
not imply that their implementations are reproduced or that a matched
state-of-the-art comparison has been completed.

## Digital jigsaw reconstruction

- T. Cho, S. Avidan and W. Freeman, *A Probabilistic Image Jigsaw Puzzle Solver*, CVPR 2010.
  Pairwise compatibility and global assembly.
- A. C. Gallagher, *Jigsaw Puzzles with Pieces of Unknown Orientation*, CVPR 2012.
  Rotation-aware compatibility and gradient continuity.
- Y. Son, J. Hays and D. Cooper, [Solving Square Jigsaw Puzzles with Loop Constraints](https://cs.brown.edu/people/jh23/papers/puzzle_eccv14.pdf), ECCV 2014.
  Geometric consistency and loop constraints.
- D. Paikin and A. Tal, [Solving Multiple Square Jigsaw Puzzles With Missing Pieces](https://openaccess.thecvf.com/content_cvpr_2015/papers/Paikin_Solving_Multiple_Square_2015_CVPR_paper.pdf), CVPR 2015.
  Reciprocal matching, component growth and incomplete puzzles.
- Y. Son et al., [Solving Small-Piece Jigsaw Puzzles by Growing Consensus](https://openaccess.thecvf.com/content_cvpr_2016/html/Son_Solving_Small-Piece_Jigsaw_CVPR_2016_paper.html), CVPR 2016.
  Contextual verification and consensus.

## Document reconstruction

- Y. Liang and X. Li, *Reassembling Shredded Document Stripes Using Word-path Metric and Greedy Composition Optimal Matching Solver*, IEEE Transactions on Multimedia, 2019.
- F. A. Paixao et al., *Fast(er) Reconstruction of Shredded Text Documents via Self-Supervised Deep Asymmetric Metric Learning*, CVPR 2020.
  Direction-dependent compatibility; learned models are outside Shardelix's CPU-classical scope.

Reported measurements from papers are not directly comparable with the
synthetic V12 cases. A valid comparison requires matched input, split,
orientation/missing policy, metrics and hardware.

## Archaeological and degraded planar fragments

- H. C. G. Leitão and J. Stolfi, [A Multiscale Method for the Reassembly of Two-Dimensional Fragmented Objects](https://doi.org/10.1109/TPAMI.2002.1033215), TPAMI 2002. Multi-resolution contour matching.
- N. Derech, A. Tal and I. Shimshoni, [Solving Archaeological Puzzles](https://arxiv.org/abs/1812.10553), Pattern Recognition 2021. Extrapolate pictorial information beyond eroded edges, estimate pairwise rigid poses and account for competing matches. The new gap-aware matcher is an independent, bounded approximation, not their solver.
- K. Zhang and X. Li, [A graph-based optimization algorithm for fragmented image reassembly](https://repository.lsu.edu/eecs_pubs/758/), 2015. Pairwise proposals and global consistency.
- C.-H. Le and X. Li, [JigsawNet: Shredded Image Reassembly Using Convolutional Neural Network and Loop-Based Composition](https://arxiv.org/abs/1809.04137), 2019. Loop-based selection motivates geometric cycle checks; the learned model is not implemented.
- S. Choi, Q.-Y. Zhou and V. Koltun, [Robust Reconstruction of Indoor Scenes](https://openaccess.thecvf.com/content_cvpr_2015/html/Choi_Robust_Reconstruction_of_2015_CVPR_paper.html), CVPR 2015. Inspiration for rejecting inconsistent graph edges and refining poses robustly; Shardelix uses soft-L1 least squares on local SE(2) components rather than their 3D line-process method.
- D. Bridger, D. Danon and A. Tal, [Solving Jigsaw Puzzles With Eroded Boundaries](https://openaccess.thecvf.com/content_CVPR_2020/html/Bridger_Solving_Jigsaw_Puzzles_With_Eroded_Boundaries_CVPR_2020_paper.html), CVPR 2020. Learned boundary completion and compatibility are a promising future optional stage, not implemented.
- [DAF Challenge official notice](https://vision.unipv.it/DAFchallenge/DAF-notice.html). The original challenge supplies the reference fresco. Blind recovery of this data in Shardelix is a different, harder protocol; reference and annotated poses are evaluation-only.

See [eroded recovery design and results](ERODED_RECOVERY.md) for implementation scope and limitations.

### Additional classical techniques for heavily eroded pieces

- M. Khoroshiltseva, L. Palmieri, S. Aslan, S. Vascon and M. Pelillo,
  [Nash Meets Wertheimer: Using Good Continuation in Jigsaw Puzzles](https://openaccess.thecvf.com/content/ACCV2024/papers/Khoroshiltseva_Nash_Meets_Wertheimer_Using_Good_Continuation_in_Jigsaw_Puzzles_ACCV_2024_paper.pdf), ACCV 2024.
  Motivates continued pictorial lines across gaps. Shardelix extracts its own
  classical multichannel structure-tensor line exits; it does not use DeepLSD
  or reproduce their game-theoretic solver.
- D. Padfield, [Masked Object Registration in the Fourier Domain](https://pubmed.ncbi.nlm.nih.gov/22203712/),
  IEEE TIP 2012. Motivates evaluating masked correlations across translations
  efficiently. Our spectral stage sweeps a discrete set of rotations, computes
  weighted bidirectional band SSD and physical overlap, then verifies SE(2)
  proposals on observed pixels.
- B. S. Reddy and B. N. Chatterji, [An FFT-Based Technique for Translation,
  Rotation, and Scale-Invariant Image Registration](https://doi.org/10.1109/83.506761),
  IEEE TIP 1996. Classical frequency-domain alignment; our solver assumes
  common working scale and does not implement Fourier-Mellin estimation.
- A. Criminisi, P. Pérez and K. Toyama, [Region Filling and Object Removal
  by Exemplar-Based Image Inpainting](https://doi.org/10.1109/TIP.2004.833105),
  IEEE TIP 2004. Classical structure propagation motivates the distinction
  between inferred appearance evidence and observed artifact material.
