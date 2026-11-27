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
