"""Regression tests for reference-free structure continuation and spectral SE(2)."""
import unittest
import math
import numpy as np
from shardelix.reconstruction.common import Fragment
from shardelix.reconstruction.scan import _shape,rotation
from shardelix.reconstruction.continuation import make_field,pose_difference,population_weights,confidence_forest
from shardelix.reconstruction.spectral_gap import fft_proposals
from scripts.benchmark_eroded import eroded_fixture


class ClassicalContinuationTests(unittest.TestCase):
    def test_fft_finds_rotated_damaged_pair(self):
        pieces,truth=eroded_fixture(seed=3,erosion=5)
        # Matching sees exactly two observed rasters, not their true poses.
        shapes=[_shape(p,2.) for p in pieces[:2]]
        fields=[make_field(s,26.) for s in shapes]
        hypotheses=fft_proposals(shapes,fields,[(0,1)],26.)[(0,1)]
        A,a=truth[0];B,b=truth[1]
        Q,u=A.T@B,A.T@(b-a)
        errors=[pose_difference(shapes,1,rotation(angle),t,Q,u) for angle,t,_,_ in hypotheses]
        self.assertTrue(any(d<10. and da<math.radians(10.) for d,da in errors))
        self.assertTrue(all(np.isfinite(f.rgb).all() for f in fields))

    def test_population_prior_is_soft_and_permutation_equivariant(self):
        y,x=np.indices((68,68))
        circle=(x-33.)**2+(y-33.)**2<=28.**2
        triangle=(x>4)&(y>4)&(x+y<62)
        def frag(i,m):
            return Fragment(str(i), np.dstack([np.full((*m.shape,3),120,np.uint8),m.astype(np.uint8)*255]))
        shapes=[_shape(frag(i,circle),2.) for i in range(22)]+[_shape(frag(22,triangle),2.)]
        w,enabled=population_weights(shapes)
        self.assertTrue(enabled)
        self.assertGreater(w[-1],0.)
        self.assertLess(w[-1],.5)
        q,_=population_weights(shapes[::-1])
        np.testing.assert_allclose(q,w[::-1])
        self.assertTrue(np.all(w[:-1]==1.))

    def test_context_growth_requires_independent_evidence(self):
        pieces,truth=eroded_fixture(seed=3,erosion=3,keep=(0,1,2,3))
        shapes=[_shape(p,2.) for p in pieces[:4]]
        def candidate(i,j,score,lines,arcs=(0,0)):
            A,a=truth[i];B,b=truth[j]
            return (score,i,j,A.T@B,A.T@(b-a),
                    {"arc_a":arcs[0],"arc_b":arcs[1],"arc_fraction_a":.12,"arc_fraction_b":.12,
                     "families":["spectral","line-contour"],"line_count":lines,"population_weight":1.})
        seed=candidate(0,1,20.,2)
        # Correct additional piece has two crossings, a single-crossing
        # proposal cannot propagate the partial assembly on its own.
        good=candidate(1,2,15.,2,(len(shapes[1].points)//2,0))
        weak=candidate(2,3,14.,1,(len(shapes[2].points)//2,0))
        groups,joins=confidence_forest(shapes,[seed],[seed,good,weak])
        self.assertEqual(set(groups[0]),{0,1,2})
        self.assertEqual(len(joins),2)
        self.assertTrue(joins[-1]["contextual_growth"])

if __name__=="__main__":unittest.main()
