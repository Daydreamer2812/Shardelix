"""Local OCR seam confidence and unverified scan result integration."""
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from shardelix.reconstruction import physical_ocr as ocr
from shardelix.workflow12 import run_job


class PhysicalOCRTests(unittest.TestCase):
    def test_only_confident_crossing_words(self):
        def row(x,w,conf,word):
            return f"5\t1\t1\t1\t1\t1\t{x}\t20\t{w}\t10\t{conf}\t{word}"
        tsv="\n".join([row(25,20,100,"paper"),row(4,10,100,"left"),
                       row(25,20,25,"weak"),row(25,20,100,"1"),
                       row(65,15,100,"text"),row(25,100,99,"overspan"),
                       row(70,12,80,"word")])
        scores=ocr.seam_scores(tsv,[[20,35,55],[60,73,100]])
        self.assertAlmostEqual(scores[0],5)
        self.assertAlmostEqual(scores[1],4+4*(40/60)**1.5)

    def test_shortlist_bounded(self):
        base=np.random.default_rng(42).random((100,100))
        np.fill_diagonal(base,np.inf)
        pairs=ocr._shortlist(base)
        self.assertLessEqual(len(pairs),900)
        self.assertTrue(all(i!=j for i,j in pairs))
        self.assertEqual(pairs,ocr._shortlist(base))

    def test_native_fallback_keeps_complete_result(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/"upload").mkdir()
            for i in range(5):
                a=np.full((320,48,4),255,np.uint8)
                for y in range(15,300,23):
                    a[y:y+5,3+i:24+i,:3]=0
                Image.fromarray(a).save(root/"upload"/f"p{i}.png")
            conf={"task":"blind","input_mode":"scan","ocr_refine":True}
            with patch("shutil.which",return_value=None):
                result=run_job(root,conf)
            self.assertEqual(result["diagnostics"]["placed"],5)
            self.assertEqual(result["diagnostics"]["scan"]["ocr"]["status"],"unavailable")
            self.assertIsNone(result["metrics"])
            self.assertFalse((root/"runs").exists())
            with zipfile.ZipFile(root/"result/result.zip") as z:
                d=json.loads(z.read("diagnostics.json"))
                self.assertEqual(d["scan"]["ocr"]["status"],"unavailable")
                self.assertFalse(d["accuracy_verified"])

    def test_global_orientation_and_apply(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/"upload").mkdir()
            for i in range(5):
                a=np.full((320,48,4),255,np.uint8)
                for y in range(15,300,23):
                    a[y:y+5,4+i:24+i,:3]=0
                Image.fromarray(a).save(root/"upload"/f"p{i}.png")
            state=run_job(root,{"task":"blind","input_mode":"scan","ocr_refine":False})
            out=root/"result"; work=root/"ocr"
            self.assertNotIn("ocr",state["diagnostics"]["scan"])
            self.assertEqual(ocr.prepare(out,work)["pieces"],5)
            manifest=ocr.batches(work,[[1],[30]])
            self.assertTrue(manifest["global_flip"])
            self.assertTrue(manifest["orientation_decisive"])
            # Synthetic high evidence tests the result writer without OCR dependencies.
            scores=[[100.]*len(b["pairs"]) for b in manifest["batches"]]
            d=ocr.finish(out,work,scores)
            self.assertEqual(d["scan"]["ocr"]["status"],"applied")
            self.assertEqual(d["algorithm"],"physical-document-ocr-wordpath")
            self.assertFalse(d["accuracy_verified"])
            self.assertEqual(d["render_collision_pixels"],0)
            with zipfile.ZipFile(out/"result.zip") as z:
                self.assertEqual(json.loads(z.read("placements.json"))["diagnostics"]["algorithm"],d["algorithm"])


if __name__=="__main__":
    unittest.main()
