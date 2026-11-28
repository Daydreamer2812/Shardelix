"""Pixel-only strip reconstruction and export in native and browser modes."""
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from shardelix.workflow import run_job, inspect_uploads


class PhysicalDocumentWorkflowTests(unittest.TestCase):
    def test_pixel_only_scan_native_and_browser(self):
        poses = []
        for browser in (False, True):
            with self.subTest(browser=browser), tempfile.TemporaryDirectory() as t:
                root = Path(t)
                (root/"upload").mkdir()
                for i in range(5):
                    a = np.full((320, 48, 4), 255, np.uint8)
                    for y in range(15, 300, 23):
                        a[y:y+5, 3+i:24+i, :3] = 0
                    Image.fromarray(a).save(root/"upload"/f"p{i}.png")
                config = {"task": "blind", "input_mode": "scan"}
                # Old clients may still send this flag. It must have no effect.
                if not browser:
                    config["ocr_refine"] = True
                with patch("shutil.which", side_effect=AssertionError("External program lookup")):
                    result = run_job(root, config, browser=browser)
                d = result["diagnostics"]
                self.assertEqual(d["algorithm"], "physical-document-foreground-multiscale")
                self.assertEqual(d["scan"]["kind"], "document-strips")
                self.assertEqual(d["placed"], 5)
                self.assertEqual(d["render_collision_pixels"], 0)
                self.assertFalse(d["accuracy_verified"])
                self.assertNotIn("ocr", d["scan"])
                self.assertIsNone(result["metrics"])
                with zipfile.ZipFile(root/"result/result.zip") as z:
                    archived = json.loads(z.read("diagnostics.json"))
                    self.assertEqual(archived["algorithm"], d["algorithm"])
                    self.assertTrue(archived["orientation_ambiguous"])
                    self.assertNotIn("ocr", archived["scan"])
                    poses.append(json.loads(z.read("placements.json"))["poses"])
        self.assertEqual(poses[0], poses[1])

    def test_color_strips_with_transparent_padding_auto_route(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/"upload").mkdir()
            y,x=np.indices((128,56))
            source=np.stack([120+80*np.sin(.13*x+.03*y),140+65*np.cos(.21*x-.06*y),
                             120+70*np.sin(.08*x-.09*y)],axis=2).astype(np.uint8)
            for i in (3,1,0,2):
                tile=np.zeros((150,100,4),np.uint8)
                tile[11:139,40:54,:3]=source[:,14*i:14*(i+1)]
                tile[11:139,40:54,3]=255
                if i%2:tile=np.rot90(tile)
                Image.fromarray(tile).save(root/"upload"/f"p{(i*7)%11}.png")
            info=inspect_uploads(sorted((root/"upload").iterdir()),True)
            self.assertEqual(info["route"],"scan-strips")
            state=run_job(root,{"task":"blind","input_mode":"auto","auto_layout":True},browser=True)
            self.assertEqual(state["diagnostics"]["algorithm"],"physical-color-strip-chain")
            self.assertEqual(state["diagnostics"]["placed"],4)
            self.assertEqual(state["diagnostics"]["render_collision_pixels"],0)
            with Image.open(root/"result/assembly.png") as im:
                self.assertEqual(im.size,(56,128))

    def test_patch_similarity_is_bounded_and_finite(self):
        from shardelix.reconstruction.strips.physical import _patch_recurrence
        images=[]
        for k in range(4):
            a=np.zeros((92,26),np.float32)
            for y in range(9,83,13):
                a[y:y+3,3:22]=1
                a[y:y+7,7+k:9+k]=1
            images.append(a)
        c=_patch_recurrence(images,6,0)
        self.assertEqual(c.shape,(4,4))
        self.assertTrue(np.isfinite(c).all())
        self.assertGreaterEqual(float(c.min()),0)
        np.testing.assert_allclose(c,_patch_recurrence(images,6,0))


if __name__ == "__main__":
    unittest.main()
