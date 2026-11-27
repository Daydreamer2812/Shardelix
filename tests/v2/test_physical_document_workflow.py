"""Pixel-only strip reconstruction and export in native and browser modes."""
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from shardelix.workflow12 import run_job


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


if __name__ == "__main__":
    unittest.main()
