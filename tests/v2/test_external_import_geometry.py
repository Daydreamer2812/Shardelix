"""External imports without case.json: detect tile layout without inferring poses."""
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image
from shardelix.workflow12 import inspect_uploads, safe_zip


def png(size, mode="RGB", transparent=False):
    im=Image.new(mode,size,"white")
    if transparent:im.putpixel((0,0),(0,0,0,0))
    f=io.BytesIO();im.save(f,format="PNG");return f.getvalue()


class ExternalImportGeometryTests(unittest.TestCase):
    def test_432_external_tiles(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);path=root/"pieces.zip";tile=png((28,28))
            with zipfile.ZipFile(path,"w") as z:
                for i in range(432):z.writestr(f"pieces/piece_{i:04}.png",tile)
            info=inspect_uploads([path],browser=True)
            self.assertIsNone(info["metadata"])
            self.assertEqual(info["observed_count"],432)
            hint=info["suggested_grid"]
            self.assertEqual((hint["tile_width"],hint["tile_height"]),(28,28))
            self.assertEqual(hint["layouts"][0],{"rows":18,"cols":24,"height":504,"width":672})
            self.assertIn({"rows":24,"cols":18,"height":672,"width":504},hint["layouts"])
            self.assertEqual(safe_zip(path,root/"case",browser=True), (None,432))

    def test_individual_opaque_and_irregular_tiles(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);files=[root/f"p{i}.png" for i in range(12)]
            for f in files:f.write_bytes(png((20,20),"RGBA"))
            info=inspect_uploads(files,browser=True)
            self.assertEqual(info["suggested_grid"]["layouts"][0],
                             {"rows":3,"cols":4,"height":60,"width":80})
            files[0].write_bytes(png((20,20),"RGBA",True))
            self.assertIsNone(inspect_uploads(files,browser=True)["suggested_grid"])
            files[0].write_bytes(png((21,20),"RGB"))
            self.assertIsNone(inspect_uploads(files,browser=True)["suggested_grid"])

    def test_authoritative_archive_metadata(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/"pieces.zip"
            meta=dict(width=80,height=60,grid=[3,4],cut="grid",piece_count=12,rotations=False)
            with zipfile.ZipFile(path,"w") as z:
                for i in range(12):z.writestr(f"pieces/piece_{i}.png",png((20,20)))
                z.writestr("case.json",json.dumps(meta))
            info=inspect_uploads([path],browser=True)
            self.assertEqual(info["metadata"],meta)
            self.assertIsNone(info["suggested_grid"])

if __name__=="__main__":unittest.main()
