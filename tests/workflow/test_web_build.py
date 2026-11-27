import unittest,tempfile,subprocess,sys,zipfile,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
class WebBuildTests(unittest.TestCase):
    def test_static_artifact_shared_core(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t)
            subprocess.run([sys.executable,str(ROOT/"scripts/build_web.py"),"--out",str(d)],check=True,capture_output=True)
            self.assertTrue((d/".nojekyll").exists())
            html=(d/"index.html").read_text()
            for text in ("Fragment generator","Reconstruction","Ground truth","Benchmark generated cases","Performance profile","./worker.js","__SESSION_TOKEN__"):
                self.assertIn(text,html)
            worker=(d/"worker.js").read_text()
            self.assertIn("browser=True",worker);self.assertIn("engine.zip",worker)
            self.assertIn("v0.27.7",worker)
            with zipfile.ZipFile(d/"engine.zip") as z:
                for f in ("shardelix/api.py","shardelix/native_core.py","shardelix/workflow12.py","shardelix/reconstruction/recovery12.py"):
                    self.assertEqual(z.read(f),(ROOT/"src"/f).read_bytes())
            p=subprocess.run([sys.executable,"-c","import shardelix;print(shardelix.__version__)"],cwd=d,
                         env={**os.environ,"PYTHONPATH":str(d/"engine.zip")},capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            self.assertEqual(p.stdout.strip(),"13.0.0")
if __name__=="__main__":unittest.main()
