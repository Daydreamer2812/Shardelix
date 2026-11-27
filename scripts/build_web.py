"""Build deterministic GitHub Pages static artifact from the shared Studio and Python package."""
from pathlib import Path
import argparse,shutil,zipfile
ROOT=Path(__file__).resolve().parents[1]
def build(output, wasm=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(ROOT/"src/shardelix/studio.html",output/"index.html")
    shutil.copyfile(ROOT/"web/worker.js",output/"worker.js")
    shutil.copyfile(ROOT/"web/core_bridge.js",output/"core_bridge.js")
    if wasm:
        from build_wasm import build as compile_wasm
        compile_wasm(output)
    with zipfile.ZipFile(output/"engine.zip","w",zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted((ROOT/"src/shardelix").rglob("*.py")):
            if "__pycache__" in p.parts:continue
            info=zipfile.ZipInfo(p.relative_to(ROOT/"src").as_posix(),date_time=(2026,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,p.read_bytes())
    (output/".nojekyll").touch()
    print("Built",output)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",default=str(ROOT/"build/pages"));p.add_argument("--wasm",action="store_true",help="Build C++ accelerator (requires Emscripten)");a=p.parse_args();build(a.out,a.wasm)
