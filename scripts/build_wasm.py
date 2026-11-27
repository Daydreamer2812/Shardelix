"""Compile identical C++ kernels for a browser Worker with Emscripten."""
from pathlib import Path
import argparse, os, shutil, subprocess
ROOT=Path(__file__).resolve().parents[1]
def build(out):
    compiler=os.environ.get("EMXX") or shutil.which("em++")
    if not compiler: raise RuntimeError("Emscripten SDK (em++) required")
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    functions="['_malloc','_free','_fl_abi','_fl_contours','_fl_arc_votes','_fl_lattice_contact','_fl_free','_fl_loop_votes','_fl_rerank','_fl_profiles']"
    subprocess.run([compiler,str(ROOT/"cpp/core.cpp"),"-std=c++20","-O3","-ffp-contract=off","-msimd128","-DNDEBUG","--no-entry","-fexceptions",
        "-sMODULARIZE=1","-sEXPORT_NAME=createShardelixCore","-sENVIRONMENT=worker","-sALLOW_MEMORY_GROWTH=1",
        "-sEXPORTED_RUNTIME_METHODS=['HEAPU8','HEAP32']", "-sEXPORTED_FUNCTIONS="+functions,"-o",str(out/"fl_core.js")],check=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);build(p.parse_args().out)
