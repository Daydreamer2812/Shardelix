"""Build the optional native C++ ABI without a Python extension dependency."""
from pathlib import Path
import argparse, os, platform, shutil, subprocess
ROOT=Path(__file__).resolve().parents[1]
def build():
    compiler=os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++")
    if not compiler: raise RuntimeError("C++20 compiler (g++ or clang++) required")
    ext={"Windows":".dll","Darwin":".dylib"}.get(platform.system(),".so")
    target=ROOT/"src/shardelix"/("_fl_core"+ext)
    args=[compiler,"-std=c++20","-O3","-ffp-contract=off","-shared","-fPIC","-DNDEBUG",str(ROOT/"cpp/core.cpp"),"-o",str(target)]
    subprocess.run(args,check=True)
    print("Built",target)
    return target
if __name__=="__main__": build()
