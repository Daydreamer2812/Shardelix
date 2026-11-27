"""Optional C++ kernels; pure Python remains a tested reference backend."""
from __future__ import annotations
import os,sys
from pathlib import Path
import numpy as np

_mode=os.getenv("SHARDELIX_CORE",os.getenv("FRAGMENTLAB_CORE","auto"))
_lib=None
_bridge=None
if _mode!="python":
    if sys.platform=="emscripten":
        try:
            from js import flNative as _bridge
            if not _bridge.available: _bridge=None
        except (ImportError,AttributeError): pass
    else:
        import ctypes as C
        for p in Path(__file__).parent.glob("_fl_core.*"):
            if p.suffix not in (".so",".dll",".dylib"):continue
            try:
                lib=C.CDLL(str(p));lib.fl_abi.restype=C.c_int
                if lib.fl_abi()!=1:continue
                ptr=C.c_void_p;I=C.c_int
                lib.fl_contours.argtypes=[ptr,I,I];lib.fl_contours.restype=ptr
                lib.fl_free.argtypes=[ptr];lib.fl_free.restype=None
                lib.fl_profiles.argtypes=[ptr,I,I,I,ptr];lib.fl_profiles.restype=I
                lib.fl_loop_votes.argtypes=[ptr,I,I,ptr];lib.fl_loop_votes.restype=I
                lib.fl_rerank.argtypes=[ptr,ptr,ptr,ptr,ptr,I,I,I,I,I,ptr];lib.fl_rerank.restype=I
                if hasattr(lib,"fl_arc_votes"):
                    lib.fl_arc_votes.argtypes=[ptr,I,I,I,I,I,I];lib.fl_arc_votes.restype=ptr
                if hasattr(lib,"fl_lattice_contact"):
                    lib.fl_lattice_contact.argtypes=[ptr,I,I,ptr,I,I,I,I,I,ptr];lib.fl_lattice_contact.restype=I
                _lib=lib;break
            except OSError:pass
def backend():
    return "cpp-wasm" if _bridge is not None else "cpp-native" if _lib is not None else "python-reference"
def _ptr(a):return a.ctypes.data
def _js(a):
    from pyodide.ffi import to_js
    return to_js(a.tobytes())
def _array(value,dtype):
    return np.frombuffer(bytes(value.to_py()),dtype=dtype).copy()
def contours(mask):
    if _lib is None and _bridge is None:return None
    a=np.ascontiguousarray(mask,dtype=np.uint8)
    h,w=a.shape
    if _lib is not None:
        p=_lib.fl_contours(_ptr(a),h,w)
        if not p:raise MemoryError("C++ contour allocation failed")
        try:
            n=C.c_int32.from_address(p).value
            raw=np.ctypeslib.as_array((C.c_int32*n).from_address(p)).copy()
        finally:_lib.fl_free(p)
    else:raw=_array(_bridge.contours(_js(a),h,w),np.int32)
    out=[];i=1
    while i<len(raw):
        n=int(raw[i]);i+=1
        d=raw[i:i+3*n].reshape(-1,3);i+=3*n
        out.append((d[:,:2].copy(),d[:,2].astype(np.uint8)))
    return out
def loop_votes(idx):
    if _lib is None and _bridge is None:return None
    a=np.ascontiguousarray(idx,dtype=np.int32);m,k=a.shape
    if not m:return np.zeros_like(a)
    if _lib is not None:
        out=np.empty_like(a)
        if not _lib.fl_loop_votes(_ptr(a),m,k,_ptr(out)):raise ValueError("Invalid loop-vote input")
        return out
    return _array(_bridge.loopVotes(_js(a),m,k),np.int32).reshape(m,k)
def rerank(e,g,f,h,jj,step):
    if _lib is None and _bridge is None:return None
    arrays=[np.ascontiguousarray(a,dtype=np.float32) for a in (e,g,f,h)]
    ids=np.ascontiguousarray(jj,dtype=np.int32);n,k=ids.shape;L=e.shape[1];targets=len(f)
    if _lib is not None:
        out=np.empty((n,k),np.float32)
        if not _lib.fl_rerank(*[_ptr(a) for a in arrays],_ptr(ids),n,targets,k,L,step,_ptr(out)):raise ValueError("Invalid seam inputs")
        return out
    return _array(_bridge.rerank(*[_js(a) for a in arrays],_js(ids),n,targets,k,L,step),np.float32).reshape(n,k)

def profiles(pieces):
    if _lib is None and _bridge is None:return None
    from collections import defaultdict
    groups=defaultdict(list)
    for i,p in enumerate(pieces):groups[p.rgba.shape[:2]].append(i)
    result=[None]*len(pieces)
    for (h,w),ids in groups.items():
        total=2*(h+w)
        # Keep bridge and native temporary buffers bounded.
        batch=max(1,min(1024,12_000_000//max(1,h*w*4+total*24)))
        for lo in range(0,len(ids),batch):
            ii=ids[lo:lo+batch];a=np.stack([pieces[i].rgba for i in ii])
            if _lib is not None:
                out=np.empty((len(ii),total,6),np.float32)
                if not _lib.fl_profiles(_ptr(a),len(ii),h,w,_ptr(out)):raise ValueError("Invalid image descriptors")
            else:out=_array(_bridge.profiles(_js(a),len(ii),h,w),np.float32).reshape(len(ii),total,6)
            for j,i in enumerate(ii):
                off=0;row=[]
                for L in (w,h,w,h):
                    edge=out[j,off:off+L];off+=L
                    row.append((edge[:,:3],edge[:,3:]))
                result[i]=row
    return [edge for p in result for edge in p]

def arc_votes(packed,k,bucket,shards,record_cap,vote_cap):
    """Compact exact chain-code votes; None means kernel not available."""
    a=np.ascontiguousarray(packed,dtype=np.int32)
    if _lib is not None and hasattr(_lib,"fl_arc_votes"):
        p=_lib.fl_arc_votes(_ptr(a),a.size,k,bucket,shards,record_cap,vote_cap)
        if not p:raise MemoryError("Arc index exceeded its allocated record/vote budget")
        try:
            n=C.c_int32.from_address(p).value
            return np.ctypeslib.as_array((C.c_int32*n).from_address(p))[1:].reshape(-1,6).copy()
        finally:_lib.fl_free(p)
    if _bridge is not None and hasattr(_bridge,"arcVotes"):
        result=_bridge.arcVotes(_js(a),a.size,k,bucket,shards,record_cap,vote_cap)
        raw=_array(result,np.int32) if result is not None else np.empty(0,np.int32)
        if not raw.size:raise MemoryError("Arc index exceeded its allocated record/vote budget")
        return raw[1:].reshape(-1,6)
    return None

def arc_votes_available():
    return bool((_lib is not None and hasattr(_lib,"fl_arc_votes")) or
                (_bridge is not None and hasattr(_bridge,"arcVotes")))

def lattice_contact(a,b,r,t):
    """Return (contact count, median RGB error, p80 error), False if invalid."""
    if _lib is not None and hasattr(_lib,"fl_lattice_contact"):
        A=np.ascontiguousarray(a,dtype=np.uint8);B=np.ascontiguousarray(b,dtype=np.uint8)
        out=np.empty(3,np.float64)
        valid=_lib.fl_lattice_contact(_ptr(A),*A.shape[:2],_ptr(B),*B.shape[:2],
                                      int(r),int(t[0]),int(t[1]),_ptr(out))
        return tuple(map(float,out)) if valid else False
    if _bridge is not None and hasattr(_bridge,"latticeContact"):
        A=np.ascontiguousarray(a,dtype=np.uint8);B=np.ascontiguousarray(b,dtype=np.uint8)
        raw=_array(_bridge.latticeContact(_js(A),*A.shape[:2],_js(B),*B.shape[:2],int(r),int(t[0]),int(t[1])),np.float64)
        return tuple(map(float,raw)) if raw.size else False
    return None
