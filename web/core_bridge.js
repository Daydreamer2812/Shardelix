// Flat byte-buffer bridge for C++ code compiled by Emscripten.
"use strict";
async function initFragmentLabCore(){
  try{
    importScripts("./fl_core.js");
    const m=await createFragmentLabCore({locateFile:f=>new URL(f,self.location.href).href});
    if(m._fl_abi()!==1)throw Error("C++ ABI mismatch");
    const copy=(v,ptrs)=>{const a=v instanceof Uint8Array?v:new Uint8Array(v);const p=m._malloc(a.byteLength);if(!p)throw Error("WASM allocation failed");ptrs.push(p);m.HEAPU8.set(a,p);return p;};
    const alloc=(n,ptrs)=>{const p=m._malloc(n);if(!p)throw Error("WASM allocation failed");ptrs.push(p);return p;};
    const withPtrs=fn=>{const ptrs=[];try{return fn(ptrs)}finally{for(const p of ptrs)m._free(p)}};
    self.flNative={
      available:true,
      contours:(a,h,w)=>withPtrs(ps=>{
        const p=m._fl_contours(copy(a,ps),h,w);if(!p)throw Error("Contour allocation failed");
        try{return m.HEAPU8.slice(p,p+4*m.HEAP32[p>>>2])}finally{m._fl_free(p)}
      }),
      profiles:(a,n,h,w)=>withPtrs(ps=>{const p=copy(a,ps),sz=n*2*(h+w)*6*4,o=alloc(sz,ps);if(!m._fl_profiles(p,n,h,w,o))throw Error("Invalid edge descriptors");return m.HEAPU8.slice(o,o+sz)}),
      loopVotes:(a,n,k)=>withPtrs(ps=>{const p=copy(a,ps),o=alloc(n*k*4,ps);if(!m._fl_loop_votes(p,n,k,o))throw Error("Invalid graph");return m.HEAPU8.slice(o,o+n*k*4)}),
      rerank:(e,g,f,h,j,n,t,k,L,s)=>withPtrs(ps=>{const p=[e,g,f,h,j].map(a=>copy(a,ps)),o=alloc(n*k*4,ps);if(!m._fl_rerank(...p,n,t,k,L,s,o))throw Error("Invalid seam buffers");return m.HEAPU8.slice(o,o+n*k*4)})
    };
  }catch(e){self.flNative={available:false,error:String(e)};}
}
