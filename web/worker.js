/* Shared Python reconstruction core, in a dedicated browser worker. */
"use strict";
const BASE = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/";
let engine, loading;
const revision=new URL(self.location.href).searchParams.get("v");
function assetURL(name){
  const url=new URL(name,self.location.href);
  if(revision)url.searchParams.set("v",revision);
  return url.href;
}
const post = (type, payload={}) => self.postMessage({type,...payload});
self.sendProgress = (value) => post("progress", {status:JSON.parse(value)});
async function load() {
  if (engine) return engine;
  if (!loading) loading = (async()=>{
    post("progress",{status:{state:"running",stage:"Loading Python / WebAssembly…"}});
    importScripts(assetURL("core_bridge.js"));
    await initShardelixCore();
    importScripts(BASE+"pyodide.js");
    const p = await loadPyodide({indexURL:BASE});
    post("progress",{status:{state:"running",stage:"Loading NumPy, SciPy and Pillow…"}});
    await p.loadPackage(["numpy","scipy","pillow"]);
    const response = await fetch(assetURL("engine.zip"),{cache:"no-cache"});
    if (!response.ok) throw new Error("Engine bundle missing; run scripts/build_web.py");
    p.FS.mkdirTree("/engine");
    p.unpackArchive(await response.arrayBuffer(),"zip",{extractDir:"/engine"});
    p.runPython("import sys; sys.path.insert(0, '/engine'); import shardelix");
    // Validate the real C++/WASM buffer bridge once before claiming it.
    if(self.flNative?.available){
      p.runPython("import numpy as np\nfrom shardelix.native_core import contours,loop_votes,rerank,backend,profiles\nassert backend()=='cpp-wasm'\nassert len(contours(np.ones((2,3),dtype=bool))[0][0])==10\nassert np.array_equal(loop_votes(np.full((4,2),-1,np.int32)),np.zeros((4,2),np.int32))\na=np.zeros((1,8,3),np.float32)\nassert float(rerank(a,a,a,a,np.zeros((1,1),np.int32),1)[0,0])==0\nfrom shardelix import Fragment\nassert len(profiles([Fragment('x',np.zeros((2,3,4),np.uint8))]))==4");
    }
    return engine = p;
  })().catch(e=>{loading=null;throw e});
  return loading;
}
function safeName(value){
  const name=String(value).replaceAll("\\","/").split("/").at(-1);
  if(!name||name==="."||name===".."||name.includes("\0")) throw new Error("Invalid filename");
  return name;
}
async function putFiles(p,root,files,truth=null){
  p.globals.set("bridge_root",root);
  p.runPython("from pathlib import Path\nimport shutil\nshutil.rmtree(bridge_root,ignore_errors=True)\n(Path(bridge_root)/'upload').mkdir(parents=True)");
  const names=new Set();let size=0;
  for(const f of files){
    const name=safeName(f.name);
    if(names.has(name))throw new Error("Duplicate filename: "+name);
    names.add(name);size+=f.size;
    if(size>32*1024*1024)throw new Error("Browser upload exceeds 32 MiB");
    p.FS.writeFile(root+"/upload/"+name,new Uint8Array(await f.arrayBuffer()));
  }
  if(truth){
    size+=truth.size;
    if(size>32*1024*1024)throw new Error("Combined fragments and reference exceed 32 MiB");
    p.FS.mkdirTree(root+"/truth");
    p.FS.writeFile(root+"/truth/"+safeName(truth.name),new Uint8Array(await truth.arrayBuffer()));
  }
}
// OCR libraries and language data are downloaded; all candidate pixels stay local.
const OCR_CDN="https://cdn.jsdelivr.net/npm/";
function deadline(promise, ms){
  let timer;
  return Promise.race([promise,new Promise((_,reject)=>{timer=setTimeout(()=>reject(new Error("OCR timeout")),ms)})]).finally(()=>clearTimeout(timer));
}
async function refineScan(p,state,config){
  if(config.ocr_refine===false||state.diagnostics?.scan?.kind!=="document-strips")return state;
  const started=performance.now(), workers=[];
  const progress=stage=>post("progress",{status:{state:"running",stage,case:1,total_cases:1,elapsed_seconds:(performance.now()-started)/1000+(state.diagnostics.workflow_elapsed_seconds||0)}});
  const py=(code)=>JSON.parse(p.runPython(code));
  const sync=(diag)=>{
    // Preserve the workflow response schema while updating the final rendered result.
    for(const key of Object.keys(state.diagnostics))if(key in diag)state.diagnostics[key]=diag[key];
    return state;
  };
  try{
    progress("Preparing OCR candidate seams");
    const plan=py("from shardelix.reconstruction import physical_ocr as _ocr\nimport json\njson.dumps(_ocr.prepare('/job/result','/job/ocr-work'))");
    if(!plan)return state;
    progress("Loading local English OCR model (first run may take a moment)");
    if(!self.Tesseract)importScripts(OCR_CDN+"tesseract.js@6.0.1/dist/tesseract.min.js");
    let fail;
    const failed=new Promise((_,reject)=>{fail=reject});
    // Attach a handler immediately, even if a model fails between OCR jobs.
    failed.catch(()=>{});
    const guarded=(promise,ms=90000)=>deadline(Promise.race([promise,failed]),ms);
    const count=Math.min(2,Math.max(1,(navigator.hardwareConcurrency||2)-1));
    for(let i=0;i<count;i++){
      const worker=await guarded(self.Tesseract.createWorker("eng",1,{
        workerPath:OCR_CDN+"tesseract.js@6.0.1/dist/worker.min.js",
        corePath:OCR_CDN+"tesseract.js-core@6.0.0",
        langPath:"https://tessdata.projectnaptha.com/4.0.0_fast",
        errorHandler:e=>fail(new Error(String(e)))
      }),120000);
      workers.push(worker);
      await guarded(worker.setParameters({tessedit_pageseg_mode:"11",user_defined_dpi:"150"}));
    }
    const recognize=async(worker,batch)=>{
      const result=await guarded(worker.recognize(p.FS.readFile(batch.path),{},{text:false,tsv:true}),120000);
      if(typeof result.data.tsv!=="string")throw new Error("OCR output unavailable");
      p.globals.set("ocr_tsv",result.data.tsv);
      p.globals.set("ocr_spans",JSON.stringify(batch.spans));
      try{return py("json.dumps(_ocr.seam_scores(ocr_tsv,json.loads(ocr_spans)))")}
      finally{p.globals.delete("ocr_tsv");p.globals.delete("ocr_spans")}
    };
    progress("Checking upright direction with English OCR");
    const orientation=[];
    for(let i=0;i<plan.orientation.length;i+=workers.length){
      orientation.push(...await Promise.all(plan.orientation.slice(i,i+workers.length).map((b,j)=>recognize(workers[j],b))));
    }
    p.globals.set("ocr_orientation",JSON.stringify(orientation));
    const manifest=py("json.dumps(_ocr.batches('/job/ocr-work',json.loads(ocr_orientation)))");
    p.globals.delete("ocr_orientation");
    let cursor=0,completed=0;
    const results=new Array(manifest.batches.length);
    await Promise.all(workers.map(async worker=>{
      while(cursor<manifest.batches.length){
        const i=cursor++;
        results[i]=await recognize(worker,manifest.batches[i]);
        progress("OCR seam verification "+(++completed)+"/"+manifest.batches.length);
      }
    }));
    progress("Optimizing OCR-supported document order");
    p.globals.set("ocr_scores",JSON.stringify(results));
    p.globals.set("ocr_elapsed",(performance.now()-started)/1000);
    try{return sync(py("json.dumps(_ocr.finish('/job/result','/job/ocr-work',json.loads(ocr_scores),ocr_elapsed))"))}
    finally{p.globals.delete("ocr_scores");p.globals.delete("ocr_elapsed")}
  }catch(e){
    // Offline/model failure must never discard a valid pixel-based solution.
    console.warn("Optional OCR refinement unavailable",String(e));
    p.globals.set("ocr_elapsed",(performance.now()-started)/1000);
    try{return sync(py("json.dumps(_ocr.unavailable('/job/result',ocr_elapsed))"))}
    finally{p.globals.delete("ocr_elapsed")}
  }finally{
    for(const worker of workers)try{await deadline(worker.terminate(),3000)}catch(e){}
    p.runPython("shutil.rmtree('/job/ocr-work',ignore_errors=True)");
  }
}
async function handle(m){
  try{
    const p=await load();
    if(m.action==="inspect"){
      await putFiles(p,"/inspect",m.files);
      const json=p.runPython("import json\nfrom shardelix.workflow12 import inspect_uploads\njson.dumps(inspect_uploads(list((Path(bridge_root)/'upload').iterdir()),browser=True))");
      post("reply",{id:m.id,value:JSON.parse(json)});
      p.runPython("shutil.rmtree('/inspect',ignore_errors=True)");
    } else if(m.action==="run"){
      await putFiles(p,"/job",m.files,m.truth);
      p.globals.set("bridge_config",JSON.stringify(m.config));
      const json=await p.runPythonAsync("import json\nfrom js import sendProgress\nfrom shardelix.workflow12 import run_job\njson.dumps(run_job(Path(bridge_root),json.loads(bridge_config),lambda s:sendProgress(json.dumps(s)),browser=True),allow_nan=False)");
      const state=await refineScan(p,JSON.parse(json),m.config);
      post("reply",{id:m.id,value:state});
    } else if(m.action==="file"){
      const allowed=p.runPython("import json\nfrom shardelix.workflow12 import DOWNLOADS\njson.dumps(sorted(DOWNLOADS))");
      if(!JSON.parse(allowed).includes(m.name))throw new Error("Unknown export");
      const bytes=p.FS.readFile("/job/result/"+m.name);
      const buffer=bytes.slice().buffer;
      self.postMessage({type:"reply",id:m.id,value:buffer},[buffer]);
    } else throw new Error("Unknown action");
  }catch(e){
    const detail=String(e.message||e).trim(),validation=detail.split("\n").at(-1).match(/^(?:ValueError|FileExistsError): (.+)$/);
    post("reply",{id:m.id,error:validation?validation[1]:detail.slice(-1600)})
  }
}
let queue=Promise.resolve();
self.onmessage=({data:m})=>{queue=queue.then(()=>handle(m))};
