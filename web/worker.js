/* Shared Python reconstruction core, in a dedicated browser worker. */
"use strict";
const BASE = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/";
let engine, loading;
const post = (type, payload={}) => self.postMessage({type,...payload});
self.sendProgress = (value) => post("progress", {status:JSON.parse(value)});
async function load() {
  if (engine) return engine;
  if (!loading) loading = (async()=>{
    post("progress",{status:{state:"running",stage:"Loading Python / WebAssembly…"}});
    importScripts("./core_bridge.js");
    await initShardelixCore();
    importScripts(BASE+"pyodide.js");
    const p = await loadPyodide({indexURL:BASE});
    post("progress",{status:{state:"running",stage:"Loading NumPy, SciPy and Pillow…"}});
    await p.loadPackage(["numpy","scipy","pillow"]);
    const response = await fetch(new URL("engine.zip",self.location.href));
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
      post("reply",{id:m.id,value:JSON.parse(json)});
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
