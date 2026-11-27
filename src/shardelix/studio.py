"""Dependency-free localhost Studio with a bounded, cancellable solver process.

No remote access, no publication. Uploaded paths are never extracted verbatim.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
import zipfile
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse,parse_qs
from .reconstruction.limits import GRID_MAX,MAX_PIXELS

from .workflow12 import native_upload_limit, DOWNLOADS, run_job, inspect_uploads, safe_zip
def atomic_json(path,value):
    tmp=path.with_suffix(".tmp");tmp.write_text(json.dumps(value,allow_nan=False),encoding="utf-8");tmp.replace(path)

def worker(root: Path):
    status=root/"status.json"
    try:
        conf=json.loads((root/"config.json").read_text())
        result=run_job(root,conf,lambda s:atomic_json(status,s))
        atomic_json(status,result)
        return 0
    except Exception as e:
        atomic_json(status,{"state":"error","stage":"Input or solver error","error":str(e)[:500]})
        return 1

class State:
    def __init__(self,root):
        self.root=Path(root);self.lock=threading.Lock();self.process=None;self.job=None
        self.token=secrets.token_urlsafe(20);self.started=None
    def status(self):
        if self.job is None:return {"state":"idle","stage":"Ready"}
        p=self.job/"status.json"
        try:s=json.loads(p.read_text())
        except (OSError,ValueError):s={"state":"running","stage":"Starting solver"}
        if self.process and self.process.poll() is not None and s["state"]=="running":
            s={"state":"error","stage":"Worker exited","error":"Solver process stopped; check memory or inputs."}
        if self.started is not None and s.get("state")=="running":s["elapsed_seconds"]=round(time.monotonic()-self.started,1)
        return s
    def start(self,conf,files,truth=None,matching_reference=None):
        with self.lock:
            if self.process and self.process.poll() is None:
                current=self.status()
                if current.get("state") not in {"done","error","cancelled"}:raise ValueError("A job is already running")
                # Worker may have written its final status moments before exit.
                try:self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:raise ValueError("Previous job is finishing; retry shortly")
            if self.job:shutil.rmtree(self.job,ignore_errors=True)
            self.job=Path(tempfile.mkdtemp(dir=self.root,prefix="job-"));up=self.job/"upload";up.mkdir()
            if not files or len(files)>GRID_MAX:raise ValueError("Invalid upload count")
            for filename,data in files:
                leaf=filename.replace("\\","/").split("/")[-1]
                if not leaf or leaf in {".",".."} or (up/leaf).exists():raise ValueError("Invalid or duplicate file name")
                (up/leaf).write_bytes(data)
            for dirname,item in (("truth",truth),("matching_reference",matching_reference)):
                if not item:continue
                name,data=item
                leaf=name.replace("\\","/").split("/")[-1]
                if not leaf or leaf in {".",".."}:raise ValueError("Invalid reference filename")
                folder=self.job/dirname;folder.mkdir()
                (folder/leaf).write_bytes(data)
            atomic_json(self.job/"config.json",conf)
            atomic_json(self.job/"status.json",{"state":"running","stage":"Starting worker"})
            self.started=time.monotonic()
            self.process=subprocess.Popen([sys.executable,"-m","shardelix.studio","--worker",str(self.job)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    def cancel(self):
        with self.lock:
            if self.process and self.process.poll() is None:
                self.process.terminate()
                try:self.process.wait(3)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
                atomic_json(self.job/"status.json",{"state":"cancelled","stage":"Cancelled"})

def serve(port=8765,open_browser=True):
    with tempfile.TemporaryDirectory(prefix="shardelix-studio-") as td:
        state=State(td)
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def reply(self,obj,status=200):
                data=json.dumps(obj,allow_nan=False).encode();self.send_response(status);self.send_header("Content-Type","application/json");self.send_header("Content-Length",str(len(data)));self.send_header("Cache-Control","no-store");self.end_headers();self.wfile.write(data)
            def local_host(self):
                return self.headers.get("Host","").split(":")[0] in {"127.0.0.1","localhost"}
            def do_GET(self):
                if not self.local_host():self.send_error(403);return
                u=urlparse(self.path)
                if u.path=="/":
                    html=Path(__file__).with_name("studio.html").read_text().replace("__SESSION_TOKEN__",state.token).replace("__NATIVE_UPLOAD_BYTES__",str(native_upload_limit(state.root,multipart=True))).encode()
                    self.send_response(200);self.send_header("Content-Type","text/html; charset=utf-8");self.send_header("Content-Length",str(len(html)));self.end_headers();self.wfile.write(html)
                elif u.path=="/api/status":self.reply(state.status())
                elif u.path=="/api/file":
                    name=parse_qs(u.query).get("name",[""])[0]
                    p=state.job/"result"/name if state.job and name in DOWNLOADS else None
                    if not p or not p.is_file():self.send_error(404);return
                    self.send_response(200);self.send_header("Content-Type","image/png" if name.endswith(".png") else "application/zip" if name.endswith(".zip") else "text/csv; charset=utf-8" if name.endswith(".csv") else "application/json");self.send_header("Content-Length",str(p.stat().st_size));self.send_header("Cache-Control","no-store");self.end_headers()
                    with p.open("rb") as f:shutil.copyfileobj(f,self.wfile)
                else:self.send_error(404)
            def do_POST(self):
                if not self.local_host() or self.headers.get("X-Studio-Token")!=state.token:self.reply({"error":"Invalid session"},403);return
                try:
                    if self.path=="/api/cancel":state.cancel();self.reply(state.status());return
                    if self.path not in {"/api/start","/api/inspect"}:self.send_error(404);return
                    size=int(self.headers.get("Content-Length",0))
                    limit=native_upload_limit(state.root,multipart=True)
                    if not 0<size<=limit:self.reply({"error":f"Upload exceeds currently available Studio budget ({limit//1024**2} MiB) or is empty"},413);return
                    if "multipart/form-data" not in self.headers.get("Content-Type",""):raise ValueError("Expected multipart form")
                    payload=self.rfile.read(size)
                    message=BytesParser(policy=default).parsebytes(("Content-Type: "+self.headers["Content-Type"]+"\r\nMIME-Version: 1.0\r\n\r\n").encode()+payload)
                    conf=None;files=[];truth=None;matching_reference=None
                    for part in message.iter_parts():
                        name=part.get_param("name",header="content-disposition")
                        if name=="config":conf=json.loads(part.get_payload(decode=True))
                        elif name=="files" and part.get_filename():files.append((part.get_filename(),part.get_payload(decode=True)))
                        elif name=="truth" and part.get_filename():
                            if truth is not None:raise ValueError("Only one reference allowed")
                            truth=(part.get_filename(),part.get_payload(decode=True))
                        elif name=="matching_reference" and part.get_filename():
                            if matching_reference is not None:raise ValueError("Only one matching reference allowed")
                            matching_reference=(part.get_filename(),part.get_payload(decode=True))
                    if not isinstance(conf,dict):raise ValueError("Missing configuration")
                    if self.path=="/api/inspect":
                        with tempfile.TemporaryDirectory() as tmp:
                            paths=[]
                            for filename,data in files:
                                leaf=filename.replace("\\","/").split("/")[-1]
                                path=Path(tmp)/leaf
                                if not leaf or leaf in {".",".."} or path.exists():raise ValueError("Invalid or duplicate filename")
                                path.write_bytes(data);paths.append(path)
                            self.reply(inspect_uploads(paths))
                        return
                    state.start(conf,files,truth,matching_reference);self.reply(state.status())
                except Exception as e:self.reply({"error":str(e)[:500]},400)
        server=ThreadingHTTPServer(("127.0.0.1",port),Handler)
        url=f"http://127.0.0.1:{server.server_port}"
        print(f"Shardelix Studio: {url}",flush=True)
        if open_browser:webbrowser.open(url)
        try:server.serve_forever()
        except KeyboardInterrupt:pass
        finally:state.cancel();server.server_close()
if __name__=="__main__":
    if len(sys.argv)==3 and sys.argv[1]=="--worker":raise SystemExit(worker(Path(sys.argv[2])))
    serve()
