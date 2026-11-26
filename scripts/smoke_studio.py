"""Local HTTP integration smoke test: demo, blind ZIP import, unsafe ZIP and cancel. Run from repo root."""
from pathlib import Path
import urllib.request,json,subprocess,sys,time,os,zipfile
from tempfile import TemporaryDirectory
from uuid import uuid4
import socket
with socket.socket() as sock:
    sock.bind(("127.0.0.1",0));port=sock.getsockname()[1]
base=f"http://127.0.0.1:{port}"
root=Path.cwd()
env={**os.environ,"PYTHONPATH":str(root/"src")}
p=subprocess.Popen([sys.executable,"-m","fragmentlab","studio","--port",str(port),"--no-browser"],env=env)
try:
    for i in range(80):
        try: html=urllib.request.urlopen(base).read().decode();break
        except OSError:time.sleep(.1)
    else:raise RuntimeError("server unavailable")
    import re
    token=re.search('token="([^"]+)"',html)[1]
    def submit(conf,paths):
        b="FragmentBoundary"+uuid4().hex
        body=f'--{b}\r\nContent-Disposition: form-data; name="config"\r\n\r\n{json.dumps(conf)}\r\n'.encode()
        for path in paths:
            body+=f'--{b}\r\nContent-Disposition: form-data; name="files"; filename="{path.name}"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode()+path.read_bytes()+b'\r\n'
        body+=f'--{b}--\r\n'.encode()
        req=urllib.request.Request(base+"/api/start",body,{"X-Studio-Token":token,"Content-Type":f"multipart/form-data; boundary={b}"})
        print(json.load(urllib.request.urlopen(req)))
    def status():
        for i in range(480):
            s=json.load(urllib.request.urlopen(base+"/api/status"))
            if s["state"]!="running":return s
            time.sleep(.25)
        raise TimeoutError
    conf=dict(task="demo",cut="grid",count=64,max_side=640,page=1,seed=42,rotations=True,tune=True,strategy="auto")
    submit(conf,[root/"data/examples/straight_texture.png"])
    s=status();assert s["state"]=="done",s;assert s["metrics"]["pass_95"],s
    print("demo PASS",s["metrics"]["direct_position_accuracy"],s["diagnostics"])
    # V10 explicit compact-index UI control must reach the worker.
    submit({**conf,"index_mode":"midpoint"},[root/"data/examples/straight_texture.png"])
    s=status();assert s["state"]=="done",s
    assert s["diagnostics"]["index_mode"]=="midpoint" and s["metrics"]["pass_95"],s
    print("V10 midpoint control PASS",s["metrics"]["direct_position_accuracy"])
    with TemporaryDirectory() as tmp:
        z=Path(tmp)/"fragments.zip";z.write_bytes(urllib.request.urlopen(base+"/api/file?name=fragments.zip").read())
        with zipfile.ZipFile(z) as f:assert "case.json" in f.namelist() and not any("ground_truth" in n for n in f.namelist())
        submit(dict(task="blind",use_metadata=True,rotations=True,tune=True,strategy="auto"),[z])
        s=status();assert s["state"]=="done",s;assert s["metrics"] is None and "source.png" not in s["files"]
        assert s["diagnostics"]["placed"]==64 and s["diagnostics"]["coverage"]==1.
        print("blind ZIP PASS",s["diagnostics"])
        # New V9 path: continuous angle + missing, then replay blind archive.
        submit({**conf,"angle_mode":"continuous","sampling":1,"missing":.10},[root/"data/examples/straight_texture.png"])
        s=status();assert s["state"]=="done",s
        assert s["metrics"]["expected_count"]==64 and s["metrics"]["piece_count"]==58
        pose=json.load(urllib.request.urlopen(base+"/api/file?name=placements.json"))
        assert "input_to_canvas_affine" in pose and "rectify_ccw_deg" in pose
        z.write_bytes(urllib.request.urlopen(base+"/api/file?name=fragments.zip").read())
        with zipfile.ZipFile(z) as f:
            meta=json.loads(f.read("case.json"));assert meta["expected_count"]==64 and meta["piece_count"]==58
            assert not any("ground_truth" in n for n in f.namelist())
        submit(dict(task="blind",use_metadata=True,rotations=True,tune=True,strategy="auto"),[z])
        s=status();assert s["state"]=="done" and s["metrics"] is None,s
        blind_pose=json.load(urllib.request.urlopen(base+"/api/file?name=placements.json"))
        assert blind_pose["poses"]==pose["poses"] and blind_pose["diagnostics"]["missing_count"]==6
        print("V9 continuous + missing demo and blind replay PASS")
        bad=Path(tmp)/"bad.zip"
        with zipfile.ZipFile(bad,"w") as f:f.writestr("../bad.png",b"bad")
        submit(dict(task="blind",use_metadata=True,rotations=True,tune=True,strategy="auto"),[bad])
        s=status();assert s["state"]=="error" and "Unsafe archive" in s["error"],s
        print("unsafe ZIP rejected PASS")
        submit({**conf,"count":1024},[root/"data/examples/straight_texture.png"])
        req=urllib.request.Request(base+"/api/cancel",b"",{"X-Studio-Token":token})
        s=json.load(urllib.request.urlopen(req));assert s["state"]=="cancelled",s
        print("cancel PASS")
finally:
    p.terminate();p.wait(timeout=10)
