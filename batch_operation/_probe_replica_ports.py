"""Smoke-test OCR on replica ports:  docker exec -w /app jindai \
    /app/.venv/bin/python /app/batch_operation/_probe_replica_ports.py 8080 8081 8083
"""
import base64
import io
import json
import sys
import time
import urllib.request

from PIL import Image, ImageDraw

buf = io.BytesIO()
im = Image.new("L", (640, 140), 255)
d = ImageDraw.Draw(im)
d.text((30, 50), "Replica check OK 2026", fill=0)
im.save(buf, format="PNG")
BODY = json.dumps({
    "file": base64.b64encode(buf.getvalue()).decode(),
    "fileType": 1,
    "returnWordBox": False,
    "visualize": False,
    "lang": "en",
}).encode()

for arg in (sys.argv[1:] or ["8080", "8081", "8083"]):
    port = int(arg)
    t0 = time.time()
    req = urllib.request.Request(
        f"http://172.24.1.230:{port}/ocr", data=BODY,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            j = json.loads(r.read())
            txt = j.get("result", {}).get("ocrResults", [{}])[0] \
                 .get("prunedResult", {}).get("rec_texts", [])
            print(f"{port}: {r.status} {time.time()-t0:.2f}s "
                  f"text={' '.join(txt)[:60]!r}")
    except Exception as e:
        print(f"{port}: ERR {e}")
