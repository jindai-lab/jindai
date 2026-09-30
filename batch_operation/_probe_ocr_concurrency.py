"""Probe PaddleX /ocr service: serial latency vs concurrent wall time (+GPU util).

Run inside jindai container: /app/.venv/bin/python batch_operation/_probe_ocr_concurrency.py
Used to decide: client-side parallel OCR against one endpoint vs replicas.
"""
import io
import json
import subprocess
import threading
import time
import urllib.request

from PIL import Image, ImageDraw


def make_png() -> bytes:
    buf = io.BytesIO()
    im = Image.new("L", (800, 200), 255)
    d = ImageDraw.Draw(im)
    d.text((30, 60), "Concurrency probe OCR test 2026", fill=0)
    d.text((30, 120), "The quick brown fox jumps over the lazy dog.", fill=0)
    im.save(buf, format="PNG")
    return buf.getvalue()


import base64

PAYLOAD = {
    "file": None,  # filled below
    "fileType": 1,
    "returnWordBox": False,
    "visualize": False,
    "lang": "en",
}
PAYLOAD["file"] = base64.b64encode(make_png()).decode("ascii")
BODY = json.dumps(PAYLOAD).encode()


def one(i: int, out: dict, url: str = "http://172.24.1.230:8080/ocr") -> None:
    t0 = time.time()
    req = urllib.request.Request(
        url, data=BODY, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            out[i] = (round(time.time() - t0, 2), r.status)
    except Exception as e:
        out[i] = (round(time.time() - t0, 2), "ERR:" + str(e)[:70])


def gpu_sampler(stop: threading.Event, samples: list) -> None:
    while not stop.is_set():
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            samples.append(int(r.stdout.strip()))
        except Exception:
            pass
        time.sleep(0.4)


def main() -> None:
    stop = threading.Event()
    samples: list = []
    th = threading.Thread(target=gpu_sampler, args=(stop, samples), daemon=True)
    th.start()

    out = {}
    one(0, out); print("serial1:", out[0])
    one(1, out); print("serial2:", out[1])

    for w in (4, 8):
        out2 = {}
        ts = [threading.Thread(target=one, args=(i, out2)) for i in range(w)]
        t0 = time.time()
        for t in ts: t.start()
        for t in ts: t.join()
        wall = round(time.time() - t0, 2)
        print(f"parallel{w}: wall={wall} results={out2}")

    stop.set(); th.join(timeout=2)
    if samples:
        print(f"gpu_util samples: n={len(samples)} max={max(samples)} "
              f"avg={sum(samples)//len(samples)} last10={samples[-10:]}")


if __name__ == "__main__":
    main()
