"""Real-page OCR timing: serial path vs prep+http over ThreadPool(4).

  docker exec -w /app jindai /app/.venv/bin/python \
      batch_operation/_probe_ocr_realpage.py "OneDrive/.../x.pdf" 5 50 100
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

import common  # noqa: F401  # inserts APP_ROOT into sys.path

from jindai.storage import storage
from jindai.pdfutils import open_pdf, render_pdf_page
from jindai.ocrutils import PaddleOCRClient

URL = PaddleOCRClient().base_url + "ocr"


def ocr(b64: str) -> tuple[float, int, str]:
    payload = {"file": b64, "fileType": 1, "returnWordBox": False,
               "visualize": False, "lang": "en"}
    t0 = time.time()
    r = httpx.post(URL, json=payload, timeout=120.0)
    dt = time.time() - t0
    rec = r.json().get("result", {}).get("ocrResults", [])
    texts = rec[0].get("prunedResult", {}).get("rec_texts", []) if rec else []
    return dt, r.status_code, " ".join(texts)


def main() -> int:
    path = sys.argv[1]
    pages = [int(x) for x in sys.argv[2:]] or [5, 50, 100, 150, 200, 250]
    stream = storage.open(path, "rb")
    doc = open_pdf(stream)
    pages = [p for p in pages if p < doc.page_count]
    client = PaddleOCRClient()

    rend = prep_t = 0.0
    imgs: list[str] = []
    http_serial = 0.0
    for p in pages:
        txt = len(doc[p].get_text() or "")
        t0 = time.time()
        img = render_pdf_page(stream, page_num=p, zoom=2.0)
        rend += time.time() - t0
        t0 = time.time()
        b64 = client._prepare_image(img)
        prep_t += time.time() - t0
        t0 = time.time()
        dt, code, head = ocr(b64)
        http_serial += time.time() - t0
        imgs.append(b64)
        print(f"page {p}: textlayer={txt} render={rend/len(imgs):.2f}s_avg "
              f"prep={prep_t/len(imgs):.2f}s_avg http={dt:.2f}s http_status={code} "
              f"head={head[:70]!r}")

    n = len(imgs)
    serial_total = rend + prep_t + http_serial
    print(f"\nSERICAL ({n} pages): {serial_total:.1f}s = render {rend:.1f} "
          f"+ prep {prep_t:.1f} + http {http_serial:.1f}")

    for w in (4, 6):
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=w) as ex:
            list(ex.map(ocr, imgs))
        wall = time.time() - t0
        # what the patch achieves: render serial, prep+http parallel
        est = rend + wall + prep_t  # prep parallelizes too; add serial as upper
        est_p = rend + wall + prep_t / w
        print(f"PARALLEL w={w}: prep+http wall={wall:.1f}s "
              f"-> file total ~= render {rend:.1f} + wall {wall:.1f} "
              f"+ prep/{w} {prep_t/w:.1f} = {est_p:.1f}s "
              f"(speedup {serial_total/max(est_p,0.1):.1f}x, "
              f"http throughput {n/max(wall,0.1):.1f} pages/s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
