"""补抓 Zotero：① 顶层条目带 key；② 附件（文件名 + parentItem）。

产出 reports/zotero_items.json（key/title/creators）与
     reports/zotero_attachments.json（filename/parentItem）
用于按附件文件名反查作者（title 对不上的那批）。
"""
import json
import os
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import yaml

CFG = yaml.safe_load(open("/home/zhuth/jindai-lab/jindai/config.yaml"))["BibliographyPlugin"]
LID, KEY = CFG["zotero_library_id"], CFG["zotero_api_key"]
BASE = f"https://api.zotero.org/users/{LID}/items"
OUT_DIR = "/home/zhuth/jindai-lab/jindai/batch_operation/reports"  # 宿主路径（bind-mount 到 /app/batch_operation/reports）


def fetch(url: str, tries: int = 3):
    req = urllib.request.Request(url, headers={"Zotero-API-Key": KEY})
    for i in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r), r.headers.get("Total-Results")
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                print("[fail]", url[-40:], type(e).__name__, e)
                return [], None
    return [], None


def page(kind: str, start: int):
    if kind == "top":
        url = (f"{BASE}?format=json&limit=100&start={start}"
               f"&include=data&itemType=-attachment%20||%20note")
    else:
        url = f"{BASE}?format=json&limit=100&start={start}&include=data&itemType=attachment"
    return fetch(url)


def run(kind: str, out_name: str, keep) -> None:
    items, total = page(kind, 0)
    if not items:
        print(kind, "first page failed")
        return
    total = int(total)
    pages = list(range(100, total, 100))
    print(kind, "total", total, "pages", len(pages) + 1)
    lock = threading.Lock()
    all_items = list(items)
    n = [1]

    def work(s):
        return page(kind, s)

    with ThreadPoolExecutor(max_workers=4) as ex:
        for res in ex.map(work, pages):
            with lock:
                all_items.extend(res[0])
                n[0] += 1
                if n[0] % 50 == 0:
                    print(kind, "pages", n[0], "/", len(pages) + 1, "items", len(all_items))
    slim = keep(all_items)
    json.dump(slim, open(os.path.join(OUT_DIR, out_name), "w"), ensure_ascii=False)
    print(kind, "saved", len(slim), "->", out_name)


def keep_top(items):
    out = []
    for it in items:
        d = it.get("data", {})
        out.append({"key": it.get("key", ""), "title": d.get("title", ""),
                    "creators": d.get("creators", []),
                    "type": d.get("itemType", ""), "date": d.get("date", "")})
    return out


def keep_att(items):
    out = []
    for it in items:
        d = it.get("data", {})
        fn = d.get("filename") or d.get("linkMode") or ""
        out.append({"filename": fn, "parent": d.get("parentItem", ""),
                    "path": d.get("path", "")})
    return out


if __name__ == "__main__":
    run("top", "zotero_items.json", keep_top)
    run("att", "zotero_attachments.json", keep_att)
