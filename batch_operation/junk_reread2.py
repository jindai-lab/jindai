#!/usr/bin/env python3
"""对疑似脏值行补读首页：让模型原样照抄页面上的作者署名。"""
import json, re, ssl, os, base64, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
import fitz

R = "/home/zhuth/jindai-lab/jindai/batch_operation/reports/"
STORE = "/misc/Public/jindai-storage"
CFG = open("/home/zhuth/.hermes/profiles/collector/config.yaml", encoding="utf-8").read()
BASE, KEY = re.search(r"newapi-local:\s*\n\s*api:\s*(\S+)\s*\n\s*api_key:\s*(\S+)", CFG).groups()
CTX = ssl.create_default_context(); CTX.check_hostname = False; CTX.verify_mode = ssl.CERT_NONE
MODEL = os.environ.get("VLMODEL", "qwen3.8-omni-flash")
OUT = R + "junk_vision2.jsonl"
SYS = ("你是书目著录助手。看图，把页面上印的**作者署名原文**照抄出来（保持原文语言、顺序、缩写）。"
       "只回一行 JSON：{\"printed\":[\"...\"],\"evidence\":\"在页面哪个位置看到的\"}。"
       "若页面上确实没有作者署名，printed 返回空数组。")


def render(path, n=1):
    p = path[7:] if path.startswith("../app/") else path
    full = os.path.join(STORE, p)
    if not os.path.exists(full):
        return None
    d = fitz.open(full)
    try:
        return [base64.b64encode(d[i].get_pixmap(dpi=110).tobytes("png")).decode()
                for i in range(min(n, d.page_count))]
    finally:
        d.close()


def ask(imgs):
    content = [{"type": "text", "text": f"这是文献前 {len(imgs)} 页的图像。"}]
    content += [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + b}} for b in imgs]
    body = json.dumps({"model": MODEL, "temperature": 0, "max_tokens": 400,
                       "messages": [{"role": "system", "content": SYS},
                                    {"role": "user", "content": content}]}).encode()
    req = urllib.request.Request(BASE + "/chat/completions", data=body,
                                 headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=180, context=CTX).read())["choices"][0]["message"]["content"]


def work(row):
    rid, path, au, title = row
    rec = {"id": rid, "old": au, "path": path, "title": title}
    imgs = render(path, 1)
    if imgs is None:
        rec["raw"], rec["err"] = "", "no_file"
        return rec
    for a in range(4):
        try:
            rec["raw"], rec["err"] = ask(imgs), ""
            break
        except Exception as e:
            rec["raw"], rec["err"] = "", type(e).__name__
            time.sleep(4 + a * 4)
    if rec.get("raw") and re.search(r'"printed"\s*:\s*\[\s*\]', rec["raw"]):
        try:
            rec["raw"] = ask(render(path, 2))
        except Exception:
            pass
    return rec


def main():
    done = set()
    if os.path.exists(OUT):
        done = {json.loads(l)["id"] for l in open(OUT, encoding="utf-8")}
    rows = json.load(open(R + "junk_last_rows.json", encoding="utf-8"))
    todo = [r for r in rows if r[0] not in done]
    print(f"待读 {len(todo)}（已完成 {len(done)}）", flush=True)
    with ThreadPoolExecutor(max_workers=4) as ex:
        for i, rec in enumerate(ex.map(work, todo), 1):
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if i % 10 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}", flush=True)
    print("完成", flush=True)


main()
