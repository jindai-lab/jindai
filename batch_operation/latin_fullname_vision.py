"""只有姓的拉丁作者：把 PDF 前 2 页渲染成图像，让 openrouter/free 读完整作者名。

主机侧运行（host python3 有 PyMuPDF；storage 直挂 /misc/Public/jindai-storage）。
结果逐条落盘 reports/latin_vision.jsonl，按 id 断点续跑。

用法：python3 latin_fullname_vision.py [并发=3] [最多条数=0]
"""
import base64, json, os, re, ssl, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
import fitz

STORE = "/misc/Public/jindai-storage"
ROWS = "/home/zhuth/jindai-lab/jindai/batch_operation/reports/latin_surname_rows.tsv"
OUT = "/home/zhuth/jindai-lab/jindai/batch_operation/reports/latin_vision.jsonl"
CFG = "/home/zhuth/.hermes/profiles/collector/config.yaml"
DPI, PAGES = 130, 2

SYS = ("你是文献元数据助手。用户给你一篇论文/书籍 PDF 开头几页的图像。"
       "请读出**作者姓名**，按原顺序、用原文写法（西文用拉丁字母原文，不要译名）："
       "只要作者本人，不要译者、编者、机构、期刊名、参考文献中的人名。"
       "姓在前还是名在前按原文印的来。"
       "严格只输出 JSON：{\"authors\":[\"Naoko Tosa\",\"Koji Kimura\"],"
       "\"confidence\":\"high|medium|low\",\"evidence\":\"图上哪里看到的\"}；"
       "图上没有署名就 authors 返回空数组。")

_lock = threading.Lock()


def creds():
    cfg = open(CFG, encoding="utf-8").read()
    m = re.search(r"newapi-local:\s*\n\s*api:\s*(\S+)\s*\n\s*api_key:\s*(\S+)", cfg)
    return m.group(1).rstrip("/"), m.group(2)


BASE, KEY = creds()
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE


def render(path, pages=PAGES):
    path = path[7:] if path.startswith("../app/") else path   # 容器相对前缀归一化
    p = os.path.join(STORE, path)
    if not os.path.exists(p):
        return None
    d = fitz.open(p)
    try:
        return [base64.b64encode(d[i].get_pixmap(dpi=DPI).tobytes("png")).decode()
                for i in range(min(pages, d.page_count))]
    finally:
        d.close()


def ask(imgs, timeout=180):
    content = [{"type": "text", "text": f"这是该文献前 {len(imgs)} 页的图像。"}]
    content += [{"type": "image_url",
                 "image_url": {"url": "data:image/png;base64," + b}} for b in imgs]
    body = json.dumps({"model": os.environ.get("VLMODEL", "qwen3.8-omni-flash"), "temperature": 0,
                       "messages": [{"role": "system", "content": SYS},
                                    {"role": "user", "content": content}]}).encode()
    req = urllib.request.Request(BASE + "/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {KEY}",
                                          "Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=timeout, context=CTX).read())
    return d["choices"][0]["message"]["content"]


def work(row):
    rid, path, au, surs = row
    rec = {"id": rid, "path": path, "old": au, "姓": surs}
    try:
        imgs = render(path, 1)
    except Exception as e:
        rec["raw"], rec["err"] = "", f"render {type(e).__name__}"
        return rec
    if imgs is None:
        rec["raw"], rec["err"] = "", "no_file"
        return rec
    for attempt in range(5):
        try:
            rec["raw"], rec["err"] = ask(imgs), ""
            break
        except Exception as e:
            rec["raw"], rec["err"] = "", f"{type(e).__name__}"
            time.sleep(6 + attempt * 6)
    # 自适应：第 1 页读不到署名 → 补发前 2 页
    if rec.get("raw") and '"authors"' in rec["raw"] and len(imgs) == 1:
        import re as _re
        m = _re.search(r'"authors"\s*:\s*\[([^\]]*)\]', rec["raw"])
        if m and not m.group(1).strip():
            try:
                rec["raw"] = ask(render(path, 2))
            except Exception as e:
                rec["err"] = f"retry2 {type(e).__name__}"
    with _lock:
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    rows = [l.rstrip("\n").split("\t") for l in open(ROWS, encoding="utf-8") if l.strip()][1:]
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT, encoding="utf-8"):
            try:
                done.add(json.loads(l)["id"])
            except Exception:
                pass
    todo = [r for r in rows if r[0] not in done]
    if lim:
        todo = todo[:lim]
    print(f"待跑 {len(todo)}（已完成 {len(done)}），并发 {nw}", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=nw) as ex:
        for i, rec in enumerate(ex.map(work, todo), 1):
            if i % 10 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  {round(time.time()-t0)}s  err={rec.get('err','')}",
                      flush=True)
    print("完成", flush=True)


main()
