#!/usr/bin/env python3
"""中文译名 → 标准西文全名：仅当结果包含原拉丁姓氏时才采纳，其余转人工（绝不猜）。"""
import json, re, ssl, csv, os, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

R = "/home/zhuth/jindai-lab/jindai/batch_operation/reports/"
CFG = open("/home/zhuth/.hermes/profiles/collector/config.yaml", encoding="utf-8").read()
BASE, KEY = re.search(r"newapi-local:\s*\n\s*api:\s*(\S+)\s*\n\s*api_key:\s*(\S+)", CFG).groups()
CTX = ssl.create_default_context(); CTX.check_hostname = False; CTX.verify_mode = ssl.CERT_NONE
OUT = R + "cn2latin.jsonl"
MODEL = os.environ.get("MODEL", "deepseek-flash")

SYS = ("你是书目考证助手。给你一本中译本的佐证信息，请判断作者的标准西文全名。"
       "只回一行 JSON：{\"name\":\"西文全名\",\"confidence\":\"high|low\"}。"
       "证据不足或无法确定（例如信息互相矛盾）时 confidence 必须为 low。不要编造。")


def cn_names(rid, vision):
    r = vision.get(rid)
    if not r:
        return []
    m = re.search(r'"authors"\s*:\s*\[([^\]]*)\]', r.get("raw") or "")
    if not m:
        return []
    got = [x.strip().strip('"') for x in m.group(1).split(",") if x.strip()]
    return [g for g in got if not re.search(r"[A-Za-z]{3}", g) and re.search(r"[\u4e00-\u9fff]", g)]


def ask(title, surname, cn):
    q = (f"书名（中译）：{title}\n该书原本作者姓氏（拉丁字母）：{surname}\n"
         f"中译本书页上读到的作者中文名：{cn}")
    body = json.dumps({"model": MODEL, "temperature": 0, "max_tokens": 200,
                       "messages": [{"role": "system", "content": SYS},
                                    {"role": "user", "content": q}]}).encode()
    req = urllib.request.Request(BASE + "/chat/completions", data=body,
                                 headers={"Authorization": "Bearer " + KEY,
                                          "Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120, context=CTX).read())["choices"][0]["message"]["content"]


def work(row):
    rid, path, au, title, cn = row
    rec = {"id": rid, "old": au, "path": path, "title": title, "cn": cn}
    for _ in range(3):
        try:
            raw = ask(title, au, cn); rec["raw"] = raw; rec["err"] = ""; break
        except Exception as e:
            rec["raw"], rec["err"] = "", type(e).__name__; time.sleep(4)
    return rec


def main():
    done = set()
    if os.path.exists(OUT):
        done = {json.loads(l)["id"] for l in open(OUT, encoding="utf-8")}
    rows = json.load(open(R + "cn2latin_todo.json", encoding="utf-8"))
    todo = [r for r in rows if r[0] not in done]
    print(f"待跑 {len(todo)}（已完成 {len(done)}），模型 {MODEL}", flush=True)
    with ThreadPoolExecutor(max_workers=4) as ex:
        for i, rec in enumerate(ex.map(work, todo), 1):
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if i % 20 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}", flush=True)
    print("完成", flush=True)


main()
