#!/usr/bin/env python3
"""盘上重复后缀文件重命名：x.pdf.pdf → x.pdf（同名冲突则跳过）。--apply 才动手。"""
import json, os, re, sys, csv, collections, datetime

IDX = "/home/zhuth/jindai-lab/jindai/batch_operation/reports/disk_index.jsonl"
HOST = "/misc/Public/jindai-storage"
EXT = r"(pdf|epub|djvu|mobi|azw3|txt|docx|doc|jpg|png|zip|html)"
pat = re.compile(r"^(.*)\.(%s)\.(%s)$" % (EXT, EXT), re.I)
apply = "--apply" in sys.argv
cands = []
extra_dirs = ["/misc/Public/OneDrive/zotero", "/misc/Public/OneDrive/Zotero"]
names = []
for l in open(IDX, encoding="utf-8"):
    d = json.loads(l)
    names.append((d["b"], d["p"]))
for d in extra_dirs:
    if os.path.isdir(d):
        for f in os.listdir(d):
            names.append((f, "OneDrive/zotero/" + f))
for b, p in names:
    m = pat.match(b)
    if m and m.group(2).lower() == m.group(3).lower():
        cands.append((p, b, b[: -(len(m.group(3)) + 1)]))
print(f"候选（双后缀）{len(cands)} 个")
cnt = collections.Counter(c[-1].rsplit(".", 1)[-1].lower() for c in cands)
print("按扩展名:", cnt.most_common())
for p, old, new in cands[:8]:
    print(f"   {old[:64]}\n      → {new[:62]}")
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
rep = f"/home/zhuth/jindai-lab/jindai/batch_operation/reports/rename_dupext_{stamp}.tsv"
with open(rep, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["rel_path", "old_name", "new_name", "结果"])
    if not apply:
        for p, old, new in cands:
            w.writerow([p, old, new, "dry-run"])
        print(f"[dry-run] 未执行；清单 → {rep}")
        sys.exit(0)
    ok = conflict = nf = err = 0
    for p, old, new in cands:
        if p.startswith("OneDrive/zotero/"):
            src = os.path.join("/misc/Public/OneDrive/zotero", old)
            dst = os.path.join("/misc/Public/OneDrive/zotero", new)
        else:
            src = os.path.join(HOST, p)
            dst = os.path.join(os.path.dirname(src), new)
        if os.path.exists(dst):
            w.writerow([p, old, new, "跳过-目标已存在"])
            conflict += 1
            continue
        if not os.path.exists(src):
            w.writerow([p, old, new, "跳过-源不存在"])
            nf += 1
            continue
        try:
            os.rename(src, dst)
            w.writerow([p, old, new, "已重命名"])
            ok += 1
        except Exception as e:
            w.writerow([p, old, new, f"失败 {type(e).__name__}"])
            err += 1
    print(f"[apply] 重命名 {ok} | 冲突跳过 {conflict} | 源缺失 {nf} | 失败 {err}")
    print(f"台账 → {rep}")
