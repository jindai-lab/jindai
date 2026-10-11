"""单汉字作者第二轮计划：manual 档里用「作者简介/文//记者」模式 + 首页词频收敛

规则：
  * 模式命中（作者简介/姓名/通讯作者/文//记者）→ 取首个干净候选；
    `文/` 档再剥尾部的动词残字（`苏伟体现在` → 苏伟）
  * 全篇词频 → 只认「干净候选（2–3 汉字、无标点）且出现在首页前 120 字内」的首个
输出 reports/onechar_plan2.tsv（id/path/old/new/src）
"""
import csv, re, collections

PLAN = "/app/batch_operation/reports/onechar_plan.tsv"
M2 = "/app/batch_operation/reports/onechar_manual2.tsv"
EVID = "/app/batch_operation/reports/onechar_evidence.tsv"
OUT = "/app/batch_operation/reports/onechar_plan2.tsv"
CJK = r"\u4e00-\u9fff"
CLEAN = re.compile(rf"^[{CJK}]{{2,3}}$")
TRAIL = re.compile(r"(?:体现在|认为|指出|表示|看来|谈到|说到|声称|主张|强调|提出|发现|"
                   r"看来|来说|而言|在|对|与|和|及|等|著|译|编|摘)$")

def main():
    plan = [r for r in csv.DictReader(open(PLAN, encoding="utf-8"), delimiter="\t")]
    man = {r["path"]: r for r in plan if not r["new_authors"]}
    head = {}
    for l in open(EVID, encoding="utf-8"):
        f = l.rstrip("\n").split("\t")
        if len(f) > 5:
            head[f[0]] = f[5]
    rows = [dict(zip(["path", "姓", "候选", "模式", "片段"], l.rstrip("\n").split("\t")))
            for l in open(M2, encoding="utf-8") if l.strip()]

    stat = collections.Counter()
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "path", "old", "new", "src", "证据"])
        for r in rows:
            m = man.get(r["path"])
            if not m:
                continue
            sur = r["姓"]
            cands = [c for c in (r["候选"].split(",") if r["候选"] else []) if c]
            pick, why = "", ""
            if r["模式"] in ("作者简介", "姓名", "通讯作者", "文/", "记者"):
                for c in cands:
                    c = re.sub(rf"^{sur}[0-9]*", sur, c)
                    if r["模式"] == "文/":
                        c = TRAIL.sub("", c)
                    if CLEAN.match(c):
                        pick, why = c, r["模式"]
                        break
            elif r["模式"] == "全篇词频":
                h = (head.get(r["path"], "") or "")[:120]
                for c in cands:
                    c = re.sub(rf"^{sur}[0-9]*", sur, c)
                    if CLEAN.match(c) and c in h:
                        pick, why = c, f"首页词频({h[:24]}…)"
                        break
            if not pick:
                stat["still_manual"] += 1
                continue
            old = m["old_authors"].split("|")
            new = [pick if a == sur or a.startswith(sur + " et al") else a for a in old]
            w.writerow([m["id"], r["path"], m["old_authors"], "|".join(new), "M2", why])
            stat["apply"] += 1
    print(dict(stat))

main()
