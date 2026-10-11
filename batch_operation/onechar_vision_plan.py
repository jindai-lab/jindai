"""把 openrouter/free 的视觉读名结果转成计划 → reports/onechar_plan6.tsv

校验：每个单字姓槽位都要能配到一个「以该姓开头、2–4 字、不含虚字」的姓名，
全部槽位都配上才落计划；否则列入 manual。
"""
import csv, json, re, collections

VJ = "/home/zhuth/jindai-lab/jindai/batch_operation/reports/onechar_vision.jsonl"
OUT = "/home/zhuth/jindai-lab/jindai/batch_operation/reports/onechar_plan6.tsv"
CJK = r"\u4e00-\u9fff"
ONE = re.compile(rf"^([{CJK}])(?: et al\.?)?$")
NAMEOK = re.compile(rf"^[{CJK}][{CJK}A-Za-z·\.\- ]{{1,11}}$")
BAD = re.compile(r"[的了著译编说话是们等]|老师|教授|先生|女士|同学|记者|作者|摘要|期刊")

def main():
    recs = [json.loads(l) for l in open(VJ, encoding="utf-8") if l.strip()]
    seen, stat = set(), collections.Counter()
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "path", "old", "new", "src", "证据"])
        for r in recs:
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            raw = r.get("raw") or ""
            m = re.search(r"\{[\s\S]*\}", raw)
            if not m:
                stat["no_json"] += 1
                continue
            try:
                au = [a.strip() for a in json.loads(m.group()).get("authors", []) if isinstance(a, str)]
            except Exception:
                stat["bad_json"] += 1
                continue
            au = [re.sub(rf"(?<=[{CJK}])\s+(?=[{CJK}])", "", a) for a in au]
            au = [a for a in au if NAMEOK.match(a) and not BAD.search(a)]
            old = r["old"].split("|")
            idx = [k for k, a in enumerate(old) if ONE.match(a)]
            names, used = {}, set()
            for k in idx:
                sur = ONE.match(old[k]).group(1)
                for j, a in enumerate(au):
                    if j in used or not a.startswith(sur):
                        continue
                    names[k], _ = a, used.add(j)
                    break
            if len(names) == len(idx):
                new = list(old)
                for k, v in names.items():
                    new[k] = v
                w.writerow([r["id"], r["path"], "|".join(old), "|".join(new), "V", ",".join(au)])
                stat["apply"] += 1
            else:
                stat[f"manual({len(names)}/{len(idx)})"] += 1
    print(dict(stat))

main()
