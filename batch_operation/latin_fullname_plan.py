"""把「读图得完整作者名」的结果转成计划 → reports/latin_fullname_plan.tsv

匹配规则：模型返回的人名里，必须有一个把原姓当**独立词**包含（忽略重音/大小写/标点）；
取那一个作为全名。逗号顺序（Tosa, Naoko）统一成「名 姓」。取不到则列 manual。
"""
import csv, json, re, unicodedata, collections

VJ = "/home/zhuth/jindai-lab/jindai/batch_operation/reports/latin_vision.jsonl"
OUT = "/home/zhuth/jindai-lab/jindai/batch_operation/reports/latin_fullname_plan.tsv"


def norm(s):
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z ]", " ", s.lower())


def tokens(s):
    return [t for t in norm(s).split() if t]


def fix_order(name):
    """Tosa, Naoko -> Naoko Tosa；去多余空白"""
    name = re.sub(r"\s+", " ", name).strip(" ,;")
    if "," in name:
        parts = [p.strip() for p in name.split(",") if p.strip()]
        if len(parts) == 2:
            name = f"{parts[1]} {parts[0]}"
        else:
            name = " ".join(parts)
    return name


def main():
    recs = [json.loads(l) for l in open(VJ, encoding="utf-8") if l.strip()]
    seen, stat = set(), collections.Counter()
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "path", "old", "new", "src", "命中名"])
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
                au = [a.strip() for a in json.loads(m.group()).get("authors", [])
                      if isinstance(a, str) and a.strip()]
            except Exception:
                stat["bad_json"] += 1
                continue
            au = [fix_order(a) for a in au]
            au = [a for a in au if 3 <= len(a) <= 60 and not re.search(r"\d", a)]
            old = r["old"].split("|")
            surs = [s for s in r["姓"].split(",") if s]
            names, used, hit = {}, set(), []
            for s in surs:
                st = tokens(s)[-1] if tokens(s) else ""
                cand = None
                for a in au:
                    if a in used:
                        continue
                    if st and st in tokens(a) and len(tokens(a)) >= 2:
                        cand = a
                        break
                if cand:
                    names[s] = cand
                    used.add(cand)
                    hit.append(f"{s}->{cand}")
            new = list(old)
            for k, v in enumerate(new):
                if v in names:
                    new[k] = names[v]
            # ① 按归一化去重（保序）；② 某元素词集是另一元素的真子集 → 丢短的
            # 归一化 key：拉丁名用词集；纯中文名用原串（空词集会让中文名互相误判重复）
            ded, keys, toks = [], [], []
            for x in new:
                t = tokens(x)
                k = tuple(t) if t else ("CJK", x)
                if k in keys:
                    continue
                ded.append(x)
                keys.append(k)
                toks.append(set(t))
            drop = set()
            for i, ti in enumerate(toks):
                for j, tj in enumerate(toks):
                    if i != j and ti and tj and ti < tj:   # 两侧都必须有拉丁词才判子集
                        drop.add(i)
            new = [x for i, x in enumerate(ded) if i not in drop]
            # ③ 还有两个元素共享同一尾词（姓）→ 疑似原值被拆开，转人工
            tails = [tokens(x)[-1] for x in new if tokens(x)]   # 只统计拉丁名
            if len(tails) != len(set(tails)):
                stat["manual(共享姓)"] += 1
                continue
            if new != old:
                w.writerow([r["id"], r["path"], "|".join(old), "|".join(new), "V", "; ".join(hit)])
                stat["apply"] += 1
            else:
                stat[f"manual({len(names)}/{len(surs)})"] += 1
    print(dict(stat))

main()
