"""authors_t2_resolve.py — 回填 T2 那批「丛书名/刊期号当作者」行的真实著者（2026-10-11）。

证据来源（按优先级）：
  1. PDF 题名页文本（dump_t2_titlepages.py 已落 reports/t2_titlepages.txt）；
  2. `file_metadata.title` 里封面著录串 `[丛书]题名.责任人.出版社(年)`；
  3. 作者目录名 / Calibre 目录名（OneDrive/<作者>/、Authors/<作者>/）；
  4. 文件名前缀（Journals/<作者> (年) - <篇名>）。
都取不到的保持原值并计入 unresolved，绝不写空、绝不猜。

**口径（用户超时未答，按推荐项执行，显式标注）**：题名/目录能确定**原作者**的写原作者
（丘特切夫、洛尔卡、博纳富瓦、奈保尔、叔本华、埃德蒙·伯克、Hermann Hesse、阿格农…）；
多人诗选/译文集无单一原作者的，写封面点名的**译者/编者**（王佐良、卞之琳、魏荒弩…），
比「未知」有信息量且可回溯到 title 字符串；完全没有责任者的写空数组。
备份 reports/authors_t2_backup_<date>.tsv 可整体回退。

dry-run 默认；--apply 写库（UPDATE 带 title/旧值护栏）。
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import datetime as dt
import json
import os
import re
import sys

sys.path.insert(0, "/app")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common as C  # noqa: E402

log = C.setup_logging(
    "authors_t2_resolve", "batch_operation/reports/authors_t2_resolve.log")

PLAN = "batch_operation/reports/authors_quality_plan.json"

# ---- 显式定案：非 A 组（PDF 题名页 / 文件名前缀 / 目录名给出的著者） --------
# path -> (new_authors, new_title|None, 依据)
EXPLICIT: dict[str, tuple[list[str], str | None, str]] = {
    # Journals/GRKG：文件名前缀是 `<作者> (<年>) - <篇名>`，题名页给出全名
    "Journals/GRKG - Grundlagenstudien aus Kybernetik und Geisteswissenschaft/1964/Cyon, E. & Ludwig, C. (1866) - Die Reflexe eines der sensiblen Nerven des Herzens auf die motorischen Blutgefäße - .GRKG-1964-Bd5-Beiheft.pdf":
        (["E. Cyon", "C. Ludwig"],
         "Die Reflexe eines der sensiblen Nerven des Herzens auf die motorischen Blutgefäße",
         "PDF题名页: 'Reproduktion der Arbeit von E. Cyon und C. Ludwig'"),
    "Journals/GRKG - Grundlagenstudien aus Kybernetik und Geisteswissenschaft/1967/Rosenblut, A. u.a. (1967) - Behavior, Purpose and Teleology - GRKG-1967-Bd.8-Beiheft.pdf":
        (["Arturo Rosenblueth", "Norbert Wiener", "Julian Bigelow"],
         "Behavior, Purpose and Teleology",
         "PDF题名页: Arturo Rosenblueth, Norbert Wiener and Julian Bigelow（文件名只写 Rosenblut, A. u.a.）"),
    "Journals/GRKG - Grundlagenstudien aus Kybernetik und Geisteswissenschaft/1971/Graf, K.-D. (1971) - Systemttheoretische Untersuchung des rechnergestützten Lehrprogrammierens (GRKG-1971-Bd12-Beiheft).pdf":
        (["Klaus-Dieter Graf"], None, "PDF题名页: VON KLAUS-DIETER GRAF"),
    "Journals/GRKG - Grundlagenstudien aus Kybernetik und Geisteswissenschaft/1970/Pflüger, E. F. W. (1970) - Die teleologische Mechanik der lebendigen Natur (GRKG-1970-Bd11-Beiheft).pdf":
        (["E. F. W. Pflüger"], None, "PDF题名页: F. W. PFLÜGER；文件名 Pflüger, E. F. W."),
    "Journals/GRKG - Grundlagenstudien aus Kybernetik und Geisteswissenschaft/1963/Kaeding, J. W. (1897) - Häufigkeitswörterbuch der deutschen Sprache - GRKG-1963-Bd.4-Beiheft.pdf":
        (["J. W. Kaeding"], "Häufigkeitswörterbuch der deutschen Sprache",
         "PDF题名页(古体字) + 文件名 Kaeding, J. W."),
    "Journals/GRKG - Grundlagenstudien aus Kybernetik und Geisteswissenschaft/1966/Stachowiak, H. (1966) - Zum Problem einer logisch-semantischen Maßbestimmung des Lernerfolgs - GRKG-1966-Bd7-Beiheft.pdf":
        (["H. Stachowiak"], None, "PDF题名页 STACHOWIAK + 文件名 Stachowiak, H."),
    "Journals/GRKG - Grundlagenstudien aus Kybernetik und Geisteswissenschaft/1969/Lincke, F. (1969) - Das mechanische Relais (GRKG-1969-Bd10-Beiheft).pdf":
        (["F. Lincke"], "Das mechanische Relais (GRKG-1969-Bd10-Beiheft)",
         "PDF题名页: Eine synthetische Studie von F. Lincke"),
    "Journals/GRKG - Grundlagenstudien aus Kybernetik und Geisteswissenschaft/1961/Kraft v. Baer, K. (1864) - Die Abhängigkeit unseres Weltbilds von der Länge unseres Moments - GRKG-1962-Bd.3-Beiheft.pdf":
        (["K. Kraft v. Baer"], "Die Abhängigkeit unseres Weltbilds von der Länge unseres Moments",
         "PDF题名页 + 文件名 Kraft v. Baer, K."),
    "Journals/GRKG - Grundlagenstudien aus Kybernetik und Geisteswissenschaft/1965/Schmidt, H. (1965) - Die anthropologische Bedeutung der Kybernetik - GRKG-1965-Bd6-Beiheft.pdf":
        (["Hermann Schmidt"], None, "PDF题名页: PROFESSOR HERMANN SCHMIDT"),
    # Series/：文件名前缀 `<作者> - <书名>`，题名页确认
    "Series/数码人类学专题/(Inside Technology) Christine Hine - Systematics as Cyberscience_ Computers, Change, and  Continuity in Science (Inside Technology)-The MIT Press (2008).pdf":
        (["Christine Hine"], None, "PDF题名页 + 文件名（(Inside Technology) 是丛书名）"),
    "Series/数码人类学专题/(Internationalizing Media Studies) Rowan Wilken, Gerard Goggin, Heather A. Horst - Location Technologies in International Context-Routledge (2018).pdf":
        (["Rowan Wilken", "Gerard Goggin", "Heather A. Horst"], None,
         "PDF版权页作者行 + 文件名（(Internationalizing Media Studies) 是丛书名）"),
    "Series/数码人类学专题/(Materializing Culture) Daniel Miller (ed.) - Anthropology and the Individual_ A Material Culture Perspective-Berg (2009).pdf":
        (["Daniel Miller"], None, "PDF题名页 + 文件名（(Materializing Culture) 是丛书名，(ed.) 是角色标记）"),
    # Zotero：PDF 作者简介给出中西对照名，译者吴蘅不入 authors
    "Zotero/〔美〕约翰·福尔克1 朱迪思·科克（著） 〔中〕吴蘅（译）_2020_博物馆观众研究：过去、现状和未来.pdf":
        (["约翰·福尔克", "朱迪思·科克"], None,
         "PDF《东南文化》2020(1) 作者简介: 约翰·福尔克(John H. Falk)、朱迪思·科克(Judith Koke)；吴蘅为译者"),
    # Authors/ 树：盘上无文件，按作者目录名 + 文件名复原（只改 authors/title，绝不改 path）
    "Authors/Habermas, Jurgen/(社会理论译丛) （德）尤尔根・哈贝马斯 - 重建历史唯物主义-社会科学文献出版社 (2000).pdf":
        (["Jürgen Habermas"], "重建历史唯物主义-社会科学文献出版社 (2000)",
         "作者目录 Authors/Habermas, Jurgen + 文件名第二段；西方人取原文名"),
    "Authors/Goodman, Nelson/Springer) Nelson Goodman - The Structure of Appearance (1977.pdf":
        (["Nelson Goodman"], "The Structure of Appearance (1977)",
         "作者目录 Authors/Goodman, Nelson + 文件名第二段"),
    "Authors/Schlegel, Friedrich von/August Wilhelm Schlegel (Hrsg.), Friedrich Schlegel (Hrsg.) - Athenaeum. 1798-1800-Cotta (1960).pdf":
        (["August Wilhelm Schlegel", "Friedrich Schlegel"], "Athenaeum. 1798-1800-Cotta (1960)",
         "文件名前段（Hrsg.=编者）+ 作者目录"),
    "Authors/柄谷行人/(社会批判译丛) [日] 柄谷行人 著_ 中田友美 译 - 马克思，其可能性的中心-中央编译出版社 (2006).pdf":
        (["柄谷行人"], "马克思，其可能性的中心-中央编译出版社 (2006)",
         "文件名前段点名著者（中田友美为译者）+ 作者目录"),
    "Authors/Marcuse, Herbert/(现代西方学术文库)  - 审美之维_ 马尔库塞美学论著集-生活·读书·新知三联书店 (1989).pdf":
        (["Herbert Marcuse"], "审美之维_ 马尔库塞美学论著集-生活·读书·新知三联书店 (1989)",
         "作者目录 Authors/Marcuse, Herbert + 文件名第二段"),
    # OneDrive：PDF 题名页是中文正文，作者按题名串/目录名
    "OneDrive/И. Способин (I. Sposobin)/斯波索宾和声学教程课后作业答案 (2162)/斯波索宾和声学教程课后作业答案 - И. Способин (I. Sposobin).pdf":
        (["И. Способин"], None, "目录名 + PDF 序言（《和声学教程》原作者斯波索宾）；去掉拉丁转写括号"),
    "OneDrive/套装11册/哈佛、斯坦福、耶鲁大学经典课程分享 (20467)/哈佛、斯坦福、耶鲁大学经典课程分享（套装11册）.epub":
        ([], None, "PDF/EPUB 首页只有 11 本书的总目录，无责任者；'套装11册' 不是作者"),
}

# ---- A 组覆盖：题名点名了原作者的（原作者优先于译者） -----------------------
OVERRIDE_BY_KEY: list[tuple[str, list[str], str]] = [
    ("丘特切夫诗全集", ["丘特切夫"], "题名点名原作者；朱宪生为译者"),
    ("洛尔卡诗选", ["洛尔卡"], "题名点名原作者；赵振江为译者"),
    ("博纳富瓦诗选", ["博纳富瓦"], "题名点名原作者；郭宏安、树才为译者"),
    ("奈保尔家书", ["奈保尔"], "题名点名原作者；北塔、常文祺为译者"),
    ("悲喜人生：叔本华论说文集", ["叔本华"], "题名点名原作者；范进等为译者"),
    ("埃德蒙·伯克读本", ["埃德蒙·伯克"], "题名点名原作者；陈志瑞、石斌为编者"),
    ("黑塞小说散文选", ["Hermann Hesse"],
     "目录名 OneDrive/Hermann Hesse + Calibre 著者 + 题名'张佩芬、王克澄等译'"),
    ("一个简单的故事", ["阿格农"], "题名与目录名点名原作者（阿格农/Agnon）；徐崇亮、郑军荣为译者"),
    ("凯旋门前的桐叶", ["柳鸣九"], "题名点名著者 柳鸣九著"),
]

RESP_RE = re.compile(r"\.([^.]*(?:编译|编著|译|编|著|校))(?=[.（(])")


def parse_responsibles(title: str) -> list[str] | None:
    """`[丛书]题名.责任人.出版社(年)` → 单个人列表；取不到返回 None。"""
    m = RESP_RE.search(title or "")
    if not m:
        return None
    seg = m.group(1)
    seg = re.sub(r"(编译|编著|译|编|著|校)$", "", seg).strip()
    seg = re.sub(r"等$", "", seg).strip()
    if not seg:
        return None
    parts = [p.strip() for p in re.split(r"[、，,]", seg) if p.strip()]
    return parts or None


def build_plan() -> list[dict]:
    data = json.load(open(PLAN, encoding="utf-8"))
    t2 = [x for x in data["plans"] if x["tier"] == "T2"]
    out = []
    for x in t2:
        path, old_title, old_authors = x["path"], x["old_title"], x["old_authors"]
        if path in EXPLICIT:
            new_a, new_t, why = EXPLICIT[path]
        else:
            new_a, new_t, why = None, None, None
            for key, authors, note in OVERRIDE_BY_KEY:
                if key in path or key in old_title:
                    new_a, why = authors, note
                    break
            if new_a is None:
                new_a = parse_responsibles(old_title)
                if new_a:
                    why = f"题名串点名的责任者（封面著录）: {old_title[:60]}"
        if new_a is None:
            out.append({**x, "tier": "UNRESOLVED", "new_authors": old_authors,
                        "new_title": old_title, "why": "取不到著者，保持原值"})
            continue
        out.append({**x, "tier": "T3", "new_authors": new_a,
                    "new_title": new_t if new_t else old_title, "why": why})
    return out


async def run(apply: bool = False) -> tuple[dict, list[dict]]:
    plan = build_plan()
    todo = [p for p in plan if p["tier"] == "T3" and
            (p["new_authors"] != p["old_authors"] or p["new_title"] != p["old_title"])]
    unres = [p for p in plan if p["tier"] == "UNRESOLVED"]
    summary = {"applied": apply, "t2_total": len(plan),
               "to_update": len(todo), "unresolved": len(unres),
               "set_empty": sum(1 for p in todo if not p["new_authors"])}
    os.makedirs("batch_operation/reports", exist_ok=True)
    with open("batch_operation/reports/authors_t2_plan.json", "w",
              encoding="utf-8") as f:
        json.dump({"summary": summary, "plans": plan}, f, ensure_ascii=False, indent=1)
    with open("batch_operation/reports/authors_t2_plan.tsv", "w",
              encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["path", "old_title", "new_title", "old_authors",
                    "new_authors", "why"])
        for p in plan:
            w.writerow([p["path"], p["old_title"], p["new_title"],
                        "{" + ",".join(p["old_authors"]) + "}",
                        "{" + ",".join(p["new_authors"]) + "}", p["why"]])

    if apply and todo:
        day = dt.date.today().strftime("%Y%m%d")
        bak = f"batch_operation/reports/authors_t2_backup_{day}.tsv"
        with open(bak, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, delimiter="\t")
            w.writerow(["id", "path", "title", "authors"])
            for p in todo:
                w.writerow([p["id"], p["path"], p["old_title"],
                            "{" + ",".join(p["old_authors"]) + "}"])
        conn = await C.get_conn()
        try:
            # 空数组用 text[] 参数显式写 '{}'，不要 NULL
            await conn.executemany(
                "UPDATE file_metadata SET authors = $2::text[], title = $3 "
                "WHERE id = $1 AND coalesce(title,'') = $4",
                [(p["id"], p["new_authors"], p["new_title"], p["old_title"])
                 for p in todo])
        finally:
            await conn.close()
        summary["backup"] = bak
        log.info("[apply] updated=%d backup=%s", len(todo), bak)

    log.info("[summary] %s", json.dumps(summary, ensure_ascii=False))
    return summary, plan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--show", type=int, default=0)
    args = ap.parse_args()
    summary, plan = asyncio.run(run(apply=args.apply))
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    for p in plan[:args.show]:
        print("-", p["tier"], p["path"][:78])
        print("    T:", p["old_title"][:60], "=>", p["new_title"][:60])
        print("    A:", p["old_authors"], "=>", p["new_authors"])
        print("    依据:", p["why"])


if __name__ == "__main__":
    main()
