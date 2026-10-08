"""truly_gone 977 条：准确提取书名 → 对比 OneDrive 同题书目 → 判是否同一本书 →
是则改用 OneDrive 路径（2026-10-09 用户指令）。

题名提取：优先 file_metadata.title，否则文件名按 Calibre 约定 `Title - Author.pdf`
取 `rsplit(' - ', 1)[0]`；两侧都归一化（NFKC/去变音符/下划线转空格/去标点小写）。
匹配：归一化题名完全相等，或（截断导致的）互为前缀且较短侧 ≥12 字符。
作者相容：路径作者目录 token 化（重音折叠、CJK 整串）→ 交集/较短侧 ≥0.5。

分类（--apply 只落前两类，逐条一个事务，可续跑）：
  path_takeover  同题同作者、目标路径有盘上文件但无 file_metadata 行 → 老行 UPDATE path
  dup_merge      同题同作者、目标路径已有行 → 复用 fix_moved_paths_stage2 的
                 “保留文本多的一侧”合并（目标行抽样逐字重合 ≥0.25 才 auto）
  review_content 同题同作者但抽样重合 <0.25（疑不同版本）→ 只列
  review_multi   同题同作者命中多个 OneDrive 行 → 只列
  review_author  同题但作者目录不合 → 只列
  fuzzy          题名近似（difflib ≥0.82）→ 只列
  no_match       OneDrive 无同题 → 只列

用法（容器内 cwd /app）：
  python batch_operation/fix_gone_by_title.py                 # 分析，写 plan
  python batch_operation/fix_gone_by_title.py --apply --limit 2
  python batch_operation/fix_gone_by_title.py --apply
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import csv
import difflib
import json
import os
import re
import sys
import unicodedata

import common as C

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fix_moved_paths_stage2 import dup_decide, merge_pair  # noqa: E402

log = C.setup_logging("fix_gone_by_title",
                      "batch_operation/reports/fix_gone_by_title.log")

TSV = "batch_operation/reports/filemeta_moved_analysis.tsv"
PLAN = "batch_operation/reports/fix_gone_by_title_plan.json"
PLAN_TSV = "batch_operation/reports/fix_gone_by_title_plan.tsv"
CJK = re.compile(r"[一-鿿]")
TOK = re.compile(r"[\w一-鿿]+", re.UNICODE)
SIM_AUTO = 0.55          # 开头文本 3-gram 包含度 ≥ 此值 → 判同一本书，自动并
SIM_MIN = 0.30           # ≥ 此值 → review_mid（列出让用户定）
SIM_PREFIX = 0.45        # 题名只是前缀命中（文件名截断）时的自动门槛
SIM_AUTHORKEY = 0.70     # 题名 key 其实是作者名（文件名 Author - Title 式）时的门槛
FUZZY_MIN = 0.82
UNKNOWN_AUTHORS = {"未知", "无", "unknown", "佚名", "none", "n a"}


def norm_title(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.casefold().replace("_", " ")
    s = re.sub(r"[^\w一-鿿]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def title_keys(row_title: str, path: str) -> list[str]:
    base = path.rsplit("/", 1)[-1]
    base = re.sub(r"\.(pdf|epub|djvu|txt)$", "", base, flags=re.I)
    parsed = base.rsplit(" - ", 1)[0] if " - " in base else base
    keys = []
    for cand in (row_title, parsed):
        k = norm_title(cand)
        if len(k) >= 2 and k not in keys:
            keys.append(k)
    return keys


def keys_match(a: str, b: str) -> bool:
    if a == b:
        return True
    lo, hi = (a, b) if len(a) <= len(b) else (b, a)
    return len(lo) >= 12 and hi.startswith(lo)


def author_score(old_dir: str, new_dir: str) -> float:
    o = norm_title(old_dir)
    n = norm_title(new_dir)
    if not o or not n:
        return 0.0
    if CJK.search(o) or CJK.search(n):
        return 1.0 if (o in n or n in o) else 0.0
    ot, nt = set(TOK.findall(o)), set(TOK.findall(n))
    if not ot or not nt:
        return 0.0
    return len(ot & nt) / min(len(ot), len(nt))


def key_is_author(k: str, *dirs: str) -> bool:
    """题名 key 其实只是作者名（文件名 Author - Title 式）→ 需更高内容门槛。"""
    kt = set(TOK.findall(k))
    if not kt:
        return False
    for d in dirs:
        dt = set(TOK.findall(norm_title(d)))
        if dt and kt == dt:
            return True
    return False


async def sample_contents(conn, fid: str, limit: int) -> list[str]:
    rows = await conn.fetch(
        "SELECT content FROM paragraph WHERE source_id=$1::uuid "
        "ORDER BY id LIMIT $2", fid, limit)
    return [r["content"] or "" for r in rows]


def overlap_ratio(old_s: list[str], new_s: list[str]) -> float:
    if not old_s:
        return 1.0
    ns = set(new_s)
    matched = sum(len(x) for x in old_s if x in ns)
    total = sum(len(x) for x in old_s) or 1
    return matched / total


async def begin_text(conn, fid: str, limit: int = 80, chars: int = 6000) -> str:
    """按页序取开头正文（封面/目录/前言），老行与新行同书则高度相似。"""
    rows = await conn.fetch(
        "SELECT content FROM paragraph WHERE source_id=$1::uuid "
        "ORDER BY coalesce(source_page, 0), pagenum NULLS LAST LIMIT $2",
        fid, limit)
    return "".join(r["content"] or "" for r in rows)[:chars]


def _norm4gram(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).casefold()
    return "".join(re.findall(r"[\w一-鿿]", s))


def contain_sim(a: str, b: str, n: int = 3) -> float:
    """A 的 n-gram 有多少出现在 B 里（包含度，抗分段差异）。"""
    na, nb = _norm4gram(a), _norm4gram(b)
    if len(na) < n or len(nb) < n:
        return 0.0
    ga = {na[i:i + n] for i in range(len(na) - n + 1)}
    gb = {nb[i:i + n] for i in range(len(nb) - n + 1)}
    return len(ga & gb) / len(ga)


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fuzzy-sim", type=float, default=SIM_AUTHORKEY,
                    help="题名近似命中时的自动门槛（默认 0.70）")
    ap.add_argument("--min-sim", type=float, default=SIM_MIN,
                    help="题名精确+作者相容时的自动门槛（默认 0.30）")
    args = ap.parse_args()

    from jindai.storage import storage
    base = storage.safe_join("OneDrive")[: -len("OneDrive")]

    gone = [r["old_path"] for r in csv.DictReader(
        open(TSV, encoding="utf-8"), delimiter="\t")
        if r["class"] == "truly_gone"]
    if args.limit:
        gone = gone[:args.limit]
    log.info("[load] %d gone rows", len(gone))

    conn = await C.get_conn()
    # ---- OneDrive 侧索引（题名 key + 作者目录 + 行信息）
    od_rows = await conn.fetch(
        "SELECT id, path, title, authors::text[] AS a "
        "FROM file_metadata WHERE path LIKE 'OneDrive/%'")
    index: dict[str, list] = collections.defaultdict(list)
    for r in od_rows:
        seg = r["path"].split("/")
        adir = seg[1] if len(seg) > 2 else ""
        for k in title_keys(r["title"], r["path"]):
            index[k].append({"id": str(r["id"]), "path": r["path"],
                             "author_dir": adir, "authors": r["a"] or []})
    log.info("[index] %d OneDrive rows, %d title keys", len(od_rows), len(index))

    st: collections.Counter = collections.Counter()
    plan: list[dict] = []
    auto: list[dict] = []

    for i, old in enumerate(gone, 1):
        row = await conn.fetchrow(
            "SELECT id, title, authors::text[] AS a FROM file_metadata WHERE path=$1", old)
        if row is None:
            st["row_missing"] += 1
            continue
        old_id = str(row["id"])
        seg = old.split("/")
        old_adir = seg[1] if len(seg) > 2 else ""
        okeys = title_keys(row["title"], old)

        # 1) 同题候选：先精确 key，再题名近似（截断/变音兜底）
        exact = True
        cands: dict[str, dict] = {}
        for k in okeys:
            for c in index.get(k, []):
                cands[c["path"]] = c
        if not cands:
            exact = False
            for k in okeys:
                kc = k[:1]
                for tk in index:
                    if abs(len(tk) - len(k)) > 10 or tk[:1] != kc:
                        continue
                    if min(len(k), len(tk)) >= 10 and \
                            difflib.SequenceMatcher(None, k, tk).ratio() >= FUZZY_MIN:
                        for c in index[tk]:
                            cands[c["path"]] = c
            if not cands:
                st["no_match"] += 1
                plan.append({"cls": "no_match", "old": old, "title": okeys})
                continue
            st["fuzzy_pool"] += 1

        # 2) 作者相容（目标作者目录未知时放行，但要求题名精确）
        scored = [(author_score(old_adir, c["author_dir"]), c) for c in cands.values()]
        strict = [c for s, c in scored if s >= 0.5]
        lenient = [c for s, c in scored
                   if norm_title(c["author_dir"]) in UNKNOWN_AUTHORS]
        if strict:
            pool = strict
        elif lenient and exact:
            pool = lenient
            st["author_unknown_ok"] += 1
        else:
            st["review_author"] += 1
            plan.append({"cls": "review_author", "old": old, "title": okeys,
                         "cands": [{"path": c["path"], "score": round(s, 2)}
                                   for s, c in sorted(scored, key=lambda x: -x[0])[:4]]})
            continue

        # 3) 多候选 → 用开头文本相似度挑唯一胜者
        ob = await begin_text(conn, old_id)      # 老行开头只取一次
        tgt = None
        if len(pool) > 1:
            sims = []
            for c in pool[:6]:
                crow = await conn.fetchval(
                    "SELECT id FROM file_metadata WHERE path=$1", c["path"])
                if crow is None:
                    sims.append((c, None))
                    continue
                sims.append((c, contain_sim(ob, await begin_text(conn, str(crow)))))
            known = sorted([(c, s) for c, s in sims if s is not None],
                           key=lambda x: -x[1])
            if known and known[0][1] >= SIM_AUTO and \
                    (len(known) == 1 or known[0][1] - known[1][1] >= 0.2):
                tgt = known[0][0]
                st["multi_resolved_by_sim"] += 1
            else:
                st["review_multi"] += 1
                plan.append({"cls": "review_multi", "old": old, "title": okeys,
                             "cands": [{"path": c["path"], "sim": s} for c, s in sims]})
                continue
        else:
            tgt = pool[0]

        tgt_row = await conn.fetchrow(
            "SELECT id FROM file_metadata WHERE path=$1", tgt["path"])
        ent = {"old": old, "target": tgt["path"], "exact": exact,
               "old_id": old_id, "tgt_id": str(tgt_row["id"]) if tgt_row else None}

        if tgt_row is None:
            # 目标路径无行：仅题名精确命中时可直接接管（无从比内容）
            if not exact:
                st["review_fuzzy"] += 1
                ent["cls"] = "review_fuzzy"
                plan.append(ent)
                continue
            if not os.path.exists(os.path.join(base, tgt["path"])):
                st["target_file_missing"] += 1
                plan.append({"cls": "target_file_missing", **ent})
                continue
            st["path_takeover"] += 1
            ent["cls"] = "path_takeover"
            plan.append(ent)
            auto.append(ent)
            continue

        # 4) 开头正文 3-gram 包含度核验（抗分段差异）→ 自动门槛
        sim = contain_sim(ob, await begin_text(conn, str(tgt_row["id"])))
        ent["sim"] = round(sim, 3)
        st["dup_pair_found"] += 1
        authorlike = any(key_is_author(k, old_adir, tgt["author_dir"])
                         for k in okeys)
        if not exact:
            thr = args.fuzzy_sim     # 题名只近似命中 → 内容必须高度一致
        elif authorlike:
            thr = SIM_AUTHORKEY      # 文件名是 Author - Title 式，key 实为作者名
        else:
            thr = args.min_sim       # 题名精确 + 作者相容 → 低门槛
        if sim >= thr:
            st["dup_merge"] += 1
            ent["cls"] = "dup_merge"
            plan.append(ent)
            auto.append(ent)
        else:
            st["review_fuzzy" if not exact else "review_content"] += 1
            ent["cls"] = "review_fuzzy" if not exact else "review_content"
            plan.append(ent)
        if i % 100 == 0:
            log.info("[analyze] %d/%d %s", i, len(gone),
                     json.dumps(dict(st), ensure_ascii=False))

    # ---- apply
    applied: collections.Counter = collections.Counter()
    if args.apply and auto:
        for ent in auto:
            try:
                if ent["cls"] == "path_takeover":
                    async with conn.transaction():
                        tag = await conn.execute(
                            "UPDATE file_metadata SET path=$1 WHERE id=$2::uuid "
                            "AND NOT EXISTS (SELECT 1 FROM file_metadata WHERE path=$1) "
                            "AND EXISTS (SELECT 1 FROM file_metadata WHERE id=$2::uuid)",
                            ent["target"], ent["old_id"])
                        applied["path_takeover" if int(tag.split()[-1]) else "path_takeover_skip"] += 1
                    continue
                # dup_merge：目标行以 apply 时实时为准（可能前一条已占位）
                owner = await conn.fetchval(
                    "SELECT id FROM file_metadata WHERE path=$1 AND id<>$2::uuid",
                    ent["target"], ent["old_id"])
                if owner is None:
                    async with conn.transaction():
                        tag = await conn.execute(
                            "UPDATE file_metadata SET path=$1 WHERE id=$2::uuid "
                            "AND NOT EXISTS (SELECT 1 FROM file_metadata WHERE path=$1)",
                            ent["target"], ent["old_id"])
                        applied["takeover_after_merge" if int(tag.split()[-1]) else "takeover_skip"] += 1
                    continue
                await merge_pair(conn, ent["old"], ent["target"],
                                 ent["old_id"], str(owner), applied)
                applied["dup_merge"] += 1
            except Exception as ex:
                applied["failed"] += 1
                log.warning("[apply] %s: %s", ent["old"][:70], str(ex)[:200])

    summary = {"mode": "apply" if args.apply else "dry-run",
               "counters": dict(st), "applied": dict(applied), "plan": plan}
    with open(PLAN, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    with open(PLAN_TSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["class", "old_path", "target_path", "sim", "overlap"])
        for p in plan:
            w.writerow([p.get("cls"), p.get("old"), p.get("target") or
                        ";".join(c if isinstance(c, str) else c.get("path", "")
                                 for c in p.get("cands", []))[:400],
                        p.get("sim", ""), p.get("overlap", "")])
    log.info("[done] mode=%s counters=%s applied=%s", summary["mode"],
             json.dumps(dict(st), ensure_ascii=False),
             json.dumps(dict(applied), ensure_ascii=False))
    await conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
