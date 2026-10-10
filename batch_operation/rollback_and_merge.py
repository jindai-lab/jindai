"""回滚 5 处有害目录改名 + Authors 孤儿行合并到 OneDrive（2026-10-10 用户决定）。

用户三项决定：
  1 回滚     → `OneDrive/未知`、`György Lukács`、`Émile Durkheim`、
               `Gabriel García Márquez`、`László Moholy-Nagy` 被整目录改名的 5 对全部还原
  2 无文件的只改作者 → Authors/ 下磁盘没文件的行**不动路径**，仅作者字段已由 A–D 规范化
  3 明确对应 OneDrive 的 → 合并：把 Authors 行的空字段填进对应 OneDrive 行，然后删掉
               Authors 孤儿行（先备份 JSON）；盘上有文件但 DB 没行 → 直接把该行路径改过去

匹配口径（"明确对应"）：清洗文件名（去扩展名/标点/变音符）→ 词元+CJK 二元组倒排取
前 30 候选 → difflib 相似度 ≥0.90 且领先次优 ≥0.03；中西文件名均要求归一后 ≥8 字符。
只扫 path LIKE 'Authors/%' 且磁盘无文件的行（=2 号决定覆盖的集合）。

--dry-run 默认：只报数字与样例。
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import unicodedata
from collections import defaultdict

import common as C

log = C.setup_logging(
    "rollback_and_merge", "batch_operation/reports/rollback_and_merge.log"
)
STORAGE = "/storage"

# (被改成的目录, 应还原的原目录) ——整目录 rename + 前缀式 UPDATE 都要还原
ROLLBACK = [
    ("OneDrive/F.H. Bradley", "OneDrive/未知"),
    ("OneDrive/白锡堃", "OneDrive/György Lukács"),
    ("OneDrive/渠东", "OneDrive/Émile Durkheim"),
    ("OneDrive/Luisa Rivera", "OneDrive/Gabriel García Márquez"),
    ("OneDrive/Moholy-Nagy Laszlo", "OneDrive/László Moholy-Nagy"),
]

FIELDS = [  # 只"填空"，绝不覆盖 OneDrive 行已有值
    "title", "authors", "abstract_note", "publication", "date", "volume",
    "issue", "pages", "publisher", "place", "doi", "url", "isbn", "issn",
    "archive", "archive_location", "library_catalog", "call_number",
    "language", "short_title", "series", "series_title", "notes", "tags",
    "related", "file_attachments", "item_type", "publication", "date_added",
]


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"\.[a-z0-9]{1,5}$", "", s)
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", s).strip()


def tokens(s: str) -> set:
    n = norm(s)
    out = {w for w in n.split() if len(w) >= 3}
    cjk = "".join(re.findall(r"[\u4e00-\u9fff]+", n))
    out |= {cjk[i:i + 2] for i in range(len(cjk) - 1)}
    return out


def key_of(path: str) -> str:
    return norm(path.rsplit("/", 1)[-1])


async def rollback(conn, apply: bool) -> dict:
    res = {"dirs": [], "rows_reverted": 0, "bindings_restored": 0}
    for new, old in ROLLBACK:
        new_abs, old_abs = f"{STORAGE}/{new}", f"{STORAGE}/{old}"
        on_disk = os.path.isdir(new_abs)
        was_absent = not os.path.isdir(old_abs)
        rows = await conn.fetch(
            "SELECT id, path, dataset_id FROM file_metadata "
            "WHERE path LIKE $1 || '/%'", new)
        miss = sum(1 for r in rows
                   if not os.path.exists(f"{STORAGE}/{r['path']}"))
        res["dirs"].append({"new": new, "old": old, "disk_dir": on_disk,
                            "old_absent": was_absent, "rows": len(rows),
                            "rows_missing_file": miss})
        if not apply:
            continue
        if on_disk and was_absent:
            os.rename(new_abs, old_abs)
        elif on_disk and os.path.isdir(old_abs):
            for ent in os.listdir(new_abs):   # 两边都在 → 逐个搬回
                s, d = f"{new_abs}/{ent}", f"{old_abs}/{ent}"
                if not os.path.exists(d):
                    os.rename(s, d)
            os.rmdir(new_abs)
        updated = 0
        for i in range(0, len(rows), 400):
            chunk = rows[i:i + 400]
            res2 = await conn.executemany(
                "UPDATE file_metadata SET path = $2 WHERE id = $1",
                [(r["id"], old + r["path"][len(new):]) for r in chunk])
            updated += len(chunk)
        res["rows_reverted"] += updated
        # 数据集：回滚后按文件夹规则 书库--<子目录>（apply#1 把这些行的绑定清空了）
        sub = old.split("/")[1]
        ds = await conn.fetchrow(
            "SELECT id FROM dataset WHERE name = $1", f"书库--{sub}")
        if ds is None and apply:
            ds = await conn.fetchrow(
                "INSERT INTO dataset(name) VALUES($1) ON CONFLICT DO NOTHING "
                "RETURNING id", f"书库--{sub}")
        if ds:
            await conn.execute(
                "UPDATE file_metadata SET dataset_id = $1 "
                "WHERE path LIKE $2 || '/%'", ds["id"], old)
            res["bindings_restored"] += len(rows)
    return res


async def orphans(conn) -> list:
    """Authors/ 下磁盘无文件的行（= 用户 2 号决定覆盖的集合）。"""
    rows = await conn.fetch(
        "SELECT id, path, authors, title, notes, tags, abstract_note "
        "FROM file_metadata WHERE path LIKE 'Authors/%'")
    return [r for r in rows if not os.path.exists(f"{STORAGE}/{r['path']}")]


async def build_index(conn) -> tuple[list, dict]:
    """OneDrive 行（盘上确有文件）→ 倒排索引。"""
    rows = await conn.fetch(
        "SELECT id, path, authors, title FROM file_metadata "
        "WHERE split_part(path, '/', 1) = 'OneDrive'")
    docs, inv = [], defaultdict(set)
    kept = 0
    for r in rows:
        if not os.path.exists(f"{STORAGE}/{r['path']}"):
            continue
        i = len(docs)
        docs.append(r)
        kept += 1
        for t in tokens(r["path"].rsplit("/", 1)[-1]):
            inv[t].add(i)
    log.info("[index] OneDrive 在盘行 %d / 取回 %d", kept, len(rows))
    return docs, inv


def match(qpath: str, docs: list, inv: dict, topn: int = 30):
    """返回 (best_idx|None, best_ratio, second_ratio)。"""
    q = key_of(qpath)
    if len(q) < 8:
        return None, 0.0, 0.0
    qt = tokens(qpath.rsplit("/", 1)[-1])
    if not qt:
        return None, 0.0, 0.0
    score: dict[int, int] = defaultdict(int)
    for t in qt:
        for i in inv.get(t, ()):
            score[i] += 1
    if not score:
        return None, 0.0, 0.0
    cov = {i: s / len(qt) for i, s in score.items()}
    cand = sorted(cov, key=lambda i: cov[i], reverse=True)[:topn]
    scored = [
        (difflib.SequenceMatcher(None, q, key_of(docs[i]["path"])).ratio(), i)
        for i in cand
    ]
    scored.sort(reverse=True)
    best_r, best_i = scored[0]
    second = scored[1][0] if len(scored) > 1 else 0.0
    if best_r < 0.90 or best_r - second < 0.03:
        return None, best_r, second
    return best_i, best_r, second


async def run(apply: bool = False) -> dict:
    conn = await C.get_conn()
    try:
        out: dict = {"applied": apply}
        out["rollback"] = await rollback(conn, apply)

        rows = await orphans(conn)
        log.info("[orphans] Authors 磁盘无文件 %d 行", len(rows))
        docs, inv = await build_index(conn)

        matches, unmatched, deps = [], [], defaultdict(int)
        for r in rows:
            i, ratio, second = match(r["path"], docs, inv)
            if i is None:
                unmatched.append(r["path"])
                continue
            d = docs[i]
            matches.append({"old_id": str(r["id"]), "old": r["path"],
                            "new_id": str(d["id"]), "new": d["path"],
                            "ratio": round(ratio, 3),
                            "second": round(second, 3)})

        # 依赖检查（合并=删行，先确认没有嵌入/绑定/跳页挂在这些 id 上）
        ids = [m["old_id"] for m in matches]
        if ids:
            n_emb = await conn.fetchval(
                "SELECT count(*) FROM text_embeddings WHERE source_id = ANY("
                "$1::uuid[])", ids)
            n_ds = await conn.fetchval(
                "SELECT count(*) FROM file_dataset WHERE file_id = ANY("
                "$1::uuid[])", ids)
            n_skip = await conn.fetchval(
                "SELECT count(*) FROM page_skip_state WHERE source_id = ANY("
                "$1::uuid[])", ids)
            deps.update(embeddings=int(n_emb or 0),
                        bindings=int(n_ds or 0), page_skip=int(n_skip or 0))

        out["orphans"] = len(rows)
        out["matched"] = len(matches)
        out["unmatched_keep_author_only"] = len(unmatched)
        out["deps_on_matched"] = dict(deps)
        if ids:
            new_ids = [m["new_id"] for m in matches]
            for tbl, col in (("paragraph", "source_id"),
                             ("text_embeddings", "source_id")):
                for tag, arr in (("old", ids), ("new", new_ids)):
                    n = await conn.fetchval(
                        f"SELECT count(*) FROM {tbl} WHERE {col} = ANY("
                        f"$1::uuid[])", arr)
                    deps[f"{tbl}_{tag}"] = int(n or 0)
            out["deps_on_matched"] = dict(deps)
        json.dump(matches, open(
            "batch_operation/reports/authors_orphan_matches.json", "w"),
            ensure_ascii=False, indent=1)
        out["match_examples"] = matches[:5]
        out["unmatched_examples"] = unmatched[:5]

        if apply and matches:
            backup = []
            filled = 0
            moved_rows = [0]
            for m in matches:
                src = await conn.fetchrow(
                    "SELECT * FROM file_metadata WHERE id = $1",
                    m["old_id"])
                dst = await conn.fetchrow(
                    "SELECT * FROM file_metadata WHERE id = $1",
                    m["new_id"])
                if not src or not dst:
                    continue
                backup.append({k: str(v) for k, v in dict(src).items()})
                sets, args = [], []
                for f in FIELDS:
                    if f in dict(src) and f in dict(dst) and \
                            (dst[f] in (None, "", "{}", "[]")) and \
                            src[f] not in (None, "", "{}", "[]"):
                        args.append(src[f]); sets.append(f"{f} = ${len(args)}")
                if sets:
                    args.append(m["new_id"])
                    await conn.execute(
                        f"UPDATE file_metadata SET {', '.join(sets)} "
                        f"WHERE id = ${len(args)}", *args)
                    filled += 1
                # 用户指令(2026-10-10)：先在关联表改 source_id，再删不用的行
                import uuid as _uuid
                oid = _uuid.UUID(m["old_id"])
                nid = _uuid.UUID(m["new_id"])
                # paragraph PK=(id,source_id)，text_embeddings FK(id,source_id)
                # -> paragraph(id,source_id) ON DELETE CASCADE：两表必须在**同一条
                # 语句**内一起改，RI 检查在语句末尾统一触发才不会互相拦截。
                await conn.execute(
                    "WITH u1 AS (UPDATE paragraph SET source_id = $2 "
                    "WHERE source_id = $1 RETURNING id), "
                    "u2 AS (UPDATE text_embeddings SET source_id = $2 "
                    "WHERE source_id = $1 RETURNING id) "
                    "SELECT (SELECT count(*) FROM u1) "
                    "+ (SELECT count(*) FROM u2)", oid, nid)
                for stmt in (
                    "UPDATE embedding_pending_queue SET source_id = $2 "
                    "WHERE source_id = $1",
                    "UPDATE page_skip_state SET source_id = $2 "
                    "WHERE source_id = $1",
                ):
                    await conn.execute(stmt, oid, nid)
                # 绑定：把旧绑定搬到新行，已存在的跳过，最后清空旧行
                await conn.execute(
                    "INSERT INTO file_dataset(file_id, dataset_id) "
                    "SELECT $1, dataset_id FROM file_dataset WHERE file_id=$2 "
                    "ON CONFLICT DO NOTHING", nid, oid)
                await conn.execute(
                    "DELETE FROM file_dataset WHERE file_id = $1", oid)
                await conn.execute(
                    "DELETE FROM file_metadata WHERE id = $1", oid)
                moved_rows[0] += 1
            json.dump(backup, open(
                "batch_operation/reports/authors_orphan_merged_20261010.json",
                "w"), ensure_ascii=False, indent=1)
            out["merged_deleted"] = len(backup)
            out["metadata_rows_repointed_then_deleted"] = moved_rows[0]
            out["rows_filled_from_orphan"] = filled
            out["orphans_after"] = len(rows) - len(matches)
        log.info("[summary] %s", json.dumps(
            {k: v for k, v in out.items()
             if k not in ("match_examples", "unmatched_examples")},
            ensure_ascii=False))
        return out
    finally:
        await conn.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    import asyncio
    asyncio.run(run(apply=args.apply))


if __name__ == "__main__":
    main()
