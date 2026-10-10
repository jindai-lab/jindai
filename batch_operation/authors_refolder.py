"""按规范化后的作者名重整作者文件夹（2026-10-10 用户指令第 5 步）。

范围：`OneDrive/`、`Authors/`、`Supplementary Books/` 下的作者目录（路径第 2 段）。
每本书的目标作者 =
    ① 该文件规整后的作者（只有 1 个 → 它；多个 → 首作者）
    ② 拿不到 → 目录名本身做格式修整（'Roy, Arundhati' → 'Arundhati Roy'、去尾随
       空格/下划线）
    ③ 仍不像人名 → 不动
目标目录名 ≠ 现目录名 → 移动（整目录共识时只 rename 目录；混合目录逐文件散开到
各自作者目录，保持 `<title (id)>` 层级）；随后同步：
    file_metadata.path、数据集绑定（书库--旧作者 → 书库--新作者）

dry-run 默认；--apply 执行（每本书一个事务式更新，先搬盘、后改库、再逐条验证）。
  python batch_operation/authors_refolder.py
  python batch_operation/authors_refolder.py --apply
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import Counter, defaultdict

import common as C

log = C.setup_logging(
    "authors_refolder", "batch_operation/reports/authors_refolder.log"
)

STORAGE = "/storage"
TOPS = ("OneDrive", "Authors", "Supplementary Books")
SELECT_SQL = """
    SELECT id, path, coalesce(authors, '{}') AS authors
    FROM file_metadata
    WHERE split_part(path, '/', 1) = ANY($1)
      AND array_length(string_to_array(path, '/'), 1) >= 3
      AND path NOT LIKE '%#zip/%'
      AND path NOT LIKE '../%'
"""

LATIN_RE = re.compile(r"^[A-Za-z][A-Za-z .'\-]*$")
CJK_RE = re.compile(r"[一-鿿㐀-䶿]")


def to_western(s: str) -> str:
    s = (s or "").strip()
    if "," in s and LATIN_RE.fullmatch(s.split(",", 1)[0].strip()):
        a, b = s.split(",", 1)
        return f"{b.strip()} {a.strip()}".strip()
    return s


def looks_person(s: str) -> bool:
    s = (s or "").strip()
    if not s or len(s) > 60 or re.search(r"\d", s):
        return False
    if any(ch in s for ch in "()[]{}「」《》") or "_" in s:
        return False
    if CJK_RE.search(s):
        return 1 <= len(s) <= 15
    if LATIN_RE.fullmatch(s):
        return 1 <= len(s.split()) <= 6
    return False


def key(s: str) -> str:
    return re.sub(r"[\s.\-'’]+", "", (s or "").lower())


def sanitize(s: str) -> str:
    """目录名合法化：去首尾空白/下划线、去掉路径非法字符。"""
    s = (s or "").strip().strip("_").strip()
    s = re.sub(r"[/\\:\0]", "", s)
    return re.sub(r"\s+", " ", s).strip()


JUNK = {"未知", "unknown", "佚名", "无名", "其他", "none", "n.a.", "na",
        "tmp", "test", "未署名", "无作者", "?", "-", "libgen", "libgen.lc",
        "libgen.li", "ocred", "扫描", "scan"}


def toks(s: str) -> list:
    return sorted(re.findall(r"[\w]+", (s or "").lower(), flags=re.UNICODE))


def is_latin(s: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z .\'-]*", s or ""))


def is_cjk(s: str) -> bool:
    return bool(CJK_RE.search(s or ""))


def target_for(path: str, authors: list, folder: str, existing: set,
               variant: dict):
    """返回 (target_dir, reason) 或 (None, why_skipped)。

    2026-10-10 收紧（dry-run 暴露的三类危险移动全部禁止）：
      ✗ 多作者按首作者搬走 —— OneDrive 目录是 Calibre 主著者，不因合著改属
      ✗ 反写目录（Flichy Patrice / J. Tambiah Stanley / Jr. E. San Juan）
        —— 与原目录词元完全相同、只是顺序不同的一律不改
      ✗ 搬进 未知/未知作者 目录、或目标目录从未存在过的新名字
    只允许三种：① 格式修整（'Roy, Arundhati'→'Arundhati Roy'、去尾随符号）
                ② 单作者且目标目录**已存在**（并入既有目录）
                ③ 单作者且语种不同（译名 → 原文名 / 片假名 → 简体中文）
    """
    top = path.split("/")[0]
    authors = [variant.get(a.strip(), a.strip())
               for a in authors if a and a.strip()]
    # 作者值本身是 未知/Unknown/libgen 之类占位 → 不作为目标目录名
    authors = [a for a in authors if sanitize(a).lower() not in JUNK]
    folder_c = sanitize(to_western(folder))
    if looks_person(folder_c) and key(folder_c) != key(folder):
        return folder_c, "folder_fixup"                 # ① 格式修整

    if folder.lower() in JUNK or not looks_person(folder):
        for a in authors:                                # ② 从垃圾目录挑出作者
            if looks_person(a) and key(a) != key(folder):
                return sanitize(a), "from_junk"
        return None, "junk_folder"

    if len(authors) != 1 or not looks_person(authors[0]):
        return None, "multi_or_no_author"                # 多作者：不搬
    cand = sanitize(authors[0])
    if key(cand) == key(folder):
        return None, "same"
    if toks(cand) == toks(folder):                       # 反写/同词元 → 不搬
        return None, "token_same"
    top_dir = f"{top}/{cand}"
    if top_dir in existing:
        return cand, "merge_existing"                    # ② 并入既有目录
    if is_latin(cand) != is_latin(folder) or is_cjk(cand) != is_cjk(folder):
        return cand, "script_unify"                      # ③ 语种统一
    return None, "target_not_established"


async def run(apply: bool = False) -> dict:
    conn = await C.get_conn()
    try:
        rows = await conn.fetch(SELECT_SQL, list(TOPS))
        # 目录共识：同一作者目录内「单作者」文件的众数
        single_by_dir: dict[str, Counter] = defaultdict(Counter)
        for r in rows:
            seg = r["path"].split("/")
            d = "/".join(seg[:2])
            a = [x.strip() for x in r["authors"] if x and x.strip()]
            if len(a) == 1:
                single_by_dir[d][a[0]] += 1
        consensus_of = {}
        for d, cnt in single_by_dir.items():
            top, n = cnt.most_common(1)[0]
            if n >= max(2, sum(cnt.values()) * 0.5) and looks_person(top):
                consensus_of[d] = top

        # 既有目录集合（用于『并入既有目录』判定）+ 作者异写映射
        existing_dirs = {"/".join(r["path"].split("/")[:2]) for r in rows}
        try:
            variant = json.load(open(
                "batch_operation/reports/authors_plan.json"))["variant_map"]
        except Exception:                                # noqa: BLE001
            variant = {}
        moves = []          # {id, old, new, dir, why}
        skip = Counter()
        for r in rows:
            seg = r["path"].split("/")
            folder = seg[1]
            d = "/".join(seg[:2])
            tgt, why = target_for(r["path"], list(r["authors"]), folder,
                                  existing_dirs, variant)
            if tgt is None:
                skip[why] += 1
                continue
            new_seg = list(seg)
            new_seg[1] = tgt
            new_path = "/".join(new_seg)
            if new_path == r["path"]:
                skip["same"] += 1
                continue
            moves.append({"id": str(r["id"]), "old": r["path"],
                          "new": new_path, "dir": d, "why": why})

        by_dir = defaultdict(list)
        for m in moves:
            by_dir[m["dir"]].append(m)
        summary = {"rows": len(rows), "moves": len(moves),
                   "dirs_affected": len(by_dir),
                   "dirs_full_rename": sum(
                       1 for d, ms in by_dir.items()
                       if len({m["new"].split("/")[1] for m in ms}) == 1
                       and d.split("/")[1] != ms[0]["new"].split("/")[1]),
                   "skipped": dict(skip),
                   "why": dict(Counter(m["why"] for m in moves))}
        os.makedirs("batch_operation/reports", exist_ok=True)
        with open("batch_operation/reports/authors_refolder_plan.json", "w",
                  encoding="utf-8") as f:
            json.dump({"summary": summary, "moves": moves,
                       "consensus": consensus_of}, f, ensure_ascii=False, indent=1)
        log.info("[plan] %s", json.dumps(summary, ensure_ascii=False))
        for m in moves[:8]:
            log.info("[move] %s -> %s (%s)", m["old"], m["new"], m["why"])

        if not apply or not moves:
            return summary

        # ------------------------- 执行 -------------------------
        done, failed = 0, []
        # 1) 先按目录聚合，整目录共识的只 rename 目录（一次搞定一批）
        dir_targets: dict[str, set] = defaultdict(set)
        for m in moves:
            dir_targets[m["dir"]].add(m["new"].split("/")[1])
        renames = [(d, list(t)[0]) for d, t in dir_targets.items() if len(t) == 1]
        renamed = {}
        for old_dir, new_dir in renames:
            src, dst = f"{STORAGE}/{old_dir}", f"{STORAGE}/{old_dir.split('/')[0]}/{new_dir}"
            try:
                if os.path.isdir(dst):
                    if not os.path.exists(src):
                        # 磁盘早已改成目标名（历史改名没同步 DB）→ 只补写 DB
                        renamed[old_dir] = new_dir
                        log.info("[mkdir] already-on-disk %s -> %s (只补 DB)",
                                 old_dir, new_dir)
                        continue
                    raise FileExistsError(dst)
                if not os.path.exists(src):
                    raise FileNotFoundError(src)
                os.rename(src, dst)
                renamed[old_dir] = new_dir
                log.info("[mkdir] rename dir %s -> %s", old_dir, new_dir)
            except Exception as ex:  # noqa: BLE001
                log.warning("[mkdir] failed %s -> %s (%s); 改走逐文件",
                            old_dir, new_dir, ex)

        # 2) 逐文件：目录没整体改名的、以及改名失败的
        # 目录已整体改名且该文件的目标正是新目录名 → 由 rename 覆盖；
        # 其余（混合目录、rename 失败）逐文件搬
        per_file = [m for m in moves
                    if renamed.get(m["dir"]) != m["new"].split("/")[1]]
        for m in per_file:
            src = f"{STORAGE}/{m['old']}"
            dst = f"{STORAGE}/{m['new']}"
            if not os.path.exists(src) and os.path.exists(dst):
                # 源已不在、目标已在 → 磁盘侧已完成，只需回写 DB（不计入 failed）
                done += 1
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.exists(dst) and os.path.exists(src) and not os.path.samefile(src, dst):
                base, ext = os.path.splitext(dst)
                dst = f"{base} ({m['id'][:8]}){ext}"
                m["new"] = os.path.relpath(dst, STORAGE)
            try:
                shutil.move(src, dst)
            except Exception as ex:  # noqa: BLE001
                failed.append({**m, "error": str(ex)[:160]})
                continue
            done += 1

        # 3) 整目录改名覆盖的文件数
        done += len(moves) - len(per_file)

        # 4) 更新 file_metadata.path（整目录改名的按前缀批量）
        upd = 0
        for old_dir, new_dir in renamed.items():
            top = old_dir.split("/")[0]
            res = await conn.execute(
                "UPDATE file_metadata SET path = $2 || substring(path from length($1)+1) "
                "WHERE path LIKE $1 || '/%' AND split_part(path,'/',1) = $3",
                old_dir, f"{top}/{new_dir}", top)
            upd += int(res.split()[-1]) if res else 0
        for m in per_file:
            if m["id"] and not any(f["id"] == m["id"] for f in failed):
                res = await conn.execute(
                    "UPDATE file_metadata SET path = $2 WHERE id = $1 AND path = $3",
                    m["id"], m["new"], m["old"])
                upd += int(res.split()[-1]) if res else 0

        # 5) 数据集：书库--旧作者 → 书库--新作者
        pairs = sorted({(m["old"].split("/")[1], m["new"].split("/")[1])
                        for m in moves if m["old"].split("/")[1] != m["new"].split("/")[1]})
        ds_upd = 0
        for old_a, new_a in pairs:
            old_ds, new_ds = f"书库--{old_a}", f"书库--{new_a}"
            await conn.execute(
                "INSERT INTO dataset (name) VALUES ($1) ON CONFLICT (name) DO NOTHING",
                new_ds)
            res = await conn.execute(
                "UPDATE file_dataset fd SET dataset_id = (SELECT id FROM dataset WHERE name=$1) "
                "WHERE fd.dataset_id = (SELECT id FROM dataset WHERE name=$2) "
                "AND EXISTS (SELECT 1 FROM file_metadata f WHERE f.id = fd.file_id "
                "             AND (split_part(f.path,'/',2) = $1 OR f.path LIKE $3))",
                new_ds, old_ds, f"%/{new_a}/%")
            ds_upd += int(res.split()[-1]) if res else 0

        # 6) 读回验证：更新后的行必须在盘上存在
        missing = []
        for m in moves:
            if m["id"] in {f["id"] for f in failed}:
                continue
            if not os.path.exists(f"{STORAGE}/{m['new']}"):
                missing.append(m["new"])
        summary.update({"missing_on_disk": missing[:20],
                        "missing_n": len(missing),
                        "applied": True, "files_moved": done,
                        "dir_renames": len(renamed), "failed": failed[:20],
                        "failed_n": len(failed), "rows_path_updated": upd,
                        "dataset_rows_repointed": ds_upd})
        with open("batch_operation/reports/authors_refolder.json", "w",
                  encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
        log.info("[done] %s", json.dumps(
            {k: v for k, v in summary.items() if k != "failed"},
            ensure_ascii=False))
        return summary
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
