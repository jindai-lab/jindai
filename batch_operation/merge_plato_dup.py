"""合并柏拉图著作的重复行 + 搬离 OneDrive/张智仁（2026-10-20 用户指令「执行」）。

背景：OneDrive/张智仁/ 下 10 个柏拉图书目目录，作者字段被写成译者「张智仁」；
其中 9 本在 OneDrive/Plato/ 下另有一行（路径盘上不存在、但带段落文本）→ 同书两行。

规则（沿用 fix_stale_paths 的同书两行口径）：
  * 每对保留【文本总字符更多】的一侧；平手时优先 authors 不含「张智仁」的一侧（元数据更好），
    再平手保留路径真实存在的一侧。
  * 保留行的 path 指向盘上真实文件（搬目录后的 OneDrive/Plato/<书名 (id)>/…）。
  * authors 含「张智仁」的统一改成 {Plato}；已有的非张智仁作者（如 Allan Bloom）原样保留。
  * 删除侧按序：text_embeddings → paragraph → embedding_pending_queue → page_skip_state
    → file_dataset → file_metadata（守卫：段落必须为 0 才删行）。
  * 物理目录 OneDrive/张智仁/<书名 (id)> → OneDrive/Plato/<书名 (id)>（容器内 /storage，
    先行预检 src 存在/dst 不存在；搬完 DB 若失败则把目录搬回去）。
  * 仅存在于一侧的行：真实侧只改路径前缀，Plato 侧独有且盘上无文件的只列出不删。

必须在导入器停机时运行。Dry-run 默认；--apply 写入。
"""
import asyncio, json, os, sys, logging, argparse, shutil

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/batch_operation")
logging.disable(logging.CRITICAL)
import common as C

SRC_TOP = "OneDrive/张智仁"
DST_TOP = "OneDrive/Plato"
STORAGE = "/storage"

def cpath(rel):
    return os.path.join(STORAGE, rel)

def parse_pg_array(s):
    """PostgreSQL text[] 文本表示 → list：{'Plato'} / {"Allan Bloom",Plato} / {} """
    s = (s or "").strip()
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    if not s:
        return []
    out, cur, inq = [], "", False
    i = 0
    while i < len(s):
        c = s[i]
        if inq:
            if c == "\\" and i + 1 < len(s):
                cur += s[i + 1]; i += 2; continue
            if c == '"':
                inq = False; i += 1; continue
            cur += c
        else:
            if c == '"':
                inq = True
            elif c == ",":
                out.append(cur); cur = ""
            else:
                cur += c
        i += 1
    out.append(cur)
    return [x.strip() for x in out if x.strip()]

async def main(apply=False):
    conn = await C.get_conn()

    # 导入器必须停机
    out = os.popen("ps auxww").read()
    if "batch_import_direct" in out or "import_onedrive_pdfs" in out:
        print("ABORT: importer running")
        return

    rows = await conn.fetch(
        "SELECT id, path, title, authors::text AS a, extension, item_type FROM file_metadata "
        "WHERE path LIKE $1 OR path LIKE $2 ORDER BY path",
        SRC_TOP + "/%", DST_TOP + "/%")

    # 按「书名 (id) 目录」分组
    def dir_of(p):
        parts = p.split("/")
        return parts[2] if len(parts) > 3 else None

    src_groups, dst_groups = {}, {}
    for r in rows:
        d = dir_of(r["path"])
        if d is None:
            continue
        (src_groups if r["path"].startswith(SRC_TOP) else dst_groups).setdefault(d, []).append(r)

    shared = sorted(set(src_groups) & set(dst_groups))
    only_src = sorted(set(src_groups) - set(dst_groups))
    only_dst = sorted(set(dst_groups) - set(src_groups))
    print("src dirs: %d, dst dirs: %d, shared: %d, only_src: %d, only_dst: %d"
          % (len(src_groups), len(dst_groups), len(shared), len(only_src), len(only_dst)))

    # 段落统计
    all_ids = [r["id"] for g in list(src_groups.values()) + list(dst_groups.values()) for r in g]
    stats = {}
    for i in range(0, len(all_ids), 2000):
        got = await conn.fetch(
            "SELECT source_id, count(*) AS n, coalesce(sum(length(content)),0) AS ch "
            "FROM paragraph WHERE source_id = ANY($1::uuid[]) GROUP BY source_id",
            all_ids[i:i + 2000])
        for g in got:
            stats[g["source_id"]] = (g["n"], g["ch"])

    plan = []   # each: kind, keep_id, drop_id, keep_path(new), drop_path, keep_authors
    for d in shared:
        src_by_base = {os.path.basename(r["path"]): r for r in src_groups[d]}
        dst_by_base = {os.path.basename(r["path"]): r for r in dst_groups[d]}
        for base in sorted(set(src_by_base) & set(dst_by_base)):
            s, t = src_by_base[base], dst_by_base[base]
            ns, ch_s = stats.get(s["id"], (0, 0))
            nt, ch_t = stats.get(t["id"], (0, 0))
            # 平手优先级：非张智仁作者 > 路径真实存在（src 真实）
            s_ok = "张智仁" not in s["a"]
            t_ok = "张智仁" not in t["a"]
            if ch_s != ch_t:
                keep, drop = (s, t) if ch_s > ch_t else (t, s)
            elif s_ok != t_ok:
                keep, drop = (s, t) if s_ok else (t, s)
            else:
                keep, drop = s, t
            drop_stats = stats.get(drop["id"], (0, 0))
            keep_stats = stats.get(keep["id"], (0, 0))
            ka = parse_pg_array(keep["a"])
            ka = ["Plato"] if ("张智仁" in ka or not ka) else ka
            plan.append({
                "kind": "merge", "dir": d, "base": base,
                "keep_id": str(keep["id"]), "drop_id": str(drop["id"]),
                "keep_old_path": keep["path"], "drop_path": drop["path"],
                "keep_new_path": "%s/%s/%s" % (DST_TOP, d, base),
                "keep_authors": ka,
                "keep_stats": keep_stats, "drop_stats": drop_stats,
            })
        for base in sorted(set(src_by_base) - set(dst_by_base)):
            s = src_by_base[base]
            ka = parse_pg_array(s["a"])
            ka = ["Plato"] if ("张智仁" in ka or not ka) else ka
            plan.append({"kind": "move_only", "dir": d, "base": base,
                         "keep_id": str(s["id"]), "old_path": s["path"],
                         "new_path": "%s/%s/%s" % (DST_TOP, d, base),
                         "keep_authors": ka})
        for base in sorted(set(dst_by_base) - set(src_by_base)):
            t = dst_by_base[base]
            plan.append({"kind": "dst_only", "dir": d, "base": base,
                         "drop_path": t["path"], "exists_on_disk": os.path.exists(cpath(t["path"]))})

    # 只在张智仁侧存在的目录（如 法律篇 (5148)）：整目录搬过去 + 改 path/authors
    for d in only_src:
        for r in sorted(src_groups[d], key=lambda x: x["path"]):
            base = os.path.basename(r["path"])
            ka = parse_pg_array(r["a"])
            ka = ["Plato"] if ("张智仁" in ka or not ka) else ka
            plan.append({"kind": "move_only", "dir": d, "base": base,
                         "keep_id": str(r["id"]), "old_path": r["path"],
                         "new_path": "%s/%s/%s" % (DST_TOP, d, base),
                         "keep_authors": ka})

    nm = sum(1 for p in plan if p["kind"] == "merge")
    mo = sum(1 for p in plan if p["kind"] == "move_only")
    do = [p for p in plan if p["kind"] == "dst_only"]
    print("PLAN: merge %d, move_only %d, dst_only %d (listed, not touched)" % (nm, mo, len(do)))
    for p in plan:
        if p["kind"] == "merge":
            print("  merge  keep=%s (%d段/%d字)  drop=%s (%d段/%d字)  -> %s | authors=%s"
                  % ("DST行" if p["keep_old_path"].startswith(DST_TOP) else "SRC行",
                     p["keep_stats"][0], p["keep_stats"][1], p["drop_path"],
                     p["drop_stats"][0], p["drop_stats"][1], p["keep_new_path"], p["keep_authors"]))
        elif p["kind"] == "move_only":
            print("  move   %s -> %s | authors=%s" % (p["old_path"], p["new_path"], p["keep_authors"]))
        else:
            print("  dst_only(不动) %s disk=%s" % (p["drop_path"], p["exists_on_disk"]))

    json.dump(plan, open("/app/batch_operation/reports/plato_merge_plan.json", "w"),
              ensure_ascii=False, indent=1)
    if not apply:
        print("DRY-RUN (no --apply)")
        return

    # ---- 备份：行 + 数据集绑定 ----
    bak = open("/app/batch_operation/reports/plato_merge_backup.tsv", "w")
    bak.write("id\tpath\tauthors\textension\titem_type\n")
    for r in rows:
        bak.write("%s\t%s\t%s\t%s\t%s\n" % (r["id"], r["path"], r["a"], r["extension"], r["item_type"]))
    bak.close()
    bds = await conn.fetch(
        "SELECT fd.file_id, fd.dataset_id, d.name FROM file_dataset fd JOIN dataset d ON d.id=fd.dataset_id "
        "WHERE fd.file_id = ANY($1::uuid[])", all_ids)
    with open("/app/batch_operation/reports/plato_merge_dataset_backup.tsv", "w") as f:
        for r in bds:
            f.write("%s\t%s\t%s\n" % (r["file_id"], r["dataset_id"], r["name"]))
    print("backup: rows=%d dataset_bindings=%d" % (len(rows), len(bds)))

    results = {"merged": 0, "moved": 0, "failed": []}
    dirs_moved = {}   # dir -> True (已搬)

    def move_dir(d):
        src = cpath("%s/%s" % (SRC_TOP, d))
        dst = cpath("%s/%s" % (DST_TOP, d))
        if not os.path.isdir(src):
            if os.path.isdir(dst):
                return "already"
            raise RuntimeError("src missing: " + src)
        if os.path.isdir(dst):
            raise RuntimeError("dst exists: " + dst)
        os.rename(src, dst)
        return "moved"

    for p in plan:
        if p["kind"] == "dst_only":
            continue
        d = p["dir"]
        try:
            if d not in dirs_moved:
                dirs_moved[d] = move_dir(d)
            if p["kind"] == "merge":
                # 1) 删 drop（先子表）
                await conn.execute("DELETE FROM text_embeddings WHERE source_id = $1::uuid", p["drop_id"])
                await conn.execute("DELETE FROM paragraph WHERE source_id = $1::uuid", p["drop_id"])
                left = await conn.fetchval("SELECT count(*) FROM paragraph WHERE source_id = $1::uuid", p["drop_id"])
                if left:
                    raise RuntimeError("paragraph still present: %d" % left)
                await conn.execute("DELETE FROM embedding_pending_queue WHERE source_id = $1::uuid", p["drop_id"])
                await conn.execute("DELETE FROM page_skip_state WHERE source_id = $1::uuid", p["drop_id"])
                await conn.execute("DELETE FROM file_dataset WHERE file_id = $1::uuid", p["drop_id"])
                await conn.execute("DELETE FROM file_metadata WHERE id = $1::uuid", p["drop_id"])
                # 2) keep 改 path/authors（旧路径守卫）
                tag = await conn.execute(
                    "UPDATE file_metadata SET path=$2, authors=$3::text[] WHERE id=$1::uuid AND path=$4",
                    p["keep_id"], p["keep_new_path"], p["keep_authors"], p["keep_old_path"])
                if tag.split()[-1] != "1":
                    raise RuntimeError("keep path update mismatch: %s" % tag)
                results["merged"] += 1
            else:
                tag = await conn.execute(
                    "UPDATE file_metadata SET path=$2, authors=$3::text[] WHERE id=$1::uuid AND path=$4",
                    p["keep_id"], p["new_path"], p["keep_authors"], p["old_path"])
                if tag.split()[-1] != "1":
                    raise RuntimeError("path update mismatch: %s" % tag)
                results["moved"] += 1
            # 3) 同目录下其余行（cover.jpg/metadata.opf 等）改前缀
            pref_old = "%s/%s/%s" % (SRC_TOP, d, "")
            await conn.execute(
                "UPDATE file_metadata SET path = $2 || substring(path from length($1)+1) "
                "WHERE path LIKE $1 || '%%'", pref_old, DST_TOP + "/")
            print("  OK %s %s" % (p["kind"], p["base"]))
        except Exception as e:
            print("  FAIL %s %s :: %s" % (p["kind"], p["base"], str(e)[:200]))
            # 目录已搬则搬回（保持盘与库一致）
            if dirs_moved.get(d) == "moved":
                try:
                    os.rename(cpath("%s/%s" % (DST_TOP, d)), cpath("%s/%s" % (SRC_TOP, d)))
                    dirs_moved[d] = "rolled_back"
                    print("    dir rolled back to", SRC_TOP + "/" + d)
                except Exception as e2:
                    print("    ROLLBACK FAILED:", str(e2)[:200])
            results["failed"].append(p.get("base"))

    print("RESULT:", json.dumps(results, ensure_ascii=False))
    json.dump(results, open("/app/batch_operation/reports/plato_merge_result.json", "w"),
              ensure_ascii=False, indent=1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    asyncio.run(main(apply=a.apply))
