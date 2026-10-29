"""删除文献记录时，把关联文件移入存储根目录下的回收站 ``.trash``。

内置插件 ``plugins/trash`` 的核心模块（2026-10-29 从 ``jindai`` 核心迁入；
其它插件直接 ``from plugins.trash import move_to_trash``，不再有旧路径副本）。

约定（与导入器口径配套，2026-10-09）：

* 回收站根 = ``<storage>/.trash``（宿主 ``/misc/Public/jindai-storage/.trash``）。
  ``.trash`` 是**隐藏目录**：导入器的磁盘 upsert 会剪掉所有隐藏目录
  （``import_onedrive_pdfs.py`` 的 ``dirnames`` 过滤），其余全盘 walk
  （``register_untracked_pdfs.py`` / ``pdf_disk_inventory.py`` /
  ``fix_moved_paths.py`` / ``purge_sync_conflict_copies.py``）也一律跳过隐藏目录，
  所以回收站里的文件**不会**被重新登记成 ``file_metadata`` 行、不会进导入队列，
  也不会被“陈旧路径修复”脚本当成别的记录的落脚点。
* 每次删除一个批次：``.trash/<YYYYmmdd-HHMMSS>-<item_id 前 8 位>/<原相对路径>``。
  保留原相对路径 → 同名文件不互撞，且一眼能看出文件原本在哪。
* 批次目录里写 ``manifest.json``（原路径/新路径/题名/作者/时间/条目 id）；
  同时向 ``batch_operation/reports/trash_manifest.jsonl`` 追加一行，
  这样回收站被每周清空之后仍然留有可审计的痕迹。
* 每周清空：常驻 ``plugins/trash/purge.py``（每周一 06:00，随 app lifespan
  启动，config ``trash_purge_enabled`` 开关；2026-10-25 起取代原 Hermes
  cronjob）。手动演练/补跑仍用 ``batch_operation/purge_trash.py [--apply]``。

本模块只碰文件系统、不碰数据库，便于单独测试；“这个路径是否还被别的记录引用”
的判断放在调用方（bibliography 插件）里，因为它需要 session。
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime
from typing import Iterable, List, Optional
from pathlib import Path

from jindai.storage import storage

TRASH_DIRNAME = ".trash"

log = logging.getLogger(__name__)


def trash_root() -> str:
    """回收站根目录的绝对路径（不自动创建）。"""
    return storage.safe_join(TRASH_DIRNAME)


MANIFEST_LOG = str(Path(trash_root()) / "trash_manifest.jsonl")


def is_trash_path(rel_path: str) -> bool:
    """判断 storage 相对路径是否已经落在回收站里。"""
    p = (rel_path or "").lstrip("/")
    return p == TRASH_DIRNAME or p.startswith(TRASH_DIRNAME + "/")


def _unique_target(batch_dir: str, rel: str) -> str:
    """批次目录内按原相对路径落位；撞名时加 ``~1``、``~2``。"""
    base = os.path.join(batch_dir, rel)
    if not os.path.exists(base):
        return base
    stem, ext = os.path.splitext(base)
    for i in range(1, 1000):
        cand = f"{stem}~{i}{ext}"
        if not os.path.exists(cand):
            return cand
    raise FileExistsError(f"无法为 {rel} 在回收站里找到空位")


def move_to_trash(
    rel_paths: Optional[Iterable[str]],
    *,
    item_id: str,
    title: Optional[str] = None,
    authors: Optional[Iterable[str]] = None,
    dry_run: bool = False,
) -> dict:
    """把一批 storage 相对路径移进回收站。

    不会因为单个文件失败而中断：跳过/失败的条目原样留在返回值里
    （原地留存 = 删除前的老行为，由调用方记日志）。

    Returns:
        ``{"batch_dir": <回收站内的相对批次目录或 None>,
          "moved": [{"from","to","size_bytes"}...],
          "skipped": [{"path","reason"}...]}``
    """
    result: dict = {"batch_dir": None, "moved": [], "skipped": []}
    paths: List[str] = []
    for p in rel_paths or []:
        p = (p or "").strip()
        if not p:
            continue
        if p.startswith("bib:") or "://" in p:
            result["skipped"].append({"path": p, "reason": "non_file_ref"})
            continue
        if is_trash_path(p):
            result["skipped"].append({"path": p, "reason": "already_in_trash"})
            continue
        if p not in paths:
            paths.append(p)
    if not paths:
        return result

    # 先解析、确认哪些真的能搬，避免建出空的批次目录。
    movable = []
    for p in paths:
        try:
            src = storage.safe_join(p)
        except ValueError:
            result["skipped"].append({"path": p, "reason": "outside_storage"})
            continue
        if not os.path.isfile(src):
            result["skipped"].append({"path": p, "reason": "not_a_file"})
            continue
        movable.append((p, src))
    if not movable:
        return result

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    batch_name = f"{stamp}-{str(item_id)[:8]}"
    root = trash_root()
    batch_dir = os.path.join(root, batch_name)
    result["batch_dir"] = f"{TRASH_DIRNAME}/{batch_name}"

    if dry_run:
        for p, src in movable:
            result["moved"].append({
                "from": p,
                "to": f"{result['batch_dir']}/{p}",
                "size_bytes": os.path.getsize(src),
            })
        return result

    os.makedirs(batch_dir, exist_ok=True)
    entries = []
    for p, src in movable:
        try:
            dst = _unique_target(batch_dir, p)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            size = os.path.getsize(src)
            shutil.move(src, dst)
            rel_dst = os.path.relpath(dst, root).replace(os.sep, "/")
            result["moved"].append({"from": p, "to": f"{TRASH_DIRNAME}/{rel_dst}",
                                    "size_bytes": size})
            entries.append({"from": p, "to": f"{TRASH_DIRNAME}/{rel_dst}",
                            "size_bytes": size})
        except Exception as ex:  # noqa: BLE001 - 单文件失败不该中断删除
            result["skipped"].append({"path": p, "reason": f"{type(ex).__name__}: {ex}"})
            log.warning("回收站搬运失败 %s: %s", p, ex)

    if not entries:
        # 一个都没搬走就别留空目录
        try:
            os.rmdir(batch_dir)
        except OSError:
            pass
        result["batch_dir"] = None
        return result

    manifest = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "item_id": str(item_id),
        "title": title,
        "authors": list(authors or []),
        "files": entries,
    }
    try:
        with open(os.path.join(batch_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=1)
    except Exception as ex:  # noqa: BLE001
        log.warning("回收站 manifest 写入失败: %s", ex)
    try:
        os.makedirs(os.path.dirname(MANIFEST_LOG), exist_ok=True)
        with open(MANIFEST_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(manifest, ensure_ascii=False) + "\n")
    except Exception as ex:  # noqa: BLE001
        log.warning("回收站审计日志追加失败: %s", ex)

    log.info(
        "删除记录 %s（%s）→ 回收站 %s：%d 个文件，%d 个跳过",
        item_id, title or "", result["batch_dir"],
        len(result["moved"]), len(result["skipped"]),
    )
    return result
