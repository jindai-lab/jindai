"""磁盘水位守卫工具（2026-10-26；2026-10-29 由 ``disk_guard.py`` 更名）。

背景：pgdata 与 /app 同在宿主 pve-root（1.9T）。库现 530GB，text_embeddings
外推全量还要 ~1.25TB——照此跑下去 embedding 会把盘写满，届时 WAL/半成品
vchordrq 索引比缺空间本身更难收拾。守卫策略：占用 ≥ pause 阈值挂起消费方，
≤ resume 阈值自动恢复（滞回防抖），全程不重启容器、不改 config。

⚠️ 检查路径必须与 pgdata 同一文件系统：容器里看不到 pgdata 挂载，但
/app（宿主 ~/jindai-lab/jindai）与 pgdata（宿主 ~/jindai-lab/pgdata）同在
pve-root，所以默认查 /app。若日后 pgdata 迁盘，改 config.yaml 的
disk_guard_path 指向与新盘同盘的容器内可见路径。

注：模块名与 ``diskutils`` 对齐（与 ocrutils / pdfutils 同一命名族），但
config 键仍是历史名 ``disk_guard_path`` / ``disk_guard_pause_pct`` /
``disk_guard_resume_pct``——改键要同步动 config.yaml 与两个 worker 的
getattr 默认值，故保持不动。
"""
import logging
import shutil
from typing import Optional


def used_pct(path: str) -> Optional[float]:
    """path 所在文件系统的已用百分比（0-100）；拿不到返回 None。"""
    try:
        du = shutil.disk_usage(path)
    except OSError:
        logging.warning("diskutils: cannot stat %s", path, exc_info=True)
        return None
    if not du.total:
        return None
    return du.used / du.total * 100.0
