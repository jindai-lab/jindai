# Built-in plugins (`plugins/`)

Jindai 的内置插件目录。`jindai.helpers.get_context("plugins", Plugin)` 会扫描
`plugins/*.py` 与 `plugins/*/__init__.py`，把其中所有 `jindai.plugin.Plugin` 的
子类注册进 `PluginManager`；`config.yaml` 的 `plugins: ['*']` 表示全部启用。

## 常驻后台 worker 插件（2026-10-29 整理）

| 插件包 | 类 | 职责 | 开关（config.yaml） | 状态/路由 |
| --- | --- | --- | --- | --- |
| `plugins/embedding` | `EmbeddingWorkerPlugin` | 轮询 `embedding_pending_queue` 补算段落向量（磁盘水位守卫 + 空闲指数退避）；`local_backend.py` 是 `embedding_api_base: local` 的本机 ONNX 后端 | `embedding_worker_enabled` | `GET /api/v2/embeddings/`、`POST /api/v2/embeddings`、`POST /api/v2/embeddings/{id}` |
| `plugins/importer` | `ImportWorkerPlugin` | 监听清单/磁盘变动，以子进程拉起导入引擎（段错误隔离）、崩溃归因与毒文件黑名单；同包的 `import_lists`（清单/磁盘登记/盖章）、`import_engine`（stage 链 + 文件级并发）、`import_cli`（argparse 入口）是导入链本体 | `import_worker_enabled` | `GET /api/v2/import/`、`POST /api/v2/import/pause` |
| `plugins/trash` | `TrashPurgeWorkerPlugin` | 每周一 06:00 清空 `<storage>/.trash`（仍被 `file_metadata` 引用的批次跳过）；`core.py` 提供 `move_to_trash()` 给其它插件用 | `trash_purge_enabled` | `GET /api/v2/trash/` |

统一状态：`GET /api/v2/plugins/workers`（`PluginManager.worker_status()`）。

## 生命周期

插件实例在 **import 期**构造（此时没有事件循环），后台任务必须在 lifespan 的
`PluginManager.startup(app)` 里创建、在 `shutdown(app)` 里取消：

```python
from jindai.plugin import WorkerPlugin


class MyWorkerPlugin(WorkerPlugin):
    worker_name = "my worker"
    enabled_key = "my_worker_enabled"     # config.yaml 开关（可被 enabled: 覆盖）
    enabled_default = False               # 开关缺省值

    async def worker_loop(self) -> None:   # 循环体：自己兜异常，直到被 cancel
        while True:
            await do_one_round()
            await asyncio.sleep(60)
```

`WorkerPlugin` 负责：开关判断、`asyncio.create_task`、取消 + `gather`、异常退出后
按指数退避重启（`restart_base_sleep` / `restart_max_sleep`）、`worker_status()`
（可覆盖 `worker_details()` 追加字段）。只声明 `abstract = True` 的基类不会被实例化。

## 依赖方向

接口唯一：回收站搬运用 `from plugins.trash import move_to_trash`，本机嵌入后端用
`from plugins.embedding import local_backend`（`jindai.models` 也在运行时按需 import
它），导入清单/磁盘登记/引擎/回填用 `from plugins.importer.import_lists import ...`
/ `plugins.importer.fill_nontitle`（`plugins/trash/purge.py`、`batch_operation/*.py`
都从这里取 `REPORTS` / `get_conn`）。依赖方向单向：plugins → jindai，
`jindai/` 核心不再 import 本目录，也不再保留这些模块的副本或兼容薄壳。
