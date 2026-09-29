# batch_operation — jindai 批处理脚本

容器内执行（`/app/batch_operation` = 本目录；宿主机 `~/jindai-lab/jindai/batch_operation`）。
所有脚本复用应用自身的 config / ORM / 分块规则，输出统一写到 `reports/`。

```bash
# 通用调用形式
docker exec -w /app jindai /app/.venv/bin/python /app/batch_operation/<script>.py [args]
```

## 脚本

| 脚本 | 作用 | 常用参数 |
|---|---|---|
| `stats.py` | 管道体检：API 健康、缺 paragraph 的 PDF、缺 embedding 的段落、队列总量 | `--deep`（stale/orphan 细分，慢 ~10min）、`--json f.json` |
| `import_onedrive_pdfs.py` | 导入 OneDrive 下尚无 paragraph 的 PDF（跑存储任务「导入 OneDrive PDF」，断点续跑） | `--dry-run`、`-n 并发数`、`--task 名称` |
| `embed_missing.py` | 给所有缺 embedding 的段落补向量（分块 200/50 与后端一致，写入 `ON CONFLICT DO NOTHING`，完成后消费队列行） | `--scope all\\|onedrive`、`--dry-run`、`--batch 256`、`--inflight 4`、`--limit-files N`、`--no-queue` |
| `clean_queue.py` | 清理队列中的 stale（已嵌入）与 orphan（段落已删）行 | `--dry-run`、`--what stale\\|orphan\\|both`、`--batch 20000` |
| `common.py` | 共享：config/DSN/EmbeddingClient/分块/进度 | — |

## 标准流程

```bash
# 0) 状态
docker exec -w /app jindai /app/.venv/bin/python /app/batch_operation/stats.py

# 1) 导入 OneDrive 缺 paragraph 的 PDF（可中断、可重跑；--dry-run 只看计数）
docker exec -w /app jindai /app/.venv/bin/python /app/batch_operation/import_onedrive_pdfs.py

# 2) 补 embedding（先小批试跑，再全量后台跑）
docker exec -w /app jindai /app/.venv/bin/python /app/batch_operation/embed_missing.py --dry-run
docker exec -w /app jindai /app/.venv/bin/python /app/batch_operation/embed_missing.py --scope all --limit-files 5
nohup docker exec -w /app jindai /app/.venv/bin/python /app/batch_operation/embed_missing.py \
  > ~/jindai-lab/jindai/batch_operation/reports/embed_missing.out 2>&1 &

# 3) 收尾清理队列（可选）
docker exec -w /app jindai /app/.venv/bin/python /app/batch_operation/clean_queue.py --dry-run
```

## 语义要点

- **断点续跑**：两个批处理都用 anti-join（没有就做、有就跳过），中断后直接重跑。
- **队列对账**：`embedding_pending_queue` 每行 = 一个待嵌入段落；脚本嵌入成功后删除对应行，
  UI 的 queued 计数随之下降。触发器在 paragraph 插入时自动入队。
- **与轮询协程共存**：`jindai/resources.py` 的 Embedding Polling Loop 也是消费者；
  写入均为 `ON CONFLICT DO NOTHING`，重复消费幂等、不报错。
- **嵌入端点**：`config.yaml` 的 `embedding_api_base` 指向本机 bge_m3
  （llama-server, `http://172.24.1.230:8090/v1`，1024 维）；rerank 走
  `rerank_api_base`/`rerank_api_key`（云端，代码见 `jindai/models.py`）。

## 配套后端修复（2026-09-29，见 git diff）

- `jindai/maintenance.py::update_text_embeddings`：反连接跳过已嵌入 + 消费队列行 +
  stall 兜底（原版会无限重选同 1 万行，最终撞主键崩溃）；`_do` 改为分块/重试/幂等插入；
  新增 `consume_embedding_queue`。
- `jindai/resources.py::polling_loop`：单轮异常不再杀死整个轮询协程。
- `jindai/models.py`：`batch_encode(timeout=...)`；rerank 支持独立 `rerank_api_base/key`。
- `jindai/config.py`：`-c` 参数不再误伤 `python -c`（argv[0]="-c" 时 IndexError）。
