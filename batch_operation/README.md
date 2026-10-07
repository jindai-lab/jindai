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
- **嵌入端点**：`config.yaml` 的 `embedding_api_base: local` → 进程内跑 HF 模型
  `embedding_model`（BAAI/bge-m3，onnxruntime-gpu CUDA EP，1024 维，见
  `jindai/local_embedding.py`）；旧的本机 bge_m3 服务（端口 8090）已于 2026-10-07 删除。
  rerank 走 `rerank_api_base`/`rerank_api_key`（云端，代码见 `jindai/models.py`）。
  **精度**：`embedding_onnx_path` 指向低精度 ONNX，现设 `quant/bge-m3-fp16.onnx`（1.06GiB，
  HF 官方 fp32 是 2.11GiB）；留空则回落到官方 fp32 导出。fp16 版转换脚本
  `~/jindai-lab/scripts/build_fp16_onnx.py`（容器内跑）。
  **注意**：`embedding_worker_enabled: false` 时轮询协程不跑（当前就是如此，等 OCR
  导入完成后再打开），批处理脚本不受影响、照常消费队列。

## 配套后端修复（2026-09-29，见 git diff）

- `jindai/maintenance.py::update_text_embeddings`：反连接跳过已嵌入 + 消费队列行 +
  stall 兜底（原版会无限重选同 1 万行，最终撞主键崩溃）；`_do` 改为分块/重试/幂等插入；
  新增 `consume_embedding_queue`。
- `jindai/resources.py::polling_loop`：单轮异常不再杀死整个轮询协程。
- `jindai/models.py`：`batch_encode(timeout=...)`；rerank 支持独立 `rerank_api_base/key`。
- `jindai/config.py`：`-c` 参数不再误伤 `python -c`（argv[0]="-c" 时 IndexError）。

## 本机 bge_m3 服务（2026-09-29 调优定稿；**2026-10-07 已删除**）

> 该服务与 `/mnt/repo/llms/bge-m3/bge-m3-FP16.gguf` 已删除，`~/stacks/bge-m3/` 也已移除；
> 嵌入改为进程内 local 后端（见上「嵌入端点」）。以下命令仅存档备查，重跑需先重新下载 GGUF。

```bash
docker run -d --name bge_m3 --restart unless-stopped --gpus all \
  -p 8090:8080 -v /mnt/repo/llms:/models:ro --network bge-m3_default \
  ghcr.nju.edu.cn/ggml-org/llama.cpp:server-cuda13 \
  -m /models/bge-m3/bge-m3-FP16.gguf --embeddings --pooling cls \
  --host 0.0.0.0 --port 8080 -ngl 99 -b 8192 -ub 8192 --no-warmup \
  -c 16384 -np 8 -t 12 -tb 12
```

要点（慢的根因与修复）：
- 默认 `-c 8192 -np 16` → n_ctx_slot=512，一次 256 条数组请求拼接 prompt 远超
  512 → 溢出路径把吞吐打到 1-15 texts/s，GPU util 0%（跑在 CPU 慢路径）。
  改 `-c 16384 -np 8`（2048 tok/槽）+ `-t 12 -tb 12` 后：
  单请求 arr256 = 0.78s（327 texts/s），24×64 并发饱和 559 texts/s，约 230 倍。
- `--gpus all` 必须显式传（重建容器时 DeviceRequests 不会自动带上）。
