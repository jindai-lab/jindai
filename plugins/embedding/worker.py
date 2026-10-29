"""常驻向量补算 worker（原 ``jindai/resources.py`` 的 EmbeddingManager）。

职责：
1. 轮询 ``embedding_pending_queue``：有待补条目就调用
   ``maintenance_manager.update_text_embeddings()`` 分批算向量、落库；
2. 磁盘水位守卫：占用 ≥ ``disk_guard_pause_pct`` 挂起本 worker，
   ≤ ``disk_guard_resume_pct`` 自动恢复（滞回防抖，不重启容器）；
3. 空闲 / 无进展时指数退避（2s → 60s），有进展立刻续跑，避免打死网关；
4. 对外提供 ``/embeddings`` 路由（统计 / 批量向量 / 单段重算）。

开关：``config.yaml`` 的 ``embedding_worker_enabled``（也可在本插件的配置段
``EmbeddingWorkerPlugin: {enabled: false}`` 里覆盖）。启停由 app lifespan →
``PluginManager.startup()/shutdown()`` 统一管理。
"""

from __future__ import annotations

import asyncio
import base64
import logging
import struct
from typing import Any, cast

from fastapi import APIRouter, Body, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from jindai import disk_guard
from jindai.config import config
from jindai.maintenance import maintenance_manager
from jindai.models import (EmbeddingPendingQueue, Paragraph, TextEmbeddings,
                           get_db, get_db_session)
from jindai.plugin import WorkerPlugin


class EmbeddingWorkerPlugin(WorkerPlugin):
    """补算段落向量（常驻 worker）+ 嵌入相关路由。"""

    worker_name = "embedding worker"
    enabled_key = "embedding_worker_enabled"
    enabled_default = True

    # ------------------------------------------------------------------ 主循环
    async def worker_loop(self) -> None:
        """后台循环：待补队列非空 → 跑一轮补算；否则指数退避。"""
        if not getattr(config, "embedding_worker_enabled", True):
            logging.info(
                "Embedding Polling Loop disabled "
                "(config.embedding_worker_enabled = false)."
            )
            return
        logging.info("Embedding Polling Loop started.")
        # Dynamic inter-cycle sleep: work done -> back almost immediately;
        # no progress (queue empty, all flushes failed, or exception) -> exponential
        # backoff to a 60s cap so a dead gateway isn't hammered.
        base_sleep = 2.0
        max_sleep = 60.0
        idle_streak = 0
        # 磁盘守卫（2026-10-26）：库的大头是 text_embeddings（~6.4KB/条，
        # 全量外推还要 ~1.25TB），写满盘比慢更糟（WAL/半成品索引难收拾）。
        # ≥ pause 挂起本 worker，≤ resume 恢复（滞回防抖），无需重启。
        guard_path = str(getattr(config, "disk_guard_path", "/app"))
        guard_pause = float(getattr(config, "disk_guard_pause_pct", 85))
        guard_resume = min(float(getattr(config, "disk_guard_resume_pct", 80)),
                           guard_pause)
        disk_paused = False
        disk_cycles = 0
        while True:
            pct = disk_guard.used_pct(guard_path)
            if pct is not None:
                if disk_paused:
                    if pct <= guard_resume:
                        logging.info(
                            "disk guard: usage %.1f%% <= %.1f%% — resuming "
                            "embedding worker", pct, guard_resume)
                        disk_paused = False
                        disk_cycles = 0
                    else:
                        disk_cycles += 1
                        if disk_cycles == 1 or disk_cycles % 15 == 0:
                            logging.warning(
                                "disk guard: usage %.1f%% > %.1f%% — embedding "
                                "worker still paused (cycle %d)", pct,
                                guard_resume, disk_cycles)
                        await asyncio.sleep(120)
                        continue
                elif pct >= guard_pause:
                    logging.warning(
                        "disk guard: usage %.1f%% >= %.1f%% — pausing embedding "
                        "worker (auto-resume at <= %.1f%%)", pct, guard_pause,
                        guard_resume)
                    disk_paused = True
                    disk_cycles = 0
                    await asyncio.sleep(120)
                    continue
            added = 0
            try:
                async with get_db_session() as session:
                    pending = await session.execute(
                        select(EmbeddingPendingQueue).limit(1)
                    )
                    has_pending = pending.first() is not None
                if has_pending:
                    logging.info("Pending queue not empty")
                    added = await maintenance_manager.update_text_embeddings()
            except asyncio.CancelledError:
                logging.info("Embedding Polling Loop is being cancelled.")
                return
            except Exception as ex:  # noqa: BLE001 - keep the worker alive
                logging.exception(
                    f"Embedding polling iteration failed: {type(ex).__name__}: {ex}")
            if added > 0:
                if idle_streak:
                    logging.info(
                        "embedding cycle made progress again after %d "
                        "idle cycle(s)", idle_streak)
                idle_streak = 0
                sleep_s = base_sleep
            else:
                idle_streak += 1
                sleep_s = min(base_sleep * (2 ** idle_streak), max_sleep)
                if idle_streak <= 3 or idle_streak % 6 == 0:
                    logging.info(
                        "embedding idle cycle %d (no embeddings added), "
                        "sleeping %.0fs", idle_streak, sleep_s)
            await asyncio.sleep(sleep_s)

    def worker_details(self) -> dict:
        """GET /plugins/workers 额外状态（运维排查用）。"""
        return {
            "queue": "see GET /embeddings/ (queued)",
            "disk_pause_pct": getattr(config, "disk_guard_pause_pct", None),
            "disk_resume_pct": getattr(config, "disk_guard_resume_pct", None),
            "api_base": ("local (in-process ONNX)"
                         if self._local_backend_selected()
                         else config.embedding_api_base),
            "model": config.embedding_model,
        }

    @staticmethod
    def _local_backend_selected() -> bool:
        """True when embedding_api_base selects this package's local backend."""
        from .local_backend import is_local
        return is_local(config.embedding_api_base)

    # -------------------------------------------------------------------- 路由
    def register_routes(self, router: APIRouter) -> None:
        """注册 /embeddings 相关路由。

        Args:
            router: APIRouter to register routes with.
        """

        @router.get("/embeddings/", tags=["Embeddings"])
        async def stat_embeddings(session: AsyncSession = Depends(get_db)):
            """Get embedding statistics.

            Args:
                session: Database session.

            Returns:
                Dictionary with finished and queued counts.
            """
            finished = None
            queued = (
                await session.execute(
                    select(EmbeddingPendingQueue).with_only_columns(
                        func.count(EmbeddingPendingQueue.id)
                    )
                )
            ).scalar()
            return {"finished": finished, "queued": queued}

        @router.post("/embeddings", tags=["Embeddings"])
        async def create_embeddings(
            data: dict = Body(...),
            session: AsyncSession = Depends(get_db),
        ):
            """Create embeddings for input text(s) using configured model.

            This endpoint follows OpenAI's embeddings API style:
            https://platform.openai.com/docs/api-reference/embeddings

            Args:
                data: Request body with:
                    - model: Model name to use for embeddings (optional, uses config model)
                    - input: Text or list of texts to embed
                    - encoding_format: Output format ("float" or "base64")
                session: Database session.

            Returns:
                Dictionary with embeddings in OpenAI format.
            """
            # Extract input texts (can be string or list of strings)
            input_data = data.get("input", [])
            if isinstance(input_data, str):
                input_texts = [input_data]
            elif isinstance(input_data, list):
                input_texts = input_data
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid input: must be a string or list of strings",
                )

            if not input_texts:
                raise HTTPException(status_code=400, detail="Invalid input: empty list")

            # Get model name (use config if not provided)
            model_name = data.get("model", config.embedding_model)

            # Compute embeddings using the configured model
            embeddings = []
            for text in input_texts:
                if not isinstance(text, str):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid input: expected string, got {type(text).__name__}",
                    )
                # Use the existing get_embedding method from TextEmbeddings
                embedding = await TextEmbeddings.get_embedding(text)
                embeddings.append(embedding)

            # Determine encoding format (default to float)
            encoding_format = data.get("encoding_format", "float")

            # Build response in OpenAI format
            data_entries = []
            for i, embedding in enumerate(embeddings):
                # Convert to base64 if requested
                if encoding_format == "base64":
                    # Pack floats as binary and encode as base64
                    packed = struct.pack(f"{len(embedding)}f", *embedding)
                    embedding_b64 = base64.b64encode(packed).decode("utf-8")
                    data_entries.append(
                        {"object": "embedding", "embedding": embedding_b64, "index": i}
                    )
                else:
                    # Default: return as float list
                    data_entries.append(
                        {"object": "embedding", "embedding": embedding, "index": i}
                    )

            # Calculate token usage (approximate: 1 token ~ 4 characters)
            total_tokens = sum(len(text) // 4 for text in input_texts)

            return {
                "object": "list",
                "data": data_entries,
                "model": model_name,
                "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens},
            }

        @router.post("/embeddings/{resource_id}", status_code=201, tags=["Embeddings"])
        async def update_single_embedding(
            resource_id: str, session: AsyncSession = Depends(get_db)
        ):
            """Update embedding for a single paragraph.

            Args:
                resource_id: Paragraph ID.
                session: Database session.

            Returns:
                Dictionary with paragraph ID.

            Raises:
                HTTPException: If paragraph not found.
            """
            para = await session.get(Paragraph, resource_id)
            if not para or not para.content:
                raise HTTPException(404)
            emb_val = await TextEmbeddings.get_embedding(para.content)
            te = (
                await session.execute(
                    select(TextEmbeddings).filter(TextEmbeddings.id == para.id)
                )
            ).scalar_one_or_none()
            if te:
                te.embedding = cast(Any, emb_val)
            else:
                session.add(TextEmbeddings(id=resource_id, embedding=emb_val))
            await session.commit()
            return {"id": resource_id}
