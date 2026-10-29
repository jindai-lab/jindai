"""常驻向量补算 worker（原 ``jindai/resources.py`` 的 EmbeddingManager）。

职责：
1. 轮询 ``embedding_pending_queue``：有待补条目就调用
   ``update_text_embeddings()`` 分批算向量、落库；
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
from typing import Any, Optional, cast

import httpx
from fastapi import APIRouter, Body, Depends, HTTPException
from sqlalchemy import TIMESTAMP
from sqlalchemy import cast as sqlcast
from sqlalchemy import delete, exists, func, select, text, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from jindai import diskutils, embedutils
from jindai.config import config
from jindai.maintenance import maintenance_manager
from jindai.models import (EmbeddingPendingQueue, Paragraph, QueryFilters,
                           TextEmbeddings, get_db, get_db_session)
from jindai.plugin import WorkerPlugin


class EmbeddingWorkerPlugin(WorkerPlugin):
    """补算段落向量（常驻 worker）+ 嵌入相关路由。"""

    worker_name = "embedding worker"
    enabled_key = "embedding_worker_enabled"
    enabled_default = True    
    
    async def consume_embedding_queue(self, pids: list, expected: dict) -> int:
        """Remove queue rows for paragraphs whose embeddings are complete.

        A paragraph is complete when it has no chunks to embed (empty
        content) or when its row count in ``text_embeddings`` has reached
        the expected chunk count.

        Args:
            pids: Paragraph ids processed in the last flush group.
            expected: Mapping of paragraph id -> expected chunk count.

        Returns:
            Number of queue rows removed.
        """
        if not pids:
            return 0
        async with get_db_session() as session:
            rows = await session.execute(
                select(TextEmbeddings.id, func.count(TextEmbeddings.chunk_id))
                .where(TextEmbeddings.id.in_(pids))
                .group_by(TextEmbeddings.id)
            )
            counts = {row[0]: row[1] for row in rows}
            complete = [
                pid for pid in pids
                if counts.get(pid, 0) >= expected.get(pid, 0)
            ]
            if not complete:
                return 0
            result = await session.execute(
                delete(EmbeddingPendingQueue).where(
                    EmbeddingPendingQueue.id.in_(complete)
                )
            )
            return result.rowcount or 0
    
    async def update_text_embeddings(self, filters: Optional[QueryFilters] = None) -> int:
        """Update text embeddings for paragraphs in the pending queue.

        Processes at most 10000 queued paragraphs per cycle, skips
        paragraphs that already have embeddings, and removes queue rows
        once a paragraph's chunks are all present. Without those guards
        the loop re-selects the same rows forever (queue rows were never
        consumed) and eventually dies on duplicate primary keys.

        Args:
            filters: Optional QueryFilters to limit which paragraphs to process.

        Returns:
            Number of embeddings added.
        """
        if isinstance(filters, dict):
            filters = QueryFilters(**filters)

        if filters:
            # legacy path: narrow user filter; order by the queue PK so the
            # merge/index plan is eligible (never ORDER BY paragraph.id alone)
            stmt = await Paragraph.build_query(filters)
            stmt = (
                stmt.join(
                    EmbeddingPendingQueue,
                    (EmbeddingPendingQueue.id == Paragraph.id)
                    & (EmbeddingPendingQueue.source_id == Paragraph.source_id),
                )
                .where(~exists(select(1).where(TextEmbeddings.id == Paragraph.id)))
                .with_only_columns(Paragraph.id, Paragraph.source_id, Paragraph.content)
                .order_by(EmbeddingPendingQueue.id, EmbeddingPendingQueue.source_id)
                .limit(10000)
            )

        added = 0
        stall_rounds = 0
        while True:

            if filters:
                async with get_db_session() as session:
                    results = await session.execute(stmt)
                    results = results.mappings().all()
                stale_ids = []
            else:
                # Step 1: 10000 queue keys via PK (index-only, instant).
                async with get_db_session() as session:
                    key_rows = (
                        await session.execute(
                            select(
                                EmbeddingPendingQueue.id,
                                EmbeddingPendingQueue.source_id,
                            )
                            .order_by(
                                EmbeddingPendingQueue.id,
                                EmbeddingPendingQueue.source_id,
                            )
                            .limit(10000)
                        )
                    ).all()
                if not key_rows:
                    return added
                keys = [(r[0], r[1]) for r in key_rows]

                # Step 2: fetch contents through a VALUES join. The planner
                # sees an exact row count (uuid MCV stats otherwise mis-est
                # the queue x paragraph join as ~1000 rows and it picks a
                # parallel hash join + sort of tens of millions of rows --
                # one cycle then ran 8+ minutes and wedged the loop). Measured:
                # 10000 keys -> paragraph PK probes + anti-join = ~4s.
                values_sql = ", ".join(
                    f"(CAST(:p{i} AS UUID), CAST(:s{i} AS UUID))"
                    for i in range(len(keys))
                )
                # already-embedded; the fm INNER JOIN skips paragraphs whose
                # file_metadata row was deleted (2813 orphan sources sit at the
                # head of the queue because queue ids are time-ordered; their
                # inserts violate fk_text_embeddings_source and used to wedge
                # every cycle). Keys dropped by either join are consumed below.
                fetch_sql = text(
                    "SELECT p.id, p.source_id, p.content "
                    "FROM (VALUES " + values_sql + ") AS v(id, source_id) "
                    "JOIN paragraph p ON p.id = v.id AND p.source_id = v.source_id "
                    "JOIN file_metadata fm ON fm.id = p.source_id "
                    "WHERE NOT EXISTS ("
                    "  SELECT 1 FROM text_embeddings e WHERE e.id = p.id)"
                )
                params = {}
                for i, (pid, sid) in enumerate(keys):
                    params[f"p{i}"] = str(pid)
                    params[f"s{i}"] = str(sid)
                async with get_db_session() as session:
                    results = (await session.execute(fetch_sql, params)).mappings().all()

                # keys dropped by the joins (already embedded, or file_metadata
                # deleted -> FK would reject any embedding row) are consumed so
                # the queue keeps moving forward.
                fetched_ids = {r["id"] for r in results}
                stale_ids = [pid for pid, _sid in keys if pid not in fetched_ids]
                if stale_ids:
                    async with get_db_session() as session:
                        await session.execute(
                            delete(EmbeddingPendingQueue).where(
                                EmbeddingPendingQueue.id.in_(stale_ids)
                            )
                        )

            results_len = len(results)

            if not results_len:
                if stale_ids:
                    logging.info(
                        f"embedding cycle: all {len(stale_ids)} selected queue "
                        "rows already embedded, consumed them")
                    stall_rounds = 0
                    continue
                return added

            new_bulk = []
            expected = {}
            group = []
            consumed = 0
            failed_flushes = 0

            for i, p in enumerate(results, start=1):
                chunks = TextEmbeddings.get_chunks(p["content"] or "", 200, 50)
                expected[p["id"]] = len(chunks)
                group.append(p["id"])
                for chunk_id, chunk in enumerate(chunks, start=1):
                    new_bulk.append(
                        {
                            "id": str(p["id"]),
                            "source_id": str(p["source_id"]),
                            "chunk_id": chunk_id,
                            "content": chunk,
                        }
                    )
                # Bigger flush groups: each flush costs one API round-trip plus
                # a queue-consume pass. Benchmarked against the local bge-m3:
                # 512-text requests run ~3x faster per text than 256, and the
                # server gains nothing from concurrent requests (it already
                # fans a single request across its 8 slots).
                if i % 500 == 0 or i == results_len:
                    pending = len(new_bulk)
                    added += await self.update_text_embeddings_do(bulk=new_bulk)
                    done = await self.consume_embedding_queue(
                        group, expected
                    )
                    consumed += done
                    if pending and done == 0:
                        failed_flushes += 1
                    new_bulk = []
                    expected = {}
                    group = []
                    if failed_flushes >= 3:
                        logging.error(
                            "embedding API unavailable, aborting cycle after "
                            f"{failed_flushes} failed flushes")
                        return added

            if consumed == 0:
                stall_rounds += 1
                if stall_rounds >= 3:
                    logging.warning(
                        f"embedding cycle made no progress ({results_len} queued "
                        "rows still pending), backing off")
                    return added
            else:
                stall_rounds = 0

    async def update_text_embeddings_do(self, bulk: list) -> int:
        """Embed a batch of chunk texts and persist them idempotently.

        The batch is split into bounded API sub-requests, transient
        failures are retried with backoff, and rows are inserted with
        ON CONFLICT DO NOTHING so duplicate primary keys (e.g. a second
        consumer such as batch_operation/embed_missing.py) can never
        crash the polling loop.

        Args:
            bulk: List of paragraph chunk data dictionaries.

        Returns:
            Number of embedding rows inserted (0 on failure).
        """
        if not bulk:
            return 0
        inserted = 0
        # 512 per sub-request: measured ~5.9 texts/s vs ~2.0 at 256 against
        # the local bge-m3 server (single request already saturates it).
        sub_size = 512
        for start in range(0, len(bulk), sub_size):
            sub = bulk[start:start + sub_size]
            vectors = None
            for attempt in range(3):
                try:
                    vectors = await embedutils.get_client().encode(
                        [p["content"] for p in sub], timeout=180.0
                    )
                    break
                except (httpx.HTTPError, asyncio.TimeoutError) as ex:
                    wait = 2 ** attempt
                    logging.warning(
                        f"embedding API call failed ({type(ex).__name__}: {ex}), "
                        f"retry {attempt + 1}/3 in {wait}s")
                    await asyncio.sleep(wait)
                except Exception as ex:
                    logging.error(
                        f"embedding encode failed: {type(ex).__name__}: {ex}")
                    break
            if vectors is None:
                break
            if len(vectors) != len(sub):
                logging.warning(
                    f"embedding API returned {len(vectors)}/{len(sub)} vectors")
                sub = sub[:len(vectors)]
            stmt = (
                insert(TextEmbeddings)
                .values(
                    [
                        {
                            "id": p["id"],
                            "source_id": p["source_id"],
                            "chunk_id": p["chunk_id"],
                            "embedding": emb,
                        }
                        for p, emb in zip(sub, vectors)
                    ]
                )
                .on_conflict_do_nothing(
                    index_elements=["id", "source_id", "chunk_id"]
                )
            )
            try:
                async with get_db_session() as session:
                    result = await session.execute(stmt)
                    inserted += result.rowcount or 0
            except Exception as ex:
                logging.error(
                    "embedding insert failed: "
                    f"{type(ex).__name__}: {str(ex)[:300]}")
                break
        logging.info(f"embeddings {inserted} added")
        return inserted

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
            pct = diskutils.used_pct(guard_path)
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
                    added = await self.update_text_embeddings()
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
        """True when embedding_api_base selects the in-process backend."""
        return embedutils.is_local(config.embedding_api_base)

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

            # Validate inputs, then embed the whole batch through the client
            # (it paces + splits the request to the gateway limits).
            for text in input_texts:
                if not isinstance(text, str):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid input: expected string, got {type(text).__name__}",
                    )
            embeddings = await embedutils.get_client().encode(input_texts)

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
            emb_val = await embedutils.get_client().encode_one(para.content)
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
