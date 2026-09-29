"""Maintenance and administrative tasks for Jindai application.

This module provides:
- MaintenanceManager: Class for database maintenance operations
- File metadata synchronization
- Dataset merging and cleanup
- Text embedding updates
- OCR processing
- Background task management
"""

import asyncio
import datetime
import logging
import httpx
import os
import tempfile
from pathlib import Path
from typing import Any, Optional, cast

from fastapi import APIRouter, BackgroundTasks, Body, Depends
from sqlalchemy import (TIMESTAMP, cast as sqlcast, delete, exists, func,
                        select, text, update)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from .config import config
from .models import (Dataset, EmbeddingPendingQueue, FileDataset, FileMetadata,
                     Paragraph, QueryFilters, TaskDBO, Terms, TextEmbeddings,
                     get_current_admin, get_db_session)
from .pdfutils import get_pdf_page_count
from .storage import storage


class MaintenanceManager:
    """Manager for database and file maintenance operations.

    Provides methods for:
    - Synchronizing data sources
    - Syncing terms/keywords
    - Merging datasets
    - Updating metadata from URLs
    - File metadata population
    - Text embedding updates
    - OCR processing
    """

    async def sync_sources(self, dataset: str = "", folder: str = "") -> None:
        """Synchronize data sources by checking file existence.

        Marks paragraphs as offline if their source files no longer exist,
        and removes datasets with no associated files.

        Args:
            dataset: Optional dataset name to filter.
            folder: Optional folder path to filter.
        """
        assert dataset or folder, "Must specify at least one condition"

        async with get_db_session() as session:
            query = (
                select(FileMetadata)
                .join(Paragraph, Paragraph.source_id == FileMetadata.id)
            )
            if folder:
                query = query.filter(FileMetadata.path.startswith(folder))
            if dataset:
                query = query.join(
                    FileDataset, FileDataset.file_id == FileMetadata.id
                ).join(
                    Dataset, Dataset.id == FileDataset.dataset_id
                ).filter(
                    (Dataset.name == dataset)
                    | Dataset.name.startswith(dataset + "--")
                )

            query = query.distinct(FileMetadata.path).with_only_columns(
                FileMetadata.path, FileMetadata.id
            )
            rows = (await session.execute(query)).all()
            dataset_ids = set()

            for path, file_id in rows:
                if '://' in path:
                    continue
                if '#' in path:
                    path = path[:path.find('#')]
                joined = storage.safe_join(path)
                if not os.path.exists(joined):
                    logging.info(
                        f"{path} does not exist any more. Mark related paragraphs."
                    )
                    # Collect datasets associated with this file
                    fd_ids = (
                        await session.execute(
                            select(FileDataset.dataset_id).filter(
                                FileDataset.file_id == file_id
                            )
                        )
                    ).scalars().all()
                    dataset_ids.update(fd_ids)
                    await session.execute(
                        update(Paragraph)
                        .filter(Paragraph.source_id == file_id)
                        .values(
                            extdata=Paragraph.extdata.op("||")(
                                func.jsonb_build_object("offline", True)
                            )
                        )
                    )

            stmt = delete(Dataset).filter(
                ~exists().where(FileDataset.dataset_id == Dataset.id),
                Dataset.id.in_(list(dataset_ids)),
            )
            await session.execute(stmt)

    async def cleanup_unused_datasets(self) -> None:
        """Clean up unused datasets.

        Deletes all datasets that have no associated files,
        except for the default dataset with name='' (empty string).
        """
        async with get_db_session() as session:
            # Find datasets with no associated files, excluding the default dataset with name=''
            stmt = delete(Dataset).filter(
                ~exists().where(FileDataset.dataset_id == Dataset.id),
                Dataset.name != ''  # Exclude default dataset
            )
            result = await session.execute(stmt)
            deleted_count = cast(Any, result).rowcount
            if deleted_count > 0:
                logging.info(f"Cleaned up {deleted_count} unused datasets.")
            else:
                logging.info("No unused datasets found.")

    async def sync_terms(self) -> None:
        """Synchronize terms/keywords from paragraphs to the terms table.

        Extracts unique keywords from all paragraphs and upserts them
        into the Terms table.
        """
        async with get_db_session() as session:
            unnested_query = (
                select(text("unnest(keywords)").label("word"))
                .select_from(text("paragraph"))
                .subquery()
            )

            # Select unique words that aren't null or empty
            distinct_words_stmt = (
                select(unnested_query.c.word)
                .distinct()
                .where(unnested_query.c.word != None)
                .where(unnested_query.c.word != "")
            )

            result = await session.execute(distinct_words_stmt)
            unique_words = result.scalars().all()

            if not unique_words:
                logging.info("No keywords found to sync.")
                return

            # Perform a Bulk Upsert into the Terms table.
            # We use on_conflict_do_nothing to ignore words already in the
            # Terms table; 600-byte cap guards pk_term's 2704-byte btree limit
            # (OCR digit-string garbage used to abort the whole insert).
            data = [
                {"term": w}
                for w in unique_words
                if w and len(w.encode("utf-8")) <= 600
            ]
            if not data:
                logging.info("No indexable terms after length filter.")
                return

            insert_stmt = insert(Terms).values(data)
            upsert_stmt = insert_stmt.on_conflict_do_nothing(index_elements=["term"])

            await session.execute(upsert_stmt)
            logging.info(
                f"Sync complete. Processed {len(unique_words)} unique keywords."
            )

    async def merge_datasets(self) -> None:
        """Merge datasets with similar names.

        Identifies datasets with matching or similar names (e.g., "Book--Title"
        and "Title") and merges them by renaming the source dataset to the
        target dataset name.
        """
        import re
        import unicodedata

        def normalize_string(s: str) -> str:
            """Normalize string: remove diacritics, dots, and standardize."""
            if not s:
                return ""

            # Decompose diacritics and remove them (e.g., é -> e)
            s = unicodedata.normalize("NFD", s)
            s = "".join([c for c in s if unicodedata.category(c) != "Mn"])

            # Replace ". " with space, handle trailing dots
            s = re.sub(r"\. +", " ", s)

            s = re.sub(r"\s+", " ", s).strip().lower()

            return s

        def assess_similarity(str1: str, str2: str) -> tuple[str, Optional[str]]:
            """Assess if two strings are the same or possibly the same person.

            Args:
                str1: First string to compare.
                str2: Second string to compare.

            Returns:
                Tuple of (comparison result, message).
                Result is "MATCH", "POTENTIAL", or "DIFFERENT".
            """
            norm1 = normalize_string(str1)
            norm2 = normalize_string(str2)

            # Case 1: Exactly the same (after normalization)
            if norm1.lower() == norm2.lower():
                return "MATCH", norm1

            # Case 2: Detect abbreviation matching (e.g., Immanuel Kant vs I Kant)
            # Split into words and check if it matches "initial + last name" pattern
            parts1 = norm1.split()
            parts2 = norm2.split()

            if len(parts1) == len(parts2) and len(parts1) > 1:
                is_potential = True
                for p1, p2 in zip(parts1, parts2):
                    p1l, p2l = p1.lower(), p2.lower()
                    # If one is the initial of the other, or both are identical, consider potential match
                    if not (
                        p1l == p2l
                        or (len(p1) == 1 and p2l.startswith(p1l))
                        or (len(p2) == 1 and p1l.startswith(p2l))
                    ):
                        is_potential = False
                        break
                if is_potential:
                    return "POTENTIAL", f"'{str1}' and '{str2}' might be the same."

            return "DIFFERENT", None

        async with get_db_session() as session:
            ds = (await session.execute(select(Dataset))).scalars().all()
            calibre = [
                d for d in ds if d.name.startswith("书库--") or "--" not in d.name
            ]
            noncalibre = [d for d in ds if d not in calibre]
        for d1 in calibre:
            n1 = d1.name.split("--")[-1]
            for d2 in noncalibre:
                n2 = d2.name.split("--")[-1]
                cmp, _ = assess_similarity(n2, n1)
                if cmp == "MATCH":
                    await d1.rename_dataset(d2.name)
                    logging.info(f"[{d1.name}] merged to [{d2.name}]")
                    break

    async def update_author_from_url(self, pattern: str) -> None:
        """Update author field from URL using regex pattern.

        Args:
            pattern: Regex pattern to extract author from the source file path.
        """
        async with get_db_session() as session:
            sql = text(
                """
                WITH extracted_data AS (
                    SELECT p.id, substring(fm.path FROM :p) AS extracted_author
                    FROM paragraph p
                    JOIN file_metadata fm ON p.source_id = fm.id
                    WHERE fm.path ~* :p
                )
                UPDATE paragraph
                SET author = extracted_data.extracted_author
                FROM extracted_data
                WHERE paragraph.id = extracted_data.id
                AND (paragraph.author IS NULL OR paragraph.author = '')
            """
            )

            result = await session.execute(sql, {"p": pattern})
            logging.info(f"Successfully updated {cast(Any, result).rowcount} records.")

    async def update_pdate_from_url(self, dataset: str) -> None:
        """Update publication date from URL using regex pattern.

        Args:
            dataset: Dataset name to filter.
        """
        async with get_db_session() as session:
            reg = r"(18|19|20)\d{2}"

            q = (
                select(
                    Paragraph.id,
                    (
                        func.regexp_matches(FileMetadata.path, reg)[1] + "-01-01"
                    ).label("pdate_str"),
                )
                .join(FileMetadata, Paragraph.source_id == FileMetadata.id)
                .join(FileDataset, FileDataset.file_id == FileMetadata.id)
                .join(Dataset, Dataset.id == FileDataset.dataset_id)
                .where(Dataset.name == dataset)
                .where(FileMetadata.path.op("~*")(reg))
                .cte("q")
            )

            # Execute UPDATE ... FROM
            stmt = (
                update(Paragraph)
                .where(Paragraph.id == q.c.id)
                .values(pdate=sqlcast(q.c.pdate_str, TIMESTAMP))
            )

            await session.execute(stmt)
            
    def _build_metadata(self, path: str):
        if '://' in path:
            return FileMetadata(
                path=path,
                extension='',
                size_bytes=0,
            )
        
        # Decide what to store as the primary key "filename"
        dbpath = storage.relative_path(path)
        if not os.path.exists(path):
            return FileMetadata(
                path=dbpath,
                extension='',
                size_bytes=0,
                )

        stat = os.stat(path)
        size_bytes = stat.st_size

        # Extension (lowercase, no dot)
        ext = Path(path).suffix.lower().lstrip(".")
        
        # Build model instance (all other fields are auto-managed by SQLAlchemy)
        metadata_obj = FileMetadata(
            path=dbpath,
            extension=ext,
            size_bytes=size_bytes,
            extdata={},
        )

        return metadata_obj

    def _scan_storage(self, storage_root: str):
        """Scan storage directory and yield file metadata objects.

        Args:
            storage_root: Root directory to scan.

        Yields:
            FileMetadata objects for each file found.
        """
        logging.info(f"🚀 Starting thorough scan of: {storage_root}")

        for root, dirs, files in os.walk(storage_root):
            # You can uncomment to skip hidden directories if you want
            # dirs[:] = [d for d in dirs if not d.startswith(".")]

            for filename in files:
                full_path = os.path.join(root, filename) or ""
                try:
                    yield self._build_metadata(full_path)
                except PermissionError:
                    logging.info(f"   ⏭️  Permission denied: {full_path}")
                except Exception as e:
                    logging.info(f"   ❌ Error processing {full_path}: {e}")
                    continue

    def get_storage_handler(self):
        """Get a file system event handler for storage monitoring.

        Returns:
            StorageHandler: FileSystemEventHandler for monitoring storage changes.
        """
        class StorageHandler(FileSystemEventHandler):
            def __init__(self):
                self.root = config.storage

            def on_any_event(self, event):
                # We ignore directory events for most cases — focus on files
                if event.is_directory:
                    return

                path = Path(str(event.src_path))
                relative_path = str(path.relative_to(self.root))

                session = cast(Any, self).get_session()

                try:
                    if event.event_type in ('created', 'modified', 'moved'):
                        # For moved: we treat it roughly as "new file appeared"
                        # (you can also handle dest_path if you want precise rename tracking)

                        if not path.exists():
                            # Race condition — file already gone
                            return

                        stat = path.stat()
                        size_bytes = stat.st_size

                        ext = path.suffix.lower().lstrip('.') or ''

                        # Get or create the FileMetadata, auto-matching/creating a
                        # dataset based on the containing folder if newly created.
                        asyncio.run(
                            FileMetadata.get(
                                relative_path,
                                extension=ext,
                                size_bytes=size_bytes,
                            )
                        )
                        session.close()
                        logging.info(f"↑ Updated DB: {relative_path}  ({event.event_type})")

                    elif event.event_type == 'deleted':
                        # File or directory was deleted → remove from DB if it's a file we track
                        q = session.query(FileMetadata).filter(FileMetadata.filename == relative_path)
                        count = q.delete()
                        if count > 0:
                            session.commit()
                            logging.info(f"🗑 Deleted from DB: {relative_path}")

                except Exception as e:
                    logging.info(f"Error processing {relative_path}: {e}")
                finally:
                    session.close()

        return StorageHandler()
    
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
                if i % 100 == 0 or i == results_len:
                    pending = len(new_bulk)
                    added += await self.update_text_embeddings_do(bulk=new_bulk)
                    done = await MaintenanceManager.consume_embedding_queue(
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

    @staticmethod
    async def consume_embedding_queue(pids: list, expected: dict) -> int:
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
        sub_size = 256
        for start in range(0, len(bulk), sub_size):
            sub = bulk[start:start + sub_size]
            vectors = None
            for attempt in range(3):
                try:
                    vectors = await TextEmbeddings.batch_encode(
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
        logging.info(f"{inserted} added")
        return inserted

    async def custom_task(self, task_id: str = "", **params) -> Optional[dict]:
        """Execute a custom task.

        Args:
            task_id: Optional task ID to load from database.
            **params: Task parameters if task_id not provided.

        Returns:
            Task execution result.
        """
        from .task import Task

        if task_id:
            dbo = await TaskDBO.get(task_id)
            assert dbo
            dbo.last_run = datetime.datetime.now()
            async with get_db_session() as session:
                await session.merge(dbo)
                await session.commit()
        else:
            dbo = TaskDBO(**params)
        task = Task.from_dbo(dbo)
        return await task.execute_async()

    async def ocr(
        self, input_path: str, output_path: str, lang: str, monochrome: bool = False
    ) -> Optional[str]:
        """Perform OCR on a PDF file.

        Args:
            input_path: Path to input PDF file.
            output_path: Path to output OCR'd PDF file.
            lang: Language code for OCR.
            monochrome: Convert to black and white before OCR.

        Returns:
            Path to the output OCR'd PDF file.
        """
        from .pdfutils import (convert_pdf_to_tiff_group4,
                               merge_images_from_folder)

        temps = []
        input_path = storage.safe_join(input_path)

        if os.path.isdir(input_path):
            fo = tempfile.NamedTemporaryFile("wb", delete=False)
            images = merge_images_from_folder(input_path, fo)
            fo.close()
            temps.append(fo.name)

            if not images:
                os.unlink(fo.name)
                for fn in storage.glob(os.path.join(input_path, "*.pdf")):
                    await self.ocr(
                        input_path=fn,
                        output_path=fn[:-4] + "_ocred.pdf",
                        lang=lang,
                        monochrome=monochrome,
                    )
                return
            input_path = fo.name

        if monochrome:
            fo = tempfile.NamedTemporaryFile("wb", delete=False)
            with open(input_path, "rb") as fi:
                convert_pdf_to_tiff_group4(fi, fo)
            fo.close()
            input_path = fo.name
            temps.append(fo.name)

        output_path = storage.safe_join(output_path)
        if output_path.endswith("/"):
            output_path += os.path.basename(input_path).rsplit(".", 1)[0] + "_ocred"
        if not output_path.endswith(".pdf"):
            output_path += ".pdf"

        try:
            import ocrmypdf

            await asyncio.to_thread(
                ocrmypdf.ocr,
                input_path,
                output_path,
                plugins=["ocrmypdf_paddleocr_remote"],
                language=lang,
                paddle_remote=config.paddle_remote,
                jobs=2,
                force_ocr=True,
            )
        finally:
            for f in temps:
                if os.path.exists(f):
                    os.unlink(f)
        return output_path

    async def test_task(self) -> None:
        """Test task for debugging."""
        logging.info("Test Task Started")
        await asyncio.sleep(10)
        logging.info("Test Task Ended")

    def register_routes(self, target: APIRouter):
        """Get FastAPI router for maintenance endpoints.

        Returns:
            APIRouter with maintenance endpoints.
        """
        router = APIRouter(
            prefix="/maintenance",
            tags=["Maintenance"],
            dependencies=[Depends(get_current_admin)],
        )

        @router.put("/{task_name}")
        async def call_task_in_background(
            task_name: str, background_tasks: BackgroundTasks, params: dict = Body(...)
        ):
            """Schedule a maintenance task to run in the background.

            Args:
                task_name: Name of the maintenance task.
                background_tasks: FastAPI BackgroundTasks.
                params: Task parameters.

            Returns:
                Task scheduling confirmation.
            """
            func = {
                "sync-pdate": self.update_pdate_from_url,
                "sync-terms": self.sync_terms,
                "text-embeddings": self.update_text_embeddings,
                "ocr": self.ocr,
                "custom": self.custom_task,
                "sync-sources": self.sync_sources,
                "cleanup-unused-datasets": self.cleanup_unused_datasets,
            }.get(task_name)
            if func:
                background_tasks.add_task(func, **params)
                return {
                    "message": "Maintenance task started in background",
                    "task_name": task_name,
                    "params": params,
                }
            else:
                return {
                    "error": "Maintenance task not found",
                    "task_name": task_name,
                    "params": params,
                }

        target.include_router(router)


maintenance_manager = MaintenanceManager()
