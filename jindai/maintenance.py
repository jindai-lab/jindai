"""Maintenance and administrative tasks for Jindai application.

This module provides:
- MaintenanceManager: Class for database maintenance operations
- File metadata synchronization
- Dataset merging and cleanup
- Background task management

2026-10-29 曾与导入链一起包化成 ``jindai/maintenance/``；2026-10-30 用户指令
把导入域（``import_lists`` / ``import_engine`` / ``import_cli`` /
``fill_nontitle``）全部归到 ``plugins/importer/``，本模块随之退回**单文件**
（``jindai/maintenance.py``）。这里只留库内维护：``/api/v2/maintenance/*``
路由、worker 任务表、embedding 补算、OCR 任务。
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

from . import embedutils
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
