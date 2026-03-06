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
import os
import tempfile
from typing import Optional
import hashlib
from pathlib import Path
from fastapi import APIRouter, BackgroundTasks, Body, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import TIMESTAMP, cast, exists, func, select, text, update, delete
from sqlalchemy.dialects.postgresql import insert
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .app import get_current_admin
from .config import config
from .storage import storage
from .pdfutils import get_pdf_page_count
from .models import (
    Dataset,
    EmbeddingPendingQueue,
    Paragraph,
    TaskDBO,
    Terms,
    TextEmbeddings,
    FileMetadata,
    get_db_session,
    QueryFilters,
)


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
        and removes datasets with no associated paragraphs.

        Args:
            dataset: Optional dataset name to filter.
            folder: Optional folder path to filter.
        """
        assert dataset or folder, "Must specify at least one condition"

        async with get_db_session() as session:
            query = select(Paragraph)
            if folder:
                query = query.filter(Paragraph.source_url.startswith(folder))
            if dataset:
                query = query.filter(
                    Paragraph.dataset.in_(
                        select(Dataset)
                        .filter(
                            (Dataset.name == dataset)
                            | Dataset.name.startswith(dataset + "--")
                        )
                        .with_only_columns(Dataset.id)
                    )
                )

            query = query.distinct(Paragraph.source_url).with_only_columns(
                Paragraph.source_url
            )
            sources: list[str] = [_ for _, in await session.execute(query)]
            datasets = []

            for source in sources:
                if '://' in source:
                    continue
                if '#' in source:
                    source = source[:source.find('#')]
                joined = storage.safe_join(source)
                if not os.path.exists(joined):
                    logging.info(
                        f"{source} does not exist any more. Mark related paragraphs."
                    )
                    datasets.extend(
                        (
                            await session.execute(
                                select(Paragraph)
                                .filter(Paragraph.source_url == source)
                                .distinct(Paragraph.dataset)
                                .with_only_columns(Paragraph.dataset)
                            )
                        )
                        .scalars()
                        .all()
                    )
                    await session.execute(
                        update(Paragraph)
                        .filter(Paragraph.source_url == source)
                        .values(
                            extdata=Paragraph.extdata.op("||")(
                                func.jsonb_build_object("offline", True)
                            )
                        )
                    )

            stmt = delete(Dataset).filter(
                ~exists().where(Paragraph.dataset == Dataset.id),
                Dataset.id.in_(list(set(datasets))),
            )
            await session.execute(stmt)

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

            # Perform a Bulk Upsert into the Terms table
            # We use on_conflict_do_nothing to ignore words already in the Terms table
            data = [{"term": w} for w in unique_words]

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
            pattern: Regex pattern to extract author from source_url.
        """
        async with get_db_session() as session:
            sql = text(
                """
                WITH extracted_data AS (
                    SELECT id, substring(source_url FROM :p) AS extracted_author
                    FROM paragraph
                    WHERE source_url ~* :p
                )
                UPDATE paragraph
                SET author = extracted_data.extracted_author
                FROM extracted_data
                WHERE paragraph.id = extracted_data.id
                AND (paragraph.author IS NULL OR paragraph.author = '')
            """
            )

            result = await session.execute(sql, {"p": pattern})
            logging.info(f"Successfully updated {result.rowcount} records.")

    async def update_pdate_from_url(self, dataset: str) -> None:
        """Update publication date from URL using regex pattern.

        Args:
            dataset: Dataset name to filter.
        """
        async with get_db_session() as session:
            dataset_id_subquery = (
                select(Dataset.id).where(Dataset.name == dataset).scalar_subquery()
            )

            reg = r"(18|19|20)\d{2}"

            q = (
                select(
                    Paragraph.id,
                    (
                        func.regexp_matches(Paragraph.source_url, reg)[1] + "-01-01"
                    ).label("pdate_str"),
                )
                .where(Paragraph.dataset == dataset_id_subquery)
                .where(Paragraph.source_url.op("~*")(reg))
                .cte("q")
            )

            # Execute UPDATE ... FROM
            stmt = (
                update(Paragraph)
                .where(Paragraph.id == q.c.id)
                .values(pdate=cast(q.c.pdate_str, TIMESTAMP))
            )

            await session.execute(stmt)

    def _compute_sha1(self, file_path: str) -> str:
        """Compute SHA-1 hash of a file in chunks for memory efficiency.

        Args:
            file_path: Path to the file.

        Returns:
            Hexadecimal SHA-1 hash string.
        """
        sha1 = hashlib.sha1()
        with open(file_path, "rb") as f:
            while chunk := f.read(16 << 10):
                sha1.update(chunk)
        return sha1.hexdigest()

    def _scan_storage(self, storage_root: str):
        """Scan storage directory and yield file metadata objects.

        Args:
            storage_root: Root directory to scan.

        Yields:
            FileMetadata objects for each file found.
        """
        print(f"🚀 Starting thorough scan of: {storage_root}")

        for root, dirs, files in os.walk(storage_root):
            # You can uncomment to skip hidden directories if you want
            # dirs[:] = [d for d in dirs if not d.startswith(".")]

            for filename in files:
                full_path = os.path.join(root, filename)

                # Decide what to store as the primary key "filename"
                db_filename = storage.relative_path(full_path)

                try:
                    stat = os.stat(full_path)
                    size_bytes = stat.st_size

                    sha1 = self._compute_sha1(full_path)

                    # Extension (lowercase, no dot)
                    ext = Path(filename).suffix.lower().lstrip(".")

                    # PDF page count (only when relevant)
                    page_count = get_pdf_page_count(full_path) if ext == "pdf" else None

                    # Build model instance (all other fields are auto-managed by SQLAlchemy)
                    metadata_obj = FileMetadata(
                        filename=db_filename,
                        sha1=sha1,
                        extension=ext,
                        size_bytes=size_bytes,
                        page_count=page_count,
                        extdata={},
                    )

                    yield metadata_obj

                except PermissionError:
                    print(f"   ⏭️  Permission denied: {full_path}")
                except Exception as e:
                    print(f"   ❌ Error processing {full_path}: {e}")
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

                path = Path(event.src_path)
                relative_path = str(path.relative_to(self.root))

                session = self.get_session()

                try:
                    if event.event_type in ('created', 'modified', 'moved'):
                        # For moved: we treat it roughly as "new file appeared"
                        # (you can also handle dest_path if you want precise rename tracking)

                        if not path.exists():
                            # Race condition — file already gone
                            return

                        stat = path.stat()
                        size_bytes = stat.st_size

                        # Compute SHA-1 (same as your initial scanner)
                        sha1 = self._compute_sha1(path)

                        ext = path.suffix.lower().lstrip('.') or ''

                        # Optional: page_count for PDFs (reuse your logic)
                        page_count = None
                        if ext == 'pdf':
                            page_count = self._get_pdf_page_count(path)  # implement if needed

                        metadata = FileMetadata(
                            filename=relative_path,     # PK = relative path
                            sha1=sha1,
                            extension=ext,
                            size_bytes=size_bytes,
                            page_count=page_count,
                            extdata={},                 # can enrich later
                        )

                        session.merge(metadata)         # UPSERT
                        session.commit()
                        print(f"↑ Updated DB: {relative_path}  ({event.event_type})")

                    elif event.event_type == 'deleted':
                        # File or directory was deleted → remove from DB if it's a file we track
                        q = session.query(FileMetadata).filter(FileMetadata.filename == relative_path)
                        count = q.delete()
                        if count > 0:
                            session.commit()
                            print(f"🗑 Deleted from DB: {relative_path}")

                except Exception as e:
                    print(f"Error processing {relative_path}: {e}")
                finally:
                    session.close()

        return StorageHandler()

    async def populate_file_metadata(self) -> None:
        """Populate file metadata table by scanning storage directory.

        Performs a thorough recursive scan of the storage directory and
        upserts file metadata into the database.
        """
        async def scan_storage_and_populate_db(
            session: AsyncSession,
            commit_every: int = 500,
        ) -> None:
            """Thorough recursive scan of storage directory.

            Args:
                session: Async database session.
                commit_every: Number of files to process before committing.
            """
            storage_root = storage.safe_join('./')
            if not os.path.isdir(storage_root):
                raise ValueError(f"❌ Not a directory: {storage_root}")

            print(f"🚀 Starting thorough scan of: {storage_root}")
            processed = 0

            for metadata_obj in self._scan_storage(storage_root):

                await session.merge(metadata_obj)   # UPSERT

                processed += 1

                if processed % commit_every == 0:
                    await session.commit()
                    print(f"   ✅ Processed {processed:,} files...")

            # Final commit
            await session.commit()

            print("\n🎉 Scan finished!")
            print(f"   Total files processed : {processed:,}")
            print(f"   Database updated via UPSERT on filename (PK)")

        async with get_db_session() as session:
            await scan_storage_and_populate_db(
                session=session,
                commit_every=500,
            )

    async def update_text_embeddings(self, filters: Optional[QueryFilters] = None) -> int:
        """Update text embeddings for paragraphs.

        Args:
            filters: Optional QueryFilters to limit which paragraphs to process.

        Returns:
            Number of embeddings added.
        """
        if isinstance(filters, dict):
            filters = QueryFilters(**filters)

        if filters:
            stmt = await Paragraph.build_query(filters)
        else:
            stmt = select(Paragraph)

        stmt = (
            stmt.join(
                EmbeddingPendingQueue,
                (EmbeddingPendingQueue.id == Paragraph.id)
                & (EmbeddingPendingQueue.dataset == Paragraph.dataset),
            )
            .with_only_columns(Paragraph.id, Paragraph.dataset, Paragraph.content)
            .limit(10000)
        )

        added = 0

        while True:

            async with get_db_session() as session:
                results = await session.execute(stmt)
                results = results.mappings().all()

            results_len = len(results)

            if not results_len:
                return added

            new_bulk = []
            for i, p in enumerate(results, start=1):
                for chunk_id, chunk in enumerate(
                    TextEmbeddings.get_chunks(p["content"], 200, 50), start=1
                ):
                    new_bulk.append(
                        {
                            "id": str(p["id"]),
                            "dataset": str(p["dataset"]),
                            "chunk_id": chunk_id,
                            "content": chunk,
                        }
                    )
                if i % 320 == 0 or i == results_len:
                    added += await self.update_text_embeddings_do(bulk=new_bulk)
                    new_bulk.clear()

    async def update_text_embeddings_do(self, bulk: list) -> int:
        """Process a batch of embeddings.

        Args:
            bulk: List of paragraph data dictionaries.

        Returns:
            Number of embeddings added.
        """
        embs = []
        for p, emb in zip(
            bulk,
            await TextEmbeddings.batch_encode([p["content"] for p in bulk]),
        ):
            embs.append(
                TextEmbeddings(
                    id=p["id"],
                    dataset=p["dataset"],
                    chunk_id=p["chunk_id"],
                    embedding=emb,
                )
            )
        async with get_db_session() as session:
            session.add_all(embs)
            await session.commit()
        logging.info(f"{len(embs)} added")
        return len(embs)

    async def custom_task(self, task_id: str = "", **params) -> dict:
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
    ) -> str:
        """Perform OCR on a PDF file.

        Args:
            input_path: Path to input PDF file.
            output_path: Path to output OCR'd PDF file.
            lang: Language code for OCR.
            monochrome: Convert to black and white before OCR.

        Returns:
            Path to the output OCR'd PDF file.
        """
        from .pdfutils import convert_pdf_to_tiff_group4, merge_images_from_folder

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

    def get_router(self) -> APIRouter:
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

        return router


maintenance_manager = MaintenanceManager()
