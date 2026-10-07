"""Bibliography Plugin for Jindai Application.

This module provides:
- FileMetadata: ORM model for bibliographic items with comprehensive fields
- BibliographyPlugin: Plugin with CRUD API endpoints for FileMetadata management,
  including PDF upload (files stored under the configurable upload base
  directory, default ``OneDrive/<normalized first author>/`` — i.e.
  ``/storage/OneDrive/...`` inside the container)
- CalibreDataSource: DataSourceStage for importing from Calibre library
- ZoteroDataSource: DataSourceStage for importing from Zotero API
- FileMetadataSave: Pipeline stage to save Paragraph information to FileMetadata

The `authors` submodule contains pure helpers separating two concerns:
clean_authors preserves author names as entered for DATABASE STORAGE
(symbols such as "." and "," are kept), while normalize_first_author
etc. build safe PATH SEGMENTS (natural-name order, hostile characters
replaced with "_") for the per-author directory name.  It also provides
file naming / default title extraction from PDF file names (a trailing
"_cropped" suffix is dropped).
"""

import os
import re
import shutil
import logging
import json
from typing import Dict, Any, List, Optional, cast

from fastapi import APIRouter, Body, Depends, File, Form, UploadFile
from sqlalchemy import select, func, or_, and_, text
from uuid import UUID, uuid4

from jindai.pipeline import PipelineStage
from jindai.task import Task
from jindai.plugin import Plugin
from jindai.helpers import get_context, jieba
from jindai.models import get_db_session, get_db, FileMetadata, redis_client
from jindai.storage import storage
from jindai.worker import worker_manager

from .authors import (
    clean_authors,
    extract_default_title,
    normalize_first_author_directory,
    sanitize_filename,
)

# Redis cache for the distinct-author list served by GET /bibliography/authors.
# The list is produced by unnest over file_metadata.authors (O(table size),
# 585k+ rows / 6k+ distinct authors here), so it is cached in Redis and
# invalidated on API writes; the TTL is a safety net for direct DB writers
# such as the batch importer (which bypasses the API hooks).
AUTHORS_CACHE_KEY = "bibliography:authors:list"
AUTHORS_CACHE_TTL = 600  # seconds


class BibliographyPlugin(Plugin):
    """Plugin for managing bibliographic items.

    Provides:
    - Full CRUD operations for FileMetadata records via API
    - PDF upload endpoint storing files under the configurable upload base
      directory (default ``OneDrive/<normalized first author>/``)
      and creating the corresponding FileMetadata records
    - Data sources for importing from Calibre and Zotero
    - Deduplication utilities for merging duplicate entries
    - Pipeline stages for saving Paragraphs to BibItems
    - Configuration management for Calibre library paths, Zotero API tokens
      and the upload base directory (default ``OneDrive/``)
    """

    def __init__(self, pmanager, **config) -> None:
        """Initialize the BibliographyPlugin.

        Args:
            pmanager: The pipeline manager instance.
            **config: Additional configuration options passed to the parent Plugin.
        """
        super().__init__(pmanager, **config)
        ctx = get_context(os.path.join("plugins", "bibliography"), PipelineStage)
        self.register_pipelines(ctx)

        # Initialize configuration
        self._config = {
            # Base directory (relative to the storage root) for uploaded and
            # relocated PDF files; the default "OneDrive" maps to
            # /storage/OneDrive/ inside the container.
            "upload_base_dir": "OneDrive",
            "calibre_library_paths": [],
            "zotero_api_key": "",
            "zotero_library_id": "",
            "zotero_library_type": "user",
        }

        # Load saved configuration from storage
        self._load_config(config)

        # Register task
        self._register_tasks()

    def _load_config(self, config_dict) -> None:
        """Load configuration from dict."""
        for key in self._config:
            if key in self._config and key in config_dict:
                self._config[key] = config_dict[key]

    def _save_config(self) -> None:
        """Save configuration to storage."""
        try:
            config_path = storage.safe_join("plugins", "bibliography", "config.json")
            cast(Any, storage).write_json(config_path, self._config)
        except Exception as e:
            logging.error(f"Error saving bibliography config: {e}")

    def _upload_base_dir(self) -> str:
        """Configured base directory for uploaded / relocated PDF files.

        The value is relative to the application storage root, so the
        default ``OneDrive`` corresponds to ``/storage/OneDrive/`` inside
        the container.  Empty values and unsafe segments (``.`` / ``..``)
        fall back to ``OneDrive``.

        Returns:
            Normalized base directory without leading/trailing slashes,
            e.g. ``OneDrive``.
        """
        raw = str(self._config.get("upload_base_dir") or "").replace("\\", "/")
        segments = [seg for seg in raw.split("/") if seg and seg not in (".", "..")]
        return "/".join(segments) or "OneDrive"

    async def _sync_from_calibre(self) -> Dict[str, Any]:
        """Internal method to synchronize bibliographic data from Calibre libraries.

        Returns:
            Dictionary with synchronization results:
            - success: Whether the operation succeeded
            - count: Number of items imported
            - message: Status message
        """

        paths = self._config.get("calibre_library_paths", [])
        if not paths:
            return {
                "success": False,
                "count": 0,
                "message": "No Calibre library paths configured",
            }

        # Create pipeline
        task = Task(
            {},
            [
                (
                    "CalibreDataSource",
                    {
                        "content": "\n".join(paths),
                        "scan_for_moved": True,
                    },
                ),
                ("FileMetadataSave", {}),
            ],
            log=logging.info,
        )
        await task.execute_async()
        return {
            "success": True,
            "count": 0,
            "message": "Synchronization completed",
        }

    async def _sync_from_zotero(self) -> Dict[str, Any]:
        """Internal method to synchronize bibliographic data from Zotero.

        Returns:
            Dictionary with synchronization results:
            - success: Whether the operation succeeded
            - count: Number of items imported
            - message: Status message
        """

        api_key = self._config.get("zotero_api_key", "")
        library_id = self._config.get("zotero_library_id", "")
        library_type = self._config.get("zotero_library_type", "user")

        if not api_key or not library_id:
            return {
                "success": False,
                "count": 0,
                "message": "Zotero API key or library ID not configured",
            }

        # Create pipeline
        task = Task(
            {},
            [
                (
                    "ZoteroDataSource",
                    {
                        "api_key": api_key,
                        "library_id": library_id,
                        "library_type": library_type,
                    },
                ),
                ("FileMetadataSave", {}),
            ],
        )
        await task.execute_async()
        return {
            "success": True,
            "count": 0,
            "message": "Synchronization completed",
        }

    def _register_tasks(self) -> None:
        worker_manager.register_task(self._sync_from_calibre, "sync_calibre")
        worker_manager.register_task(self._sync_from_zotero, "sync_zotero")

    async def _upload_pdf(
        self,
        file: UploadFile,
        authors: List[str],
        title: str = "",
        item_type: str = "",
    ) -> Dict[str, Any]:
        """Save an uploaded PDF and create the corresponding FileMetadata.

        The file is stored at the relative path
        ``<upload base dir>/<normalized first author>/<filename>.pdf`` (the
        base directory is configurable, default ``OneDrive/`` which maps to
        ``/storage/OneDrive/`` inside the container) and a FileMetadata
        record is created with the default title extracted from the file
        name (unless an explicit title is given).

        Author handling: the FileMetadata record stores the names AS
        ENTERED (cleaned only for whitespace — ``.`` and ``,`` etc. are
        preserved); only the directory name is normalized for file-system
        safety and natural-name ordering (see normalize_author_name,
        e.g. ``A/B Cage`` -> directory ``A_B Cage``).

        Args:
            file: Uploaded file (must be a PDF).
            authors: Author names as entered by the user.
            title: Optional explicit title overriding the file name.
            item_type: Optional item type for the FileMetadata record.

        Returns:
            Dictionary with upload results:
            - success: Whether the operation succeeded
            - message: Status message
            - path: Relative storage path of the uploaded file
            - title: Title of the created record
            - result: The created FileMetadata as a dictionary
        """
        filename = file.filename or ""
        if not filename.lower().endswith(".pdf"):
            return {"success": False, "message": "Only PDF files are allowed"}

        # Verify the magic header to reject non-PDF payloads
        try:
            head = await file.read(5)
            await file.seek(0)
        except Exception as e:
            return {
                "success": False,
                "message": f"Error reading uploaded file: {str(e)}",
            }
        if head != b"%PDF-":
            return {
                "success": False,
                "message": "Invalid PDF file: missing %PDF- header",
            }

        default_title = extract_default_title(filename)
        final_title = re.sub(r"\s+", " ", title or "").strip() or default_title or "Untitled"

        base_dir = self._upload_base_dir()
        dir_name = normalize_first_author_directory(authors)
        stem = sanitize_filename(default_title or "untitled")

        target_abs = None
        rel_path = None
        try:
            async with get_db_session() as session:
                # Resolve name collisions against the file system and the
                # database (FileMetadata.path is NOT NULL UNIQUE).
                candidates = [stem] + [f"{stem}_{i}" for i in range(1, 100)]
                for candidate in candidates:
                    candidate_rel = f"{base_dir}/{dir_name}/{candidate}.pdf"
                    candidate_abs = storage.safe_join(candidate_rel)
                    if os.path.exists(candidate_abs):
                        continue
                    exists = (
                        await session.execute(
                            select(FileMetadata)
                            .where(FileMetadata.path == candidate_rel)
                            .limit(1)
                        )
                    ).scalar_one_or_none()
                    if exists is not None:
                        continue
                    rel_path = candidate_rel
                    target_abs = candidate_abs
                    break
                if rel_path is None:
                    rel_path = f"{base_dir}/{dir_name}/{stem}_{uuid4().hex[:8]}.pdf"
                    target_abs = storage.safe_join(rel_path)

                os.makedirs(os.path.dirname(target_abs), exist_ok=True)
                # Stream the upload to disk (Starlette spooled it already)
                with open(target_abs, "wb") as out:
                    shutil.copyfileobj(file.file, out, 1024 * 1024)
                size_bytes = os.path.getsize(target_abs)

                new_item = FileMetadata(
                    path=rel_path,
                    extension="pdf",
                    size_bytes=size_bytes,
                    title=final_title,
                    authors=clean_authors(authors) or None,
                    item_type=(item_type or "").strip() or None,
                    file_attachments=[rel_path],
                )
                session.add(new_item)
                await session.flush()
                await session.refresh(new_item)
                result = new_item.as_dict()

            self._invalidate_authors_cache()
            return {
                "success": True,
                "message": f"PDF saved to {rel_path}",
                "path": rel_path,
                "title": final_title,
                "result": result,
            }
        except Exception as e:
            logging.error(f"Error uploading PDF: {e}")
            return {"success": False, "message": f"Error uploading PDF: {str(e)}"}

    def _invalidate_authors_cache(self) -> None:
        """Drop the cached distinct-author list (best effort).

        Called after every API operation that may add, change or remove
        authors so the next autocomplete request rebuilds the list.  Failures
        (e.g. Redis being unavailable) are logged and swallowed: a stale list
        still self-heals through AUTHORS_CACHE_TTL.
        """
        try:
            redis_client.delete(AUTHORS_CACHE_KEY)
        except Exception as e:
            logging.warning(f"Failed to invalidate authors cache: {e}")

    async def _all_authors(self) -> List[str]:
        """Get all distinct author names across FileMetadata records.

        Served from the Redis cache when available; on a cache miss the list
        is rebuilt with a single ``SELECT DISTINCT unnest(authors)`` query
        (kept out of the request hot path on purpose: unnest over the full
        table is not cheap) and written back with AUTHORS_CACHE_TTL.

        Returns:
            Sorted (case-insensitive) list of distinct author names.
        """
        try:
            cached = redis_client.get(AUTHORS_CACHE_KEY)
            if cached:
                names = json.loads(cached)
                if isinstance(names, list):
                    return names
        except Exception as e:
            logging.warning(f"Failed to read authors cache: {e}")

        async with get_db_session() as session:
            rows = (
                await session.execute(
                    text(
                        "SELECT DISTINCT unnest(authors) FROM file_metadata "
                        "WHERE authors IS NOT NULL"
                    )
                )
            ).scalars().all()
        names = sorted(
            {(a or "").strip() for a in rows if (a or "").strip()},
            key=str.lower,
        )
        try:
            redis_client.setex(
                AUTHORS_CACHE_KEY,
                AUTHORS_CACHE_TTL,
                json.dumps(names, ensure_ascii=False),
            )
        except Exception as e:
            logging.warning(f"Failed to write authors cache: {e}")
        return names

    async def _relocate_item_file(self, session, item: FileMetadata) -> Optional[str]:
        """Apply the canonical path naming to the item's file on update.

        The canonical path naming is
        ``<upload base dir>/<first author>/<title>``: the base directory
        is configurable (default ``OneDrive/``), the per-author directory
        is derived from the FIRST author (normalized to natural-name
        order) and the file name is derived from the title (fallback: the
        current file stem).  The author directory is created when missing.
        Only real files inside the application storage are relocated; pure
        bibliographic entries (``bib:...``) and external URLs are left
        untouched.

        Args:
            session: Database session (used for collision checks).
            item: The FileMetadata whose fields were just updated.

        Returns:
            The new relative path if the file was moved, else None.
        """
        path = item.path or ""
        if not path or path.startswith("bib:") or "://" in path:
            return None
        try:
            old_abs = storage.safe_join(path)
        except ValueError:
            return None
        if not os.path.isfile(old_abs):
            return None

        old_stem, ext = os.path.splitext(os.path.basename(path))
        ext = ext.lower()
        if (item.title or "").strip():
            stem = sanitize_filename(item.title)
        else:
            stem = old_stem

        base_dir = self._upload_base_dir()
        dir_name = normalize_first_author_directory(item.authors)
        if f"{base_dir}/{dir_name}/{stem}{ext}" == path:
            return None

        new_rel = None
        new_abs = None
        candidates = [stem] + [f"{stem}_{i}" for i in range(1, 100)]
        for candidate in candidates:
            candidate_rel = f"{base_dir}/{dir_name}/{candidate}{ext}"
            if candidate_rel == path:
                # Same location: nothing to move.
                return None
            candidate_abs = storage.safe_join(candidate_rel)
            if os.path.exists(candidate_abs):
                continue
            exists = (
                await session.execute(
                    select(FileMetadata)
                    .where(
                        FileMetadata.path == candidate_rel,
                        FileMetadata.id != item.id,
                    )
                    .limit(1)
                )
            ).scalar_one_or_none()
            if exists is not None:
                continue
            new_rel = candidate_rel
            new_abs = candidate_abs
            break
        if new_rel is None:
            new_rel = f"{base_dir}/{dir_name}/{stem}_{uuid4().hex[:8]}{ext}"
            new_abs = storage.safe_join(new_rel)

        os.makedirs(os.path.dirname(new_abs), exist_ok=True)
        shutil.move(old_abs, new_abs)

        old_path = path
        item.path = new_rel
        if item.file_attachments and old_path in item.file_attachments:
            item.file_attachments = [
                new_rel if att == old_path else att
                for att in item.file_attachments
            ]
        logging.info(f"Relocated bibliography file: {old_path} -> {new_rel}")
        return new_rel

    def register_routes(self, target: APIRouter) -> None:
        """Register plugin-specific API routes."""

        router = APIRouter(prefix="/bibliography", tags=["Bibliography"])

        def item_type_cond(item_type: str):
            """Build a FileMetadata.item_type filter condition.

            Args:
                item_type: Item type to match. An empty string or 'all'
                    disables filtering.

            Returns:
                SQLAlchemy condition, or None when no filtering is needed.
            """
            if item_type and item_type != "all":
                return FileMetadata.item_type == item_type
            return None

        async def list_bibitem(offset: int = 0, limit: int = 100, item_type: str = ""):
            """List out BibItems
            Returns:
                Updated configuration dictionary
            """
            cond = item_type_cond(item_type)
            conds = [cond] if cond is not None else []
            async with get_db_session() as session:
                res = await session.execute(
                    select(FileMetadata).where(*conds).order_by(FileMetadata.created_at.desc()).offset(offset).limit(limit)
                )
                return {
                    "results": res.scalars().all(),
                    "count": (
                        await session.execute(
                            select(func.count()).select_from(FileMetadata).where(*conds)
                        )
                    ).scalar_one(),
                }

        @router.post("/sync/calibre")
        async def sync_from_calibre():
            """Synchronize bibliographic data from Calibre libraries.

            Returns:
                Dictionary with synchronization results:
                - success: Whether the operation succeeded
                - count: Number of items imported
                - message: Status message
            """
            result = await self._sync_from_calibre()
            if result.get("success"):
                self._invalidate_authors_cache()
            return result

        @router.post("/sync/zotero")
        async def sync_from_zotero():
            """Synchronize bibliographic data from Zotero.

            Returns:
                Dictionary with synchronization results:
                - success: Whether the operation succeeded
                - count: Number of items imported
                - message: Status message
            """
            result = await self._sync_from_zotero()
            if result.get("success"):
                self._invalidate_authors_cache()
            return result

        @router.post("/import/bibtex")
        async def import_bibtex(
            bibtex_text: str,
        ):
            """Import bibliographic items from bibtex text.

            Args:
                bibtex_text: Raw bibtex text containing one or more entries

            Returns:
                Dictionary with import results:
                - success: Whether the operation succeeded
                - count: Number of items imported
                - message: Status message
                - items: List of imported item IDs
            """
            try:
                # Get or create dataset
                async with get_db_session() as session:

                    # Parse bibtex text
                    items = FileMetadata.parse_bibtex(bibtex_text)

                    # Save items to database
                    imported_ids = []
                    for item in items:
                        session.add(item)
                        await session.flush()
                        imported_ids.append(str(item.id))

                    self._invalidate_authors_cache()
                    return {
                        "success": True,
                        "count": len(items),
                        "message": f"Imported {len(items)} items from BibTeX",
                        "items": imported_ids,
                    }
            except Exception as e:
                return {
                    "success": False,
                    "count": 0,
                    "message": f"Error importing BibTeX: {str(e)}",
                }

        @router.post("/export/bibtex")
        async def export_bibtex(item_ids: list[str]):
            """Export bibliographic items to bibtex text.

            Args:
                item_ids: List of FileMetadata IDs to export

            Returns:
                Dictionary with export results:
                - success: Whether the operation succeeded
                - bibtex: BibTeX formatted string
                - count: Number of items exported
                - message: Status message
            """
            try:
                
                async with get_db_session() as session:
                    # Fetch items by IDs
                    stmt = select(FileMetadata).where(FileMetadata.id.in_(item_ids))
                    result = await session.execute(stmt)
                    items = result.scalars().all()

                    if not items:
                        return {
                            "success": False,
                            "bibtex": "",
                            "count": 0,
                            "message": "No items found",
                        }

                    # Export each item to bibtex
                    bibtex_entries = []
                    for item in items:
                        bibtex_entries.append(item.export_bibtex())

                    bibtex_text = "\n\n".join(bibtex_entries)

                    return {
                        "success": True,
                        "bibtex": bibtex_text,
                        "count": len(items),
                        "message": f"Exported {len(items)} items to BibTeX",
                    }
            except Exception as e:
                return {
                    "success": False,
                    "bibtex": "",
                    "count": 0,
                    "message": f"Error exporting BibTeX: {str(e)}",
                }

        @router.get("/search")
        async def search_bibitems(
            query: str = "",
            type: str = "all",
            item_type: str = "book",
            offset: int = 0,
            limit: int = 100,
        ):
            """Search bibliographic items by query.

            Args:
                query: Search query string
                type: Search type - 'all' (all fields), 'title', 'author', 'tag'
                item_type: FileMetadata.item_type filter (default 'book');
                    '' or 'all' disables the filter
                offset: Number of results to skip
                limit: Maximum number of results to return

            Returns:
                Dictionary with search results:
                - results: List of matching BibItems
                - count: Total number of matching results
            """
            
            def build_cond(field, word):
                # Build search query based on type
                if field == "title":
                    cond = FileMetadata.title.ilike(f"%{word}%")
                elif field == "author":
                    cond = func.array_to_string(FileMetadata.authors, ' & ').ilike(f"%{word}%")
                elif field == 'tag':
                    cond = FileMetadata.tags.contains([word])
                else:  # 'all' - search all fields
                    cond = or_(
                        FileMetadata.title.ilike(f"%{word}%"),
                        func.array_to_string(FileMetadata.authors, ' & ').ilike(f"%{word}%"),
                        FileMetadata.abstract_note.ilike(f"%{word}%"),
                        FileMetadata.publication.ilike(f"%{word}%"),
                        FileMetadata.doi.ilike(f"%{word}%"),
                        FileMetadata.url.ilike(f"%{word}%"),
                        FileMetadata.isbn.ilike(f"%{word}%"),
                        FileMetadata.notes.ilike(f"%{word}%"),
                        FileMetadata.publisher.ilike(f"%{word}%"),
                        FileMetadata.place.ilike(f"%{word}%"),
                        FileMetadata.series.ilike(f"%{word}%"),
                        FileMetadata.series_title.ilike(f"%{word}%"),
                        FileMetadata.volume.ilike(f"%{word}%"),
                        FileMetadata.issue.ilike(f"%{word}%"),
                        FileMetadata.pages.ilike(f"%{word}%"),
                        FileMetadata.language.ilike(f"%{word}%"),
                        FileMetadata.short_title.ilike(f"%{word}%"),
                        FileMetadata.archive.ilike(f"%{word}%"),
                        FileMetadata.archive_location.ilike(f"%{word}%"),
                        FileMetadata.library_catalog.ilike(f"%{word}%"),
                        FileMetadata.call_number.ilike(f"%{word}%"),
                        FileMetadata.tags.contains([word]),
                    )
                return cond
            
            try:
                type_cond = item_type_cond(item_type)

                async with get_db_session() as session:
                    if not query:
                        # If no query, return list results
                        return await list_bibitem(offset, limit, item_type)

                    if type == 'tag':
                        words = [query]
                    else:
                        words = jieba.cut_text(query)
                    cond = and_(*[build_cond(type, word) for word in words])
                    if type_cond is not None:
                        cond = and_(cond, type_cond)

                    stmt = select(FileMetadata).where(cond).order_by(FileMetadata.date_added.desc()).offset(offset).limit(limit)
                    count_stmt = select(func.count()).select_from(FileMetadata).where(cond)
                    # Execute search
                    res = await session.execute(stmt)
                    count_res = await session.execute(count_stmt)

                    return {
                        # FileMetadata is an alias of FileMetadata: use scalars()
                        # instead of mappings() so the result does not depend
                        # on the class name.
                        "results": res.scalars().all(),
                        "count": count_res.scalar_one(),
                    }
            except Exception as e:
                return {
                    "success": False,
                    "results": [],
                    "count": 0,
                    "message": f"Error searching bibliography: {str(e)}",
                }

        @router.post("/upload/pdf")
        async def upload_pdf(
            file: UploadFile = File(...),
            authors: Optional[List[str]] = Form(None),
            title: str = Form(""),
            item_type: str = Form(""),
        ):
            """Upload a PDF file to the bibliography.

            Only PDF files are allowed.  The PDF is stored at the relative
            path ``<upload base dir>/<normalized first author>/<filename>.pdf``
            (base directory configurable, default ``OneDrive/``).  The
            default title is extracted from the PDF file name unless an
            explicit title is provided, and a FileMetadata record is
            created for the upload.

            Args:
                file: Uploaded PDF file.
                authors: Author names (repeated form field).
                title: Optional explicit title overriding the file name.
                item_type: Optional item type for the created record.

            Returns:
                Dictionary with upload results
                (see BibliographyPlugin._upload_pdf).
            """
            return await self._upload_pdf(file, list(authors or []), title, item_type)

        @router.post("/authors/normalize")
        async def normalize_authors_endpoint(data: dict = Body(...)):
            """Normalize author names (preview of the storage directory).

            Args:
                data: Dictionary with an ``authors`` list of names.

            Returns:
                Dictionary with:
                - success: Whether the operation succeeded
                - results: List of normalized author names
                - directory: Normalized FIRST author (the per-author
                  directory below the upload base directory)
                - root: Configured upload base directory (default
                  ``OneDrive``, i.e. ``/storage/OneDrive/`` in the container)
            """
            try:
                authors = data.get("authors") or []
                if isinstance(authors, str):
                    authors = [authors]
                return {
                    "success": True,
                    # results: names as they will be STORED (symbols kept);
                    # directory: normalized PATH segment preview.
                    "results": clean_authors(authors),
                    "directory": normalize_first_author_directory(authors),
                    "root": self._upload_base_dir(),
                }
            except Exception as e:
                return {
                    "success": False,
                    "results": [],
                    "directory": "",
                    "root": self._upload_base_dir(),
                    "message": f"Error normalizing authors: {str(e)}",
                }

        @router.get("/authors")
        async def list_authors(q: str = "", limit: int = 100):
            """List distinct authors for autocomplete.

            Backs the author suggest in the create/edit and PDF-upload forms.
            The full distinct list comes from the Redis-cached
            ``SELECT DISTINCT unnest(authors)`` (see
            BibliographyPlugin._all_authors); filtering happens in-process so
            each keystroke never hits the database.

            Must be registered before ``GET /{item_id}`` so the literal path
            is not swallowed by the parameterized route.

            Args:
                q: Optional case-insensitive substring filter.
                limit: Maximum number of names to return (1-1000).

            Returns:
                Dictionary with:
                - success: Whether the operation succeeded
                - results: Matching author names (sorted, capped at limit)
                - count: Total number of matches before the limit
            """
            names = await self._all_authors()
            needle = (q or "").strip().lower()
            matched = (
                [n for n in names if needle in n.lower()] if needle else names
            )
            try:
                limit = int(limit)
            except (TypeError, ValueError):
                limit = 100
            limit = max(1, min(limit, 1000))
            return {
                "success": True,
                "results": matched[:limit],
                "count": len(matched),
            }

        @router.get("/{item_id}")
        async def get_bibitem(item_id: int, session=Depends(get_db)):
            """Get a single FileMetadata by ID.

            Args:
                item_id: FileMetadata ID
                session: Database session (injected via Depends)

            Returns:
                Dictionary with FileMetadata data if found, error otherwise
            """
            try:
                stmt = select(FileMetadata).where(FileMetadata.id == item_id)
                result = await session.execute(stmt)
                item = result.scalar_one_or_none()
                
                if not item:
                    return {
                        "success": False,
                        "message": f"FileMetadata with id {item_id} not found",
                    }
                
                return {
                    "success": True,
                    "result": item.as_dict(),
                }
            except Exception as e:
                return {
                    "success": False,
                    "message": f"Error retrieving FileMetadata: {str(e)}",
                }

        @router.post("/")
        async def create_bibitem(item_data: dict, session=Depends(get_db)):
            """Create a new FileMetadata.

            Args:
                item_data: Dictionary with FileMetadata fields
                session: Database session (injected via Depends)

            Returns:
                Dictionary with created FileMetadata data
            """
            try:
                item = FileMetadata(**item_data)
                # FileMetadata is merged into FileMetadata: `path` is NOT NULL UNIQUE.
                # Synthesize a unique path for pure bibliographic entries when
                # the caller did not provide one.
                if not item.path:
                    item.path = item.doi or item.url or item.title or f"bib:{item.id}"
                # Provide safe defaults for the NOT NULL file columns when the
                # caller did not supply them (pure bibliographic entries).
                if not item.extension:
                    ext = os.path.splitext(item.path)[1].lstrip(".").lower()
                    item.extension = ext or "bib"
                if item.size_bytes is None:
                    item.size_bytes = 0
                session.add(item)
                await session.flush()
                # Move a caller-supplied file into the canonical
                # <base dir>/<author>/<title> location (creating the author
                # directory when needed), mirroring what updates do.
                await self._relocate_item_file(session, item)
                # Persist the relocated path BEFORE refresh(): refresh()
                # reloads from the DB and would otherwise discard the
                # unflushed path change made above.
                await session.flush()
                await session.refresh(item)

                self._invalidate_authors_cache()
                return {
                    "success": True,
                    "result": item.as_dict(),
                    "message": "FileMetadata created successfully",
                }
            except Exception as e:
                # Reset the failed transaction so the session dependency's
                # final commit does not raise PendingRollbackError.
                logging.exception("Error creating FileMetadata")
                await session.rollback()
                return {
                    "success": False,
                    "message": f"Error creating FileMetadata: {str(e)}",
                }

        @router.put("/{item_id}")
        async def update_bibitem(item_id: UUID, item_data: dict, session=Depends(get_db)):
            """Update an existing FileMetadata.

            Args:
                item_id: FileMetadata ID to update
                item_data: Dictionary with fields to update
                session: Database session (injected via Depends)

            Returns:
                Dictionary with updated FileMetadata data
            """
            try:
                stmt = select(FileMetadata).where(FileMetadata.id == item_id)
                result = await session.execute(stmt)
                item = result.scalar_one_or_none()
                
                if not item:
                    return {
                        "success": False,
                        "message": f"FileMetadata with id {item_id} not found",
                    }
                
                # Update fields
                for key, value in item_data.items():
                    if hasattr(item, key):
                        setattr(item, key, value)
                
                # Apply the canonical path naming
                # (<upload base dir>/<first author>/<title>.pdf) to managed
                # PDF files.
                moved_to = await self._relocate_item_file(session, item)
                
                await session.flush()
                await session.refresh(item)

                if "authors" in item_data:
                    self._invalidate_authors_cache()
                return {
                    "success": True,
                    "result": item.as_dict(),
                    "message": (
                        f"FileMetadata updated successfully; file moved to {moved_to}"
                        if moved_to
                        else "FileMetadata updated successfully"
                    ),
                    "moved": bool(moved_to),
                    "path": moved_to or item.path,
                }
            except Exception as e:
                # Reset the failed transaction so the session dependency's
                # final commit does not raise PendingRollbackError.
                await session.rollback()
                return {
                    "success": False,
                    "message": f"Error updating FileMetadata: {str(e)}",
                }

        @router.delete("/{item_id}")
        async def delete_bibitem(item_id: UUID, session=Depends(get_db)):
            """Delete a FileMetadata.

            Args:
                item_id: FileMetadata ID to delete
                session: Database session (injected via Depends)

            Returns:
                Dictionary with deletion result
            """
            try:
                stmt = select(FileMetadata).where(FileMetadata.id == item_id)
                result = await session.execute(stmt)
                item = result.scalar_one_or_none()
                
                if not item:
                    return {
                        "success": False,
                        "message": f"FileMetadata with id {item_id} not found",
                    }
                
                await session.delete(item)
                await session.flush()
                
                self._invalidate_authors_cache()
                return {
                    "success": True,
                    "message": "FileMetadata deleted successfully",
                }
            except Exception as e:
                # Reset the failed transaction so the session dependency's
                # final commit does not raise PendingRollbackError.
                await session.rollback()
                return {
                    "success": False,
                    "message": f"Error deleting FileMetadata: {str(e)}",
                }

        # Register routes with the app
        target.include_router(router)

    @property
    def upload_base_dir(self) -> str:
        """Get configured base directory for uploaded PDF files."""
        return self._upload_base_dir()

    @upload_base_dir.setter
    def upload_base_dir(self, path: str) -> None:
        """Set base directory for uploaded PDF files.

        Args:
            path: Storage-root-relative directory, e.g. ``OneDrive``
                (maps to ``/storage/OneDrive/`` inside the container).
        """
        self._config["upload_base_dir"] = path
        self._save_config()

    @property
    def calibre_library_paths(self) -> list:
        """Get configured Calibre library paths."""
        return self._config.get("calibre_library_paths", [])

    @calibre_library_paths.setter
    def calibre_library_paths(self, paths: list) -> None:
        """Set Calibre library paths.

        Args:
            paths: List of Calibre library paths
        """
        self._config["calibre_library_paths"] = paths
        self._save_config()

    @property
    def zotero_api_key(self) -> str:
        """Get configured Zotero API key."""
        return self._config.get("zotero_api_key", "")

    @zotero_api_key.setter
    def zotero_api_key(self, key: str) -> None:
        """Set Zotero API key.

        Args:
            key: Zotero API key
        """
        self._config["zotero_api_key"] = key
        self._save_config()

    @property
    def zotero_library_id(self) -> str:
        """Get configured Zotero library ID."""
        return self._config.get("zotero_library_id", "")

    @zotero_library_id.setter
    def zotero_library_id(self, library_id: str) -> None:
        """Set Zotero library ID.

        Args:
            library_id: Zotero library ID
        """
        self._config["zotero_library_id"] = library_id
        self._save_config()

    @property
    def zotero_library_type(self) -> str:
        """Get configured Zotero library type."""
        return self._config.get("zotero_library_type", "user")

    @zotero_library_type.setter
    def zotero_library_type(self, library_type: str) -> None:
        """Set Zotero library type.

        Args:
            library_type: Zotero library type ('user' or 'group')
        """
        self._config["zotero_library_type"] = library_type
        self._save_config()
