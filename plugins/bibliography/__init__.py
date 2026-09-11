"""Bibliography Plugin for Jindai Application.

This module provides:
- FileMetadata: ORM model for bibliographic items with comprehensive fields
- BibliographyPlugin: Plugin with CRUD API endpoints for FileMetadata management,
  including PDF upload (files stored under Authors/<normalized authors>/)
- CalibreDataSource: DataSourceStage for importing from Calibre library
- ZoteroDataSource: DataSourceStage for importing from Zotero API
- FileMetadataSave: Pipeline stage to save Paragraph information to FileMetadata

The `authors` submodule contains pure helpers for author name normalization
(Chinese/Japanese names kept as-is, other names reordered to
"<Family>, <Given>", punctuation other than "," replaced with "_") and for
file naming / default title extraction from PDF file names.
"""

import os
import re
import shutil
import logging
from typing import Dict, Any, List, Optional, cast

from fastapi import APIRouter, Body, Depends, File, Form, UploadFile
from sqlalchemy import select, func, or_, and_
from uuid import UUID, uuid4

from jindai.pipeline import PipelineStage
from jindai.task import Task
from jindai.plugin import Plugin
from jindai.helpers import get_context, jieba
from jindai.models import get_db_session, get_db, FileMetadata
from jindai.storage import storage
from jindai.worker import worker_manager

from .authors import (
    extract_default_title,
    normalize_authors,
    normalize_authors_directory,
    normalize_first_author_directory,
    sanitize_filename,
)


class BibliographyPlugin(Plugin):
    """Plugin for managing bibliographic items.

    Provides:
    - Full CRUD operations for FileMetadata records via API
    - PDF upload endpoint storing files under Authors/<normalized authors>/
      and creating the corresponding FileMetadata records
    - Data sources for importing from Calibre and Zotero
    - Deduplication utilities for merging duplicate entries
    - Pipeline stages for saving Paragraphs to BibItems
    - Configuration management for Calibre library paths and Zotero API tokens
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
        ``Authors/<normalized authors>/<filename>.pdf`` and a FileMetadata
        record is created with the default title extracted from the file
        name (unless an explicit title is given).

        Author normalization: Chinese/Japanese names are kept as-is, other
        names are reordered to ``<Family>, <Given>`` (e.g. ``John Cage`` ->
        ``Cage, John``); punctuation other than "," is replaced with ``_``.

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
                    candidate_rel = f"Authors/{dir_name}/{candidate}.pdf"
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
                    rel_path = f"Authors/{dir_name}/{stem}_{uuid4().hex[:8]}.pdf"
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
                    authors=normalize_authors(authors) or None,
                    item_type=(item_type or "").strip() or None,
                    file_attachments=[rel_path],
                )
                session.add(new_item)
                await session.flush()
                await session.refresh(new_item)
                result = new_item.as_dict()

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

    async def _relocate_item_file(self, session, item: FileMetadata) -> Optional[str]:
        """Apply the canonical path naming to the item's file on update.

        The canonical path naming is ``Authors/<first author>/<title>.pdf``:
        the directory is derived from the FIRST author (normalized) and the
        file name from the title (fallback: the current file stem).  Only
        real PDF files inside the application storage are relocated; pure
        bibliographic entries (``bib:...``), external URLs and non-PDF files
        are left untouched.

        Args:
            session: Database session (used for collision checks).
            item: The FileMetadata whose fields were just updated.

        Returns:
            The new relative path if the file was moved, else None.
        """
        path = item.path or ""
        if not path or path.startswith("bib:") or "://" in path:
            return None
        if item.extension and item.extension.lower() != "pdf":
            return None
        try:
            old_abs = storage.safe_join(path)
        except ValueError:
            return None
        if not os.path.isfile(old_abs):
            return None

        old_stem, ext = os.path.splitext(os.path.basename(path))
        ext = ext.lower() or ".pdf"
        if (item.title or "").strip():
            stem = sanitize_filename(item.title)
        else:
            stem = old_stem

        dir_name = normalize_first_author_directory(item.authors)
        if f"Authors/{dir_name}/{stem}{ext}" == path:
            return None

        new_rel = None
        new_abs = None
        candidates = [stem] + [f"{stem}_{i}" for i in range(1, 100)]
        for candidate in candidates:
            candidate_rel = f"Authors/{dir_name}/{candidate}{ext}"
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
            new_rel = f"Authors/{dir_name}/{stem}_{uuid4().hex[:8]}{ext}"
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

        async def list_bibitem(offset: int = 0, limit: int = 100):
            """List out BibItems
            Returns:
                Updated configuration dictionary
            """
            async with get_db_session() as session:
                res = await session.execute(select(FileMetadata).offset(offset).limit(limit))
                return {
                    "results": res.scalars().all(),
                    "count": (
                        await session.execute(select(func.count()).select_from(FileMetadata))
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
            return await self._sync_from_calibre()

        @router.post("/sync/zotero")
        async def sync_from_zotero():
            """Synchronize bibliographic data from Zotero.

            Returns:
                Dictionary with synchronization results:
                - success: Whether the operation succeeded
                - count: Number of items imported
                - message: Status message
            """
            return await self._sync_from_zotero()

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
            query: str = "", type: str = "all", offset: int = 0, limit: int = 100
        ):
            """Search bibliographic items by query.

            Args:
                query: Search query string
                type: Search type - 'all' (all fields), 'title', 'author', 'tag'
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

                async with get_db_session() as session:
                    if not query:
                        # If no query, return list results
                        return await list_bibitem(offset, limit)
                    
                    if type == 'tag':
                        words = [query]
                    else:
                        words = jieba.cut_text(query)
                    cond = and_(*[build_cond(type, word) for word in words])
                    
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
            path ``Authors/<normalized authors>/<filename>.pdf``.  The
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
                - directory: Resulting directory name under ``Authors/``
            """
            try:
                authors = data.get("authors") or []
                if isinstance(authors, str):
                    authors = [authors]
                return {
                    "success": True,
                    "results": normalize_authors(authors),
                    "directory": normalize_first_author_directory(authors),
                }
            except Exception as e:
                return {
                    "success": False,
                    "results": [],
                    "directory": "",
                    "message": f"Error normalizing authors: {str(e)}",
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
                await session.refresh(item)
                
                return {
                    "success": True,
                    "result": item.as_dict(),
                    "message": "FileMetadata created successfully",
                }
            except Exception as e:
                # Reset the failed transaction so the session dependency's
                # final commit does not raise PendingRollbackError.
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
                # (Authors/<first author>/<title>.pdf) to managed PDF files.
                moved_to = await self._relocate_item_file(session, item)
                
                await session.flush()
                await session.refresh(item)
                
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
