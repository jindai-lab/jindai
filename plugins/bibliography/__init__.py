"""Bibliography Plugin for Jindai Application.

This module provides:
- BibItem: ORM model for bibliographic items with comprehensive fields
- BibliographyPlugin: Plugin with CRUD API endpoints for BibItem management
- CalibreDataSource: DataSourceStage for importing from Calibre library
- ZoteroDataSource: DataSourceStage for importing from Zotero API
- BibItemSave: Pipeline stage to save Paragraph information to BibItem
"""

import os
from typing import Dict, Any

from fastapi import APIRouter, Depends
from sqlalchemy import select, func, or_, and_
from uuid import UUID

from jindai.pipeline import PipelineStage
from jindai.task import Task
from jindai.plugin import Plugin
from jindai.helpers import get_context, jieba
from jindai.models import get_db_session, get_db
from jindai.storage import storage
from jindai.worker import worker_manager
from jindai.app import router as api_router


# Import all components for registration
from .models import BibItem


class BibliographyPlugin(Plugin):
    """Plugin for managing bibliographic items.

    Provides:
    - Full CRUD operations for BibItem records via API
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

        # Register plugin routes
        self._register_routes()

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
            storage.write_json(config_path, self._config)
        except Exception as e:
            print(f"Error saving bibliography config: {e}")

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
                        "scan_for_moved": False,
                    },
                ),
                ("BibItemSave", {}),
            ],
            log=print,
        )
        await task.execute_async()
        return True

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
                ("BibItemSave", {}),
            ],
        )
        await task.execute_async()
        return True

    def _register_tasks(self) -> None:
        worker_manager.register_task(self._sync_from_calibre, "sync_calibre")
        worker_manager.register_task(self._sync_from_zotero, "sync_zotero")

    def _register_routes(self) -> None:
        """Register plugin-specific API routes."""

        router = APIRouter(prefix="/bibliography", tags=["Bibliography"])

        async def list_bibitem(offset: int = 0, limit: int = 100):
            """List out BibItems
            Returns:
                Updated configuration dictionary
            """
            async with get_db_session() as session:
                res = await session.execute(select(BibItem).offset(offset).limit(limit))
                return {
                    "results": [rec["BibItem"] for rec in res.mappings().all()],
                    "count": (
                        await session.execute(select(func.count()).select_from(BibItem))
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
                    items = BibItem.parse_bibtex(bibtex_text)

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
                item_ids: List of BibItem IDs to export

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
                    stmt = select(BibItem).where(BibItem.id.in_(item_ids))
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
                    cond = BibItem.title.ilike(f"%{word}%")
                elif field == "author":
                    cond = func.array_to_string(BibItem.authors, ' & ').ilike(f"%{word}%")
                elif field == 'tag':
                    cond = BibItem.tags.contains([word])
                else:  # 'all' - search all fields
                    cond = or_(
                        BibItem.title.ilike(f"%{word}%"),
                        func.array_to_string(BibItem.authors, ' & ').ilike(f"%{word}%"),
                        BibItem.abstract_note.ilike(f"%{word}%"),
                        BibItem.publication.ilike(f"%{word}%"),
                        BibItem.doi.ilike(f"%{word}%"),
                        BibItem.url.ilike(f"%{word}%"),
                        BibItem.isbn.ilike(f"%{word}%"),
                        BibItem.notes.ilike(f"%{word}%"),
                        BibItem.publisher.ilike(f"%{word}%"),
                        BibItem.place.ilike(f"%{word}%"),
                        BibItem.series.ilike(f"%{word}%"),
                        BibItem.series_title.ilike(f"%{word}%"),
                        BibItem.volume.ilike(f"%{word}%"),
                        BibItem.issue.ilike(f"%{word}%"),
                        BibItem.pages.ilike(f"%{word}%"),
                        BibItem.language.ilike(f"%{word}%"),
                        BibItem.short_title.ilike(f"%{word}%"),
                        BibItem.archive.ilike(f"%{word}%"),
                        BibItem.archive_location.ilike(f"%{word}%"),
                        BibItem.library_catalog.ilike(f"%{word}%"),
                        BibItem.call_number.ilike(f"%{word}%"),
                        BibItem.tags.contains([word]),
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
                    
                    stmt = select(BibItem).where(cond).order_by(BibItem.date_added.desc()).offset(offset).limit(limit)
                    count_stmt = select(func.count()).select_from(BibItem).where(cond)
                    # Execute search
                    res = await session.execute(stmt)
                    count_res = await session.execute(count_stmt)

                    return {
                        "results": [rec["BibItem"] for rec in res.mappings().all()],
                        "count": count_res.scalar_one(),
                    }
            except Exception as e:
                return {
                    "success": False,
                    "results": [],
                    "count": 0,
                    "message": f"Error searching bibliography: {str(e)}",
                }

        @router.get("/{item_id}")
        async def get_bibitem(item_id: int, session=Depends(get_db)):
            """Get a single BibItem by ID.

            Args:
                item_id: BibItem ID
                session: Database session (injected via Depends)

            Returns:
                Dictionary with BibItem data if found, error otherwise
            """
            try:
                stmt = select(BibItem).where(BibItem.id == item_id)
                result = await session.execute(stmt)
                item = result.scalar_one_or_none()
                
                if not item:
                    return {
                        "success": False,
                        "message": f"BibItem with id {item_id} not found",
                    }
                
                return {
                    "success": True,
                    "result": item.as_dict(),
                }
            except Exception as e:
                return {
                    "success": False,
                    "message": f"Error retrieving BibItem: {str(e)}",
                }

        @router.post("/")
        async def create_bibitem(item_data: dict, session=Depends(get_db)):
            """Create a new BibItem.

            Args:
                item_data: Dictionary with BibItem fields
                session: Database session (injected via Depends)

            Returns:
                Dictionary with created BibItem data
            """
            try:
                item = BibItem(**item_data)
                session.add(item)
                await session.flush()
                await session.refresh(item)
                
                return {
                    "success": True,
                    "result": item.as_dict(),
                    "message": "BibItem created successfully",
                }
            except Exception as e:
                return {
                    "success": False,
                    "message": f"Error creating BibItem: {str(e)}",
                }

        @router.put("/{item_id}")
        async def update_bibitem(item_id: UUID, item_data: dict, session=Depends(get_db)):
            """Update an existing BibItem.

            Args:
                item_id: BibItem ID to update
                item_data: Dictionary with fields to update
                session: Database session (injected via Depends)

            Returns:
                Dictionary with updated BibItem data
            """
            try:
                stmt = select(BibItem).where(BibItem.id == item_id)
                result = await session.execute(stmt)
                item = result.scalar_one_or_none()
                
                if not item:
                    return {
                        "success": False,
                        "message": f"BibItem with id {item_id} not found",
                    }
                
                # Update fields
                for key, value in item_data.items():
                    if hasattr(item, key):
                        setattr(item, key, value)
                
                await session.flush()
                await session.refresh(item)
                
                return {
                    "success": True,
                    "result": item.as_dict(),
                    "message": "BibItem updated successfully",
                }
            except Exception as e:
                return {
                    "success": False,
                    "message": f"Error updating BibItem: {str(e)}",
                }

        @router.delete("/{item_id}")
        async def delete_bibitem(item_id: UUID, session=Depends(get_db)):
            """Delete a BibItem.

            Args:
                item_id: BibItem ID to delete
                session: Database session (injected via Depends)

            Returns:
                Dictionary with deletion result
            """
            try:
                stmt = select(BibItem).where(BibItem.id == item_id)
                result = await session.execute(stmt)
                item = result.scalar_one_or_none()
                
                if not item:
                    return {
                        "success": False,
                        "message": f"BibItem with id {item_id} not found",
                    }
                
                await session.delete(item)
                await session.flush()
                
                return {
                    "success": True,
                    "message": "BibItem deleted successfully",
                }
            except Exception as e:
                return {
                    "success": False,
                    "message": f"Error deleting BibItem: {str(e)}",
                }

        # Register routes with the app
        api_router.include_router(router)

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
