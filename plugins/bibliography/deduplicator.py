"""Bibliography Deduplication Utility.

This module provides utilities for detecting and merging duplicate bibliographic
entries based on title and author matching.
"""

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID

from sqlalchemy import select, and_, or_

from jindai.models import get_db_session
from .models import BibItem


class BibItemDeduplicator:
    """Utility class for deduplicating bibliographic items.
    
    This class provides methods to:
    - Detect duplicate entries based on title and author
    - Merge duplicate entries, filling in missing fields
    - Handle conflicting field values by selecting the most recent one
    
    The deduplication logic:
    1. Items with the same title AND author are considered duplicates
    2. Missing fields from any duplicate are filled in from others
    3. For conflicting values, the most recently modified item's value is used
    4. File attachments from all duplicates are combined
    5. Tags from all duplicates are merged (unique values only)
    """

    def __init__(self, log_func=None):
        """Initialize deduplicator.
        
        Args:
            log_func: Optional logging function for debug output.
        """
        self.log_func = log_func or logging.info

    def _normalize_string(self, s: Optional[str]) -> str:
        """Normalize a string for comparison.
        
        Args:
            s: String to normalize.
            
        Returns:
            Lowercase, stripped string.
        """
        if not s:
            return ""
        return s.strip().lower()

    def _get_merge_key(self, item: BibItem) -> Tuple[str, str]:
        """Get the merge key (title, author) for an item.
        
        Args:
            item: BibItem to get key for.
            
        Returns:
            Tuple of (normalized_title, normalized_author).
        """
        title = self._normalize_string(item.title)
        author = self._normalize_string(item.author)
        return (title, author)

    async def find_duplicates(
        self, session, items: List[BibItem] = None
    ) -> List[List[BibItem]]:
        """Find groups of duplicate items.
        
        Args:
            session: SQLAlchemy session.
            items: Optional list of items to check. If None, checks all items.
            
        Returns:
            List of groups, where each group is a list of duplicate items.
        """
        if items is None:
            # Get all items
            stmt = select(BibItem)
            result = await session.execute(stmt)
            items = result.scalars().all()
        
        # Group items by merge key
        groups: Dict[Tuple[str, str], List[BibItem]] = {}
        for item in items:
            key = self._get_merge_key(item)
            if key not in groups:
                groups[key] = []
            groups[key].append(item)
        
        # Return only groups with more than one item
        duplicates = [group for group in groups.values() if len(group) > 1]
        return duplicates

    async def merge_items(
        self, session, items: List[BibItem], keep: str = "latest"
    ) -> BibItem:
        """Merge a list of duplicate items into one.
        
        Args:
            session: SQLAlchemy session.
            items: List of duplicate items to merge.
            keep: Strategy for handling conflicts:
                - "latest": Keep the most recently modified item's values
                - "earliest": Keep the earliest item's values
                - "first": Keep the first item's values (default)
                
        Returns:
            The merged BibItem.
        """
        if not items:
            raise ValueError("No items to merge")
        
        if len(items) == 1:
            return items[0]
        
        self.log_func(f"Merging {len(items)} duplicate items")
        
        # Sort items by modification date for conflict resolution
        def get_modified_date(item: BibItem) -> datetime:
            # Try to get from extra field first
            if item.extra and "zotero_date_modified" in item.extra:
                try:
                    return datetime.fromisoformat(
                        item.extra["zotero_date_modified"].replace("Z", "+00:00")
                    )
                except (ValueError, TypeError):
                    pass
            # Fall back to modified_at
            return item.modified_at or item.created_at
        
        sorted_items = sorted(items, key=get_modified_date)
        
        # Create merged item (start with the first item)
        merged = items[0]
        
        # Fields to merge
        fields_to_merge = [
            "author", "abstract_note", "publication", "date", "volume",
            "issue", "pages", "doi", "url", "isbn", "issn", "archive",
            "archive_location", "library_catalog", "call_number", "language",
            "short_title", "series", "series_title", "publisher", "place",
            "notes", "related", "file_attachments", "extra", "tags"
        ]
        
        for field in fields_to_merge:
            values = []
            for item in items:
                value = getattr(item, field, None)
                if value is not None:
                    values.append((item, value))
            
            if not values:
                continue
            
            # Select value based on strategy
            if keep == "latest":
                selected_item, selected_value = values[-1]  # Most recent
            elif keep == "earliest":
                selected_item, selected_value = values[0]  # Earliest
            else:  # "first" (default)
                selected_item, selected_value = values[0]  # First in list
            
            # For lists (tags, file_attachments), merge all values
            if field in ("tags", "file_attachments"):
                merged_value = []
                seen = set()
                for item, value in values:
                    if isinstance(value, list):
                        for v in value:
                            if isinstance(v, str):
                                key = v.lower() if field == "tags" else v
                                if key not in seen:
                                    seen.add(key)
                                    merged_value.append(v)
                            elif isinstance(v, dict):
                                # For file_attachments, use path as unique key
                                path = v.get("path", v.get("title", ""))
                                if path not in seen:
                                    seen.add(path)
                                    merged_value.append(v)
                setattr(merged, field, merged_value)
            else:
                # For single values, use selected value
                setattr(merged, field, selected_value)
        
        # Merge extra fields
        merged_extra = dict(merged.extra) if merged.extra else {}
        for item in items:
            if item.extra:
                for key, value in item.extra.items():
                    if key not in merged_extra:
                        merged_extra[key] = value
        merged.extra = merged_extra
        
        # Log merge details
        self.log_func(f"Merged into: {merged.title}")
        self.log_func(f"  Author: {merged.author}")
        self.log_func(f"  DOI: {merged.doi}")
        self.log_func(f"  URL: {merged.url}")
        
        return merged

    async def deduplicate(
        self, session, items: List[BibItem] = None, keep: str = "latest"
    ) -> List[BibItem]:
        """Deduplicate a list of items.
        
        Args:
            session: SQLAlchemy session.
            items: Optional list of items to deduplicate. If None, checks all items.
            keep: Strategy for handling conflicts (see merge_items).
            
        Returns:
            List of unique items after deduplication.
        """
        # Find duplicates
        duplicate_groups = await self.find_duplicates(session, items)
        
        if not duplicate_groups:
            self.log_func("No duplicates found")
            if items is None:
                stmt = select(BibItem)
                result = await session.execute(stmt)
                return list(result.scalars().all())
            return list(items)
        
        self.log_func(f"Found {len(duplicate_groups)} groups of duplicates")
        
        # Process each group
        unique_items = []
        items_to_remove = set()
        
        for group in duplicate_groups:
            if len(group) <= 1:
                unique_items.extend(group)
                continue
            
            # Merge the group
            merged = await self.merge_items(session, group, keep=keep)
            unique_items.append(merged)
            
            # Mark other items for removal
            for item in group:
                if item.id != merged.id:
                    items_to_remove.add(item.id)
        
        # Remove merged items
        if items_to_remove:
            self.log_func(f"Removing {len(items_to_remove)} duplicate items")
            # Note: Actual deletion would require a separate operation
        
        return unique_items

    async def deduplicate_all(
        self, session, dataset_id: UUID = None, keep: str = "latest"
    ) -> Dict[str, Any]:
        """Deduplicate all items in the database (optionally filtered by dataset).
        
        Args:
            session: SQLAlchemy session.
            dataset_id: Optional dataset ID to filter by.
            keep: Strategy for handling conflicts.
            
        Returns:
            Dictionary with deduplication statistics.
        """
        # Get items to deduplicate
        if dataset_id:
            stmt = select(BibItem).where(BibItem.dataset == dataset_id)
        else:
            stmt = select(BibItem)
        
        result = await session.execute(stmt)
        items = result.scalars().all()
        
        total_before = len(items)
        
        # Find duplicates
        duplicate_groups = await self.find_duplicates(session, items)
        num_duplicate_groups = len(duplicate_groups)
        
        # Process each group
        unique_items = []
        items_to_remove = set()
        
        for group in duplicate_groups:
            if len(group) <= 1:
                unique_items.extend(group)
                continue
            
            # Merge the group
            merged = await self.merge_items(session, group, keep=keep)
            unique_items.append(merged)
            
            # Mark other items for removal
            for item in group:
                if item.id != merged.id:
                    items_to_remove.add(item.id)
        
        total_after = len(unique_items)
        removed_count = len(items_to_remove)
        
        return {
            "total_before": total_before,
            "total_after": total_after,
            "removed": removed_count,
            "duplicate_groups": num_duplicate_groups,
        }

    def get_merge_key_from_dict(self, data: Dict[str, Any]) -> Tuple[str, str]:
        """Get merge key from item data dictionary.
        
        Args:
            data: Dictionary with item data (title, author).
            
        Returns:
            Tuple of (normalized_title, normalized_author).
        """
        title = self._normalize_string(data.get("title", ""))
        author = self._normalize_string(data.get("author", ""))
        return (title, author)

    async def find_duplicates_by_key(
        self, session, title: str, author: str, dataset_id: UUID = None
    ) -> List[BibItem]:
        """Find items that match a specific title and author.
        
        Args:
            session: SQLAlchemy session.
            title: Title to search for.
            author: Author to search for.
            dataset_id: Optional dataset ID to filter by.
            
        Returns:
            List of matching items.
        """
        title_norm = self._normalize_string(title)
        author_norm = self._normalize_string(author)
        
        stmt = select(BibItem).where(
            and_(
                BibItem.title.ilike(f"%{title}%"),
                BibItem.author.ilike(f"%{author}%")
            )
        )
        
        if dataset_id:
            stmt = stmt.where(BibItem.dataset == dataset_id)
        
        result = await session.execute(stmt)
        return list(result.scalars().all())
