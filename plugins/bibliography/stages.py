"""Pipeline stages for Bibliography Plugin.

This module provides pipeline stages for saving Paragraph information to FileMetadata
records and other bibliography-related operations.
"""

import logging
import uuid

from typing import Any, Dict

from jindai.models import Paragraph, FileMetadata
from jindai.pipeline import PipelineStage


class FileMetadataSave(PipelineStage):
    """Pipeline stage to save Paragraph information to FileMetadata.
    
    This stage converts Paragraph objects to FileMetadata records,
    supporting upsert behavior based on DOI or URL. It also handles
    merging file attachments from multiple sources.
    
    Attributes:
        dataset_name: Target dataset name for BibItems.
        update_existing: Whether to update existing BibItems (by DOI/URL).
        merge_attachments: Whether to merge file attachments from multiple sources.
    """
    
    def __init__(
        self,
        update_existing: bool = True,
        merge_attachments: bool = False
    ) -> None:
        """Initialize FileMetadataSave stage.
        
        Args:
            update_existing: If True, update existing BibItems by DOI/URL.
                If False, always create new records.
            merge_attachments: If True, merge file attachments from multiple sources.
        """
        super().__init__()
        self.update_existing = update_existing
        self.merge_attachments = merge_attachments
        self._log = lambda *x: logging.info(' '.join(map(str, x)))
    
    async def resolve(self, paragraph: Paragraph) -> Paragraph | None:
        """Process a Paragraph and save to FileMetadata.
        
        Args:
            paragraph: Paragraph to process.
        
        Returns:
            The same Paragraph (unchanged), or None if excluded.
        """
        if not paragraph:
            return None
        
        try:
            # Check for existing FileMetadata by DOI or URL
            existing = None
            if self.update_existing:
                    
                if existing is None and paragraph.extdata:
                    # Try DOI first
                    doi = paragraph.extdata.get("doi")
                    if doi and isinstance(doi, str):
                        existing = await FileMetadata.get_by_doi(self.dbsession, doi)
                    
                    # Try catalog & call_number combination
                    library_catalog, call_number = paragraph.extdata.get('library_catalog'), paragraph.extdata.get('call_number')
                    if library_catalog and call_number:
                        existing = await FileMetadata.get_by_catalog(self.dbsession, library_catalog, call_number)
                
                # Try URL if no DOI match
                if existing is None and paragraph.extdata and paragraph.extdata.get("url"):
                    existing = await FileMetadata.get_by_url(self.dbsession, paragraph.extdata["url"])
            
            if existing:
                # Update existing FileMetadata
                self.log(f"Updating existing FileMetadata: {existing.title}")
                self._update_bibitem_from_paragraph(existing, paragraph)
                result_item = existing
                await self.dbsession.merge(existing)
            else:
                # Create new FileMetadata
                self.log(f"Creating new FileMetadata from Paragraph: {paragraph.outline}")
                try:
                    # FileMetadata is merged into FileMetadata: path (unique, NOT NULL)
                    # must be provided. Prefer the source file path; fall back to
                    # a synthesized unique key for pure bibliographic entries.
                    new_item = FileMetadata(
                        path=(
                            paragraph.source_obj.path
                            if paragraph.source_obj is not None
                            else f"bib:{uuid.uuid4()}"
                        )
                    )
                    self._update_bibitem_from_paragraph(new_item, paragraph)
                    self.dbsession.add(new_item)
                    result_item = new_item
                except Exception as e:
                    self.log_exception('FileMetadata creation failure', e)
                    raise e
            
            # Store FileMetadata ID in Paragraph extdata for reference
            if paragraph.extdata is None:
                paragraph.extdata = {}
            paragraph.extdata["bibitem_id"] = str(result_item.id)
            
            return paragraph
        
        except Exception as e:
            self.log_exception("Error saving FileMetadata from Paragraph", e)
            raise e
    
    def _update_bibitem_from_paragraph(
        self, FileMetadata: FileMetadata, paragraph: Paragraph
    ) -> None:
        """Update FileMetadata fields from Paragraph data.
        
        Args:
            FileMetadata: FileMetadata to update.
            paragraph: Source Paragraph.
            dataset: Target dataset.
        """
        # Basic mapping
        FileMetadata.title = paragraph.outline or ""
        FileMetadata.authors = (paragraph.author or "").split(' & ')
        FileMetadata.abstract_note = paragraph.content or ""
        FileMetadata.date = paragraph.pdate
        FileMetadata.language = paragraph.lang or "zh"
        
        # Map extdata fields
        if paragraph.extdata:
            extdata = paragraph.extdata
            
            # DOI and URL
            if isinstance(extdata.get("doi"), str):
                FileMetadata.doi = extdata["doi"]
            if isinstance(extdata.get("url"), str):
                FileMetadata.url = extdata["url"]
            
            # Publication info
            if "publication" in extdata:
                FileMetadata.publication = extdata["publication"]
            if "publisher" in extdata:
                FileMetadata.publisher = extdata["publisher"]
            if "place" in extdata:
                FileMetadata.place = extdata["place"]
            if "volume" in extdata:
                FileMetadata.volume = extdata["volume"]
            if "issue" in extdata:
                FileMetadata.issue = extdata["issue"]
            if "pages" in extdata:
                FileMetadata.pages = extdata["pages"]
            if "isbn" in extdata:
                FileMetadata.isbn = extdata["isbn"]
            if "issn" in extdata:
                FileMetadata.issn = extdata["issn"]
            
            # Series
            if "series" in extdata:
                FileMetadata.series = extdata["series"]
            if "series_title" in extdata:
                FileMetadata.series_title = extdata["series_title"]
            
            # Call number and archive
            if "call_number" in extdata:
                FileMetadata.call_number = extdata["call_number"]
            if "archive" in extdata:
                FileMetadata.archive = extdata["archive"]
            if "archive_location" in extdata:
                FileMetadata.archive_location = extdata["archive_location"]
            if "library_catalog" in extdata:
                FileMetadata.library_catalog = extdata["library_catalog"]
            if "short_title" in extdata:
                FileMetadata.short_title = extdata["short_title"]
            
            # Notes and item type
            if "notes" in extdata:
                FileMetadata.notes = extdata["notes"]
            if "item_type" in extdata:
                FileMetadata.item_type = extdata["item_type"]
            
            # Tags from keywords or tags
            if isinstance(extdata.get("keywords"), list):
                FileMetadata.tags = extdata["keywords"]
            elif isinstance(extdata.get("tags"), list):
                FileMetadata.tags = extdata["tags"]
            else:
                if FileMetadata.tags is None:
                    FileMetadata.tags = []
            
            # File attachments - merge if update_existing and merge_attachments
            if isinstance(extdata.get("file_attachments"), list):
                new_attachments = extdata["file_attachments"]
                if (
                    self.update_existing and
                    self.merge_attachments and
                    FileMetadata.file_attachments
                ):
                    # Merge attachments, avoiding duplicates by path
                    existing_paths = {a for a in FileMetadata.file_attachments}
                    for path in new_attachments:
                        if path not in existing_paths:
                            FileMetadata.file_attachments.append(path)
                else:
                    FileMetadata.file_attachments = new_attachments
                    
            # Cover
            if "cover" in extdata:
                FileMetadata.cover = extdata["cover"]


class BibItemDeduplicate(PipelineStage):
    """Pipeline stage to deduplicate BibItems.
    
    This stage finds and merges duplicate BibItems based on title and author.
    It can be configured to use different strategies for handling conflicts.
    
    Attributes:
        keep: Strategy for handling conflicts:
            - "latest": Keep the most recently modified item's values
            - "earliest": Keep the earliest item's values
            - "first": Keep the first item's values (default)
    """
    
    def __init__(
        self,
        keep: str = "latest"
    ) -> None:
        """Initialize BibItemDeduplicate stage.
        
        Args:
            keep: Strategy for handling conflicts.
        """
        super().__init__(name="BibItemDeduplicate")
        self.keep = keep
        self._log = lambda *x: logging.info(' '.join(map(str, x)))
    
    @classmethod
    def get_spec(cls) -> Dict[str, Any]:
        """Get specification info for the pipeline stage.
        
        Returns:
            Dictionary with name, docstring, and argument info.
        """
        return {
            "name": cls.__name__,
            "doc": (cls.__doc__ or "").strip(),
            "args": PipelineStage._spec(cls),
        }
    
    async def resolve(self, paragraph: Paragraph) -> Paragraph | None:
        """Process a Paragraph and deduplicate BibItems.
        
        This stage doesn't modify the input paragraph but triggers
        deduplication of all BibItems.
        
        Args:
            paragraph: Paragraph to process (not modified).
        
        Returns:
            The same Paragraph (unchanged).
        """
        if not paragraph:
            return None
        
        try:
            # Import deduplicator
            from .deduplicator import BibItemDeduplicator
            deduplicator = BibItemDeduplicator(log_func=self.log)

            # Deduplicate
            stats = await deduplicator.deduplicate_all(
                self.dbsession, keep=self.keep
            )
            
            self.log(f"Deduplication complete: {stats}")
            
            return paragraph
        
        except Exception as e:
            self.log_exception("Error during FileMetadata deduplication", e)
            raise e
    