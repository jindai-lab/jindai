"""Pipeline stages for Bibliography Plugin.

This module provides pipeline stages for saving Paragraph information to BibItem
records and other bibliography-related operations.
"""

import logging

from jindai.models import Paragraph
from jindai.pipeline import PipelineStage
from .models import BibItem


class BibItemSave(PipelineStage):
    """Pipeline stage to save Paragraph information to BibItem.
    
    This stage converts Paragraph objects to BibItem records,
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
        """Initialize BibItemSave stage.
        
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
        """Process a Paragraph and save to BibItem.
        
        Args:
            paragraph: Paragraph to process.
        
        Returns:
            The same Paragraph (unchanged), or None if excluded.
        """
        if not paragraph:
            return None
        
        try:
            # Check for existing BibItem by DOI or URL
            existing = None
            if self.update_existing:
                    
                if existing is None and paragraph.extdata:
                    # Try DOI first
                    doi = paragraph.extdata.get("doi")
                    if doi and isinstance(doi, str):
                        existing = await BibItem.get_by_doi(self.dbsession, doi)
                    
                    # Try catalog & call_number combination
                    library_catalog, call_number = paragraph.extdata.get('library_catalog'), paragraph.extdata.get('call_number')
                    if library_catalog and call_number:
                        existing = await BibItem.get_by_catalog(self.dbsession, library_catalog, call_number)
                
                # Try URL if no DOI match
                if existing is None and paragraph.source_url:
                    existing = await BibItem.get_by_url(self.dbsession, paragraph.source_url)
            
            if existing:
                # Update existing BibItem
                self.log(f"Updating existing BibItem: {existing.title}")
                self._update_bibitem_from_paragraph(existing, paragraph)
                result_item = existing
                await self.dbsession.merge(existing)
            else:
                # Create new BibItem
                self.log(f"Creating new BibItem from Paragraph: {paragraph.outline}")
                try:
                    new_item = BibItem()
                    self._update_bibitem_from_paragraph(new_item, paragraph)
                    self.dbsession.add(new_item)
                    result_item = new_item
                except Exception as e:
                    self.log_exception('BibItem creation failure', e)
                    raise e
            
            # Store BibItem ID in Paragraph extdata for reference
            if paragraph.extdata is None:
                paragraph.extdata = {}
            paragraph.extdata["bibitem_id"] = str(result_item.id)
            
            return paragraph
        
        except Exception as e:
            self.log_exception("Error saving BibItem from Paragraph", e)
            raise e
    
    def _update_bibitem_from_paragraph(
        self, bibitem: BibItem, paragraph: Paragraph
    ) -> None:
        """Update BibItem fields from Paragraph data.
        
        Args:
            bibitem: BibItem to update.
            paragraph: Source Paragraph.
            dataset: Target dataset.
        """
        # Basic mapping
        bibitem.title = paragraph.outline or ""
        bibitem.author = paragraph.author or ""
        bibitem.abstract_note = paragraph.content or ""
        bibitem.date = paragraph.pdate
        bibitem.language = paragraph.lang or "zh"
        
        # Map extdata fields
        if paragraph.extdata:
            extdata = paragraph.extdata
            
            # DOI and URL
            if isinstance(extdata.get("doi"), str):
                bibitem.doi = extdata["doi"]
            if isinstance(extdata.get("url"), str):
                bibitem.url = extdata["url"]
            
            # Publication info
            if "publication" in extdata:
                bibitem.publication = extdata["publication"]
            if "publisher" in extdata:
                bibitem.publisher = extdata["publisher"]
            if "place" in extdata:
                bibitem.place = extdata["place"]
            if "volume" in extdata:
                bibitem.volume = extdata["volume"]
            if "issue" in extdata:
                bibitem.issue = extdata["issue"]
            if "pages" in extdata:
                bibitem.pages = extdata["pages"]
            if "isbn" in extdata:
                bibitem.isbn = extdata["isbn"]
            if "issn" in extdata:
                bibitem.issn = extdata["issn"]
            
            # Series
            if "series" in extdata:
                bibitem.series = extdata["series"]
            if "series_title" in extdata:
                bibitem.series_title = extdata["series_title"]
            
            # Call number and archive
            if "call_number" in extdata:
                bibitem.call_number = extdata["call_number"]
            if "archive" in extdata:
                bibitem.archive = extdata["archive"]
            if "archive_location" in extdata:
                bibitem.archive_location = extdata["archive_location"]
            if "library_catalog" in extdata:
                bibitem.library_catalog = extdata["library_catalog"]
            if "short_title" in extdata:
                bibitem.short_title = extdata["short_title"]
            
            # Notes and item type
            if "notes" in extdata:
                bibitem.notes = extdata["notes"]
            if "item_type" in extdata:
                bibitem.item_type = extdata["item_type"]
            
            # Tags from keywords or tags
            if isinstance(extdata.get("keywords"), list):
                bibitem.tags = extdata["keywords"]
            elif isinstance(extdata.get("tags"), list):
                bibitem.tags = extdata["tags"]
            else:
                if bibitem.tags is None:
                    bibitem.tags = []
            
            # File attachments - merge if update_existing and merge_attachments
            if isinstance(extdata.get("file_attachments"), list):
                new_attachments = extdata["file_attachments"]
                if (
                    self.update_existing and
                    self.merge_attachments and
                    bibitem.file_attachments
                ):
                    # Merge attachments, avoiding duplicates by path
                    existing_paths = {a for a in bibitem.file_attachments}
                    for path in new_attachments:
                        if path not in existing_paths:
                            bibitem.file_attachments.append(path)
                else:
                    bibitem.file_attachments = new_attachments
                    
            # Cover
            if "cover" in extdata:
                bibitem.cover = extdata["cover"]


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
    def get_spec(cls) -> dict[str, str]:
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
            self.log_exception("Error during BibItem deduplication", e)
            raise e
    