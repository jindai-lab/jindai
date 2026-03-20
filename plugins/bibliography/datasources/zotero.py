"""Zotero Data Source for Bibliography Plugin.

This module provides a data source implementation for importing bibliographic
data from Zotero using the official Zotero API.
"""

import datetime
import logging
from typing import List, Optional, Dict, Any
from uuid import UUID

import httpx

from jindai.models import Dataset, Paragraph, get_db_session
from jindai.pipeline import DataSourceStage, PipelineStage


class ZoteroDataSource(DataSourceStage):
    """Import bibliographic data from Zotero library via official API.
    
    This data source connects to Zotero's REST API to fetch library items
    including books, journal articles, conference papers, and more. It creates
    Paragraph objects with comprehensive metadata and file attachments.
    
    The API requires:
    - API key (from Zotero account settings)
    - Library ID (user library ID or group library ID)
    - Library type ('user' or 'group')
    
    Attributes:
        dataset_name: The name of the target dataset.
        lang: Language code for imported paragraphs.
        api_key: Zotero API key.
        library_id: Zotero library ID.
        library_type: Type of library ('user' or 'group').
        item_type: Filter by item type (optional).
        tag: Filter by tag (optional).
        include_attachments: Whether to include file attachment metadata.
        include_annotations: Whether to include annotation data.
    """

    # Zotero API base URL
    ZOTERO_API_BASE = "https://api.zotero.org"

    # Mapping of Zotero item types to BibItem item types
    ITEM_TYPE_MAPPING = {
        "book": "book",
        "bookSection": "bookSection",
        "journalArticle": "journalArticle",
        "magazineArticle": "magazineArticle",
        "newspaperArticle": "newspaperArticle",
        "thesis": "thesis",
        "letter": "letter",
        "mail": "email",
        "conferencePaper": "conferencePaper",
        "document": "document",
        "report": "report",
        "bill": "bill",
        "case": "case",
        "patent": "patent",
        "statute": "statute",
        "interview": "interview",
        "film": "film",
        "artwork": "artwork",
        "webpage": "webpage",
        "audioRecording": "audioRecording",
        "videoRecording": "videoRecording",
        "tvBroadcast": "tvBroadcast",
        "radioBroadcast": "radioBroadcast",
        "podcast": "podcast",
        "computerProgram": "computerProgram",
        "conferencePaper": "conferencePaper",
        "thesis": "thesis",
        "encyclopediaArticle": "encyclopediaArticle",
        "dictionaryEntry": "dictionaryEntry",
    }

    def __init__(self, **params) -> None:
        """Initialize Zotero data source.
        
        Args:
            **params: Parameters for data source (see class attributes).
        """
        super().__init__(**params)
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create HTTP client.
        
        Returns:
            Async HTTP client configured with Zotero API headers.
        """
        if self._client is None:
            headers = {
                "Zotero-API-Key": self.api_key,
                "Zotero-API-Version": "3",
            }
            self._client = httpx.AsyncClient(headers=headers, timeout=30.0)
        return self._client

    async def close_client(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def apply_params(
        self,
        dataset_name: str = "",
        lang: str = "auto",
        api_key: str = "",
        library_id: str = "",
        library_type: str = "user",
        item_type: str = "",
        tag: str = "",
        include_attachments: bool = True,
        include_annotations: bool = False,
        limit: int = 100,
        start: int = 0,
    ) -> None:
        """Configure the data source parameters.
        
        Args:
            dataset_name: Name of the target dataset for imported paragraphs.
            lang: Language code for imported paragraphs.
            api_key: Zotero API key (required).
            library_id: Zotero library ID (required).
            library_type: Type of library ('user' or 'group').
            item_type: Filter by item type (optional).
            tag: Filter by tag (optional).
            include_attachments: Whether to include file attachment metadata.
            include_annotations: Whether to include annotation data.
            limit: Number of items per request (max 100).
            start: Starting index for pagination.
        """
        self.dataset_name = dataset_name
        self.lang = lang
        self.api_key = api_key
        self.library_id = library_id
        self.library_type = library_type
        self.item_type = item_type
        self.tag = tag
        self.include_attachments = include_attachments
        self.include_annotations = include_annotations
        self.limit = limit
        self.start = start

    def _parse_zotero_date(self, date_str: Optional[str]) -> Optional[datetime.datetime]:
        """Parse Zotero date string to datetime.
        
        Args:
            date_str: Date string from Zotero API (YYYY-MM-DD or YYYY-MM).
            
        Returns:
            datetime object or None if parsing fails.
        """
        if not date_str:
            return None
        try:
            # Try full date first
            if len(date_str) >= 10:
                return datetime.datetime.strptime(date_str[:10], "%Y-%m-%d")
            # Try year-month
            elif len(date_str) >= 7:
                return datetime.datetime.strptime(date_str[:7], "%Y-%m")
            # Just year
            elif len(date_str) >= 4:
                return datetime.datetime.strptime(date_str[:4], "%Y")
        except ValueError:
            pass
        return None

    def _map_zotero_item_to_paragraph(
        self, item: Dict[str, Any], library_id: str
    ) -> Optional[Paragraph]:
        """Map Zotero API item to Paragraph object.
        
        Args:
            item: Item data from Zotero API.
            library_id: Zotero library ID for archive location.
            
        Returns:
            Paragraph object with mapped metadata, or None if item is invalid.
            File attachments are returned as an array of relative paths.
        """
        # Get item data
        data = item.get("data", {})
        item_type = data.get("itemType", "document")
        
        # Map item type
        bib_item_type = self.ITEM_TYPE_MAPPING.get(item_type, item_type)
        
        # Extract basic fields
        title = data.get("title", "").strip()
        if not title:
            return None  # Skip items without title
            
        # Extract authors
        creators = data.get("creators", [])
        authors = []
        for creator in creators:
            creator_type = creator.get("creatorType", "author")
            first_name = creator.get("firstName", "").strip()
            last_name = creator.get("name", "").strip()
            
            if creator_type in ("author", "editor") and last_name:
                if first_name:
                    authors.append(f"{last_name}, {first_name}")
                else:
                    authors.append(last_name)
            elif last_name:
                authors.append(last_name)
        
        author_str = " & ".join(authors) if authors else ""
        
        # Extract publication details
        publication = data.get("publicationTitle", "").strip()
        journal_abbreviation = data.get("journalAbbreviation", "").strip()
        publication = publication or journal_abbreviation
        
        publisher = data.get("publisher", "").strip()
        place = data.get("place", "").strip()
        
        # Extract identifiers
        doi = data.get("DOI", "").strip()
        url = data.get("url", "").strip()
        isbn = data.get("ISBN", "").strip()
        issn = data.get("ISSN", "").strip()
        
        # Extract dates
        date = self._parse_zotero_date(data.get("date", ""))
        
        # Extract volume/issue/pages
        volume = data.get("volume", "").strip()
        issue = data.get("issue", "").strip()
        pages = data.get("pages", "").strip()
        
        # Extract abstract
        abstract = data.get("abstractNote", "").strip()
        
        # Extract tags
        tags = []
        for tag_item in data.get("tags", []):
            tag_str = tag_item.get("tag", "").strip()
            if tag_str:
                tags.append(tag_str)
        
        # Extract series
        series = data.get("series", "").strip()
        series_title = data.get("seriesTitle", "").strip()
        series = series or series_title
        
        # Extract language
        language = data.get("language", "").strip() or "en"
        
        # Extract notes
        notes = data.get("notes", "").strip()
        if isinstance(notes, list):
            notes = "\n".join(notes)
        
        # Extract extra fields
        extra = data.get("extra", "").strip()
        
        # Build file attachments as array of relative paths
        file_attachments = []
        if self.include_attachments:
            for attachment in data.get("attachments", []):
                attachment_path = attachment.get("path", "")
                
                if attachment_path:
                    file_attachments.append(attachment_path)
        
        # Build extdata with comprehensive metadata
        extdata = {
            "item_type": bib_item_type,
            "zotero_key": data.get("key", ""),
            "zotero_version": item.get("version", 0),
            "zotero_date_added": data.get("dateAdded", ""),
            "zotero_date_modified": data.get("dateModified", ""),
            "archive": "Zotero",
            "archive_location": f"{self.library_type}_{library_id}",
            "library_catalog": data.get("libraryCatalog", ""),
            "short_title": data.get("shortTitle", ""),
            "url_access": data.get("accessDate", ""),
            "series": series,
            "series_title": series_title,
            "num_items": data.get("numItems", ""),
            "num_pages": data.get("numPages", ""),
            "language": language,
            "medium": data.get("medium", ""),
            "edition": data.get("edition", ""),
            "section": data.get("section", ""),
            "artwork_size": data.get("artworkSize", ""),
            "running_time": data.get("runningTime", ""),
            "video_record_type": data.get("videoRecordingType", ""),
            "audio_file_type": data.get("audioFileType", ""),
            "audio_format": data.get("audioFormat", ""),
            "audio_count": data.get("audioCount", ""),
            "audio_track_number": data.get("audioTrackNumber", ""),
            "place_of_publication": place,
            "date_modified": data.get("dateModified", ""),
            "date_added": data.get("dateAdded", ""),
            "extra": extra,
            "tags": tags,
            "file_attachments": file_attachments,
        }
        
        # Add optional fields if they have values
        if publication:
            extdata["publication"] = publication
        if publisher:
            extdata["publisher"] = publisher
        if volume:
            extdata["volume"] = volume
        if issue:
            extdata["issue"] = issue
        if pages:
            extdata["pages"] = pages
        if doi:
            extdata["doi"] = doi
        if url:
            extdata["url"] = url
        if isbn:
            extdata["isbn"] = isbn
        if issn:
            extdata["issn"] = issn
        if notes:
            extdata["notes"] = notes
        if abstract:
            extdata["abstract_note"] = abstract
        
        # Create Paragraph
        paragraph = Paragraph(
            author=author_str,
            pdate=date,
            outline=title,
            content=abstract or "",
            source_url=url,
            extdata=extdata,
        )
        
        return paragraph

    async def fetch_items(
        self, session, library_id: str, start: int = 0
    ) -> List[Paragraph]:
        """Fetch items from Zotero API.
        
        Args:
            session: SQLAlchemy session (for dataset lookup).
            library_id: Zotero library ID.
            start: Starting index for pagination.
            
        Returns:
            List of Paragraph objects.
        """
        # Build API URL
        if self.library_type == "group":
            url = f"{self.ZOTERO_API_BASE}/groups/{library_id}/items"
        else:
            url = f"{self.ZOTERO_API_BASE}/users/{library_id}/items"
        
        # Build query parameters
        params = {
            "format": "json",
            "limit": self.limit,
            "start": start,
        }
        
        if self.item_type:
            params["itemType"] = self.item_type
        if self.tag:
            params["tag"] = self.tag
        
        # Make API request
        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            items = response.json()
            
            # Parse items
            paragraphs = []
            for item in items:
                paragraph = self._map_zotero_item_to_paragraph(item, library_id)
                if paragraph:
                    paragraphs.append(paragraph)
            
            return paragraphs
            
        except httpx.HTTPStatusError as e:
            self.log_exception(f"Zotero API error (status {e.response.status_code})", e)
            return []
        except httpx.RequestError as e:
            self.log_exception("Zotero API request error", e)
            return []
        except Exception as e:
            self.log_exception("Unexpected error fetching Zotero items", e)
            return []

    async def fetch(self):
        """Fetch bibliographic data from Zotero library.
        
        Yields:
            Paragraph objects containing comprehensive bibliographic metadata
            with file attachments.
        """
        # Get dataset
        ds = await Dataset.get(self.dataset_name)
        
        # Fetch all items with pagination
        start = self.start
        while True:
            paragraphs = await self.fetch_items(None, self.library_id, start)
            
            if not paragraphs:
                break
            
            # Set dataset and yield
            for paragraph in paragraphs:
                paragraph.dataset = ds.id
                yield paragraph
            
            # Check if we got fewer items than limit (last page)
            if len(paragraphs) < self.limit:
                break
            
            start += self.limit

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close_client()
