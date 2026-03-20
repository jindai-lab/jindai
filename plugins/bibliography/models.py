"""BibItem ORM Model for Bibliographic Items.

This module provides the BibItem SQLAlchemy model for storing publication metadata.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    asc,
    desc,
    or_,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID, ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from jindai.models import Base, Dataset, get_db_session
from uuid import UUID


class BibItem(Base):
    """Bibliographic item model for storing publication metadata.
    
    This model provides comprehensive fields for managing bibliographic
    references including:
    - Basic information: item type, title, author, abstract, publication, date
    - Identifiers: DOI, URL, ISBN, ISSN, archive info, call number
    - Publication details: publisher, place, series, volume/issue/pages
    - Additional: notes, tags, language, short title, extra JSON data
    """
    
    __tablename__ = "bib_items"
    __table_args__ = (
        Index("idx_bibitem_title", "title"),
        Index("idx_bibitem_author", "author"),
        Index("idx_bibitem_doi", "doi"),
        Index("idx_bibitem_url", "url"),
        Index("idx_bibitem_item_type", "item_type"),
        Index("idx_bibitem_tags", "tags", postgresql_using="gin"),
        {
            "comment": "Bibliographic items table",
        },
    )
    
    # ========== Basic Information ==========
    item_type: Mapped[str | None] = mapped_column(
        String(64), comment="Item type (e.g., book, journalArticle, conferencePaper)"
    )
    title: Mapped[str] = mapped_column(Text, comment="Publication title")
    author: Mapped[str | None] = mapped_column(
        String(512), comment="Author(s) / Creator(s)"
    )
    abstract_note: Mapped[str | None] = mapped_column(
        Text, comment="Abstract or summary"
    )
    publication: Mapped[str | None] = mapped_column(
        String(512), comment="Publication name (journal, book title, etc.)"
    )
    date: Mapped[datetime | None] = mapped_column(
        DateTime, comment="Publication date"
    )
    volume: Mapped[str | None] = mapped_column(String(32), comment="Volume number")
    issue: Mapped[str | None] = mapped_column(String(32), comment="Issue number")
    pages: Mapped[str | None] = mapped_column(String(64), comment="Page range")
    
    # ========== Identifiers and Locations ==========
    doi: Mapped[str | None] = mapped_column(String(256), unique=True, comment="Digital Object Identifier")
    url: Mapped[str | None] = mapped_column(String(1024), comment="URL to publication")
    isbn: Mapped[str | None] = mapped_column(String(32), comment="ISBN")
    issn: Mapped[str | None] = mapped_column(String(16), comment="ISSN")
    archive: Mapped[str | None] = mapped_column(
        String(256), comment="Archive name (e.g., Zotero, local library)"
    )
    archive_location: Mapped[str | None] = mapped_column(
        String(512), comment="Location within archive"
    )
    library_catalog: Mapped[str | None] = mapped_column(
        String(256), comment="Library catalog name"
    )
    call_number: Mapped[str | None] = mapped_column(
        String(128), comment="Call number / shelf location"
    )
    
    # ========== Publication Details ==========
    language: Mapped[str | None] = mapped_column(
        String(32), default="zh", comment="Publication language"
    )
    short_title: Mapped[str | None] = mapped_column(
        String(256), comment="Short title abbreviation"
    )
    series: Mapped[str | None] = mapped_column(
        String(256), comment="Series name"
    )
    series_title: Mapped[str | None] = mapped_column(
        String(256), comment="Series title"
    )
    publisher: Mapped[str | None] = mapped_column(
        String(256), comment="Publisher name"
    )
    place: Mapped[str | None] = mapped_column(
        String(256), comment="Publication place"
    )
    
    # ========== Additional Information ==========
    notes: Mapped[str | None] = mapped_column(Text, comment="User notes")
    tags: Mapped[List[str] | None] = mapped_column(
        ARRAY(Text), default=list, comment="Tag list (JSON array)"
    )
    related: Mapped[str | None] = mapped_column(
        Text, comment="Related items/links"
    )
    file_attachments: Mapped[List[str] | None] = mapped_column(
        ARRAY(Text), default=list, comment="File attachments metadata (JSON)"
    )
    extra: Mapped[Dict[str, Any] | None] = mapped_column(
        JSONB, default=dict, comment="Extra fields (JSONB)"
    )
    
    # ========== Relationships ==========
    # Link to dataset for organizational purposes
    dataset: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("dataset.id"),
        nullable=False,
        comment="Associated dataset ID",
    )
    
    dataset_obj: Mapped["Dataset"] = relationship(
        "Dataset", backref="bib_items", lazy="joined"
    )
    
    def as_dict(self) -> dict:
        """Convert model instance to dictionary.
        
        Returns:
            Dictionary with all field values.
        """
        data = super().as_dict()
        # Ensure tags and file_attachments are properly serialized
        if data.get("tags") is None:
            data["tags"] = []
        if data.get("file_attachments") is None:
            data["file_attachments"] = []
        if data.get("extra") is None:
            data["extra"] = {}
        return data
    
    @classmethod
    async def get_by_doi(cls, session, doi: str) -> Optional["BibItem"]:
        """Get a BibItem by DOI.
        
        Args:
            session: SQLAlchemy session.
            doi: Digital Object Identifier.
            
        Returns:
            BibItem if found, None otherwise.
        """
        from sqlalchemy import select
        stmt = select(cls).where(cls.doi == doi)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()
    
    @classmethod
    async def get_by_url(cls, session, url: str) -> Optional["BibItem"]:
        """Get a BibItem by URL.
        
        Args:
            session: SQLAlchemy session.
            url: URL to the publication.
            
        Returns:
            BibItem if found, None otherwise.
        """
        from sqlalchemy import select
        stmt = select(cls).where(cls.url == url)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()
    
    @classmethod
    async def search_by_title_author(
        cls, session, title: str, author: str
    ) -> List["BibItem"]:
        """Search BibItems by title and author.
        
        Args:
            session: SQLAlchemy session.
            title: Title to search for.
            author: Author to search for.
            
        Returns:
            List of matching BibItems.
        """
        from sqlalchemy import select
        stmt = select(cls).where(
            cls.title.ilike(f"%{title}%"),
            cls.author.ilike(f"%{author}%")
        )
        result = await session.execute(stmt)
        return result.scalars().all()
    
    @staticmethod
    def parse_bibtex(bibtex_text: str) -> List["BibItem"]:
        """Parse bibtex text and return list of BibItem objects.
        
        Args:
            bibtex_text: Raw bibtex text (may contain multiple entries).
            
        Returns:
            List of BibItem objects parsed from the bibtex text.
        """
        import re
        from datetime import datetime
        
        items = []
        
        # Remove comments (lines starting with %)
        bibtex_text = re.sub(r'^\s*%.*$', '', bibtex_text, flags=re.MULTILINE)
        
        # Find all bibtex entries
        # Pattern matches @EntryType{key, ... }
        entry_pattern = r'@(\w+)\s*{\s*([^,]+)\s*,\s*([\s\S]*?)\s*}'
        
        for match in re.finditer(entry_pattern, bibtex_text):
            entry_type = match.group(1)
            entry_key = match.group(2).strip()
            fields_str = match.group(3)
            
            # Parse fields
            item = BibItem(item_type=entry_type)
            
            # Parse each field
            field_pattern = r'(\w+)\s*=\s*{([^}]*)}|(\w+)\s*=\s*"([^"]*)"|(\w+)\s*=\s*(\d+)'
            
            for field_match in re.finditer(field_pattern, fields_str):
                if field_match.group(1):  # {value} format
                    field_name = field_match.group(1)
                    field_value = field_match.group(2)
                elif field_match.group(3):  # "value" format
                    field_name = field_match.group(3)
                    field_value = field_match.group(4)
                elif field_match.group(5):  # numeric format
                    field_name = field_match.group(5)
                    field_value = field_match.group(6)
                else:
                    continue
                
                # Map bibtex fields to BibItem attributes
                field_name_lower = field_name.lower()
                field_value = field_value.strip()
                
                # Clean up braces and quotes
                field_value = field_value.strip('{}"')
                
                # Map common bibtex fields
                field_map = {
                    'title': 'title',
                    'author': 'author',
                    'abstract': 'abstract_note',
                    'publication': 'publication',
                    'journal': 'publication',
                    'year': 'date',
                    'volume': 'volume',
                    'number': 'issue',
                    'pages': 'pages',
                    'doi': 'doi',
                    'url': 'url',
                    'isbn': 'isbn',
                    'issn': 'issn',
                    'publisher': 'publisher',
                    'address': 'place',
                    'series': 'series',
                    'language': 'language',
                    'shorttitle': 'short_title',
                    'notes': 'notes',
                    'extra': 'extra',
                }
                
                if field_name_lower in field_map:
                    attr_name = field_map[field_name_lower]
                    if attr_name == 'date' and field_value:
                        # Parse year to datetime
                        try:
                            year = int(field_value)
                            item.date = datetime(year, 1, 1)
                        except (ValueError, TypeError):
                            item.date = None
                    elif attr_name == 'tags' and field_value:
                        # Parse tags as comma-separated values
                        item.tags = [t.strip() for t in field_value.split(',')]
                    else:
                        setattr(item, attr_name, field_value)
                else:
                    # Store unknown fields in extra
                    if not hasattr(item, '_extra_fields'):
                        item._extra_fields = {}
                    item._extra_fields[field_name_lower] = field_value
            
            items.append(item)
        
        return items
    
    def export_bibtex(self) -> str:
        """Export BibItem to bibtex string.
        
        Returns:
            Bibtex string representation of this BibItem.
        """
        import re
        from datetime import datetime
        
        # Field mapping from BibItem to bibtex
        field_map = {
            'title': 'title',
            'author': 'author',
            'abstract_note': 'abstract',
            'publication': 'journal',  # Default to journal
            'date': 'year',
            'volume': 'volume',
            'issue': 'number',
            'pages': 'pages',
            'doi': 'doi',
            'url': 'url',
            'isbn': 'isbn',
            'issn': 'issn',
            'publisher': 'publisher',
            'place': 'address',
            'series': 'series',
            'language': 'language',
            'short_title': 'shorttitle',
            'notes': 'notes',
        }
        
        lines = []
        lines.append(f"@{self.item_type or 'article'}" + " {")
        
        # Build field string
        fields = []
        
        # Add key - use first 8 chars of id or title
        key = f"{self.id.hex[:8]}" if self.id else "key"
        if self.title:
            # Create a more readable key from title
            title_key = re.sub(r'[^\w\s]', '', self.title[:30])
            title_key = re.sub(r'\s+', '_', title_key).lower()
            key = f"{title_key}_{self.id.hex[:4]}" if self.id else title_key
        fields.append(f"  key = {{{key}}}")
        
        for attr_name, bibtex_name in field_map.items():
            value = getattr(self, attr_name, None)
            if value is None:
                continue
            
            # Format value
            if attr_name == 'date' and isinstance(value, datetime):
                value = str(value.year)
            elif attr_name == 'tags' and value:
                value = ', '.join(value)
            elif isinstance(value, str):
                # Escape special characters in bibtex
                value = value.replace('\\', '\\\\')
                value = value.replace('{', '\\{').replace('}', '\\}')
                value = value.replace('&', '\\&').replace('%', '\\%')
                value = value.replace('$', '\\$').replace('#', '\\#')
                value = value.replace('_', '\\_').replace('@', '\\@')
                value = value.replace('\'', '\\\'')
                value = value.replace('"', '\\"')
            
            fields.append(f"  {bibtex_name} = {{{value}}}")
        
        # Add extra fields
        if hasattr(self, 'extra') and self.extra:
            for key, value in self.extra.items():
                if isinstance(value, str):
                    # Escape special characters
                    value = value.replace('\\', '\\\\')
                    value = value.replace('{', '\\{').replace('}', '\\}')
                    value = value.replace('&', '\\&').replace('%', '\\%')
                    value = value.replace('$', '\\$').replace('#', '\\#')
                    value = value.replace('_', '\\_').replace('@', '\\@')
                    value = value.replace('\'', '\\\'')
                    value = value.replace('"', '\\"')
                    fields.append(f"  {key} = {{{value}}}")
        
        lines.append(',\n'.join(fields))
        lines.append("}")
        
        return '\n'.join(lines)
