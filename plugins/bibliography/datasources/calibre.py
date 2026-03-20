"""Calibre Library Data Source for Bibliography Plugin.

This module provides a data source implementation for importing book metadata
from Calibre library databases (metadata.db SQLite files) with rich content
including file attachments.
"""

import datetime
import os
import urllib.parse
from typing import List, Optional, Dict, Any
from uuid import UUID

from sqlalchemy import distinct, select, update, func
from sqlalchemy.orm import Session

from jindai.storage import storage
from jindai.models import Dataset, Paragraph, get_db_session
from jindai.pipeline import DataSourceStage, PipelineStage

from .calibre_models import (
    Base, Books, Data, Authors, BooksAuthorsLink,
    Publishers, BooksPublishersLink, Series, BooksSeriesLink,
    Tags, BooksTagsLink, Comments, Languages, BooksLanguagesLink,
    Ratings, BooksRatingsLink, Identifiers, LastReadPositions,
    Annotations, CustomColumns, CustomColumn1,
    Format, CompleteBookInfo, create_book_info_from_orm_models,
    get_books_by_filter, get_all_books_generator
)


class CalibreDataSource(DataSourceStage):
    """Import book metadata from Calibre library databases with rich content.
    
    This data source reads from Calibre's metadata.db SQLite database to extract
    information about books (PDF and EPUB formats). It creates Paragraph objects
    containing comprehensive book metadata including:
    - Basic info: title, author, publication date
    - File attachments: array of relative paths to PDF/EPUB files
    - Identifiers: book_id for tracking
    - Publication details: publisher, place, series, etc.
    
    The data source supports scanning for moved files to update source URLs when
    books are relocated within the library.
    
    Attributes:
        dataset_name: The name of the target dataset.
        lang: Language code for imported paragraphs.
        paths: List of Calibre library paths to scan.
        formats: Tuple of allowed file extensions (default: ('epub', 'pdf')).
        scan_for_moved: Whether to update source URLs for moved books.
    """

    def get_calibre_books_safe(self, library_path: str) -> List[Dict[str, Any]]:
        """Safely read book metadata from a Calibre library database.
        
        Opens the metadata.db file in read-only mode to extract book information.
        Handles database errors gracefully and returns an empty list on failure.
        
        Args:
            library_path: Path to the Calibre library directory containing metadata.db.
            
        Returns:
            A list of dictionaries, each containing:
                - book_id: Database ID
                - title: Book title
                - authors: Book authors joined with ' & '
                - pubdate: Publication date or None if unknown
                - file_path: Relative file path within the library
                - file_format: File format (PDF/EPUB)
                - file_size: File size in bytes
                - publisher: Publisher name
                - publication_date: Full publication date
                - series: Series name (if any)
                - series_index: Series index number
                - isbn: ISBN
                - tags: List of tags
                - file_attachments: Array of relative file paths
        """
        db_path = os.path.abspath(os.path.join(library_path, "metadata.db"))
        if not os.path.exists(db_path):
            return []

        db_uri = f"sqlite:///{urllib.parse.quote(db_path)}?mode=ro"
        
        # Create engine with read-only URI
        from sqlalchemy import create_engine
        engine = create_engine(db_uri, echo=False)
        
        with engine.connect() as connection:
            # Create a session
            session = Session(bind=connection)
            
            # Query books with their formats (PDF/EPUB only)
            query = (
                select(
                    Books.id.label('book_id'),
                    Books.title,
                    Books.path.label('folder_path'),
                    Data.name.label('file_name'),
                    Data.format.label('file_format'),
                    Data.uncompressed_size.label('file_size'),
                    Books.pubdate,
                    Books.author_sort,
                    Books.series_index,
                    func.group_concat(distinct(Authors.name), ' & ').label('authors'),
                    Publishers.name.label('publisher'),
                    func.group_concat(Tags.name, ', ').label('tags'),
                    Series.name.label('series_name'),
                    Languages.lang_code.label('language')
                )
                .select_from(Books)
                .join(Data, Books.id == Data.book)
                .outerjoin(BooksAuthorsLink, Books.id == BooksAuthorsLink.book)
                .outerjoin(Authors, BooksAuthorsLink.author == Authors.id)
                .outerjoin(BooksPublishersLink, Books.id == BooksPublishersLink.book)
                .outerjoin(Publishers, BooksPublishersLink.publisher == Publishers.id)
                .outerjoin(BooksSeriesLink, Books.id == BooksSeriesLink.book)
                .outerjoin(Series, BooksSeriesLink.series == Series.id)
                .outerjoin(BooksTagsLink, Books.id == BooksTagsLink.book)
                .outerjoin(Tags, BooksTagsLink.tag == Tags.id)
                .outerjoin(BooksLanguagesLink, Books.id == BooksLanguagesLink.book)
                .outerjoin(Languages, BooksLanguagesLink.lang_code == Languages.id)
                .where(Data.format.in_(['PDF', 'EPUB', 'pdf', 'epub']))
                .group_by(Books.id, Data.id)
            )
            
            result = session.execute(query).fetchall()
            
            books_info: List[Dict[str, Any]] = []
            for row in result:
                (
                    book_id, title, folder_path, file_name, ext, size, pubdate,
                    author_sort, series_index, authors, publisher,
                    tag_names, series_name, language
                ) = row

                # Parse publication year
                year: Optional[int] = pubdate.year

                # Construct relative file path
                relative_file_path = os.path.join(
                    folder_path, f"{file_name}.{ext.lower()}"
                )

                # Build file attachments as array of relative paths
                file_attachments = []
                if size:
                    file_attachments.append(relative_file_path)

                # Build tags list
                tags = []
                if tag_names:
                    tags = [t.strip() for t in tag_names.split(',') if t.strip()]

                books_info.append({
                    "book_id": book_id,
                    "title": title,
                    "authors": authors or "",
                    "pubdate": pubdate,
                    "year": year,
                    "file_path": relative_file_path,
                    "file_format": ext.upper(),
                    "file_size": size,
                    "publisher": publisher or "",
                    "publication_date": pubdate,
                    "series_name": series_name or "",
                    "series_index": series_index,
                    "tags": tags,
                    "file_attachments": file_attachments,
                })

            session.close()
            return books_info

    def apply_params(
        self,
        dataset_name: str = "",
        lang: str = "auto",
        content: str = "",
        formats: str = "epub,pdf",
        scan_for_moved: bool = True
    ) -> None:
        """Configure the data source parameters.
        
        Args:
            dataset_name: Name of the target dataset for imported paragraphs.
            lang: Language code for imported paragraphs ('auto' for automatic detection).
            content: Path(s) to Calibre library directory(s), one per line.
            formats: Comma-separated list of allowed file extensions.
            scan_for_moved: If True, update source URL for books that have been moved.
        """
        self.dataset_name = dataset_name
        self.lang = lang
        self.paths = content
        self.formats = tuple(formats.lower().split(","))
        self.scan_for_moved = scan_for_moved

    async def fetch(self):
        """Fetch book metadata from configured Calibre libraries.
        
        Yields:
            Paragraph objects containing comprehensive book metadata with:
            - author: Book authors joined with ' & '
            - pdate: Publication date (year only) or None if unknown
            - outline: Book title
            - content: Absolute file path
            - extdata: Dictionary with comprehensive book metadata including:
                - book_id: Database ID
                - file_attachments: Array of relative file paths
                - publisher, series, tags, comments, etc.
        """
        paths = await PipelineStage.parse_paths(self.paths)
        dsid = (await Dataset.get(self.dataset_name)).id
        existent: dict = {}
        
        async with get_db_session() as session:
        
            if self.scan_for_moved:
                # Build mapping of book_id -> current source_url for existing books
                existent = dict(
                    (await session.execute(
                        select(Paragraph.extdata.op('->>')('book_id'),
                               Paragraph.source_url)
                               .distinct(Paragraph.source_url)
                    )).all()
                )
                
            for path in paths:
                books = self.get_calibre_books_safe(storage.safe_join(path))
                for book in books:
                    # Filter by allowed formats
                    if self.formats and book["file_path"].lower().endswith(self.formats):
                        # Convert to absolute path
                        absolute_path = os.path.join(path, book["file_path"])
                        
                        # Create Paragraph with rich metadata
                        paragraph = Paragraph(
                            author=book["authors"],
                            pdate=datetime.datetime(book["year"], 1, 1) if book["year"] else None,
                            outline=book["title"],
                            content=absolute_path,
                            extdata={
                                "book_id": book["book_id"],
                                "file_attachments": book["file_attachments"],
                                "publisher": book["publisher"],
                                "series": book["series_name"],
                                "series_index": book["series_index"],
                                "tags": book["tags"],
                                "item_type": "book",
                                "archive": "Calibre",
                                "library_catalog": path,
                            },
                        )
                        paragraph.dataset = dsid
                        book_id = str(book["book_id"])
                        
                        # Update source URL for moved books
                        if self.scan_for_moved and existent.get(book_id):
                            await session.execute(
                                update(Paragraph)
                                .filter(
                                    Paragraph.source_url == existent[book_id],
                                )
                                .values(source_url=absolute_path)
                            )
                            
                        yield paragraph
