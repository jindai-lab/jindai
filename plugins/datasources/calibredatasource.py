"""Calibre Library Data Source

This module provides a data source implementation for importing book metadata
from Calibre library databases (metadata.db SQLite files).
"""

import datetime
import sqlite3
import os
import urllib.parse
from typing import List, Optional
from uuid import UUID

from sqlalchemy import select, update

from jindai.storage import storage
from jindai.models import Dataset, Paragraph, get_db_session
from jindai.pipeline import DataSourceStage, PipelineStage


class CalibreLibraryDataSource(DataSourceStage):
    """Import book metadata from Calibre library databases.
    
    This data source reads from Calibre's metadata.db SQLite database to extract
    information about books (PDF and EPUB formats). It creates Paragraph objects
    containing book metadata such as title, author, publication date, and file paths.
    
    The data source supports scanning for moved files to update source URLs when
    books are relocated within the library.
    
    Attributes:
        dataset_name: The name of the target dataset.
        lang: Language code for imported paragraphs.
        paths: List of Calibre library paths to scan.
        formats: Tuple of allowed file extensions (default: ('epub', 'pdf')).
        scan_for_moved: Whether to update source URLs for moved books.
    """

    def get_calibre_books_safe(self, library_path: str) -> List[Paragraph]:
        """Safely read book metadata from a Calibre library database.
        
        Opens the metadata.db file in read-only mode to extract book information.
        Handles database errors gracefully and returns an empty list on failure.
        
        Args:
            library_path: Path to the Calibre library directory containing metadata.db.
            
        Returns:
            A list of Paragraph objects, each containing:
                - author: Book authors joined with ' & '
                - pdate: Publication date (year only) or None if unknown
                - outline: Book title
                - content: Relative file path within the library
                - extdata: Dictionary with 'book_id' key containing the database ID
        """
        # Convert path to absolute path for SQLite URI
        db_path = os.path.abspath(os.path.join(library_path, "metadata.db"))
        if not os.path.exists(db_path):
            return []

        # Construct read-only URI (handles Windows drive letters)
        db_uri = f"file:{urllib.parse.quote(db_path)}?mode=ro"

        try:
            # Open database in read-only mode
            conn = sqlite3.connect(db_uri, uri=True)
            cursor = conn.cursor()

            # SQL Query:
            # 1. Filters by PDF and EPUB formats (case-insensitive)
            # 2. Joins books with data table for file information
            # 3. Aggregates author names from the authors table
            query = """
            SELECT 
                b.id,
                b.title, 
                b.path, 
                d.name, 
                d.format, 
                b.pubdate,
                (SELECT GROUP_CONCAT(a.name, ' & ') 
                FROM authors a 
                JOIN books_authors_link bal ON a.id = bal.author 
                WHERE bal.book = b.id) as author_names
            FROM books b
            JOIN data d ON b.id = d.book
            WHERE LOWER(d.format) IN ('pdf', 'epub')
            """

            cursor.execute(query)
            rows = cursor.fetchall()

            books_info: List[Paragraph] = []
            for row in rows:
                book_id, title, folder_path, file_name, ext, pub_date, authors = row

                # Parse publication year (Calibre uses '0101-01-01' for unknown dates)
                year: Optional[int] = (
                    int(pub_date[:4])
                    if pub_date and not pub_date.startswith("0101")
                    else None
                )

                # Construct relative file path
                # Calibre stores path as directory relative to library root
                relative_file_path = os.path.join(
                    folder_path, f"{file_name}.{ext.lower()}"
                )

                books_info.append(
                    Paragraph(
                        author=authors,
                        pdate=datetime.datetime(year, 1, 1) if year else None,
                        outline=title,
                        content=relative_file_path,
                        extdata={"book_id": book_id},
                    )
                )

            conn.close()
            return books_info

        except sqlite3.Error as e:
            self.log_exception(f"Error while reading from database", e)
            return []

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
            scan_for_moved: If True, update source URLs for books that have been moved.
        """
        self.dataset_name = dataset_name
        self.lang = lang
        self.paths = content
        self.formats = tuple(formats.lower().split(","))
        self.scan_for_moved = scan_for_moved

    async def fetch(self):
        """Fetch book metadata from configured Calibre libraries.
        
        Yields:
            Paragraph objects containing book metadata with absolute file paths.
            When scan_for_moved is enabled, also updates existing records for
            books whose source URLs have changed.
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
                    if self.formats and book.content.lower().endswith(self.formats):
                        # Convert to absolute path
                        book.content = os.path.join(path, book.content)
                        book.dataset = dsid
                        book_id = str(book.extdata['book_id'])
                        
                        # Update source URL for moved books
                        if self.scan_for_moved and existent.get(book_id):
                            await session.execute(
                                update(Paragraph)
                                .filter(
                                    Paragraph.source_url == existent[book_id],
                                )
                                .values(source_url=book.content)
                            )
                            
                        yield book
