"""Calibre Library Data Source for Bibliography Plugin.

This module provides a data source implementation for importing book metadata
from Calibre library databases (metadata.db SQLite files) with rich content
including file attachments.
"""

import datetime
import os
import shutil
import tempfile
import urllib.parse
from contextlib import contextmanager
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID

from sqlalchemy import create_engine, delete, func, select
from sqlalchemy.orm import Session

from jindai.storage import storage
from jindai.models import (
    Dataset, FileDataset, FileMetadata, Paragraph, TextEmbeddings,
    get_db_session,
)
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
    
    The data source tracks every imported file in ``FileMetadata`` directly
    (keyed by library path, book id and format in ``FileMetadata.extdata``).
    When a book is relocated inside the library, the ``path`` of the tracked
    FileMetadata row is rewritten in place; if the new path is already
    occupied by another FileMetadata row (``path`` is unique), the stale row
    is deleted together with its bound dataset links, paragraphs and text
    embeddings, and the occupying row's ID is reused.

    Attributes:
        dataset_name: The name of the target dataset.
        lang: Language code for imported paragraphs.
        paths: List of Calibre library paths to scan.
        formats: Tuple of allowed file extensions (default: ('epub', 'pdf')).
        scan_for_moved: Whether to reconcile the path of moved books. When
            False, moved books are imported as new FileMetadata rows and the
            stale rows are left untouched.
    """
    def apply_params(
        self,
        dataset_name: str = "",
        lang: str = "auto",
        content: str = "",
        formats: str = "epub,pdf",
        scan_for_moved: bool = True,
        **params: Any
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

    def get_calibre_books_safe(self, library_path: str) -> List[Dict[str, Any]]:
        """Safely read book metadata from a Calibre library database.
        
        A snapshot copy of metadata.db is made and queried instead of the
        live file, so that a running Calibre instance cannot interfere with
        the import; the copy is removed once the query has completed.
        Handles database errors gracefully and returns an empty list on failure.
        
        Args:
            library_path: Path to the Calibre library directory containing metadata.db.
                          It is an absolute path.
            
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

        # Make a snapshot copy of the database, connect to the copy instead
        # of the live file, and have the copy removed once the query has
        # completed (see ``_connect_metadata_snapshot``).
        with self._connect_metadata_snapshot(db_path) as connection:
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
                    func.group_concat(Authors.name, '&').label('authors'),
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
                .where(Data.format.in_(['PDF', 'EPUB', 'pdf', 'epub']), Data.uncompressed_size > 0)
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
                
                if file_name.lower().endswith(f".{ext.lower()}"):
                    file_name = file_name[:-len(ext)-1]

                # Parse publication year
                year: Optional[int] = pubdate.year

                # Construct relative file path
                relative_file_path = storage.relative_path(os.path.join(
                    library_path, folder_path, f"{file_name}.{ext.lower()}"
                ))
                authors = ' & '.join(set([_.strip() for _ in authors.split('&')]))

                # Build file attachments as array of relative paths
                file_attachments = [relative_file_path]
                
                # Build tags list
                tags = []
                if tag_names:
                    tags = [t.strip() for t in tag_names.split(',') if t.strip()]

                books_info.append({
                    "book_id": str(book_id),
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

    @contextmanager
    def _connect_metadata_snapshot(self, db_path: str):
        """Copy a Calibre ``metadata.db`` to a temporary file and connect to it.

        Connecting straight to the live ``metadata.db`` can fail while the
        Calibre application is running, because it holds locks on the file
        (the previous ``nolock``/``immutable`` URI flags were only a
        workaround that could still observe a half-written state). Instead,
        the database file -- together with its WAL sidecar when one is
        present -- is copied to a temporary location, the engine is pointed
        at that private copy, and the copy is removed again as soon as the
        yielded connection has been closed.

        Args:
            db_path: Absolute path to the ``metadata.db`` file to snapshot.

        Yields:
            A SQLAlchemy connection bound to the snapshot copy.
        """
        fd, snapshot_path = tempfile.mkstemp(prefix="calibre_metadata_", suffix=".db")
        os.close(fd)
        engine = None
        try:
            shutil.copy2(db_path, snapshot_path)
            # A running Calibre keeps its most recent transactions in the
            # WAL sidecar of metadata.db; copy it over as well so that the
            # snapshot reflects the latest state of the library. The engine
            # connects without the ``immutable``/``nolock`` flags so that
            # SQLite can recover the WAL into the snapshot on open.
            if os.path.exists(db_path + "-wal"):
                shutil.copy2(db_path + "-wal", snapshot_path + "-wal")
            engine = create_engine(
                f"sqlite:///{urllib.parse.quote(snapshot_path)}", echo=False
            )
            with engine.connect() as connection:
                yield connection
        finally:
            if engine is not None:
                engine.dispose()
            for suffix in ("", "-wal", "-shm"):
                try:
                    os.remove(snapshot_path + suffix)
                except OSError:
                    pass

    async def fetch(self):  # type: ignore[override]
        """Fetch book metadata from configured Calibre libraries.

        The FileMetadata row of every book is checked and reconciled directly
        (see ``_reconcile_file_metadata``); the resulting ID is used as the
        paragraph source without any on-the-fly path lookups.

        Yields:
            Paragraph objects containing comprehensive book metadata with:
            - author: Book authors joined with ' & '
            - pdate: Publication date (year only) or None if unknown
            - outline: Book title
            - source_url = content: Absolute file path
            - extdata: Dictionary with comprehensive book metadata including:
                - book_id: Database ID
                - file_attachments: Array of relative file paths
                - publisher, series, tags, comments, etc.
        """
        paths = await PipelineStage.parse_paths(self.paths)
        dsid = (await Dataset.get(self.dataset_name)).id

        # Read all libraries up front so that the FileMetadata state can be
        # preloaded in bulk instead of being queried per book.
        libraries: List[Tuple[str, List[Dict[str, Any]]]] = []
        for path in paths:
            books = [
                book for book in self.get_calibre_books_safe(storage.safe_join(path))
                if not self.formats or book["file_path"].lower().endswith(self.formats)
            ]
            if books:
                libraries.append((path, books))

        # Preload the FileMetadata state:
        # - tracked: (library, book_id, format) -> (file id, path) for rows
        #   written by this datasource (stamped in FileMetadata.extdata);
        # - by_path: path -> file id, to detect path conflicts (the
        #   FileMetadata.path column is unique).
        tracked: Dict[Tuple[str, str, str], Tuple[UUID, str]] = {}
        by_path: Dict[str, UUID] = {}
        async with get_db_session() as session:
            if paths:
                rows = (await session.execute(
                    select(FileMetadata.id, FileMetadata.path, FileMetadata.extdata)
                    .where(FileMetadata.extdata.op('->>')('library_catalog').in_(paths))
                )).all()
                for fid, fpath, extdata in rows:
                    book_id = (extdata or {}).get('book_id')
                    fmt = (extdata or {}).get('format')
                    if book_id and fmt:
                        library = (extdata or {}).get('library_catalog')
                        tracked[(str(library), str(book_id), str(fmt))] = (fid, fpath)
            target_paths = {
                book["file_path"] for _, books in libraries for book in books
            }
            if target_paths:
                for fid, fpath in (await session.execute(
                    select(FileMetadata.id, FileMetadata.path)
                    .where(FileMetadata.path.in_(target_paths))
                )).all():
                    by_path[fpath] = fid

        for path, books in libraries:
            for book in books:
                file_path = book["file_path"]

                # Check if cover.jpg exists
                cover_path = storage.safe_join(file_path).rsplit('/', 1)[0] + "/cover.jpg"
                if os.path.exists(cover_path):
                    cover_path = storage.relative_path(cover_path)
                else:
                    cover_path = ''

                # Check FileMetadata directly and reconcile the stored path;
                # use the resulting ID as the paragraph source.
                source = await self._reconcile_file_metadata(
                    path, book, tracked, by_path
                )

                # Create Paragraph with rich metadata
                paragraph = Paragraph(
                    author=book["authors"],
                    pdate=datetime.datetime(book["year"], 1, 1) if book["year"] else None,
                    outline=book["title"],
                    content=file_path,
                    source_id=source,
                    extdata={
                        "call_number": book["book_id"],
                        "file_attachments": book["file_attachments"],
                        "publisher": book["publisher"],
                        "series": book["series_name"],
                        "series_index": book["series_index"],
                        "tags": book["tags"],
                        "item_type": "book",
                        "archive": "Calibre",
                        "library_catalog": path,
                        "cover": cover_path or ''
                    },
                )
                await FileDataset.link(source, dsid)
                yield paragraph


    async def _reconcile_file_metadata(
        self,
        library: str,
        book: Dict[str, Any],
        tracked: Dict[Tuple[str, str, str], Tuple[UUID, str]],
        by_path: Dict[str, UUID],
    ) -> UUID:
        """Check FileMetadata directly and reconcile the stored path of a book.

        The row is located by the tracking key stamped in
        ``FileMetadata.extdata`` (``library_catalog`` + ``book_id`` +
        ``format``); rows written before tracking existed are adopted via
        their path and stamped with the tracking key.

        Resolution rules:
        - No FileMetadata found: a new one is created (or an existing row
          occupying the same path is adopted) and its ID returned;
        - Found with an unchanged path: its ID is returned as-is;
        - Moved and the new path is free: ``path`` is rewritten in place,
          keeping the ID so that existing paragraphs stay valid;
        - Moved but the new path is already occupied (``path`` is unique):
          the stale row is deleted together with its bound dataset links
          (``file_dataset``), ``paragraph`` and ``text_embeddings`` rows,
          and the occupying row's ID is returned.

        Args:
            library: Calibre library path the book belongs to.
            book: Book info dict as produced by ``get_calibre_books_safe``.
            tracked: Preloaded tracking-key -> (file id, path) mapping;
                updated in place.
            by_path: Preloaded path -> file id mapping; updated in place.

        Returns:
            The FileMetadata ID to be used as ``Paragraph.source_id``.
        """
        file_path = book["file_path"]
        book_id = str(book["book_id"])
        fmt = book["file_format"].lower()
        stamp = {"library_catalog": library, "book_id": book_id, "format": fmt}

        fm_id, fm_path = tracked.get((library, book_id, fmt), (None, None))
        if fm_id is None:
            # Adopt a legacy row created before tracking was introduced
            fm_id = by_path.get(file_path)
            fm_path = file_path if fm_id is not None else None

        if fm_id is not None and fm_path != file_path and not self.scan_for_moved:
            # Moved books are left untouched; fall back to creation semantics
            fm_id, fm_path = None, None

        if fm_id is None:
            # Create a new FileMetadata row (or adopt whatever already
            # occupies the path, since ``path`` is unique).
            occupy_id = by_path.get(file_path)
            async with get_db_session() as session:
                fm = (await session.get(FileMetadata, occupy_id)) if occupy_id else None
                if fm is not None:
                    fm.extdata = {**(fm.extdata or {}), **stamp}
                    fid = fm.id
                else:
                    fm = FileMetadata(
                        path=file_path,
                        extension=fmt,
                        size_bytes=int(book.get("file_size") or 0),
                        extdata=stamp,
                    )
                    session.add(fm)
                    await session.flush()
                    fid = fm.id
            tracked[(library, book_id, fmt)] = (fid, file_path)
            by_path[file_path] = fid
            return fid

        if fm_path == file_path:
            if (library, book_id, fmt) not in tracked:
                # Adopted legacy row: stamp the tracking key
                async with get_db_session() as session:
                    fm = await session.get(FileMetadata, fm_id)
                    if fm is not None:
                        fm.extdata = {**(fm.extdata or {}), **stamp}
                tracked[(library, book_id, fmt)] = (fm_id, file_path)
            return fm_id

        # The book has been moved: the stored path differs from file_path.
        conflict_id = by_path.get(file_path)
        if conflict_id is not None and conflict_id != fm_id:
            # Changing the path would violate the unique constraint on
            # FileMetadata.path: delete the stale row with everything bound
            # to it and reuse the occupying row.
            async with get_db_session() as session:
                stale = await session.get(FileMetadata, fm_id)
                if stale is not None:
                    await session.execute(
                        delete(TextEmbeddings).where(TextEmbeddings.source_id == fm_id)
                    )
                    await session.execute(
                        delete(Paragraph).where(Paragraph.source_id == fm_id)
                    )
                    await session.execute(
                        delete(FileDataset).where(FileDataset.file_id == fm_id)
                    )
                    await session.delete(stale)
                occupant = await session.get(FileMetadata, conflict_id)
                if occupant is not None:
                    occupant.extdata = {**(occupant.extdata or {}), **stamp}
            for key, val in list(tracked.items()):
                if val[0] == conflict_id:
                    del tracked[key]
            tracked[(library, book_id, fmt)] = (conflict_id, file_path)
            by_path.pop(fm_path, None)
            by_path[file_path] = conflict_id
            return conflict_id

        # The target path is free: rewrite the path in place, keeping the ID
        # so that paragraphs already imported remain valid.
        async with get_db_session() as session:
            fm = await session.get(FileMetadata, fm_id)
            if fm is not None:
                fm.path = file_path
                fm.extension = fmt
                fm.size_bytes = int(book.get("file_size") or 0)
                fm.extdata = {**(fm.extdata or {}), **stamp}
        tracked[(library, book_id, fmt)] = (fm_id, file_path)
        by_path.pop(fm_path, None)
        by_path[file_path] = fm_id
        return fm_id

