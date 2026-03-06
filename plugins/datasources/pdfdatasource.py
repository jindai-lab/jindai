"""PDF Data Source

This module provides a data source implementation for importing paragraphs
from PDF documents using the PyMuPDF (fitz) library.
"""

from typing import Iterator, List, Optional

import fitz
import regex as re
from sqlalchemy import func, select

from jindai.app import storage
from jindai.models import Dataset, Paragraph, get_db_session
from jindai.pipeline import DataSourceStage, PipelineStage


def resolve_range(page_range: str) -> Iterator[int]:
    """Parse a page range string into an iterator of page indices.
    
    Supports multiple formats:
    - Single pages: "1", "5", "10"
    - Page ranges: "1-3", "5-10"
    - Combined: "1-3,5,7-9"
    
    Page numbers are 1-indexed in the input but converted to 0-indexed
    for PyMuPDF which uses 0-based indexing.
    
    Args:
        page_range: Page range string (e.g., "1-3", "1,3,5-6,23").
        
    Yields:
        0-based page indices for valid page specifications.
        
    Examples:
        >>> list(resolve_range("1-3"))
        [0, 1, 2]
        >>> list(resolve_range("1,3,5-6"))
        [0, 2, 4, 5]
    """
    ranges = (page_range or "").split(",")
    for rng in ranges:
        if "-" in rng:
            try:
                start, end = map(int, rng.split("-", 1))
                yield from range(start - 1, end)
            except ValueError:
                # Skip invalid range specifications
                pass
        elif rng and re.match(r"\d+", rng):
            yield int(rng) - 1


class PDFDataSource(DataSourceStage):
    """Import paragraphs from PDF documents.
    
    This data source reads PDF files page by page and creates Paragraph
    objects for each page. It supports:
    - Skipping already imported pages (for incremental updates)
    - Processing specific page ranges
    - Tracking page labels and numbers
    
    Attributes:
        dataset_name: Target dataset name.
        lang: Language code for imported paragraphs.
        files: List of PDF file paths to process.
        skip_existed: If True, skip pages already in the dataset.
        page_range: Specific pages to process (e.g., "1-3", "1,3,5-6").
    """

    def apply_params(
        self,
        dataset_name: str = "",
        lang: str = "auto",
        content: str = "",
        skip_existed: bool = True,
        page_range: str = ""
    ) -> None:
        """Configure the PDF data source.
        
        Args:
            dataset_name: Name of the target dataset for imported paragraphs.
            lang: Language code for imported paragraphs ('auto' for automatic detection).
            content: Paths to PDF files (one per line).
            skip_existed: If True, skip pages already in the dataset.
            page_range: Specific pages to process (e.g., "1-3", "1,3,5-6").
        """
        self.dataset_name = dataset_name
        self.lang = lang
        self.skip_existed = skip_existed
        self.page_range = sorted(resolve_range(page_range))
        self.files = PipelineStage.parse_paths(content)

    async def fetch(self):
        """Process configured PDF files and yield paragraphs.
        
        Yields:
            Paragraph objects, one for each page in the specified range.
            Each paragraph contains:
                - content: Extracted text from the page
                - source_url: Path to the PDF file
                - source_page: 0-based page index
                - pagenum: Page label or 1-based page number
                - dataset: Target dataset ID
                - lang: Language code
        """
        dataset = await Dataset.get(self.dataset_name)
        lang = self.lang
        files = await self.files

        # Build mapping of file -> last imported page (0-indexed)
        if self.skip_existed:
            async with get_db_session() as session:
                result = await session.execute(
                    select(
                        Paragraph.source_url,
                        func.max(Paragraph.source_page).label("max_page"),
                    ).where(Paragraph.source_url.in_(files)).group_by(Paragraph.source_url)
                )
                existent = {
                    d["source_url"]: d["max_page"]
                    for d in result.mappings()
                }
        else:
            existent = {}

        total = len(files)
        for i, filepath in enumerate(files):
            imported_pages = 0
            self.log(f"{i+1}/{total} importing {filepath}")

            # Open PDF file
            stream = storage.open(filepath, "rb")
            if hasattr(stream, "name"):
                doc = fitz.open(stream.name)
            else:
                doc = fitz.open("pdf", stream)

            # Determine page range to process
            page_range = self.page_range
            if not page_range:
                # Process pages after the last imported one
                min_page = existent.get(filepath)
                min_page = 0 if min_page is None else (min_page + 1)
                self.log("... from page", min_page)
                page_range = range(min_page, doc.page_count)

            for page in page_range:
                if page >= doc.page_count:
                    break

                try:
                    # Get page label (e.g., "i", "1", "A-1")
                    label = doc[page].get_label()
                except (RuntimeError, TypeError):
                    label = ""

                try:
                    # Extract text content
                    content = (
                        doc[page]
                        .get_text()
                        .encode("utf-8", errors="ignore")
                        .decode("utf-8")
                    )
                except Exception as ex:
                    self.log(filepath, page + 1, ex)
                    content = ""

                # Only yield pages with substantial content
                if len(content) > 10:
                    imported_pages += 1

                yield Paragraph(
                    lang=lang,
                    content=content,
                    source_url=filepath,
                    source_page=page,
                    pagenum=label or str(page + 1),
                    dataset=dataset.id,
                )

            # Log if no sufficient content was found
            if not existent.get(filepath) and imported_pages == 0:
                self.log(f"no sufficient texts found in {filepath}")
