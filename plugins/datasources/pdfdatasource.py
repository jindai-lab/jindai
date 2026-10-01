"""PDF Data Source

This module provides a data source implementation for importing paragraphs
from PDF documents. Text is extracted page by page via the unified
``jindai.pdfutils`` helpers; pages without extractable text fall back to
an external PaddleOCR remote service.

Enhanced features:
- Text cleaning (OCR error correction, whitespace normalization)
- Language detection optimized for zh, en, de, fr, ja, ru
- Cross-page re-paragraphization with sliding window
- External OCR fallback for scanned pages (no embedded text)
- Batched parallel OCR (ocr_workers) fanned out over paddle replicas
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterator, List, Optional, cast

import regex as re
from sqlalchemy import func, select

from jindai.storage import storage
from jindai.models import Dataset, FileDataset, FileMetadata, Paragraph, get_db_session
from jindai.pipeline import DataSourceStage, PipelineStage
from jindai.pdfutils import open_pdf, render_pdf_page
from jindai.ocrutils import PaddleOCRClient
from plugins.pipelines.pdf_stages import TextCleaner, PDFLanguageDetect, CrossPageReparagraphizer


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


def _norm_page_text(text: str) -> str:
    """Whitespace-stripped lowercase text for containment dedup checks."""
    return re.sub(r"\s+", "", text or "").lower()


class PDFDataSource(DataSourceStage):
    """Import paragraphs from PDF documents with enhanced text processing.
    
    This data source reads PDF files page by page and creates Paragraph
    objects for each page. It supports:
    - Skipping already imported pages (for incremental updates)
    - Processing specific page ranges
    - Tracking page labels and numbers
    - Text cleaning (OCR error correction, whitespace normalization)
    - Language detection optimized for zh, en, de, fr, ja, ru
    - Cross-page re-paragraphization with sliding window
    
    Attributes:
        dataset_name: Target dataset name.
        lang: Language code for imported paragraphs.
        files: List of PDF file paths to process.
        skip_existed: If True, skip pages already in the dataset.
        page_range: Specific pages to process (e.g., "1-3", "1,3,5-6").
        clean_text: Enable text cleaning.
        detect_language: Enable language detection.
        reparagraphize: Enable cross-page re-paragraphization.
    """

    def apply_params(
        self,
        dataset_name: str = "",
        lang: str = "auto",
        content: str = "",
        skip_existed: bool = True,
        page_range: str = "",
        # New text processing parameters
        clean_text: bool = True,
        reparagraphize: bool = True,
        # TextCleaner parameters
        remove_garbled: bool = True,
        garbled_threshold: float = 0.3,
        fix_ocr_errors: bool = True,
        fix_german_umlauts: bool = False,
        # CrossPageReparagraphizer parameters
        min_paragraph_length: int = 50,
        max_paragraph_length: int = 2000,
        short_line_threshold: int = 30,
        short_line_batch: int = 5,
        # OCR fallback parameters
        ocr_fallback: bool = False,
        ocr_workers: int = 4,
        # Page-set resume: re-import pages lacking a non-empty paragraph
        reimport_gaps: bool = True,
        **params,
    ) -> None:
        """Configure the PDF data source.
        
        Args:
            dataset_name: Name of the target dataset for imported paragraphs.
            lang: Language code for imported paragraphs ('auto' for automatic detection).
            content: Paths to PDF files (one per line).
            skip_existed: If True, skip pages already in the dataset.
            page_range: Specific pages to process (e.g., "1-3", "1,3,5-6").
            clean_text: Enable text cleaning (default: True).
            detect_language: Enable language detection (default: True).
            reparagraphize: Enable cross-page re-paragraphization (default: True).
            remove_garbled: Filter garbled text (default: True).
            garbled_threshold: Garbled detection threshold 0-1 (default: 0.3).
            fix_ocr_errors: Fix OCR errors (default: True).
            fix_german_umlauts: Fix German umlaut substitutions (Ae→Ä, etc.) (default: False).
            min_paragraph_length: Minimum paragraph length (default: 50).
            max_paragraph_length: Maximum paragraph length (default: 2000).
            short_line_threshold: Short line threshold (default: 30).
            short_line_batch: Short lines to trigger merge (default: 5).
            ocr_fallback: If True, call external OCR service for pages with
                no extractable text (default: True).
            ocr_workers: Parallel OCR workers per file, fanned out over
                paddle replicas round-robin (default: 4).
            reimport_gaps: If True (default), resume by page set: re-import
                every page without a non-empty, non-metadata paragraph
                (never imported, dropped by a failed OCR call, or stored
                empty) instead of continuing after max(page#); already
                stored text is skipped to avoid duplicates. False = legacy
                max(page#)+1 resume.
        """
        self.dataset_name = dataset_name
        self.lang = lang
        self.skip_existed = skip_existed
        self.ext_params = params
        self.page_range = sorted(resolve_range(page_range))
        
        self.log(f'File Paths: {content}')
        self.files = PipelineStage.parse_paths(content)
        
        # Text processing options
        self.clean_text = clean_text
        self.detect_language = lang == 'auto'
        self.reparagraphize = reparagraphize
        self.ocr_fallback = ocr_fallback
        self.ocr_workers = max(1, int(ocr_workers))
        self.reimport_gaps = bool(reimport_gaps)
        
        # TextCleaner options
        self.remove_garbled = remove_garbled
        self.garbled_threshold = garbled_threshold
        self.fix_ocr_errors = fix_ocr_errors
        self.fix_german_umlauts = fix_german_umlauts
        
        # CrossPageReparagraphizer options
        self.min_paragraph_length = min_paragraph_length
        self.max_paragraph_length = max_paragraph_length
        self.short_line_threshold = short_line_threshold
        self.short_line_batch = short_line_batch

    async def fetch(self):
        """Process configured PDF files and yield paragraphs.
        
        Yields:
            Paragraph objects, one for each page in the specified range.
            Each paragraph contains:
                - content: Extracted text from the page (optionally cleaned)
                - source: FileMetadata of the PDF file
                - source_page: 0-based page index
                - pagenum: Page label or 1-based page number
                - dataset: Target dataset ID
                - lang: Language code (optionally auto-detected)
        """
        
        dataset = await Dataset.get(self.dataset_name)
        lang = self.lang
        files = await self.files
        self.log(f'{len(files)} files resolved')

        # Build mapping of file -> last imported page (0-indexed)
        if self.skip_existed:
            async with get_db_session() as session:
                result = await session.execute(
                    select(
                        FileMetadata.path,
                        func.max(Paragraph.source_page).label("max_page"),
                    )
                    .join(FileMetadata, Paragraph.source_id == FileMetadata.id)
                    .where(FileMetadata.path.in_(files))
                    .group_by(FileMetadata.path)
                )
                existent = {
                    d["path"]: d["max_page"]
                    for d in result.mappings()
                }
        else:
            existent = {}

        # Initialize text processing stages
        text_cleaner = TextCleaner(
            remove_garbled=self.remove_garbled,
            garbled_threshold=self.garbled_threshold,
            normalize_spaces=True,
            remove_hyphens=True,
            fix_ocr_errors=self.fix_ocr_errors,
            fix_german_umlauts=self.fix_german_umlauts,
        ) if self.clean_text else None
        
        lang_detector = PDFLanguageDetect() if self.detect_language else None
        
        reparagraphizer = CrossPageReparagraphizer(
            min_paragraph_length=self.min_paragraph_length,
            max_paragraph_length=self.max_paragraph_length,
            short_line_threshold=self.short_line_threshold,
            short_line_batch=self.short_line_batch,
        ) if self.reparagraphize else None

        # OCR client for fallback on text-less pages (scanned documents)
        ocr_client = PaddleOCRClient() if self.ocr_fallback else None

        total = len(files)
        for i, filepath in enumerate(files):
            imported_pages = 0
            repaired_pages = 0
            dup_skipped = 0
            self.log(f"{i+1}/{total} importing {filepath}")
            source = await FileMetadata.get(storage.relative_path(filepath))
            assert source, 'Source should not be None'

            # Folder author: Calibre layout <catalog>/<author>/<title (id)>/<file>.
            # Page paragraphs carry no author column value, so derive it from
            # the path — DatasetFromField then links 书库--{author} (author
            # sub-dataset) instead of rendering a literal "书库--None" bucket.
            _rel = storage.relative_path(filepath).split('/')
            _folder_author = _rel[-3] if len(_rel) == 4 else ''
            author_kw = ({"author": _folder_author}
                         if _folder_author and "author" not in self.ext_params
                         else {})

            # Open PDF file using the unified helper
            stream = storage.open(filepath, "rb")
            doc = open_pdf(cast(Any, stream))
            try:
                # Determine page range to process
                page_range = self.page_range
                repair_blob = ""
                repair_set: set = set()
                if not page_range and self.skip_existed and self.reimport_gaps:
                    # Page-set resume: a page counts as DONE only when it
                    # has a NON-EMPTY, non-metadata paragraph row
                    # (outline IS NULL excludes the Calibre book-metadata
                    # paragraph). This finds every page without a paragraph
                    # or with an empty one -- never imported, dropped by a
                    # failed OCR call, or stored blank -- instead of just
                    # resuming after max(page#), which misses interior gaps.
                    async with get_db_session() as session:
                        have = {r[0] for r in (await session.execute(
                            select(Paragraph.source_page).where(
                                Paragraph.source_id == source.id,
                                Paragraph.source_page.is_not(None),
                                Paragraph.outline.is_(None),
                                func.btrim(func.coalesce(Paragraph.content, "")) != "",
                            )
                        )).all()}
                    gap_all = set(range(doc.page_count)) - have
                    page_range = sorted(gap_all)
                    if page_range:
                        # Dedup only for interior gaps (<= max imported
                        # page): cross-page merges store page N+1's text
                        # inside page N's row -- re-import would duplicate
                        # it. Tail pages were never imported: no dup risk.
                        have_max = max(have) if have else -1
                        repair_set = {p for p in gap_all if p <= have_max}
                        if repair_set:
                            async with get_db_session() as session:
                                contents = [r[0] for r in (await session.execute(
                                    select(Paragraph.content).where(
                                        Paragraph.source_id == source.id,
                                        Paragraph.outline.is_(None),
                                        func.btrim(func.coalesce(Paragraph.content, "")) != "",
                                    )
                                )).all()]
                            repair_blob = "".join(
                                _norm_page_text(c) for c in contents
                            )
                        self.log(
                            f"... reimport {len(page_range)} gap pages "
                            f"({len(repair_set)} interior)"
                        )
                elif not page_range:
                    # Legacy resume: continue after the max imported page
                    min_page = existent.get(filepath)
                    min_page = 0 if min_page is None else (min_page + 1)
                    self.log("... from page", min_page)
                    page_range = range(min_page, doc.page_count)

                # Accumulate pages for re-paragraphization
                page_buffer = []

                # --- Pre-extract text for the whole range (one pass) ---
                page_text: dict = {}
                for page in page_range:
                    if page >= doc.page_count:
                        break
                    try:
                        # Extract text content
                        page_text[page] = (
                            str(doc[page].get_text())
                            .encode("utf-8", errors="ignore")
                            .decode("utf-8")
                        )
                    except Exception as ex:
                        self.log(filepath, page + 1, ex)
                        page_text[page] = ""

                # --- Batched parallel OCR for text-less pages (scanned) ---
                ocr_text: dict = {}
                if ocr_client:
                    need = [
                        p for p, t in page_text.items()
                        if not t or len(t.strip()) < 10
                    ]
                    if need:
                        self.log(
                            f"{filepath}: OCR {len(need)}/{len(page_text)} "
                            f"textless pages, {self.ocr_workers} workers"
                        )
                        for start in range(0, len(need), 12):
                            chunk = need[start:start + 12]
                            imgs = []
                            for p in chunk:
                                try:
                                    imgs.append(
                                        (p, render_pdf_page(cast(Any, stream), page_num=p, zoom=2.0))
                                    )
                                except Exception as ex:
                                    self.log(f"render failed for {filepath} page {p + 1}: {ex}")
                                    ocr_text[p] = ""
                            if not imgs:
                                continue
                            with ThreadPoolExecutor(max_workers=self.ocr_workers) as pool:
                                futs = [
                                    (p, pool.submit(ocr_client.ocr_image, img, lang=lang))
                                    for p, img in imgs
                                ]
                                for p, fut in futs:
                                    try:
                                        ocr_text[p] = fut.result()
                                    except Exception as ex:
                                        self.log(f"OCR failed for {filepath} page {p + 1}: {ex}")
                                        ocr_text[p] = ""

                for page in page_range:
                    if page >= doc.page_count:
                        break

                    try:
                        # Get page label (e.g., "i", "1", "A-1")
                        label = doc[page].get_label()
                    except Exception:
                        # pymupdf raises FzErrorFormat (e.g. "cycle in page
                        # tree") on corrupt PDFs -- not RuntimeError/TypeError,
                        # it used to kill the whole import task here.
                        label = ""

                    content = page_text.get(page, "")

                    # OCR fallback: text-less pages served from batched results
                    if (not content or len(content.strip()) < 10) and ocr_client:
                        content = ocr_text.get(page, "")

                    # Skip pages with no content after OCR fallback
                    if not content or len(content.strip()) < 10:
                        continue
                    
                    # Create base paragraph
                    para = Paragraph(
                        lang=lang,
                        content=content,
                        source_id=source.id,
                        source_page=page,
                        pagenum=label or str(page + 1),
                        **author_kw,
                        **self.ext_params
                    )
                    
                    # Apply text cleaning
                    if text_cleaner:
                        para = cast(Paragraph, text_cleaner.resolve(para))
                        if para is None:  # Filtered as garbled
                            continue

                    # Apply language detection
                    if lang_detector:
                        para = cast(Paragraph, lang_detector.resolve(para))

                    # Gap repair: page N's text may already be stored
                    # inside an earlier row (cross-page reparagraphization
                    # merges page N+1 into page N's row) -- skip those to
                    # avoid duplicate content; genuine drops are new text.
                    if repair_blob and page in repair_set:
                        norm = _norm_page_text(para.content)
                        if norm and len(norm) >= 20 and (
                                norm in repair_blob
                                or norm[:40] in repair_blob
                                or norm[-40:] in repair_blob):
                            dup_skipped += 1
                            continue

                    # Buffer for re-paragraphization
                    if reparagraphize and page not in repair_set:
                        page_buffer.append((page, label, para.content))
                    else:
                        imported_pages += 1
                        if page in repair_set:
                            repaired_pages += 1
                        yield para

                # Process buffered pages for re-paragraphization
                if reparagraphizer and page_buffer:
                    for page, label, content in page_buffer:
                        for result in reparagraphizer.process_page(
                            page, label or str(page + 1), content, filepath, dataset.id
                        ):
                            imported_pages += 1
                            yield result
                    
                    # Finalize re-paragraphization
                    for result in reparagraphizer.finalize(filepath, dataset.id):
                        imported_pages += 1
                        yield result

            finally:
                doc.close()

            if self.reimport_gaps and (repaired_pages or dup_skipped):
                self.log(
                    f"reimport {filepath}: {repaired_pages} recovered, "
                    f"{dup_skipped} already-present skipped"
                )

            # Log if no sufficient content was found
            if not existent.get(filepath) and imported_pages == 0:
                self.log(f"no sufficient texts found in {filepath}")
