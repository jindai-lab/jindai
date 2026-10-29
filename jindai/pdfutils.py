"""PDF utilities for Jindai application.

This module provides functions for:
- Opening PDFs (path / bytes / file-like) and page-level access
  (page count, page text, page label)
- PDF page count extraction
- PDF rendering to images
- PDF conversion to TIFF
- Image merging into PDF
- PDF merging operations
- Unlocking "secured" PDFs

它是全站 PDF I/O 的**唯一出口**：``plugins/datasources/pdfdatasource.py``、
``jindai/storage.py``、``jindai/maintenance/manager.py`` 等调用方只允许 import
本模块，不要自己再 import fitz / PyPDF2 / pike。
"""

from __future__ import annotations

import os
import threading
from io import BytesIO
from typing import Any, cast

import fitz
from PIL import Image
from PyPDF2 import PdfReader, PdfWriter

# PyMuPDF/MuPDF is NOT thread-safe for concurrent use of one document (or of
# two documents built from the same stream): a 2026-10-16 import crash showed
# 4 OCR worker threads calling insert_pdf/get_pixmap on a shared doc, which
# produced "code=7: cannot find page" errors and finally a native
# "Segmentation fault (core dumped)" that killed the whole import round.
# Every fitz entry point here is therefore serialized. Rendering/extraction
# is cheap (~0.01s/page for page copies), so this costs nothing in practice.
#
# RLock (not Lock): `render_pdf_page` opens the doc via `open_pdf` while
# already holding the lock, and `convert_pdf_to_tiff_group4` re-enters through
# `page` accessors -- a plain Lock self-deadlocks.
_FITZ_LOCK = threading.RLock()


def open_pdf(pdf_path: str | bytes):
    """Open a PDF document from a path, bytes, or file-like object.

    Provides a single unified way to open PDFs across the codebase,
    accepting either a filesystem path, raw bytes, or an open file object.

    Args:
        pdf_path: Path to PDF file, raw PDF bytes, or file-like object.

    Returns:
        An open fitz.Document. Caller must close it.
    """
    with _FITZ_LOCK:
        if isinstance(pdf_path, str):
            return fitz.open(pdf_path)
        if hasattr(pdf_path, 'read'):
            # File-like object - fitz can read from a stream
            if hasattr(pdf_path, 'name'):
                return fitz.open(cast(Any, pdf_path).name)
            return fitz.open("pdf", cast(Any, pdf_path).read())
        return fitz.open(stream=pdf_path)


def page_count(doc: Any) -> int:
    """总页数（open_pdf 返回的 doc）。"""
    return doc.page_count


def extract_page_text(doc: Any, page_num: int) -> str:
    """抽取单页文本层；失败返回空串（坏页不该打断整本导入）。

    Args:
        doc: An open fitz.Document (opened by :func:`open_pdf`).
        page_num: 0-based page index.

    Returns:
        UTF-8 清洗后的文本；抽取异常时返回 ""。
    """
    with _FITZ_LOCK:
        try:
            return str(doc[page_num].get_text()).encode(
                "utf-8", errors="ignore").decode("utf-8")
        except Exception:
            return ""


def extract_page_label(doc: Any, page_num: int) -> str:
    """页标签（如 "i" / "1" / "A-1"）；坏 PDF 返回空串。

    pymupdf 在页树损坏时抛的是 FzErrorFormat（不是 RuntimeError/TypeError），
    以前会打死整个导入任务，所以这里统一吞掉异常。
    """
    with _FITZ_LOCK:
        try:
            return doc[page_num].get_label()
        except Exception:
            return ""


def get_pdf_page_count(pdf_path: str | bytes) -> int | None:
    """Get the number of pages in a PDF file.

    口径统一走 fitz（与 ``page_count(open_pdf(...))`` 一致），fitz 打不开时
    才回退 PyPDF2——两套库对坏 PDF 的页数判定可能不同，而盖章 / 页集续传都以
    这个页数为准，所以同进程内必须只有一个主口径。

    Args:
        pdf_path: Path to the PDF file (bytes / file-like also accepted).

    Returns:
        Number of pages, or None if the file cannot be read.
    """
    try:
        doc = open_pdf(pdf_path)
        try:
            return page_count(doc)
        finally:
            doc.close()
    except Exception:
        pass
    try:
        if isinstance(pdf_path, (bytes, bytearray)):
            return len(PdfReader(BytesIO(pdf_path)).pages)
        if hasattr(pdf_path, "read"):
            return len(PdfReader(cast(Any, pdf_path)).pages)
        with open(cast(str, pdf_path), "rb") as f:
            return len(PdfReader(f).pages)
    except Exception:
        return None


def render_pdf_page(pdf_path: str | bytes, page_num: int = 0, zoom: float = 2.0, format: str = "png") -> bytes:
    """Render a single PDF page to an image as raw bytes.

    This is the unified page-rendering helper used for OCR fallback
    and page previews. It accepts a path, raw bytes, or file-like object.

    Args:
        pdf_path: Path to PDF file, raw PDF bytes, or file-like object.
        page_num: Page number to render (0-indexed).
        zoom: Zoom factor for resolution (default: 2.0).
        format: Output image format (default: "png").

    Returns:
        Raw image bytes.
    """
    with _FITZ_LOCK:
        doc = open_pdf(pdf_path)
        try:
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            return pix.tobytes(format)
        finally:
            doc.close()


def extract_pages_pdf(doc: Any, pages: list[int]) -> bytes:
    """Build a PDF containing just the given pages (in order) of an open doc.

    Used to ship raw page data to the remote OCR service (fileType=0) instead
    of rasterizing on the client: the service renders with pypdfium2 at the
    same scale=2.0 we use, and a page copy is ~10x smaller than the rendered
    JPEG while preserving strokes the 1-bit binarization destroys.

    Args:
        doc: An open fitz.Document (opened by :func:`open_pdf`).
        pages: 0-based page indices, kept in the given order.

    Returns:
        Raw PDF bytes for the extracted pages.

    Raises:
        RuntimeError: If PyMuPDF cannot copy the requested pages.
    """
    if not pages:
        raise RuntimeError("no pages to extract")
    with _FITZ_LOCK:
        out = fitz.open()
        try:
            for p in pages:
                out.insert_pdf(doc, from_page=p, to_page=p)
            return out.tobytes()
        except Exception as ex:
            raise RuntimeError(f"page extraction failed: {ex!r}") from ex
        finally:
            out.close()


def extract_single_page_pdf(pdf_path: str, page_num: int = 0) -> bytes:
    """Build a one-page PDF from page ``page_num`` of ``pdf_path``.

    下载单页 / 预览用（原先 ``jindai/storage.py`` 内联 PyPDF2 复制了一份，
    2026-10-29 收口到本模块）。这里刻意走 PyPDF2 而不是 fitz：只需搬运页对象、
    不渲染，无 MuPDF 线程安全的顾虑。

    Args:
        pdf_path: PDF 文件路径。
        page_num: 0-based 页号。

    Returns:
        Bytes of a single-page PDF.

    Raises:
        ValueError: page 越界。
    """
    reader = PdfReader(pdf_path)
    if page_num < 0 or page_num >= len(reader.pages):
        raise ValueError("Page number out of range")
    writer = PdfWriter()
    writer.add_page(reader.pages[page_num])
    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()


def unlock_pdf(src) -> bytes:
    """Remove read protection from a "secured" PDF (原 `PDFUnlock` stage 的实现).

    用 pikepdf（不是 pyproject 里的 ``pike``：实测 venv 内 ``pike 0.2.0`` 是个
    空包，``dir(pike) == []``，旧实现 ``pike.open(...)`` 必抛 AttributeError，
    即该 stage 从未真正跑通过）。pikepdf 打开时自动解密"仅属主口令保护"的
    文档，``save()`` 不带 encryption 参数即重新写出无加密 PDF。

    Args:
        src: 文件路径 / bytes / file-like（``storage.open(...)`` 返回的流即可）。

    Returns:
        Unlocked PDF bytes.

    Raises:
        Exception: 打不开（需要用户口令 / 文件损坏）时原样抛出，由调用方决定跳过。
    """
    import pikepdf  # 惰性导入：只有解锁这条路径需要它

    with pikepdf.open(src) as pdf:
        buf = BytesIO()
        pdf.save(buf)
        return buf.getvalue()


def render_pdf_with_fitz(pdf_path: str | bytes, page_num: int = 0, format: str = "png") -> BytesIO:
    """Render a PDF page to an image using PyMuPDF (fitz).

    Wraps :func:`render_pdf_page` and returns the image as a BytesIO stream.

    Args:
        pdf_path: Path to PDF file or bytes.
        page_num: Page number to render (0-indexed).
        format: Output image format (default: "png").

    Returns:
        Image data as BytesIO.
    """
    return BytesIO(render_pdf_page(pdf_path, page_num=page_num, format=format))


def convert_pdf_to_tiff_group4(pdf, outp) -> None:
    """Convert a PDF to a 1-bit Group-4 raster PDF (historical name: "tiff").

    ⚠️ 函数名历史遗留：实际输出 **PDF**（``format="pdf"``），不是 TIFF。改名会
    牵动 ``jindai/maintenance/manager.py::ocr()`` 的 monochrome 分支，故保留。

    Args:
        pdf: PDF file object or path.
        outp: Output file object.
    """
    with _FITZ_LOCK:
        doc = fitz.open(pdf)
        image_list = []
        try:
            # Process each page
            for page in doc:
                # Render page to a pixmap (image representation)
                # Zoom factor increases resolution (e.g., 2 for 150-200 DPI approx, 4 for 300+ DPI)
                zoom = 4
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)

                # Convert pixmap to a Pillow Image
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

                # Convert to black and white (1-bit pixels) for efficient Group 4 compression
                # This is crucial for Group 4 compression to work effectively
                img = img.convert("1")
                image_list.append(img)
        finally:
            doc.close()

    if image_list:
        # Save all pages as a single multi-page TIFF with Group 4 compression
        image_list[0].save(
            outp,
            format="pdf",
            save_all=True,
            append_images=image_list[1:],
            compression="group4",
            dpi=(300, 300),  # Set desired DPI
        )


def merge_images_from_folder(folderpath: str, outp) -> int:
    """Merge images from a folder into a single PDF.

    Args:
        folderpath: Path to folder containing images.
        outp: Output file object.

    Returns:
        Number of images merged.
    """
    image_list = []
    for f in sorted(os.listdir(folderpath)):
        if f.endswith((".jpg", ".png", ".jpeg", ".tif", ".tiff")):
            im = Image.open(os.path.join(folderpath, f))
            image_list.append(im)
    if image_list:
        image_list[0].save(
            outp,
            format="pdf",
            save_all=True,
            append_images=image_list[1:],
            compression="group4",
            dpi=(300, 300),  # Set desired DPI
        )
    return len(image_list)


def read_pdf_pages(path: str, reverse: bool = False) -> list:
    """Read pages from a PDF file.

    Args:
        path: Path to PDF file.
        reverse: Whether to reverse page order.

    Returns:
        List of page objects.
    """
    reader = PdfReader(path)
    return list(reversed(reader.pages) if reverse else reader.pages)


def cross_merge_pdf(outp, pdf1, pdf2, reversed1: bool, reversed2: bool) -> None:
    """Cross merge two PDF files (alternating pages).

    Args:
        outp: Output file object.
        pdf1: First PDF file path.
        pdf2: Second PDF file path.
        reversed1: Whether to reverse first PDF.
        reversed2: Whether to reverse second PDF.

    Raises:
        ValueError: If PDFs have different page counts.
    """
    try:
        pdf1_pages, pdf2_pages = (
            read_pdf_pages(pdf1, reversed1),
            read_pdf_pages(pdf2, reversed2),
        )

        # Check if page counts match
        if len(pdf1_pages) != len(pdf2_pages):
            raise ValueError("Two PDF files have different page counts and cannot be merged")

        # Create output PDF
        pdf_writer = PdfWriter()

        # Cross merge pages
        for page1, page2 in zip(pdf1_pages, pdf2_pages):
            pdf_writer.add_page(page1)
            pdf_writer.add_page(page2)

        # Write to output
        pdf_writer.write(outp)

    except Exception as e:
        raise e


def sequential_merge_pdf(outp, pdf1, pdf2, reversed1: bool, reversed2: bool) -> None:
    """Sequentially merge two PDF files.

    Args:
        outp: Output file object.
        pdf1: First PDF file path.
        pdf2: Second PDF file path.
        reversed1: Whether to reverse first PDF.
        reversed2: Whether to reverse second PDF.
    """
    try:
        pdf1_pages, pdf2_pages = (
            read_pdf_pages(pdf1, reversed1),
            read_pdf_pages(pdf2, reversed2),
        )

        # Create output PDF
        pdf_writer = PdfWriter()

        # Sequential merge pages
        for page in pdf1_pages + pdf2_pages:
            pdf_writer.add_page(page)

        # Write to output
        pdf_writer.write(outp)

    except Exception as e:
        raise e
