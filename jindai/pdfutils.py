"""PDF utilities for Jindai application.

This module provides functions for:
- PDF page count extraction
- PDF rendering to images
- PDF conversion to TIFF
- Image merging into PDF
- PDF text extraction
- PDF merging operations
"""

import os
from io import BytesIO
from typing import Iterator

import fitz
from PIL import Image
from PyPDF2 import PdfReader, PdfWriter


def get_pdf_page_count(pdf_path: str) -> int | None:
    """Get the number of pages in a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Number of pages, or None if the file cannot be read.
    """
    try:
        from PyPDF2 import PdfReader

        with open(pdf_path, "rb") as f:
            return len(PdfReader(f).pages)
    except Exception:
        return None


def render_pdf_with_fitz(pdf_path: str | bytes, page_num: int = 0, format: str = "png") -> BytesIO:
    """Render a PDF page to an image using PyMuPDF (fitz).

    Args:
        pdf_path: Path to PDF file or bytes.
        page_num: Page number to render (0-indexed).
        format: Output image format (default: "png").

    Returns:
        Image data as BytesIO.
    """
    # Open PDF
    if isinstance(pdf_path, str):
        doc = fitz.open(pdf_path)
    else:
        doc = fitz.open(stream=pdf_path)
    # Get specified page (note: fitz is 0-indexed)
    page = doc.load_page(page_num)

    # Render page to pixel map (PixMap)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for higher resolution

    # Convert to byte stream
    img_data = pix.tobytes(format)
    return BytesIO(img_data)


def convert_pdf_to_tiff_group4(pdf, outp) -> None:
    """Convert a PDF file to a multi-page Group 4 compressed TIFF file.

    Args:
        pdf: PDF file object or path.
        outp: Output file object for the TIFF.
    """
    doc = fitz.open(pdf)
    image_list = []

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


def extract_pdf_texts(filename: str, since: int = 0) -> Iterator:
    """Extract text from PDF pages.

    Args:
        filename: Path to PDF file.
        since: Start page number (0-indexed).

    Yields:
        Tuples of (page_number, label, content).
    """
    doc = fitz.open(filename)

    for page in range(since, doc.page_count):
        try:
            label = doc[page].get_label()
        except (RuntimeError, TypeError):
            label = ""

        try:
            content = doc[page].get_text()
        except Exception as ex:
            content = ""

        yield page, label, content
