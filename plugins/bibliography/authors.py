"""Author name normalization and file name helpers for the Bibliography Plugin.

These pure helpers back the PDF upload endpoint:

- normalize_author_name: Normalize a personal name for storage and for
  building the ``Authors/<author>/`` directory hierarchy.
- normalize_authors_directory: Build a directory name from a list of authors.
- extract_default_title: Extract the default publication title from a PDF
  file name.
- sanitize_filename: Make an arbitrary file name safe for use on disk while
  preserving non-Latin (e.g. Chinese/Japanese) characters.

Author normalization rules:

- Chinese/Japanese names are kept as-is.
- Other names are converted to ``<Family>, <Given>`` form, e.g.
  ``John Cage`` -> ``Cage, John`` (the last whitespace-separated token is
  treated as the family name).  Names that already contain a comma are
  assumed to be normalized already and are kept unchanged.
- In the final directory-safe form every punctuation/symbol character except
  the comma is replaced with ``_`` (e.g. ``Donald E. Knuth`` ->
  ``Knuth, Donald E_``).
"""

import os
import re
from typing import Iterable, List

# CJK code point ranges covering Chinese characters (Unified Ideographs and
# extensions), Japanese kana and CJK punctuation/symbols.
_CJK_RANGES = (
    (0x2E80, 0x2EFF),  # CJK Radicals Supplement
    (0x2F00, 0x2FDF),  # Kangxi Radicals
    (0x3000, 0x303F),  # CJK Symbols and Punctuation
    (0x3040, 0x309F),  # Hiragana
    (0x30A0, 0x30FF),  # Katakana
    (0x31C0, 0x31EF),  # CJK Strokes
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0xFF66, 0xFF9F),  # Halfwidth and Fullwidth Forms (kana part)
)


def contains_cjk(text: str) -> bool:
    """Check whether a string contains Chinese/Japanese characters.

    Args:
        text: Input string.

    Returns:
        True if at least one character falls into a CJK range.
    """
    return any(
        any(start <= ord(ch) <= end for start, end in _CJK_RANGES)
        for ch in (text or "")
    )


def normalize_author_name(name: str) -> str:
    """Normalize a single author name.

    Chinese/Japanese names are returned unchanged.  Other names are
    reordered to ``<Family>, <Given>``.  Punctuation other than the comma is
    replaced with ``_`` so the result is safe as a path segment.

    Args:
        name: Author name as entered by the user.

    Returns:
        Normalized author name ("" for empty input).

    Examples:
        >>> normalize_author_name("John Cage")
        'Cage, John'
        >>> normalize_author_name("宇野常寛")
        '宇野常寛'
        >>> normalize_author_name("Donald E. Knuth")
        'Knuth, Donald E_'
    """
    name = re.sub(r"\s+", " ", (name or "").strip())
    if not name:
        return ""

    if "," in name or contains_cjk(name):
        # Already in "<Family>, <Given>" form, or a Chinese/Japanese name:
        # keep as-is.
        normalized = name
    else:
        parts = name.split(" ")
        if len(parts) == 1:
            normalized = name
        else:
            # Treat the last whitespace-separated token as the family name.
            normalized = f"{parts[-1]}, {' '.join(parts[:-1])}"

    # Replace punctuation/symbols with "_" (the comma is intentionally kept).
    normalized = "".join(
        ch
        if (ch == "," or ch.isspace() or ch.isalnum() or contains_cjk(ch))
        else "_"
        for ch in normalized
    )
    # Collapse runs of underscores produced by consecutive punctuation marks.
    normalized = re.sub(r"_{2,}", "_", normalized)
    return normalized.strip()


def normalize_authors(authors: Iterable[str]) -> List[str]:
    """Normalize a list of author names, dropping empty results.

    Args:
        authors: Author names as entered by the user.

    Returns:
        List of normalized author names.
    """
    normalized = [normalize_author_name(a) for a in (authors or [])]
    return [a for a in normalized if a]


def normalize_authors_directory(authors: Iterable[str]) -> str:
    """Build the ``Authors/<...>`` directory name for a list of authors.

    Multiple authors are joined with ", " (the comma is the only punctuation
    kept by the normalization rules).  Falls back to "Unknown" when no valid
    author name is provided.

    Args:
        authors: Author names as entered by the user.

    Returns:
        Directory name segment.
    """
    normalized = normalize_authors(authors)
    return ", ".join(normalized) if normalized else "Unknown"


def normalize_first_author_directory(authors: Iterable[str]) -> str:
    """Build the ``Authors/<...>`` directory name from the FIRST author.

    Only the first valid author name is used for the directory (e.g.
    ``["John Cage", "宇野常寛"]`` -> ``Cage, John``).  Falls back to
    "Unknown" when no valid author name is provided.

    Args:
        authors: Author names as entered by the user.

    Returns:
        Directory name segment.
    """
    for author in authors or []:
        normalized = normalize_author_name(author)
        if normalized:
            return normalized
    return "Unknown"


def extract_default_title(filename: str) -> str:
    """Extract the default publication title from a PDF file name.

    The title is the file name without its ``.pdf`` extension, with
    whitespace collapsed.

    Args:
        filename: Uploaded file name (e.g. ``Silence_ Lectures.pdf``).

    Returns:
        Default title ("" if nothing is left after stripping).
    """
    name = os.path.basename(filename or "").strip()
    stem = re.sub(r"\s*\.pdf$", "", name, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", stem).strip()


def sanitize_filename(name: str, max_length: int = 150) -> str:
    """Make a file name (without extension) safe for use on disk.

    Path separators, control characters and characters that are invalid on
    common file systems are replaced with ``_``.  Non-Latin characters
    (Chinese/Japanese etc.) are preserved.

    Args:
        name: File name or stem.
        max_length: Maximum length of the returned stem.

    Returns:
        Sanitized stem (never empty).
    """
    name = os.path.basename(name or "").strip()
    name = re.sub(r"[\x00-\x1f\x7f]", "_", name)  # control characters
    name = re.sub(r'[\\/:*?"<>|]', "_", name)  # file-system unsafe characters
    name = re.sub(r"\s+", " ", name).strip(" .")
    if len(name) > max_length:
        name = name[:max_length].rstrip(" ._")
    return name or "untitled"
