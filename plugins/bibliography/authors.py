"""Author name helpers for the Bibliography Plugin.

Two layers of "normalization" live here, with different rules:

- clean_authors: Clean an author list FOR DATABASE STORAGE.  Only
  whitespace is collapsed and empty entries are dropped; every symbol
  (``.``, ``,``, ``'``, ``-`` ...) is preserved exactly as entered.
- normalize_author_name / normalize_authors_directory /
  normalize_first_author_directory: Build PATH SEGMENTS (the per-author
  directory under the configurable upload base directory, default
  ``OneDrive/``).  Only the path needs to be normalize'd: names are
  reordered to natural order and file-system-hostile characters are
  replaced so the segment is safe on disk.

Other helpers:

- extract_default_title: Extract the default publication title from a PDF
  file name (a trailing ``_cropped`` suffix is dropped).
- sanitize_filename: Make an arbitrary file name safe for use on disk while
  preserving non-Latin (e.g. Chinese/Japanese) characters.

Path normalization rules (normalize_author_name):

- Chinese/Japanese names are kept as-is.
- Names written as ``<Family>, <Given>`` are converted to natural
  ``<Given> <Family>`` order (e.g. ``Cage, John`` -> ``John Cage``);
  names already in natural order are kept unchanged (``John Cage`` ->
  ``John Cage``).  This keeps the per-author directories under the
  configurable upload base dir in natural-name order, matching the
  existing folder naming across the storage tree.
- In the final directory-safe form characters outside the path whitelist
  are replaced with ``_`` while ``.`` and ``,`` are kept (they are legal
  on disk and match the existing folder naming, e.g. ``Donald E. Knuth``
  -> ``Donald E. Knuth``, but ``A/B`` -> ``A_B``).
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


def clean_authors(authors: Iterable[str]) -> List[str]:
    """Clean author names FOR DATABASE STORAGE, keeping every symbol.

    Only whitespace is collapsed, entries are stripped and empty entries
    are dropped.  Unlike the path helpers below, punctuation such as ``.``
    and ``,`` is preserved exactly as entered (``Donald E. Knuth`` stays
    ``Donald E. Knuth``, ``Cage, John`` stays ``Cage, John``).

    Args:
        authors: Author names as entered by the user.

    Returns:
        Cleaned list of author names (may be empty).

    Examples:
        >>> clean_authors(["  Donald E. Knuth ", "", "Cage, John"])
        ['Donald E. Knuth', 'Cage, John']
        >>> clean_authors(["李  飞飞"])
        ['李 飞飞']
    """
    cleaned = [re.sub(r"\s+", " ", (a or "").strip()) for a in (authors or [])]
    return [a for a in cleaned if a]


def normalize_author_name(name: str) -> str:
    """Normalize a single author name.

    Chinese/Japanese names are returned unchanged.  Names in
    ``<Family>, <Given>`` form are converted to natural
    ``<Given> <Family>`` order; names already in natural order are kept
    as-is.  Characters outside the path whitelist (letters, digits,
    spaces, ``.``, ``,``) are replaced with ``_`` so the result is safe
    as a path segment.

    Args:
        name: Author name as entered by the user.

    Returns:
        Normalized author name ("" for empty input).

    Examples:
        >>> normalize_author_name("John Cage")
        'John Cage'
        >>> normalize_author_name("Cage, John")
        'John Cage'
        >>> normalize_author_name("宇野常寛")
        '宇野常寛'
        >>> normalize_author_name("Donald E. Knuth")
        'Donald E. Knuth'
        >>> normalize_author_name("A/B Test")
        'A_B Test'
    """
    name = re.sub(r"\s+", " ", (name or "").strip())
    if not name:
        return ""

    normalized = name
    if not contains_cjk(name) and ',' in name:
        parts = name.split(", ", 1)
        if len(parts) == 1:
            normalized = name
        else:
            # Treat the last whitespace-separated token as the family name.
            normalized = f"{parts[1]} {parts[0]}"

    # Path-segment whitelist: keep letters/digits/spaces, "." and ","
    # (legal on disk and used throughout the existing folder naming, e.g.
    # "Donald E. Knuth"); replace everything else (e.g. "/" ":" "*") with
    # "_" so the segment is safe as a path.
    normalized = "".join(
        ch
        if (ch in ".," or ch.isspace() or ch.isalnum() or contains_cjk(ch))
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
    """Build a directory name from a list of authors.

    Multiple authors are joined with ", " (the comma is the only punctuation
    kept by the normalization rules).  Falls back to "Unknown" when no valid
    author name is provided.

    Note: PDF uploads / relocations use :func:`normalize_first_author_directory`
    (first author only); this all-authors variant is kept for other callers.

    Args:
        authors: Author names as entered by the user.

    Returns:
        Directory name segment.
    """
    normalized = normalize_authors(authors)
    return ", ".join(normalized) if normalized else "Unknown"


def normalize_first_author_directory(authors: Iterable[str]) -> str:
    """Build the per-author directory name from the FIRST author.

    The result becomes the directory segment below the configurable upload
    base directory (default ``OneDrive/``, i.e. ``/storage/OneDrive/`` in
    the container).

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
    whitespace collapsed.  A trailing ``_cropped`` suffix (the marker of
    cropped scans, e.g. ``Paper_cropped.pdf``) is dropped so the upload
    form is pre-filled with a clean title the user can edit.

    Args:
        filename: Uploaded file name (e.g. ``Silence_ Lectures.pdf``).

    Returns:
        Default title ("" if nothing is left after stripping).
    """
    name = os.path.basename(filename or "").strip()
    stem = re.sub(r"\s*\.pdf$", "", name, flags=re.IGNORECASE)
    # Drop a trailing "_cropped" / " cropped" suffix (possibly repeated).
    stem = re.sub(r"(?i)(?:[_\s]cropped)+$", "", stem).strip(" _")
    stem = stem.replace('_', ' ')
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
