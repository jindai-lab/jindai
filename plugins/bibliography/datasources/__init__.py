"""Data sources for bibliography plugin.

This module provides data source implementations for importing bibliographic
data from various sources including Calibre and Zotero.
"""

from .calibre import CalibreDataSource
from .zotero import ZoteroDataSource

__all__ = ["CalibreDataSource", "ZoteroDataSource"]
