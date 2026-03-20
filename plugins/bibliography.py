"""Bibliography Plugin Registration for Jindai Application.

This module registers the bibliography plugin and its components with the
plugin system. It provides:
- BibliographyPlugin: Main plugin class for managing bibliographic items
- Data sources: CalibreDataSource, ZoteroDataSource
- Pipeline stages: BibItemSave, BibItemDeduplicate
- Models: BibItem ORM model
- Deduplicator: BibItemDeduplicator utility class
"""

import os

from jindai.pipeline import PipelineStage
from jindai.plugin import Plugin
from jindai.helpers import get_context

# Import all components for registration
from .bibliography.models import BibItem
from .bibliography.datasources import CalibreDataSource, ZoteroDataSource
from .bibliography.deduplicator import BibItemDeduplicator
from .bibliography.stages import BibItemSave, BibItemDeduplicate


class BibliographyPlugin(Plugin):
    """Plugin for managing bibliographic items.
    
    Provides:
    - Full CRUD operations for BibItem records via API
    - Data sources for importing from Calibre and Zotero
    - Deduplication utilities for merging duplicate entries
    - Pipeline stages for saving Paragraphs to BibItems
    """
    
    def __init__(self, pmanager, **config) -> None:
        """Initialize the BibliographyPlugin.
        
        Args:
            pmanager: The pipeline manager instance.
            **config: Additional configuration options passed to the parent Plugin.
        """
        super().__init__(pmanager, **config)
        ctx = get_context(os.path.join('plugins', 'bibliography'), PipelineStage)
        self.register_pipelines(ctx)


def prepare_bibliography_plugin() -> dict:
    """Prepare the bibliography plugin for registration.
    
    Returns:
        Dictionary mapping plugin name to class.
    """
    return {
        "BibliographyPlugin": BibliographyPlugin,
        "CalibreDataSource": CalibreDataSource,
        "ZoteroDataSource": ZoteroDataSource,
        "BibItemSave": BibItemSave,
        "BibItemDeduplicate": BibItemDeduplicate,
    }
