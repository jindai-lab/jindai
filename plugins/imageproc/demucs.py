"""Demucs"""

from jindai.helpers import safe_import
from jindai.models import MediaItem

from .imageproc import MediaItemStage


class DemucsStage(MediaItemStage):
    """Demucs
    """

    def __init__(self) -> None:
        """"""
        super().__init__()
        self.demucs = safe_import('demucs')

    def resolve_audio(self, i : MediaItem, _):
        """Handle audio file"""
        
        return i
