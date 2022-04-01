"""OCR 光学字符识别"""

from jindai.helpers import safe_import
from jindai.models import ImageItem

from .imageproc import ImageOrAlbumStage


class TesseractOCR(ImageOrAlbumStage):
    """基于 Tesseract 的光学字符识别
    """

    def __init__(self, langs) -> None:
        """

        Args:
            langs (str): 识别的语言，可用“,”连接多种语言。常用：chi_sim, chi_tra_vert, eng, rus, jpn。
        """
        super().__init__()
        self.langs = '+'.join([{
            'chs': 'chi_sim', 'cht': 'chi_tra',
            'cht-vert': 'chi_tra_vert', 'en': 'eng',
            'ru': 'rus', 'ja': 'jpn'}.get(l, l) for l in langs.split(',')])
        self.lang = langs.split(',')[0].split('-')[0]
        self.tesseract = safe_import('pytesseract')

    def resolve_image(self, i : ImageItem, _):
        """处理图像"""
        i.content = self.tesseract.image_to_string(i.image).encode("utf-8")
        i.lang = self.lang
        return i
