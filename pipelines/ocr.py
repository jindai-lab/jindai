import pytesseract
from imageproc import ImageOrAlbumStage
from models import ImageItem


class TesseractOCR(ImageOrAlbumStage):
    """基于 Tesseract 的光学字符识别
    """

    def __init__(self, langs) -> None:
        """

        Args:
            langs (str): 识别的语言，可用“,”连接多种语言。常用：chi_sim, chi_tra_vert, eng, rus, jpn。
        """
        self.langs = '+'.join([{'chs': 'chi_sim', 'cht': 'chi_tra', 'cht-vert': 'chi_tra_vert', 'en': 'eng', 'ru': 'rus', 'ja': 'jpn'}.get(l, l) for l in langs.split(',')])
        self.lang = langs.split(',')[0].split('-')[0]

    def resolve_image(self, p : ImageItem):
        p.content = pytesseract.image_to_string(p.image).encode("utf-8")
        p.lang = self.lang
        return p
