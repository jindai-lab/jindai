from PIL import Image
import pytesseract
from pipeline import PipelineStage


class TesseractOCR(PipelineStage):
    """基于 Tesseract 的光学字符识别
    """

    def __init__(self, langs) -> None:
        """

        Args:
            langs (str): 识别的语言，可用“,”连接多种语言。常用：chi_sim, chi_tra_vert, eng, rus, jpn。
        """
        self.langs = langs.replace(',', '+')
        self.lang = langs.split(',')[0]
        self.lang = {'chi_sim': 'chs',
                     'chi_tra': 'cht',
                     'chi_tra_vert': 'cht', 
                     'jpn': 'ja'}.get(self.lang, self.lang[:2])

    def resolve(self, p):
        p.content = pytesseract.image_to_string(p.image).encode("utf-8")
        p.lang = self.lang
        return p
