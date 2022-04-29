"""机器翻译
"""
import requests

from jindai.helpers import safe_import
from jindai.models import Paragraph
from jindai import PipelineStage, Plugin


class RemoteTranslation(PipelineStage):
    """调用远程 API 进行机器翻译"""

    def __init__(self, translator_url, to_lang='chs'):
        """
        Args:
            to_lang (LANG): 目标语言标识
            translator_url (str): 远程 API 网址
        """
        super().__init__()
        self.url = translator_url
        self.to_lang = 'ZH' if to_lang in (
            'chs', 'cht') else to_lang.upper()

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        """机器翻译段落"""
        if not paragraph.content.strip():
            return None

        resp = requests.post(self.url, json={
                             'text': paragraph.content, 'source_lang': paragraph.lang.upper() if paragraph.lang != 'auto' else 'auto', 'target_lang': self.to_lang})
        result = (resp.json() or {}).get('data')
        paragraph.content = result
        paragraph.id = None
        return paragraph


class MachineTranslation(PipelineStage):
    """机器翻译"""

    def __init__(self, to_lang='chs', model='opus-mt') -> None:
        """
        Args:
            to_lang (LANG): 目标语言标识
            model (较快速度:opus-mt|较高准确度:mbart50_m2m): 机器翻译所使用的模型
        """
        super().__init__()

        self.model = safe_import('easynmt').EasyNMT(model)

        self.opencc = None
        if to_lang == 'chs':
            to_lang = 'zh'
        elif to_lang == 'cht':
            to_lang = 'zh'
            self.opencc = safe_import(
                'opencc', 'opencc-python-reimplemented').OpenCC('s2t')

        self.to_lang = to_lang

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        """处理段落"""
        result = self.model.translate(paragraph.content, source_lang=paragraph.lang if paragraph.lang not in (
            'chs', 'cht') else 'zh', target_lang=self.to_lang)
        if self.opencc:
            result = self.opencc.convert(result)
        paragraph.content = result
        paragraph.id = None
        return paragraph


class MachineTranslationPlugin(Plugin):
    """Plugin for machin translations
    """

    def __init__(self, pmanager, **config):
        super().__init__(pmanager, **config)
        self.register_pipelines(globals())
