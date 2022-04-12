"""机器翻译
"""
from jindai.helpers import safe_import
from jindai.models import Paragraph
from jindai import PipelineStage, Plugin


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
        t = self.model.translate(paragraph.content, source_lang=paragraph.lang if paragraph.lang not in (
            'chs', 'cht') else 'zh', target_lang=self.to_lang)
        if self.opencc:
            t = self.opencc.convert(t)
        paragraph.content = t
        if paragraph.id:
            paragraph._id = None
            del paragraph._orig['_id']
        return paragraph


class MachineTranslationPlugin(Plugin):
    """Plugin for machin translations
    """

    def __init__(self, pmanager, **config):
        super().__init__(pmanager, **config)
        self.register_pipelines(globals())
