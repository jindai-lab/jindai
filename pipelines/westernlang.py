from models import Paragraph
from pipeline import PipelineStage
from .utils import language_iso639

import re
spliter = re.compile(r'[^\w]')


class WesternStemmer(PipelineStage):
    """附加词干到 tokens 中（需要先进行切词）
    """

    _language_stemmers = {}

    @staticmethod
    def get_stemmer(lang):
        from nltk.stem.snowball import SnowballStemmer 
        if lang not in WesternStemmer._language_stemmers:
            stemmer = SnowballStemmer(language_iso639.get(lang, lang).lower())
            WesternStemmer._language_stemmers[lang] = stemmer
        return WesternStemmer._language_stemmers[lang]

    def __init__(self, append=True):
        """
        Args:
            append (bool): 将词干添加到结尾，否则直接覆盖
        """
        self.append = append

    def resolve(self, p : Paragraph) -> Paragraph:
        tokens = [WesternStemmer.get_stemmer(p.lang).stem(_) for _ in p.tokens]
        if self.append:
            p.tokens += tokens
        else:
            p.tokens = tokens
        return p


class RussianTransliterate(PipelineStage):
    """转写为拉丁字母的单词（需要先进行切词）
    """
    def __init__(self, language, append=True):
        """
        Args:
            append (bool): 是添加到结尾还是覆盖
        """
        self.append = append
        
    def resolve(self, p : Paragraph) -> Paragraph:
        from transliterate import translit
        tokens = [translit(_, 'ru', reversed=True).lower() for _ in p.tokens]
        if self.append:
            p.tokens += tokens
        else:
            p.tokens = tokens
        return p
