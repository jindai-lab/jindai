from models import Paragraph
from pipeline import PipelineStage

import re
spliter = re.compile(r'[^\w]')


class WesternStemmer(PipelineStage):
    """附加词干到 tokens 中（需要先进行切词）
    """

    def __init__(self, language, append=True):
        """
        Args:
            language (Arabic|Danish|Dutch|English|Finnish|French|German|Hungarian|Italian|Norwegian|Portuguese|Romanian|Russian|Spanish|Swedish): 指定语言
            append (bool): 是添加到结尾还是覆盖
        """
        from nltk.stem.snowball import SnowballStemmer 
        self.stemmer = SnowballStemmer(language.lower())
        self.append = append

    def resolve(self, p : Paragraph) -> Paragraph:
        tokens = [self.stemmer.stem(_) for _ in p.tokens]
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
