from pipeline import PipelineStage
from plugin import Plugin
from models import Paragraph
from helpers import safe_import
from storage import expand_path
import os
import config

_init = lambda *x: x


class JiayanWordCut(PipelineStage):

    tokenizer = None

    def __init__(self):
        if JiayanWordCut.tokenizer is None:
            JiayanWordCut.tokenizer = _init(expand_path('models_data/jiayan.klm'))

    def resolve(self, p : Paragraph):
        p.tokens = list(JiayanWordCut.tokenizer.tokenize(p.content))


class JiayanPOSTagger(PipelineStage):

    postagger = None

    def __init__(self):
        from jiayan import CRFPOSTagger
        if JiayanPOSTagger.postagger is None:
            JiayanPOSTagger.postagger = CRFPOSTagger()
            JiayanPOSTagger.postagger.load(expand_path('models_data/pos_model'))
        
    def resolve(self, p : Paragraph):
        p.pos = JiayanPOSTagger.postagger.postag(p.tokens)


class JiayanPlugin(Plugin):
    
    def __init__(self, app, **config) -> None:
        global _init
        
        safe_import('kenlm', 'https://github.com/kpu/kenlm/archive/master.zip')
        jiayan = safe_import('jiayan')
        load_lm = jiayan.load_lm
        CharHMMTokenizer = jiayan.CharHMMTokenizer
        _init = lambda *x: CharHMMTokenizer(load_lm(*x))
        
        self.register_pipelines(globals())
        super().__init__(app, **config)
