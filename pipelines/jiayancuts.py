from pipeline import PipelineStage
from models import Paragraph
from helpers import safe_import
import os
import config


jiayan = safe_import('jiayan')
safe_import('kenlm', 'https://github.com/kpu/kenlm/archive/master.zip')
load_lm = jiayan.load_lm
CharHMMTokenizer = jiayan.CharHMMTokenizer


class JiayanWordCut(PipelineStage):

    tokenizer = None

    def __init__(self):
        if JiayanWordCut.tokenizer is None:
            JiayanWordCut.tokenizer = CharHMMTokenizer(load_lm(os.path.join(config.rootpath, 'models_data', 'jiayan.klm')))

    def resolve(self, p : Paragraph):
        p.tokens = list(JiayanWordCut.tokenizer.tokenize(p.content))


class JiayanPOSTagger(PipelineStage):

    postagger = None

    def __init__(self):
        from jiayan import CRFPOSTagger
        if JiayanPOSTagger.postagger is None:
            JiayanPOSTagger.postagger = CRFPOSTagger()
            JiayanPOSTagger.postagger.load(os.path.join(config.rootpath, 'models_data', 'pos_model'))
        
    def resolve(self, p : Paragraph):
        p.pos = JiayanPOSTagger.postagger.postag(p.tokens)

