from pipeline import PipelineStage
from models import Paragraph
import os
import config


class JiayanWordCut(PipelineStage):

    tokenizer = None

    def __init__(self):
        from jiayan import load_lm, CharHMMTokenizer
        if JiayanWordCut.tokenizer is None:
            JiayanWordCut.tokenizer = CharHMMTokenizer(load_lm(os.path.join(config.rootpath, 'jiayan_models', 'jiayan.klm')))

    def resolve(self, p : Paragraph):
        p.tokens = list(JiayanWordCut.tokenizer.tokenize(p.content))


class JiayanPOSTagger(PipelineStage):

    postagger = None

    def __init__(self):
        from jiayan import CRFPOSTagger
        if JiayanPOSTagger.postagger is None:
            JiayanPOSTagger.postagger = CRFPOSTagger()
            JiayanPOSTagger.postagger.load(os.path.join(config.rootpath, 'jiayan_models', 'pos_model'))
        
    def resolve(self, p : Paragraph):
        p.pos = JiayanPOSTagger.postagger.postag(p.tokens)

