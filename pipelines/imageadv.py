"""图像高级处理
"""

import struct
import imagehash
from pipeline import PipelineStage
import numpy as np
from PIL import Image, ImageOps
from models import Paragraph


def dhash(im):
    dh = imagehash.dhash(im)
    dh = bytes.fromhex(str(dh))
    return dh


def whash(im):
    dh = imagehash.whash(im)
    dh = bytes.fromhex(str(dh))
    return dh


def bitcount(x):
    return bin(x).count('1')


class Hashing(PipelineStage):
    """图像哈希
    """

    def __init__(self):
        if 'dhash' not in Paragraph.fields:
            Paragraph._fields.append('dhash')
        if 'whash' not in Paragraph.fields:
            Paragraph._fields.append('whash')

    def resolve(self, p):
        dh, wh = p.dhash, p.whash
        if dh and wh: return
        if not f: return

        if not dh:
            f.seek(0)
            im = Image.open(f)
            p.dhash = dhash(im)
            
        if not wh:
            f.seek(0)
            im = Image.open(f)
            p.whash = whash(im)


class AutoRating(PipelineStage):
    """应用特定模型对图像进行打分
    """

    def __init__(self, target_field='rating', model='autorating_best_state.pth'):
        from autorating.inference_model import InferenceModel, load_state
        state = load_state(os.path.join(config.rootpath, 'models_data', model))
        self.model = InferenceModel(state)
        self.target_field = target_field

    def resolve(self, p):
        setattr(p, self.target_field, float(model.predict_from_pil_image(p.image).numpy()[0]))
        return p
