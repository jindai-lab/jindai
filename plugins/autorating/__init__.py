import os
from PIL import Image
import config
from helpers import safe_import
from storage import expand_path
from plugin import Plugin
from pipelines.imageproc import ImageOrAlbumStage
from models import ImageItem
from .inference_model import InferenceModel, load_state

model = None


def predict(i):
    global model
    if model is None:
        state = load_state(expand_path('models_data/autorating_best.pth'))
        model = InferenceModel(state)
       
    return float(model.predict_from_pil_image(i).numpy()[0])


class AutoRating(ImageOrAlbumStage):
    """使用自训练模型进行打分"""
        
    def resolve_image(self, i: ImageItem, context):
        i.rating = predict(i.image)
        i.save()
        return i


class OpenNsfw(ImageOrAlbumStage):
    """使用 OpenNSFW 模型进行打分"""

    def __init__(self):
        safe_import('opennsfw_standalone', 'opennsfw-standalone')
        from opennsfw_standalone import OpenNSFWInferenceRunner
        self.runner = OpenNSFWInferenceRunner.load()
        ImageItem.set_field('nsfw', float)
        
    def resolve_image(self, i: ImageItem, context):
        i.nsfw = float(self.runner.infer(i.image_raw.read()))
        i.save()
        return i
        

class AutoRatingPlugin(Plugin):

    def __init__(self, app, **config) -> None:
        super().__init__(app, **config)
        self.register_pipelines(globals())
