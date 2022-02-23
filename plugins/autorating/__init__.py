import os
from PIL import Image
import config
from plugin import Plugin

from pipelines.imageproc import ImageOrAlbumStage
from models import ImageItem
from .inference_model import InferenceModel, load_state

model = None


def predict(i):
    global model
    if model is None:
        state = load_state(os.path.join(config.rootpath, 'models_data', 'autorating_best.pth'))
        model = InferenceModel(state)
       
    return float(model.predict_from_pil_image(i).numpy()[0])


class AutoRating(ImageOrAlbumStage):
    """使用自训练模型进行打分"""
        
    def resolve_image(self, i: ImageItem):
        i.rating = predict(i.image)
        i.save()
        return i


class OpenNsfw(ImageOrAlbumStage):
    """使用 OpenNSFW 模型进行打分"""

    def __init__(self):
        from opennsfw_standalone import OpenNSFWInferenceRunner
        self.runner = OpenNSFWInferenceRunner.load()
        
    def resolve_image(self, i: ImageItem):
        i.nsfw = float(self.runner.infer(i.image_raw.read()))
        i.save()
        return i
        

class AutoRatingPlugin(Plugin):

    def __init__(self, app, **config) -> None:
        super().__init__(app, **config)
        super().register_pipelines('自动打分', globals())
