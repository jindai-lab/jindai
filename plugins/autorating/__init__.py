"""自动图像评分"""

from PIL.Image import Image
from jindai import Plugin, expand_path
from jindai.helpers import safe_import
from jindai.models import ImageItem
from plugins.gallery import ImageOrAlbumStage

from .inference_model import InferenceModel, load_state

try:
    import torch
except ImportError as e:
    print("Please install pytorch first.")
    raise e


class AutoRating(ImageOrAlbumStage):
    """使用自训练模型进行打分"""

    def __init__(self) -> None:
        super().__init__()
        state = load_state(expand_path('models_data/autorating_best.pth'))
        self.model = InferenceModel(state)

    def predict(self, i: Image):
        """获取图像评分"""
        return float(self.model.predict_from_pil_image(i).numpy()[0])

    def resolve_image(self, i: ImageItem, _):
        i.rating = self.predict(i.image)
        i.save()
        return i


class OpenNsfw(ImageOrAlbumStage):
    """使用 OpenNSFW 模型进行打分"""

    def __init__(self):
        super().__init__()
        self.runner = safe_import(
            'opennsfw_standalone', 'opennsfw-standalone').OpenNSFWInferenceRunner.load()
        ImageItem.set_field('nsfw', float)

    def resolve_image(self, i: ImageItem, _):
        i.nsfw = float(self.runner.infer(i.image_raw.read()))
        i.save()
        return i


class AutoRatingPlugin(Plugin):
    """自动评分插件"""

    def __init__(self, app, **config) -> None:
        super().__init__(app, **config)
        self.register_pipelines(globals())
