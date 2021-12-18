"""基于神经网络模型的图像处理
"""

from models import ImageItem
from pipelines.imageproc import ImageOrAlbumStage


class OpenNsfw(ImageOrAlbumStage):
    """使用 OpenNSFW 模型进行打分"""

    def __init__(self):
        from opennsfw_standalone import OpenNSFWInferenceRunner
        self.runner = OpenNSFWInferenceRunner.load()
        
    def resolve_image(self, i: ImageItem):
        i.nsfw = float(self.runner.infer(i.image_raw.read()))
        i.save()
        return i
        

class AutoRating(ImageOrAlbumStage):
    """使用自训练模型进行打分"""
        
    def resolve_image(self, i: ImageItem):
        from plugins.autorating import predict
        i.rating = predict(i.image)
        i.save()
        return i


class FaceDet(ImageOrAlbumStage):
    """人脸检测"""

    def __init__(self) -> None:
        from plugins.facedet import FaceDet
        self.det = FaceDet(None)

    def resolve_image(self, i: ImageItem):
        self.det.faces(i)
