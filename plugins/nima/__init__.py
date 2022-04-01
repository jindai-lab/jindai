from jindai import Plugin
from plugins.gallery import ImageOrAlbumStage
from jindai import  expand_path
from jindai.models import ImageItem

MODEL = expand_path('models_data/nima.pkl')


class NIMAEval(ImageOrAlbumStage):
    """图像质量自动评价
    """
    
    def __init__(self):
        from .test import predict as nima_predict, load_state as nima_init
        nima_init(MODEL)
        self.predict = nima_predict

    def resolve_image(self, i: ImageItem, context):
        for (_, mean) in self.predict([i.image]):
            i.ava_eval = mean
            i.save()


class NIMAPlugin(Plugin):

    def __init__(self, app, **config):
        ImageItem.set_field('ava_eval', float)
        self.register_pipelines([NIMAEval])        
