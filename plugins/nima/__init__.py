import os
from plugin import Plugin
from pipelines.imageproc import ImageOrAlbumStage
import config
from models import ImageItem

MODEL = os.path.join(config.rootpath, 'models_data', 'nima.pkl')


class NIMAEval(ImageOrAlbumStage):
    def __init__(self):
        from .test import predict as nima_predict, load_state as nima_init
        nima_init(MODEL)
        self.predict = nima_predict

    def resolve_image(self, i: ImageItem):
        for (_, mean) in self.predict([i.image]):
            i.ava_eval = mean
            i.save()


class NIMAPlugin(Plugin):

    def __init__(self, app, **config):
        ImageItem.set_field('ava_eval', float)
        self.register_pipelines([NIMAEval])        
