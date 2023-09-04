"""
Image Captioning
@zhs 为图像自动添加描述
"""
from jindai import Plugin
from jindai.pipeline import PipelineStage
from jindai.models import MediaItem, Paragraph
from jindai.helpers import safe_import


class ImageCaptioning(PipelineStage):
    """Image Captioning
    @zhs 自动为图片添加描述（英文）
    """

    def __init__(self, labels='', topk=10) -> None:
        '''
        Args:
            labels (LINES): candidate labels
            topk (int): top results
        '''
        super().__init__()
        safe_import('clip_interrogator')
        from clip_interrogator import Config, Interrogator, LabelTable
        ci = Interrogator(Config(caption_model_name='blip-base'))
        if labels:
            self.table = LabelTable(labels.split(), 'terms', ci)
            self.topk = topk
        else:
            self.table = None
            self.topk = -1
        self.ci = ci

    def resolve(self, paragraph: Paragraph):
        if paragraph.caption or not paragraph.images:
            return paragraph
        try:
            for i in paragraph.images:
                caption = self.caption(i.image)
                paragraph.content += '\n' + caption
                paragraph.caption = caption
                paragraph.save()
                return paragraph
        except Exception as ex:
            self.log_exception('', ex)
            return

    def caption(self, image):
        image = image.convert('RGB')
        image.thumbnail((800, 800))
        if self.table:
            features = self.ci.image_to_features(image)
            prompt = ' '.join(self.table.rank(features, top_count=self.topk))
        else:
            prompt = self.ci.interrogate_fast(image)
        self.log(prompt)
        return prompt


class ImageCaptioningPlugin(Plugin):
    """Image captioning plugin
    """

    def __init__(self, pmanager, **conf) -> None:
        super().__init__(pmanager, **conf)
        Paragraph.ensure_index('caption')
        self.register_pipelines([ImageCaptioning])
