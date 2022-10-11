"""人脸检测"""
import base64
from email.mime import image
from io import BytesIO
from bson import ObjectId
from PIL import Image
import numpy as np

from jindai import Plugin
from jindai.helpers import safe_import
from jindai.models import MediaItem, Paragraph, F, Fn, Var, DbObjectCollection
from plugins.imageproc import MediaItemStage
from plugins.hashing import single_item, bitcount, to_int, whash, HashingBase


safe_import('deepface')

class FaceDet(MediaItemStage):
    """人脸检测"""
    
    def __init__(self, method="opencv") -> None:
        """
        Args:
            method (opencv|dlib|mtcnn|ssd|retinaface): Face detection method.
                @chs 人脸检测方法
        """        
        self.method = method
        from deepface.detectors.FaceDetector import detect_faces, build_model
        self.detector = build_model(method)
        super().__init__()

    def crop_faces(self, buf):
        """Crop faces from image in buffer"""
        from deepface.detectors.FaceDetector import detect_faces, build_model
        image = Image.open(buf)
        image.thumbnail((1024, 1024))
        for face in detect_faces(self.detector, self.method, np.array(image)):
            im, _ = face
            yield Image.fromarray(im)

    def resolve_image(self, i: MediaItem, _):
        """Resolve image"""
        try:
            data = i.data
            if not data:
                return
            i.faces = []
            for face in self.crop_faces(data):
                i.faces.append(whash(face))

            i.save()
        except OSError:
            pass
        
        return i


class FaceDetPlugin(HashingBase):
    """人脸检测插件"""

    def __init__(self, pmanager, **_):
        super().__init__(pmanager)
        
        self.det = FaceDet()
        self.register_pipelines([FaceDet])
        self.register_filter(
            'face', 'e', 'face/{mediaitem._id}', 'mdi-emoticon-outline', self.handle_filter)
        MediaItem.set_field('faces', DbObjectCollection(bytes))
        
    def handle_filter_check(self, context):
        m = MediaItem.first(F.id == context.iid)
        if not m.faces:
            FaceDet().resolve_image(m, None)
        
        if not m.faces:
            return False
        
        fid = context.args[0] if len(context.args) > 0 else ''
        fid = 0 if not fid else int(fid)
        
        context.fdh = [to_int(f) for f in m.faces]
        if fid:
            context.fdh = [context.fdh[fid-1]]
            
        if not context.fdh:
            return []
        
        paragraph_faces = context.sticky_paragraphs
        
        source_paragraph = paragraph_faces[0]
        for face in self.det.crop_faces(source_paragraph.images[0].data):
            saved = BytesIO()
            face.save(saved, format='JPEG')
            paragraph_faces.append(
                Paragraph(
                    _id=source_paragraph.id,
                    images=[
                        MediaItem(source={
                                    'url': 'data:image/jpeg;base64,'
                                    + base64.b64encode(saved.getvalue()).decode('ascii')}, item_type='image')
                    ]
                )
            )

        if fid:
            paragraph_faces = [paragraph_faces[0], paragraph_faces[fid]]

        context.sticky_paragraphs = paragraph_faces
        return True
        
    def handle_filter_item(self, image_item, context):
        if not image_item or not isinstance(image_item, MediaItem) \
            or not image_item.faces or image_item.id == context.iid:
            return
        image_item.score = min([
            min([bitcount(to_int(i) ^ j) for j in context.fdh])
            for i in image_item.faces
        ])
        return image_item
        
    def handle_filter(self, dbq, iid='', *args):
        """Handle page"""
        if iid == '':
            dbq.limit = 0
            dbq.raw = False
            dbq.groups = 'none'

            dbq.aggregator.addFields(
                images=Fn.filter(input=Var.images, as_='item',
                                 cond=Fn.size(Fn.ifNull('$$item.faces', [])))
            ).match(F.images != [])

            result_set = dbq.fetch()
            return result_set

        else:
            return super().handle_filter(dbq, iid, *args)
