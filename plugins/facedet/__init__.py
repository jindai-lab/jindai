"""人脸检测"""
import base64
from io import BytesIO

import numpy as np
from PIL import Image

from jindai.helpers import safe_import
from jindai.models import DbObjectCollection, F, Fn, MediaItem, Paragraph, Var
from plugins.hashing import HashingBase, bitcount, to_int, whash
from plugins.imageproc import MediaItemStage


class FaceDet(MediaItemStage):
    """人脸检测"""

    def __init__(self, method="retinaface") -> None:
        """
        Args:
            method (opencv|retinaface): Face detection method.
                @zhs 人脸检测方法
        """
        self.method = method
        if method == 'opencv':
            cv2 = safe_import('cv2', 'opencv-python-headless')
            self.face_classifier = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        elif method == 'retinaface':
            safe_import('retinaface', 'retina-face')
            from retinaface import RetinaFace
            self.face_classifier = lambda input_image: RetinaFace.extract_faces(input_image, align=True)
            
        super().__init__()

    def crop_faces(self, buf):
        """Crop faces from image in buffer"""
        image = Image.open(buf)
        image.thumbnail((1024, 1024))
        if self.method == 'opencv':
            gray_image = np.array(image.convert('L'))
            for (x, y, w, h) in self.face_classifier.detectMultiScale(
                        gray_image, minSize=(40, 40)):
                yield image.crop((x, y, x+w, y+h))
        elif self.method == 'retinaface':
            for im in self.face_classifier(np.array(image)):
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
            'face', 'e', 'face,o"{mediaitem._id}"', 'mdi-emoticon-outline', self.handle_filter)
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
