# facedet
import base64
from io import BytesIO

from PyMongoWrapper.dbo import DbObjectCollection


from models import Paragraph, ImageItem
from pipelines.imageproc import ImageOrAlbumStage
from plugins.gallery import single_item
from PIL import Image
from plugin import Plugin
from plugins.hashing import bitcount, whash, v
from PyMongoWrapper import F, Fn, ObjectId, Var


class FaceDet(ImageOrAlbumStage):
    """人脸检测"""

    def __init__(self) -> None:
        from plugins.facedet import FaceDet
        self.det = FaceDet(None)

    def resolve_image(self, i: ImageItem):
        self.det.faces(i)


class FaceDetPlugin(Plugin):
    
    def __init__(self, app, **config):
        super().__init__(app)
        super().register_pipelines('人脸识别', [FaceDet])
        if 'faces' not in ImageItem.fields:
            ImageItem.fields['faces'] = DbObjectCollection(bytes)
            ImageItem.faces = ImageItem.fields['faces']

    def get_special_pages(self):
        return ['face']
    
    def faces(self, i):
        if hasattr(i, 'faces') and i.faces != b'':
            return i

        f = i.image_raw
        if not f: return
        i.faces = []
        for face in self.crop_faces(f):
            i.faces.append(whash(face))

        i.save()
        return i

    def crop_faces(self, buf):
        from . import facedetectcnn
        image = Image.open(buf)
        image.thumbnail((1024, 1024))
        for x, y, w, h, confi in facedetectcnn.facedetect_cnn(image):
            if confi < 75: continue
            bufi = image.crop((x, y, x + w, y + h))
            yield bufi
       
    def handle_special_page(self, ds, post_args):
        
        groups = ds.groups
        archive = ds.archive

        offset = ds.order.get('offset', 0)
        limit = ds.limit
        ds.limit = 0
        ds.raw = False

        if len(post_args) == 1:
            ds.aggregator.addFields(
                images=Fn.filter(input=Var.images, as_='item', cond=Fn.size(Fn.ifNull('$$item.faces', [])))
            ).match(F.images != [])            

            rs = ds.fetch()
            return rs, {}, {}

        else:
            dots = post_args[1].split('.')
            fid = 0
            if len(dots) == 1:
                iid = dots[0]
            elif len(dots) == 2:
                iid, fid = dots
                fid = int(fid)
            else:
                return 'Invalid arguments', 400
            iid = ObjectId(iid)
            if groups:
                ps = single_item('', iid)
                p = ps[0]
                for face in self.crop_faces(p.images[0].image_raw):
                    saved = BytesIO()
                    face.save(saved, format='JPEG')
                    ps.append(
                        Paragraph(
                            _id=p.id,
                            images=[
                                ImageItem(source={'url': 'data:image/jpeg;base64,' + base64.b64encode(saved.getvalue()).decode('ascii')})
                            ]
                        )
                    )

                if fid: ps = [ps[0], ps[fid]]

                return ps, {}, {}
            else:
                fdh = [v(f) for f in ImageItem.first(F.id == iid).faces]
                if fid: fdh = [fdh[fid-1]]
                if not fdh: return [], {}, {}

                groupped = {}
                results = []
                for rp in ds.fetch():
                    for ri in rp.images:
                        if not ri or not isinstance(ri, ImageItem) or ri.flag != 0 or not ri.faces or ri.id == iid: continue
                        ri.score = min([
                            min([bitcount(v(i) ^ j) for j in fdh])
                            for i in ri.faces
                        ])
                        rpo = Paragraph(**rp.as_dict())
                        rpo.images = [ri]
                        if archive:
                            pgs = [g for g in rp.keywords if g.startswith('*')]
                            for g in pgs or [rp.source['url']]:
                                if g not in groupped or groupped[g][0] > ri.score:
                                    groupped[g] = (ri.score, rpo)
                        else:
                            results.append((ri.score, rpo))
                
                if archive:
                    results = list(groupped.values())
                return [r for _, r in sorted(results, key=lambda x: x[0])[offset:offset+limit]], \
                    {'keys': ['offset'], 'offset': max(0, offset-limit)}, {'keys': ['offset'], 'offset': offset + limit}
