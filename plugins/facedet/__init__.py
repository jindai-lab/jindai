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

    def crop_faces(self, buf):
        from . import facedetectcnn
        image = Image.open(buf)
        image.thumbnail((1024, 1024))
        for x, y, w, h, confi in facedetectcnn.facedetect_cnn(image):
            if confi < 75: continue
            bufi = image.crop((x, y, x + w, y + h))
            yield bufi
    
    def resolve_image(self, i: ImageItem, context):
        f = i.image_raw
        if not f: return
        i.faces = []
        for face in self.crop_faces(f):
            i.faces.append(whash(face))
        
        i.save()
        return i


class FaceDetPlugin(Plugin):
    
    def __init__(self, app, **config):
        super().__init__(app)
        self.det = FaceDet()
        self.register_pipelines([FaceDet])
        ImageItem.set_field('faces', DbObjectCollection(bytes))
        
    def handle_page(self, ds, iid='', fid=''):
        groups = ds.groups in ('both', 'group')
        archive = ds.groups in ('both', 'source')

        offset = ds.skip
        limit = ds.limit
        ds.limit = 0
        ds.raw = False

        if iid == '':
            ds.aggregator.addFields(
                images=Fn.filter(input=Var.images, as_='item', cond=Fn.size(Fn.ifNull('$$item.faces', [])))
            ).match(F.images != [])            

            rs = ds.fetch()
            return rs

        else:
            fid = 0 if not fid else int(fid)
            iid = ObjectId(iid)
            if groups:
                ps = single_item('', iid)
                p = ps[0]
                for face in self.det.crop_faces(p.images[0].image_raw):
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

                return ps
            else:
                fdh = [v(f) for f in ImageItem.first(F.id == iid).faces]
                if fid: fdh = [fdh[fid-1]]
                if not fdh:
                    return []

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
                return [r for _, r in sorted(results, key=lambda x: x[0])[offset:offset+limit]]
    
    def get_pages(self):
        return {
            'face': {
                'format': 'face/{imageitem._id}',
                'shortcut': 'e',
                'icon': 'mdi-emoticon-outline'
            }
        }
