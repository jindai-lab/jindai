# facedet
import base64
import struct
from io import BytesIO

from gallery import Album, ImageItem, single_item
from PIL import Image
from plugin import Plugin
from plugins.hashing import bitcount, whash
from PyMongoWrapper import F, Fn, ObjectId, Var


class FaceDet(Plugin):
    
    def __init__(self, app, **config):
        if 'faces' not in ImageItem.fields:
            ImageItem._fields.append('faces')
        setattr(ImageItem, 'faces', lambda: None)

    def get_special_pages(self):
        return ['face']
    
    def faces(self, i):
        if hasattr(i, 'faces') and i.faces is not None:
            return i

        f = i.read_image()
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
       
    def special_page(self, ds, post_args):
        
        def _v(x):
            if isinstance(x, bytes):
                return struct.unpack('>q', x)[0]
            elif isinstance(x, str):
                return int(x, 16)
            elif isinstance(x, int):
                return x
            else:
                raise TypeError(x)

        groups = ds.groups
        archive = ds.archive

        offset = ds.order.get('offset', 0)
        limit = ds.limit
        ds.raw = False

        if len(post_args) == 1:
            ds.aggregator.addFields(
                items=Fn.filter(input=Var.items, as_='item', cond=Fn.size(Fn.ifNull('$$item.faces', [])))
            ).match(F.items != [])            

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
                for face in self.crop_faces(p.items[0].read_image()):
                    saved = BytesIO()
                    face.save(saved, format='JPEG')
                    ps.append(
                        Album(
                            _id=p.id,
                            items=[
                                ImageItem(source={'url': 'data:image/jpeg;base64,' + base64.b64encode(saved.getvalue()).decode('ascii')})
                            ]
                        )
                    )

                if fid: ps = [ps[0], ps[fid]]

                return ps, {}, {}
            else:
                fdh = [_v(f) for f in ImageItem.first(F.id == iid).faces]
                if fid: fdh = [fdh[fid-1]]
                if not fdh: return [], {}, {}

                groupped = {}
                results = []
                for rp in ds.fetch():
                    for ri in rp.items:
                        if not ri or not isinstance(ri, ImageItem) or ri.flag != 0 or not ri.faces or ri.id == iid: continue
                        ri.score = min([
                            min([bitcount(_v(i) ^ j) for j in fdh])
                            for i in ri.faces
                        ])
                        rpo = Album(**rp.as_dict())
                        rpo.items = [ri]
                        if archive:
                            pgs = [g for g in rp.tags if g.startswith('*')]
                            for g in pgs or [rp.source['url']]:
                                if g not in groupped or groupped[g][0] > ri.score:
                                    groupped[g] = (ri.score, rpo)
                        else:
                            results.append((ri.score, rpo))
                
                if archive:
                    results = list(groupped.values())
                return [r for _, r in sorted(results, key=lambda x: x[0])[offset:offset+limit]], \
                    {'offset': max(0, offset-limit), 'limit': limit}, {'offset': offset + limit, 'limit': limit}