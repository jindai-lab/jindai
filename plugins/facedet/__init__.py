# facedet
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from PyMongoWrapper import ObjectId, F, Fn, Var
import struct
import base64

from gallery import single_item, Item, Post
from plugin import Plugin, PluginContext
from plugins.hashing import whash, bitcount


class FaceDet(Plugin):
    
    def __init__(self, app, **config):
        if 'faces' not in Item.fields:
            Item._fields.append('faces')
        setattr(Item, 'faces', lambda: None)

    def get_callbacks(self):
        return ['check-images']
    
    def get_tools(self):
        return ['faces']
    
    def get_special_pages(self):
        return ['face']
    
    def faces(self, ctx, *args):
        self.check_images_callback(ctx,
            Item.aggregator.match(
                (F.flag == 0) & (F.storage == True) & (F.faces == None) & F.url.regex(r'\.jpe?g$')
            ).perform()
        )

    def crop_faces(self, buf):
        from . import facedetectcnn
        image = Image.open(buf)
        image.thumbnail((1024, 1024))
        for x, y, w, h, confi in facedetectcnn.facedetect_cnn(image):
            if confi < 75: continue
            bufi = image.crop((x, y, x + w, y + h))
            yield bufi

    def check_images_callback(self, ctx, items):
       
        for i in tqdm(items):
            if hasattr(i, 'faces') and i.faces is not None:
                return

            try:
                f = i.read_image()
                if not f: return
                i.faces = []
                for face in self.crop_faces(f):
                    ctx.log(i.id, 'found face')
                    i.faces.append(whash(face))

                i.save()

            except KeyboardInterrupt:
                exit()

            except Exception as ex:
                ctx.log(ex)

    def special_page(self, aggregate, params, orders_params, **vars):
        
        def _v(x):
            if isinstance(x, bytes):
                return struct.unpack('>q', x)[0]
            elif isinstance(x, str):
                return int(x, 16)
            elif isinstance(x, int):
                return x
            else:
                raise TypeError(x)

        post1 = params['post']
        groups = params['groups']
        archive = params['archive']

        offset = params['order'].get('offset', 0)
        limit = params['limit']

        if post1 == 'face/':
            aggregate = aggregate.project(
                _id=1, liked_at=1, created_at=1, source_url=1, tags=1,
                items=Fn.filter(input=Var.items, as_='item', cond=Fn.size(Fn.ifNull('$$item.faces', [])))
            ).match(F.items != [])            

            if orders_params:
                order_conds = orders_params['order_conds']
                orders = orders_params['orders']
                
                if order_conds:
                    aggregate.match(order_conds[0])

                aggregate.sort(orders)
                if offset:
                    aggregate.skip(offset)
                
                aggregate.limit(limit)
            rs = aggregate.perform()
            return rs, {}, {}

        elif post1.startswith('face/'):
            dots = post1.split('/')[1].split('.')
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
                        Post(
                            _id=p.id,
                            items=[
                                Item(url='data:image/jpeg;base64,' + base64.b64encode(saved.getvalue()).decode('ascii'))
                            ]
                        )
                    )

                if fid: ps = [ps[0], ps[fid]]

                return ps, {}, {}
            else:
                fdh = [_v(f) for f in Item.first(F.id == iid).faces]
                if fid: fdh = [fdh[fid-1]]
                if not fdh: return [], {}, {}

                groupped = {}
                results = []
                for rp in aggregate.perform():
                    for ri in rp.items:
                        if not ri or not isinstance(ri, Item) or ri.flag != 0 or not ri.faces or ri.id == iid: continue
                        ri.score = min([
                            min([bitcount(_v(i) ^ j) for j in fdh])
                            for i in ri.faces
                        ])
                        rpo = Post(**rp.as_dict())
                        rpo.items = [ri]
                        if archive:
                            pgs = [g for g in rp.tags if g.startswith('*')]
                            for g in pgs or [rp.source_url]:
                                if g not in groupped or groupped[g][0] > ri.score:
                                    groupped[g] = (ri.score, rpo)
                        else:
                            results.append((ri.score, rpo))
                
                if archive:
                    results = list(groupped.values())
                return [r for _, r in sorted(results, key=lambda x: x[0])[offset:offset+limit]], \
                    {'offset': max(0, offset-limit), 'limit': limit}, {'offset': offset + limit, 'limit': limit}
