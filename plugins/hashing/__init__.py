import imagehash
from PIL import Image
from PyMongoWrapper.dbo import DbObjectInitializer
from plugins.gallery import *
from plugin import Plugin
from storage import *
from models import ImageItem


def dhash(im):
    if isinstance(im, BytesIO):
        im.seek(0)
        im = Image.open(im)
    dh = imagehash.dhash(im)
    dh = bytes.fromhex(str(dh))
    return dh


def whash(im):
    if isinstance(im, BytesIO):
        im.seek(0)
        im = Image.open(im)
    dh = imagehash.whash(im)
    dh = bytes.fromhex(str(dh))
    return dh


def bitcount(x):
    return bin(x).count('1')


def v(x):
    return int(x.hex(), 16)


def flip(x, i):
    x ^= 1 << i
    return x


def flips(x, n, lm=0):
    for i in range(lm, 64):
        _x = flip(x, i)
        if n == 1:
            yield _x
        else:
            for __x in flips(_x, n - 1, i + 1):
                yield __x


class Hashing(Plugin):

    def __init__(self, app):
        super().__init__(app)
        ImageItem.set_field('dhash', bytes)
        ImageItem.set_field('whash', bytes)
            
        @app.route('/api/plugins/compare.tsv')
        def _compare_tsv():
            if not os.path.exists('compare.tsv'):
                return Response('')

            def __parse_compare_results():
                with open('compare.tsv') as fi:
                    for l in fi:
                        r = l.strip().split('\t')
                        if len(r) < 3:
                            continue
                        id1, id2, score = r
                        if id1 == id2:
                            continue
                        yield id1, id2, int(score)

            def __get_items():
                ids = set()
                for id1, id2, s in __parse_compare_results():
                    ids.add(id1)
                    ids.add(id2)
                items = {}
                for i in ImageItem.query(F.flag.eq(0) & F.id.in_([ObjectId(_) for _ in ids])):
                    items[str(i.id)] = i
                return items

            buf = ''
            slimit = int(request.args.get('q', 10))
            items = __get_items()
            for id1, id2, score in sorted(__parse_compare_results(),
                                          key=lambda x: x[2]):
                if score > slimit:
                    continue
                if id1 in items and id2 in items:
                    buf += '{} {} {}\n'.format(id1, id2, score)
            return Response(buf)

        @app.route('/api/plugins/compare')
        def _compare_html():
            return serve_file(os.path.join(os.path.dirname(__file__), 'compare.html'))

    def handle_page(self, ds, iid):
        limit = ds.limit
        offset = ds.order.get('offset', 0)
        ds.limit = 0
        ds.raw = False
        
        if ds.groups:
            return make_gallery_response(single_item('', iid))
        else:
            it = ImageItem.first(F.id == iid)
            if it.dhash is None:
                return
            pgroups = [g for g in (Paragraph.first(F.images == ObjectId(iid)) or Paragraph()).keywords if g.startswith(
                '*')] or [(Paragraph.first(F.images == ObjectId(iid)) or Paragraph()).source.get('url', '')]
            dha, dhb = v(it.dhash), v(it.whash)
            results = []
            groupped = {}

            for p in ds.fetch():
                for i in p.images:
                    if i.id == it.id:
                        continue
                    if i.flag != 0 or i.dhash is None or i.dhash == b'':
                        continue
                    dha1, dhb1 = v(i.dhash), v(i.whash)
                    i.score = bitcount(dha ^ dha1) + bitcount(dhb ^ dhb1)
                    po = Paragraph(**p.as_dict())
                    po.images = [i]
                    po.score = i.score
                    if ds.archive:
                        pgs = [g for g in p.keywords if g.startswith('*')]
                        for g in pgs or [po.source['url']]:
                            if g not in pgroups and (g not in groupped or groupped[g].score > po.score):
                                groupped[g] = po
                    else:
                        results.append(po)

            if ds.archive:
                results = list(groupped.values())

            results = sorted(results, key=lambda x: x.score)[
                offset:offset + limit]
            return make_gallery_response(results, {'keys': ['offset'], 'offset': max(offset - limit, 0), 'limit': limit}, {'keys': ['offset'], 'offset': offset + limit,
                                                                                                        'limit': limit}, False, ds.direction, ds.order)    
    def get_pages(self):
        return {
            'sim': {
                'format': 'sim/{imageitem._id}',
                'shortcut': 's',
                'icon': 'mdi-image'
            }
        }
