import struct

import imagehash
from gallery import *
from plugin import Plugin
from storage import *


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
    if isinstance(x, bytes): return struct.unpack('>q', x)[0]
    return int(x, 16) if isinstance(x, str) else x


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

    def __init__(self, app, method='dhash'):
        self.method = method

        @app.route('/api/gallery/compare.tsv')
        def _compare_tsv():
            if not os.path.exists('compare.tsv'):
                return Response('')

            def __parse_compare_results():
                with open('compare.tsv') as fi:
                    for l in fi:
                        r = l.strip().split('\t')
                        if len(r) < 3: continue
                        id1, id2, score = r
                        if id1 == id2: continue
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
                if score > slimit: continue
                if id1 in items and id2 in items:
                    buf += '{} {} {}\n'.format(id1, id2, score)
            return Response(buf)

        @app.route('/api/gallery/compare')
        def _compare_html():
            return serve_file(os.path.join(os.path.dirname(__file__), 'compare.html'))
    
    def get_special_pages(self):
        return ['sim']
    
    def special_page(self, ds, post_args):
        groups = ds.groups
        archive = ds.archive
        limit, offset = ds.limit, ds.order.get('offset', 0)
        ds.limit = 0

        if post_args[0] not in ('sim',):
            return

        iid = post_args[1]

        if post_args[0] == 'sim':
            if groups:
                return single_item('', iid), None, None
            else:
                it = ImageItem.first(F.id == iid)
                if not hasattr(it, self.method): return
                pgroups = [g for g in (Album.first(F.items == ObjectId(iid)) or Album()).keywords if g.startswith('*')] or [(Album.first(F.items == ObjectId(iid)) or Album()).source.get('url', '')]
                dha, dhb = v(it.dhash), v(it.whash)
                results = []
                groupped = {}
                ds.raw = False
                
                for p in ds.fetch():
                    for i in p.items:
                        if i.id == it.id: continue
                        if i.flag != 0 or i[self.method] is None or i[self.method] == '': continue
                        dha1, dhb1 = v(i.dhash), v(i.whash)
                        i.score = bitcount(dha ^ dha1) + bitcount(dhb ^ dhb1)
                        po = Album(**p.as_dict())
                        po.items = [i]
                        po.score = i.score
                        if archive:
                            pgs = [g for g in p.keywords if g.startswith('*')]
                            for g in pgs or [po.source['url']]:
                                if g not in pgroups and (g not in groupped or groupped[g].score > po.score):
                                    groupped[g] = po
                        else:
                            results.append(po)

                if archive:
                    results = list(groupped.values())
                
                results = sorted(results, key=lambda x: x.score)[offset:offset + limit]
                return results, {'keys': ['offset'], 'offset': max(offset - limit, 0), 'limit': limit}, {'keys': ['offset'], 'offset': offset + limit,
                                                                                    'limit': limit}
