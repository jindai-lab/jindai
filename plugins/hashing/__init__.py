import imagehash
from PIL import Image
from plugins.gallery import *
from queue import deque
from plugin import Plugin
from storage import *
from pipelines.imageproc import ImageOrAlbumStage
from models import ImageItem
import tempfile


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


class ImageHash(ImageOrAlbumStage):
    """建立图像哈希检索
    """

    def resolve_image(self, i : ImageItem):
        try:
            dh, wh = i.dhash, i.whash
            if dh and wh: return i

            f = i.image_raw
            if not f: return

            if not dh:
                dh = dhash(f) or ''
            if not wh:
                wh = whash(f) or ''

            i.dhash, i.whash = dh, wh
        except (IOError, AssertionError):
            pass
        i.save()


class ImageHashDuplications(ImageOrAlbumStage):
    """进行图像哈希去重
    """
    
    def __init__(self) -> None:
        self.results = deque()
        self.result_pairs = set()

    def _unv(self, x):
        from bson import binary
        return binary.Binary(bytes.fromhex(f'{x:016x}'))

    def resolve_image(self, i: ImageItem):
        if not i.dhash: return
        if not isinstance(i.dhash, bytes): self.logger(type(i).__name__, i.as_dict())
        dh2 = v(i.dhash)
        dhb2 = v(i.whash)
        h2, w2 = i.height, i.width
        for j in ImageItem.query(F.dhash.in_(
                [self._unv(x) for x in [dh2] + list(flips(dh2, 1)) + list(flips(dh2, 2))])):
            id1 = j.id
            if id1 == i.id or f'{i.id}-{id1}' in self.result_pairs or f'{id1}-{i.id}' in self.result_pairs: continue
            self.result_pairs.add(f'{id1}-{i.id}')
            a, b = i.id, id1
            if j.width * j.height < w2 * h2: b, a = a, b
            r = '{}\t{}\t{}'.format(a, b, bitcount(v(i.dhash) ^ dh2) + bitcount(v(j.whash) ^ dhb2))
            self.logger(r)
            self.results.append(r + '\n')
        return i

    def summarize(self, r):
        k = tempfile.mktemp()
        self.fo = open(k + '.tsv', 'w')
        for l in self.results:
            self.fo.write(l)
        self.fo.close()
        return {'redirect': '/api/plugins/compare?' + k}
        

class Hashing(Plugin):

    def __init__(self, app):
        super().__init__(app)
        ImageItem.set_field('dhash', bytes)
        ImageItem.set_field('whash', bytes)
        self.register_pipelines([ImageHashDuplications, ImageHash])
            
        @app.route('/api/plugins/compare.tsv')
        def _compare_tsv():
            p = request.args.get('key', '') + '.tsv'
            if not os.path.exists(p):
                return Response('')

            def __parse_compare_results():
                with open(p) as fi:
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

        @app.route('/api/plugins/jquery.min.js')
        def _jquery_js():
            return serve_file(os.path.join(os.path.dirname(__file__), 'jquery.min.js'))

    def handle_page(self, ds, iid):
        limit = ds.limit
        offset = ds.skip
        ds.limit = 0
        ds.raw = False
        
        groups = ds.groups in ('both', 'group')
        archive = ds.groups in ('both', 'source')

        if groups:
            return single_item('', iid)
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
                    if archive:
                        pgs = [g for g in p.keywords if g.startswith('*')]
                        for g in pgs or [po.source['url']]:
                            if g not in pgroups and (g not in groupped or groupped[g].score > po.score):
                                groupped[g] = po
                    else:
                        results.append(po)

            if archive:
                results = list(groupped.values())

            results = sorted(results, key=lambda x: x.score)[
                offset:offset + limit]
            return results
    
    def get_pages(self):
        return {
            'sim': {
                'format': 'sim/{imageitem._id}',
                'shortcut': 's',
                'icon': 'mdi-image'
            }
        }
