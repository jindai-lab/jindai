import struct

import imagehash
from tqdm import tqdm
from PIL import Image

from gallery import *
from storage import *
from plugin import Plugin

def dhash(im):
    dh = imagehash.dhash(im)
    dh = bytes.fromhex(str(dh))
    return dh


def whash(im):
    dh = imagehash.whash(im)
    dh = bytes.fromhex(str(dh))
    return dh


def bitcount(x):
    return bin(x).count('1')


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
                for i in Item.query(F.flag.eq(0) & F.id.in_([ObjectId(_) for _ in ids])):
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


    def check_images_callback(self, ctx, items):
        for i in tqdm(items):
            ctx.log(i.id)
            if not i.storage: continue
            try:
                dh, wh = i.dhash, i.whash
                if dh and wh: continue

                try: f = i.read_image()
                except: f = None
                if not f: continue

                if not dh:
                    f.seek(0)
                    im = Image.open(f)
                    dh = dhash(im) or ''
                if not wh:
                    f.seek(0)
                    im = Image.open(f)
                    wh = whash(im) or ''

                i.dhash, i.whash = dh, wh
                i.save()
            except IOError:
                pass
            except AssertionError:
                i.storage = None
                i.save()
            except KeyboardInterrupt:
                break

    def get_tools(self):
        return ['hash', 'hash-compare']
    
    def get_special_pages(self):
        return ['sim', 'group_ratings']
    
    def get_callbacks(self):
        return ['check-images']

    def hash(self, ctx):
        self.check_images_callback(ctx, Item.query(Item.valid_item() & F.storage.eq(True) & (F[self.method].eq('') | F[self.method].empty())))
        
    def hash_compare(self, ctx, tspan=''):

        if tspan == '':
            tspan = 0
        else:
            tspan = queryparser.parse_dt_span(tspan)
        min_id = ObjectId.from_datetime(datetime.datetime.fromtimestamp(tspan))
        ctx.log('min_id', min_id)
                
        def _tod(x):
            return x if isinstance(x, int) else int(x, 16) if x else -1
    
        def _flip(x, i):
            x ^= 1 << i
            return x

        def _flips(x, n, lm=0):
            for i in range(lm, 64):
                _x = _flip(x, i)
                if n == 1:
                    yield _x
                else:
                    for __x in _flips(_x, n - 1, i + 1):
                        yield __x

        from collections import defaultdict
        d = set()
        d2 = defaultdict(list)
    
        for i in tqdm(Item.query(F[self.method].exists(1) & ~F[self.method].empty() & F.flag.eq(0) & (F.width > 0) & F.url.regex(r'\.(jpe?g|gif|png|tiff)$'))):
            if not i[self.method]: continue
            dha = _tod(i.dhash)
            dhb = _tod(i.whash)
            if i.id > min_id:
                d.add(dha)
            d2[dha].append((i.id, i.width, i.height, dhb))
        total = len(d2)
        ctx.log(total)
        ctx.log('loaded {} unique hashes, {} to compare'.format(len(d2), len(d)))

        fo = open('compare.tsv', 'w')
        for dh2 in tqdm(d):
            ls2 = d2[dh2]
            for id2, w2, h2, dhb2 in ls2:
                for dh1, sc in [(dh2, 0)] + list(zip(_flips(dh2, 1), [1] * 64)) + list(zip(_flips(dh2, 2), [2] * 2080)):
                    if dh1 not in d2: continue
                    for id1, w1, h1, dhb1 in d2[dh1]:
                        if id1 >= id2 or w1 == 0: continue
                        a, b = id2, id1
                        if w1 * h1 < w2 * h2: b, a = a, b
                        r = '{}\t{}\t{}'.format(a, b, sc + bitcount(dhb1 ^ dhb2))
                        ctx.log(r)
                        fo.write(r + '\n')
        fo.close()

    def special_page(self, rs, params, **vars):
        groups = params['groups']
        archive = params['archive']
        post1 = params['post']
        limit, offset = int(params['limit']), int(params['order'].get('offset', 0))

        if post1.split('/')[0] not in ('sim', 'group_ratings'):
            return

        iid = post1.split('/')[1]

        def _v(x):
            if isinstance(x, bytes): return struct.unpack('>q', x)[0]
            return int(x, 16) if isinstance(x, str) else x

        if post1.startswith('sim/'):
            if groups:
                return single_item('', iid), None, None
            else:
                iid = ObjectId(iid) if len(iid) == 24 else iid
                it = Item.first(F._id == iid)
                if not hasattr(it, self.method): return
                pgroups = [g for g in (Post.first(F.items == iid) or Post()).tags if g.startswith('*')]
                dha, dhb = _v(it.dhash), _v(it.whash)
                results = []
                groupped = {}

                for p in rs.perform():
                    for i in p.items:
                        if i.id == it.id: continue
                        if i.flag != 0 or i[self.method] is None or i[self.method] == '': continue
                        dha1, dhb1 = _v(i.dhash), _v(i.whash)
                        i.score = bitcount(dha ^ dha1) + bitcount(dhb ^ dhb1)
                        po = Post(**p.as_dict())
                        po.items = [i]
                        po.score = i.score
                        if archive:
                            pgs = [g for g in p.tags if g.startswith('*')]
                            for g in pgs or [po.source_url]:
                                if g not in pgroups and (g not in groupped or groupped[g].score > po.score):
                                    groupped[g] = po
                        else:
                            results.append(po)

                if archive:
                    results = list(groupped.values())
                results = sorted(results, key=lambda x: x.score)[offset:offset + limit]
                return results, {'offset': max(offset - limit, 0), 'limit': limit}, {'offset': offset + limit,
                                                                                    'limit': limit}

        elif post1.startswith('group_ratings/'):
            if groups:
                return [], {}, {}
            return Post.aggregator.match(F.tags.regex(r'^\*')).lookup(
                from_=F.item, localField=F.items, foreignField=F._id, as_=F.items2
            ).addFields(
                group_id=Fn.filter(input=Var.tags, as_='t', cond=Fn.substrCP(Var._t, 0, 1) == '*')
            ).unwind(
                path=Var.group_id
            ).addFields(
                group_id=Fn.ifNull(Var.group_id, Var.source_url)
            ).group(
                _id=Var.group_id,
                id=Fn.first(Var._id),
                liked_at=Fn.first(Var.liked_at),
                created_at=Fn.first(Var.created_at),
                source_url=Fn.first(Var.source_url),
                items=Fn.push(Var.items),
                tags=Fn.first(Var.tags),
                counts=Fn.sum(Fn.size(Var.items))
            ).addFields(
                items=Fn.cond(Var.items2 == [], Var.items, Var.items2)
            ).addFields(
                ratings=Fn.sum(Var['items.rating'])
            ).sort(ratings=-1).limit(100).perform(), {}, {}
                
