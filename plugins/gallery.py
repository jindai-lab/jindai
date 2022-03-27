import hashlib
from typing import List
from bson import ObjectId
from helpers import *
from models import Paragraph, ImageItem, StorageManager
from plugin import Plugin
from PyMongoWrapper import F, Fn


# HELPER FUNCS
def single_item(pid: str, iid: str) -> List[Paragraph]:
    """Return a single-item paragraph object with id = `pid` and item id = `iid`

    Args:
        pid (str): Paragraph ID
        iid (str): ImageItem ID

    Returns:
        List[Paragraph]: a list with at most one element, i.e., the single-item paragraph object
    """
    if pid:
        pid = ObjectId(pid)
        p = Paragraph.first(F.id == pid)
    elif iid:
        iid = ObjectId(iid)
        p = Paragraph.first(F.images == iid)

    if iid and p:
        p.images = [i for i in p.images if i.id == iid]
        p.group_id = f"id={p['_id']}"
        return [p]
    else:
        return []
  

class Gallery(Plugin):

    def __init__(self, app):
        super().__init__(app)

        # ALBUM OPERATIONS

        @app.route('/api/gallery/grouping', methods=["GET", "POST", "PUT"])
        @rest()
        def grouping(ids, group='', delete=False):
            """Grouping selected paragraphs

            Returns:
                Response: 'OK' if succeeded
            """
            def gh(x): return hashlib.sha256(
                x.encode('utf-8')).hexdigest()[-9:]

            paras = list(Paragraph.query(
                F.id.in_([ObjectId(_) if len(_) == 24 else _ for _ in ids])))

            if delete:
                group_id = ''
                for p in paras:
                    p.keywords = [
                        _ for _ in p.keywords if not _.startswith('*')]
                    p.save()

            else:
                if not paras:
                    return True
                gids = []
                for p in paras:
                    gids += [_ for _ in p.keywords if _.startswith('*')]
                named = [_ for _ in gids if not _.startswith('*0')]

                if group:
                    group_id = '*' + group
                elif named:
                    group_id = min(named)
                elif gids:
                    group_id = min(gids)
                else:
                    group_id = '*0' + gh(min(map(lambda p: str(p.id), paras)))

                for p in paras:
                    if group_id not in p.keywords:
                        p.keywords.append(group_id)
                        p.save()

                gids = list(set(gids) - set(named))
                if gids:
                    for p in Paragraph.query(F.keywords.in_(gids)):
                        for id0 in gids:
                            if id0 in p.keywords:
                                p.keywords.remove(id0)
                        if group_id not in p.keywords:
                            p.keywords.append(group_id)
                        p.save()

            return group_id


'''
Some formerly used util functions, need further clean up
'''


def fetch_items(stored=True):
    """Fetch all items from database

    Args:
        stored (bool, optional): return only items with Storage = True or thumbnailed

    Returns:
        Set: item id strings
    """
    cond = {} if not stored else (
        F['source.file'].exists(1) | (F.thumbnail != None))
    return {str(i['_id']) for i in ImageItem.query(cond).rs}


def remove_unused_items(log=print):
    """Remove unused items
    """
    items = fetch_items(False)
    for p in Paragraph.query({}).rs:
        for ii in p['images']:
            if str(ii) in items:
                items.remove(str(ii))
    log(len(items))
    ImageItem.query(F.id.in_(list(items))).delete()


def find_latest_id():
    """Find latest id in storage
    """
    with StorageManager() as mgr:
        kmax = ''
        for f in mgr.files:
            k = max([_ for _ in f['data'] if len(_) == 24])
            kmax = max(k, kmax)
        return kmax


def push(since, target_url):
    """Fetch items data since id
    """
    from tqdm import tqdm
    from io import BytesIO
    import base64
    
    items = ImageItem.query(F.id > since)
    pbar = tqdm(total=items.count())

    def _push(k, data: BytesIO, url):
        data = base64.b64encode(data.getvalue())
        requests.put(url + '/api/imageitem/put_storage/' + k, data)
        pbar.set_description(f"{k} len={len(data)}")
        pbar.update(1)

    with StorageManager() as mgr:
        for i in items:
            k = str(i.id)
            try:
                _push(k, mgr.read(k), target_url)
            except:
                pass


def fix_items():
    """Remove non-existent items
    """
    aa = {}
    for a in Paragraph.aggregator.lookup(from_=F.imageitem, localField=F.images, foreignField=F.id, as_=F.items2).project(items2=1, c1=Fn.size(Var.images), c2=Fn.size(Var.items2), _id=1).match(Fn.expr(Var.c1 != Var.c2)).perform(raw=True):
        aa[a['_id']] = [_['_id'] for _ in a['items2']]

    for a in aa:
        Paragraph.query(F.id == a).update(Fn.set(images=aa[a]))

    Paragraph.query(F.images == []).delete()


def merge_items(log=print):
    """Merge items w.r.t. their urls
    """
    item_urls = defaultdict(list)
    for i in ImageItem.query({}):
        item_urls[i.source['url']].append(i)
    for k, v in item_urls.items():
        if len(v) > 1:
            stored = [i for i in v if i.source.get('file')]
            if len(stored) > 0:
                replace_to = stored[0]
            else:
                replace_to = v[0]
            for i in v:
                if i.id != replace_to.id:
                    Paragraph.query(F.images == i.id).update(
                        Fn.push(images=ObjectId(replace_to.id)))
                    Paragraph.query(F.images == i.id).update(
                        Fn.pull(images=ObjectId(i.id)))
                    i.delete()
                    log('replace', i.id, 'to', replace_to.id)


def merge_albums(filter_expr='', log=print):
    """Merge albums w.r.t their items and urls
    """
    def do_merge(albums):
        albums = [_ for _ in albums if _.id]
        if len(albums) < 2:
            return
        albums = sorted(albums, key=lambda p: p.id)
        for p in albums[1:]:
            albums[0].keywords += p.keywords
            albums[0].images += p.images
            p.delete()
            p._id = ''
        albums[0].images = sorted(
            set(albums[0].images), key=lambda x: x.source['url'])
        albums[0].save()

    _urls = defaultdict(list)
    _items = defaultdict(list)

    for p in Paragraph.query(parser.eval(filter_expr)):
        _urls[p.source['url']].append(p)
        for i in p.images:
            _items[i.id].append(p)

    for v in _urls.values():
        if len(v) > 1:
            do_merge(v)
            log('merge', *v)

    for v in _items.values():
        if len(v) > 1:
            do_merge(v)
            log('merge', *v)
