#!/usr/bin/python3
from collections import defaultdict
from gallery import *
from models import Paragraph 
import numpy as np
import h5py
import glob
import fire
from tqdm import tqdm

log = print
find_k = lambda k: k.split('.')[0] # {k}[.{ext}]


def fetch_items(stored=True):
    """Fetch all items from database

    Args:
        stored (bool, optional): return only items with Storage = True or thumbnailed

    Returns:
        Set: item id strings
    """    
    cond = {} if not stored else (F['source.file'].exists(1) | (F.thumbnail != None))
    return {str(i['_id']) for i in ImageItem.query(cond).rs}


def remove_unused_items():
    """Remove unused items
    """    
    items = fetch_items(False)
    for p in Paragraph.query({}).rs:
        for ii in p['images']:
            if str(ii) in items: items.remove(str(ii))
    log(len(items))
    ImageItem.query(F.id.in_(list(items))).delete()


def find_latest_id():
    """Find latest id in storage
    """
    with StorageManager() as mgr:
        kmax = ''
        for f in mgr.files:
            k = max([_ for _ in f['data'] if len(_) == 24])
            print(f, k)
            kmax = max(k, kmax)
        print(kmax)
        return kmax


def push(since, target_url):
    """Fetch items data since id
    """
    items = ImageItem.query(F.id > since)
    pbar = tqdm(total=items.count())

    def _push(k, data : BytesIO, url):
        data = base64.b64encode(data.getvalue())
        requests.put(url + '/api/gallery/imageitem/put_storage/' + k, data)
        pbar.set_description(f"{k} len={len(data)}")
        pbar.update(1)

    with StorageManager() as mgr:
        for i in items:
            k = str(i.id)
            try:
                _push(k, mgr.read(k), target_url)
            except: pass


def fix_items():
    """Remove non-existent items
    """
    aa = {}
    for a in Paragraph.aggregator.lookup(from_=F.imageitem, localField=F.images, foreignField=F.id, as_=F.items2).project(items2=1, c1=Fn.size(Var.images), c2=Fn.size(Var.items2), _id=1).match(Fn.expr(Var.c1 != Var.c2)).perform(raw=True):
        aa[a['_id']] = [_['_id'] for _ in a['items2']]

    for a in aa:
        Paragraph.query(F.id == a).update(Fn.set(images=aa[a]))    

    Paragraph.query(F.images == []).delete()

        
def merge_items():
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
                    Paragraph.query(F.images == i.id).update(Fn.push(images=ObjectId(replace_to.id)))
                    Paragraph.query(F.images == i.id).update(Fn.pull(images=ObjectId(i.id)))
                    i.delete()
                    log('replace', i.id, 'to', replace_to.id)
                    
                    
def merge_albums(expr=''):
    """Merge albums w.r.t their items and urls
    """    
    def do_merge(albums):
        albums = [_ for _ in albums if _.id]
        if len(albums) < 2: return
        albums = sorted(albums, key=lambda p: p.id)
        for p in albums[1:]:
            albums[0].keywords += p.keywords
            albums[0].images += p.images
            p.delete()
            p._id = ''
        albums[0].images = sorted(set(albums[0].images), key=lambda x: x.source['url'])
        albums[0].save()
    
    _urls = defaultdict(list)
    _items = defaultdict(list)
    
    print(parser.eval(expr))
    for p in Paragraph.query(parser.eval(expr)):
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


if __name__ == '__main__':
    fire.Fire()
