#!/usr/bin/python3
from collections import defaultdict
from gallery import *
from models import Album 
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
    cond = {} if not stored else (F.storage == True) | (F.thumbnail != None)
    return {str(i['_id']) for i in ImageItem.query(cond).rs}


def check_blocks_items():
    """Check itmes in h5 files

    Returns:
        Iterable: item id strings
    """    
    items = set()
    
    for g in glob.glob('blocks*.h5'):
        log(g)
        h = h5py.File(g, 'r')
        for k in h['data']:
            kid = find_k(k)
            items.add(kid)
        h.close()
    
    return items


def extract(items, h, t):
    """Extract specified items from h5 file `h` to `t`

    Args:
        items (Iterable): items to extract
        h (h5py.File): source
        t (h5py.File): target

    Returns:
        [type]: [description]
    """    
    if 'data' not in h:
        return t
    
    count = 0
    total = 0
    for k in tqdm(h['data']):
        count += 1
        if count % 100 == 0 and t is not None:
            t.flush()
        kid = find_k(k)
        if kid in items:
            try:
                dat = h[f'data/{k}']
                total += len(dat)
                if t is not None:
                    t[f'data/{k}'] = np.frombuffer(dat[:].tobytes(), dtype='uint8')
            except Exception as ex:
                log(ex)

    log('total:', total, 'bytes')
    return t


def merge(*files, dry_run=False, output='tmp.h5'):
    """Merge h5 files

    Args:
        dry_run (bool, optional): Dry run, makes no change to file system
        output (str, optional): Output file name
    """    
    items = fetch_items()    
    log(len(items), 'items.')

    if dry_run:
        t = None
        log('dry run.')
    else:
        t = h5py.File(output, 'w')
    
    for g in files:
        log('checking', g, '...')
        h = h5py.File(g, 'r')
        extract(items, h, t)
        h.close()
    
    if not dry_run:
        t.close()


def check_storage():
    """Sync storage status to items
    """    
    items = fetch_items()
    blocks = check_blocks_items()
    diff = items.difference(blocks)
    log(len(diff))
    for i in diff:
        ImageItem.query(F.id == i).update(Fn.set(storage=None))


def remove_unused_items():
    """Remove unused items
    """    
    items = fetch_items(False)
    for p in Album.query({}).rs:
        for ii in p['items']:
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
        requests.put(url + '/api/item/put_storage/' + k, data)
        pbar.set_description(f"{k} len={len(data)}")
        pbar.update(1)

    with StorageManager() as mgr:
        for i in items:
            k = str(i.id)
            try:
                _push(k, mgr.read(k), target_url)
            except: pass

        
def merge_items():
    """Merge items w.r.t. their urls
    """    
    item_urls = defaultdict(list)
    for i in ImageItem.query({}):
        item_urls[i.source['url']].append(i)
    for k, v in item_urls.items():
        if len(v) > 1:
            stored = [i for i in v if i.storage]
            if len(stored) > 0:
                replace_to = stored[0]
            else:
                replace_to = v[0]
            for i in v:
                if i.id != replace_to.id:
                    Album.query(F.items == i.id).update(Fn.push(items=ObjectId(replace_to.id)))
                    Album.query(F.items == i.id).update(Fn.pull(items=ObjectId(i.id)))
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
            albums[0].items.a += p.items.a
            p.delete()
            p._id = ''
        albums[0].items.a = sorted(set(albums[0].items.a), key=lambda x: x.source['url'])
        albums[0].save()
    
    _urls = defaultdict(list)
    _items = defaultdict(list)
    
    print(parser.eval(expr))
    for p in Album.query(parser.eval(expr)):
        _urls[p.source['url']].append(p)
        for i in p.items:
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
