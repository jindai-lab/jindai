import click
from PyMongoWrapper import F, Fn
import h5py
from models import Meta, User, TaskDBO, MongoJSONEncoder, ImageItem
from typing import Dict, Iterable
from task import Task
from tqdm import tqdm
import requests, zipfile
from bson import ObjectId
import base64, json
import datetime
from io import BytesIO
import os
import numpy as np


def mongodb(coll):
    return Meta.db.database[coll]


@click.group()
def cli():
    pass


@cli.command('export')
@click.option('--output')
@click.option('--query')
def export(query, output):

    task = Task(datasource=('DBQueryDataSource', {'query': query}), pipeline=[
        ('AccumulateParagraphs', {}),
        ('Export', {'format': 'xlsx', 'inp': 'return'})
    ])

    xlsx = task.execute()

    with open(output, 'wb') as fo:
        fo.write(xlsx)


@cli.command('task')
@click.argument('task_id')
def task(task_id):
    task = Task.from_dbo(TaskDBO.first(F.id == task_id))
    result = task.execute()
    print(result)
    

@cli.command('enqueue')
@click.option('--id')
def task_enqueue(id):
    j = requests.put('http://localhost:8370/api/queue/', json={'id': id}, headers={'Accept-ContentType': 'text/plain'})
    if j.status_code == 200:
        print(j.json())
    else:
        print(j.content)


@cli.command('user')
@click.option('--add', default='')
@click.option('--setrole', default='')
@click.option('--delete', default='')
@click.argument('roles', nargs=-1)
def user(add, delete, setrole, roles):
    if add:
        print('Password: ', end='')
        password = input()
        if not User.first(F.username == add):
            u = User(username=add)
            u.set_password(password)
            u.save()
        else:
            print('User already exists.')
    elif delete:
        User.query(F.username == delete).delete()
    elif setrole:
        u = User.first(F.username == setrole)
        u.roles = roles
        u.save()


@cli.command('meta')
@click.option('--key')
@click.option('--value', default='')
def meta(key, value):
    r = Meta.first({}) or Meta()
    if value:
        setattr(r, key, value)
        r.save()
    else:
        print(r[key] if hasattr(r, key) else '')


@cli.command('storage-merge')
@click.option('--output', '-o', default='tmp.h5')
@click.argument('infiles', nargs=-1)
def storage_merge(infiles, output):
    items = {str(i.id) for i in ImageItem.query(F['source.file'] == 'blocks.h5')}
    fo = h5py.File(output, 'r+' if os.path.exists(output) else 'w')
    total = 0
    for f in infiles:
        f = h5py.File(f, 'r')
        for k in tqdm(f['data']):
            if k[:24] in items:
                dat = f[f'data/{k}']
                total += len(dat)
                fo[f'data/{k}'] = np.frombuffer(dat[:].tobytes(), dtype='uint8')
    fo.close()
    print('Total:', total, 'bytes')


@cli.command('storage-sync')
@click.argument('infiles', nargs=-1)
def storage_sync(infiles):
    items = {str(i.id) for i in ImageItem.query(F['source.file']=='blocks.h5')}
    for f in infiles:
        f = h5py.File(f, 'r')
        for k in tqdm(f['data']):
            _id = k[:24]
            if _id in items: items.remove(_id)
    print(len(items))
    ImageItem.query(F.id.in_(list(items))).update([Fn.set(F['source.file']=='blocks.h5')])
                

@cli.command('dump')
@click.option('--output', default='')
@click.argument('colls', nargs=-1)
def dump(output, colls):
    """Dump the current status of database to a zip file of jsons.

    Args:
        output (str, optional): output zip filename
        colls (optional): list of collections
    """
    if not output:
        output = f'dump-{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}{("," + ",".join(colls)) if colls else ""}.zip'
    def _hook(s):
        if isinstance(s, datetime.datetime): return s.isoformat()
        else: return s

    jsonenc = MongoJSONEncoder(ensure_ascii=False)
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as z:
        for coll in colls or Meta.db.database.list_collection_names():
            fo = BytesIO()
            for p in tqdm(mongodb(coll).find(), total=mongodb(coll).count()):
                fo.write(jsonenc.encode(p).encode('utf-8') + b'\n')
            fo.seek(0)
            z.writestr(coll, fo.read())
    with open('auditor.log', 'w'):
        pass


@cli.command('restore')
@click.option('--infile', default='')
@click.option('--force', type=bool, default=False)
@click.argument('colls', nargs=-1)
def restore(infile, colls, force):
    """Restore the status of database from a zip file of jsons.

    Args:
        infile (str): input zip filename.
        colls (optional): list of collections
        force (bool, optional): ignore errors
    """
    def _hook(dic: Dict):
        """JSON decoder hook for restoring collections.

        Args:
            dic (dict): decoded dict

        Returns:
            dict: dic
        """
        if '_id' in dic:
            dic['_id'] = ObjectId(dic['_id'])
        if 'items' in dic and isinstance(dic['items'], list):
            dic['items'] = [ObjectId(_) for _ in dic['items']]
        for hw in ('dhash', 'whash'):
            if hw in dic:
                if isinstance(dic[hw], int):
                    dic[hw] = '%016x' % dic[hw]
                    dic[hw] = bytes.fromhex(dic[hw])
                elif isinstance(dic[hw], str):
                    if len(dic[hw]) == 12:
                        dic[hw] = base64.b64decode(dic[hw])
                    else:
                        dic[hw] = bytes.fromhex(dic[hw])
        return dic

    def _save_db(coll: str, ps: Iterable[Dict]):
        """Write items to database.

        Args:
            ps (list[dict]): list of decoded dicts from jsons.
        """
        try:
            if force:
                mongodb(coll).delete_many({'_id': {'$in': [p['_id'] for p in ps]}})
            mongodb(coll).insert_many(ps, ordered=False,
                                        bypass_document_validation=True)
        except Exception as ex:
            print(ex)

    with zipfile.ZipFile(infile, 'r') as z:
        restore_albums = set()
        restore_items = set()
        collections = []
        
        if not colls: colls = z.namelist()
        for coll in colls:
            if ':' in coll:
                coll, cid = coll.split(':', 1)
                if len(cid) == 24: cid = ObjectId(cid)
                if coll == 'imageitem':
                    restore_items.add(cid)
                else:
                    restore_albums.add(cid)
                collections.append('album')
                collections.append('imageitem')

            if coll not in collections: collections.append(coll)

        for coll in collections:
            print('restoring', coll, '...')
            fi = BytesIO(z.read(coll))
            ps = []
            for line in tqdm(fi):
                p = json.loads(line.decode('utf-8'), object_hook=_hook)
                if (not restore_items and not restore_albums) or (
                    restore_items and (
                        (coll == 'imageitem' and p['_id'] in restore_items) or (coll == 'album' and restore_items.intersection(set(p['items'])))
                    )
                ):
                    # print('\nfound match', p['_id'])
                    ps.append(p)
                elif restore_albums and (
                        coll == 'album' and (p['_id'] in restore_albums or restore_albums.intersection(set(p['tags'])))
                    ):
                    # print('\nfound match', p['_id'], p['items'])
                    ps.append(p)
                    for i in p['items']:
                        restore_items.add(i)

                if len(ps) > 100:
                    _save_db(coll, ps)
                    ps = []
            if ps:
                _save_db(coll, ps)


if __name__ == '__main__':
    cli()
    
