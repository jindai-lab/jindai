"""命令行界面"""

import base64
import datetime
import json
import os
import re
import zipfile
from io import BytesIO
from typing import Dict, Iterable

import click
import h5py
import numpy as np
import requests
from bson import ObjectId
from flask import Flask
from tqdm import tqdm

from PyMongoWrapper.dbo import create_dbo_json_encoder
from . import Plugin, PluginManager, Task, safe_open
from .helpers import get_context
from .models import F, ImageItem, Meta, TaskDBO, User

MongoJSONEncoder = create_dbo_json_encoder(json.encoder.JSONEncoder)


def _mongodb(coll):
    """Get mongodb cursors"""
    return Meta.db.database[coll]


def _init_plugins():
    """Inititalize plugins"""
    PluginManager(get_context('plugins', Plugin), Flask(__name__))


@click.group()
def cli():
    """Cli group"""


@cli.command('export')
@click.option('--output')
@click.option('--query')
def export(query, output_file):
    """Export query results to json"""
    _init_plugins()
    task_obj = Task(stages=[
        ('DBQueryDataSource', {'query': query}),
        ('AccumulateParagraphs', {}),
        ('Export', {'format': 'xlsx', 'inp': 'return'})
    ], params={'query': query})

    xlsx = task_obj.execute()

    with safe_open(output_file, 'wb') as output_file:
        output_file.write(xlsx)



@cli.command('task')
@click.argument('task_id')
@click.option('--concurrent', type=int, default=10)
def run_task(id_or_name, concurrent=0):
    """Run task according to id or name"""
    _init_plugins()
    task = Task.from_dbo(TaskDBO.first((F.id == id_or_name) if re.match(
        r'[0-9a-f]{24}', id_or_name) else (F.name == id_or_name)), logger='print', verbose=True)
    if concurrent > 0:
        task.pipeline.concurrent = concurrent
    result = task.execute()
    print(result)



@cli.command('enqueue')
@click.option('--item-id')
@click.option('--host')
@click.option('--ssl')
def task_enqueue(item_id, host='localhost:8370', ssl=False):
    """Enqueue task to server"""
    j = requests.put(f'http{"s" if ssl else ""}://{host}/api/queue/',
                     json={'id': item_id}, headers={'Accept-ContentType': 'text/plain'})
    if j.status_code == 200:
        print(j.json())
    else:
        print(j.content)


@cli.command('user')
@click.option('--add', default='')
@click.option('--setrole', default='')
@click.option('--delete', default='')
@click.argument('roles', nargs=-1)
def user_manage(add, delete, setrole, roles):
    """User management"""
    if add:
        print('Password: ', end='')
        password = input()
        if not User.first(F.username == add):
            user = User(username=add)
            user.set_password(password)
            user.save()
        else:
            print('User already exists.')
    elif delete:
        User.query(F.username == delete).delete()
    elif setrole:
        user = User.first(F.username == setrole)
        user.roles = roles
        user.save()


@cli.command('meta')
@click.option('--key')
@click.option('--value', default='')
def meta(key, value):
    """Get/set meta settings"""
    record = Meta.first({}) or Meta()
    if value:
        setattr(record, key, value)
        record.save()
    else:
        print(record[key] if hasattr(record, key) else '')


@cli.command('storage-merge')
@click.option('--output', '-o', default='tmp.h5')
@click.argument('infiles', nargs=-1)
def storage_merge(infiles, output):
    """Merge h5df storage files"""

    items = {str(i['_id']) for i in ImageItem.aggregator.match(
        F['source.file'] == 'blocks.h5').project(_id=1).perform(raw=True)}

    items = items.union({i.thumbnail[:24] for i in ImageItem.query(
        F.thumbnail.exists(1) & (F.thumbnail != '')) if i.thumbnail})

    print(len(items), 'items')
    output_file = h5py.File(output, 'r+' if os.path.exists(output) else 'w')
    total = 0
    for filename in infiles:
        filename = h5py.File(filename, 'r')
        for k in tqdm(filename['data']):
            if k[:24] in items:
                dat = filename[f'data/{k}']
                total += len(dat)
                output_file[f'data/{k}'] = np.frombuffer(dat[:].tobytes(),
                                                dtype='uint8')
    output_file.close()
    print('Total:', total, 'bytes')


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
        output = f'dump-{datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")}'
        if colls:
            output += '{("," + ",".join(colls))}.zip'
        else:
            output += '.zip'

    jsonenc = MongoJSONEncoder(ensure_ascii=False)
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for coll in colls or Meta.db.database.list_collection_names():
            output = BytesIO()
            for record in tqdm(_mongodb(coll).find(), total=_mongodb(coll).count_documents({})):
                output.write(jsonenc.encode(record).encode('utf-8') + b'\n')
            output.seek(0)
            zip_file.writestr(coll, output.read())


def _restore_hook(dic: Dict):
    """JSON decoder hook for restoring collections.

    Args:
        dic (dict): decoded dict

    Returns:
        dict: dic
    """
    if '_id' in dic:
        dic['_id'] = ObjectId(dic['_id'])
    if 'images' in dic and isinstance(dic['images'], list):
        dic['images'] = [ObjectId(_) for _ in dic['images']]
    for hash_method in ('dhash', 'whash'):
        if hash_method in dic:
            if isinstance(dic[hash_method], int):
                dic[hash_method] = f'{dic[hash_method]:016x}'
                dic[hash_method] = bytes.fromhex(dic[hash_method])
            elif isinstance(dic[hash_method], str):
                if len(dic[hash_method]) == 12:
                    dic[hash_method] = base64.b64decode(dic[hash_method])
                else:
                    dic[hash_method] = bytes.fromhex(dic[hash_method])
    return dic


def _save_db(coll: str, records: Iterable[Dict], force):
    """Write items to database.

    Args:
        records (list[dict]): list of decoded dicts from jsons.
    """
    try:
        if force:
            _mongodb(coll).delete_many(
                {'_id': {'$in': [p['_id'] for p in records]}})
        _mongodb(coll).insert_many(records, ordered=False,
                                    bypass_document_validation=True)
    except Exception as ex:
        print(ex.__class__.__name__)


@cli.command('restore')
@click.option('--infile', default='')
@click.option('--force', type=bool, default=False)
@click.argument('colls', nargs=-1)
def restore(infile, colls, force):
    """Restore the status of database from a zip file of jsons.

    Args:
        infile (str): input zip filename.
        colls (optional): list of collections
        force (bool, optional): delete before import
    """

    with zipfile.ZipFile(infile, 'r') as zip_file:
        restore_albums = set()
        restore_items = set()
        collections = []

        if not colls:
            colls = zip_file.namelist()
        for coll in colls:
            if ':' in coll:
                coll, cid = coll.split(':', 1)
                if len(cid) == 24:
                    cid = ObjectId(cid)
                if coll == 'imageitem':
                    restore_items.add(cid)
                else:
                    restore_albums.add(cid)
                collections.append('paragraph')
                collections.append('imageitem')

            if coll not in collections:
                collections.append(coll)

        for coll in collections:
            print('restoring', coll, '...')
            buf = BytesIO(zip_file.read(coll))
            records = []
            for line in tqdm(buf):
                record = json.loads(line.decode('utf-8'), object_hook=_restore_hook)
                if (not restore_items and not restore_albums) or (
                    restore_items and (
                        (coll == 'imageitem' and record['_id'] in restore_items) or (
                            coll == 'paragraph' and restore_items.intersection(
                                set(record['images'])))
                    )
                ):
                    # print('\nfound match', p['_id'])
                    records.append(record)
                elif restore_albums and (
                    coll == 'paragraph' and (
                        record['_id'] in restore_albums or \
                            restore_albums.intersection(set(record['keywords'])))
                ):
                    # print('\nfound match', p['_id'], p['images'])
                    records.append(record)
                    for i in record['images']:
                        restore_items.add(i)

                if len(records) > 100:
                    _save_db(coll, records, force)
                    records = []

            if records:
                _save_db(coll, records, force)


if __name__ == '__main__':
    cli()
