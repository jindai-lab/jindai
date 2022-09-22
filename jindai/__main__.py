"""CLI for jindai"""

import base64
import datetime
import json
import glob
import os
import re
import zipfile
from io import BytesIO
from typing import Dict, Iterable

import click
import h5py
import numpy as np
from flask import Flask
from tqdm import tqdm

from PyMongoWrapper import ObjectId
from PyMongoWrapper.dbo import create_dbo_json_encoder
from . import Plugin, PluginManager, Task, storage, config
from .api import run_service, prepare_plugins
from .helpers import get_context, safe_import
from .models import F, MediaItem, Meta, TaskDBO, User

MongoJSONEncoder = create_dbo_json_encoder(json.encoder.JSONEncoder)

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def _mongodb(coll):
    """Get mongodb cursors"""
    return Meta.db.database[coll]


def _init_plugins():
    """Inititalize plugins"""
    return PluginManager(get_context('plugins', Plugin), Flask(__name__))


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
        ('Export', {'output_format': 'xlsx', 'inp': 'return'})
    ], params={'query': query})

    xlsx = task_obj.execute()

    with open(output_file, 'wb') as output_file:
        output_file.write(xlsx)


@cli.command('task')
@click.argument('task_id')
@click.option('-n', '--concurrent', type=int, default=0)
@click.option('-v', '--verbose', type=bool, flag_value=True)
def run_task(task_id, concurrent, verbose):
    """Run task according to id or name"""
    dbo = TaskDBO.first((F.id == task_id) if re.match(
        r'[0-9a-f]{24}', task_id) else (F.name == task_id))
    if not dbo:
        print(f'Task {task_id} not found')
        return
    
    _init_plugins()
    task = Task.from_dbo(dbo, verbose=verbose)
    if concurrent > 0:
        task.concurrent = concurrent
    result = task.execute()
    print(result)


@cli.command('user')
@click.option('--add', '-a', default='')
@click.option('--setrole', '-g', default='')
@click.option('--delete', '-d', default='')
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

    items = {str(i['_id']) for i in MediaItem.aggregator.match(
        F['source.file'] == 'blocks.h5').project(_id=1).perform(raw=True)}

    items = items.union({i.thumbnail[:24] for i in MediaItem.query(
        F.thumbnail.exists(1) & (F.thumbnail != '')) if i.thumbnail})

    print(len(items), 'items')
    output_file = h5py.File(output, 'r+' if os.path.exists(output) else 'w')
    total = 0
    for filename in infiles:
        filename = h5py.File(filename, 'r')
        for k in tqdm(filename['data']):
            try:
                if k[:24] in items:
                    dat = filename[f'data/{k}']
                    total += len(dat)
                    output_file[f'data/{k}'] = np.frombuffer(dat[:].tobytes(),
                                                             dtype='uint8')
            except Exception as ex:
                print(k, ex)
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


@cli.command('replace-tag')
@click.option('--from', '-f', 'from_')
@click.option('--to', '-t')
def replace_tag(from_, to):
    from plugins.autotagging import AutoTag
    from .models import Paragraph, Term, Fn
    Paragraph.query(F.keywords == from_, F.author == from_).update(Fn.set(author=to))
    qs = Paragraph.query(F.keywords == from_)
    qs.update(Fn.addToSet(keywords=to))
    qs.update(Fn.pull(keywords=from_))
    AutoTag.query(F.tag == from_).update(Fn.set(tag=to))
    Term.query(F.term == from_, F.field == 'keywords').update(Fn.set(F.term == to))
    print('OK')


@cli.command('keywords-fix')
@click.option('--cond', '-c')
def replace_tag(cond):
    from .models import Paragraph, Fn
    from .dbquery import parser
    rs = Paragraph.query(parser.parse(cond)).update([Fn.set(keywords=Fn.setIntersection('$keywords', '$keywords'))])
    print('OK', rs.modified_count)


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
        print(str(ex)[:100])


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
                if coll == 'mediaitem':
                    restore_items.add(cid)
                else:
                    restore_albums.add(cid)
                collections.append('paragraph')
                collections.append('mediaitem')

            if coll not in collections:
                collections.append(coll)

        for coll in collections:
            print('restoring', coll, '...')
            buf = BytesIO(zip_file.read(coll))
            records = []
            for line in tqdm(buf):
                record = json.loads(line.decode('utf-8'),
                                    object_hook=_restore_hook)
                if (not restore_items and not restore_albums) or (
                    restore_items and (
                        (coll == 'mediaitem' and record['_id'] in restore_items) or (
                            coll == 'paragraph' and restore_items.intersection(
                                set(record['images'])))
                    )
                ):
                    # print('\nfound match', p['_id'])
                    records.append(record)
                elif restore_albums and (
                    coll == 'paragraph' and (
                        record['_id'] in restore_albums or
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


@cli.command('plugin-install')
@click.argument('url')
def plugin_install(url: str):
    """Install plugin

    :param url: install from
    :type url: str
    """
    pmanager = _init_plugins()
    pmanager.install(url)


@cli.command('plugin-export')
@click.option('--output', '-o')
@click.argument('infiles', nargs=-1)
def plugin_export(output: str, infiles):
    """Export plugin

    :param output: output file name
    :type output: str
    :param infiles: includes path
    :type infiles: path
    """
        
    def _all_files(path):
        if os.path.isfile(path):
            yield path
        else:
            for base, _, files in os.walk(path):
                if base == '__pycache__': continue
                for f in files:
                    yield os.path.join(base, f)
                
    def _export_one(outputzip, filelist):
        if not outputzip.startswith('jindai.plugins.'):
            outputzip = f'jindai.plugins.{outputzip}'
        if outputzip.endswith('.zip'):
            outputzip = outputzip[:-4]
            
        print('output to', outputzip)
        with zipfile.ZipFile(outputzip + '.zip', 'w', zipfile.ZIP_DEFLATED) as zout:
            for filepath in filelist:
                for filename in _all_files(filepath):
                    print(' ...', filename)
                    zout.write(filename, filename)
                
    if len(infiles) > 0:
        _export_one(output, infiles)
    else:
        for p in glob.glob('plugins/*'):
            pname = os.path.basename(p)
            if pname.startswith(('_', 'temp_')) \
                or ('.' in p and not p.endswith('.py')) \
                or ('.' not in p and os.path.isfile(p)):
                continue
            if pname in ('datasources', 'hashing', 'imageproc', 'pipelines', 'shortcuts',
                     'taskqueue.py', 'onedrive.py', 'scheduler.py', 'autotagging.py'):
                continue
            
            _export_one(os.path.basename(p).split('.')[0], [p])


@cli.command('clear-duplicates')
@click.option('--limit', '-l', type=int, default=0)
@click.option('--offset', '-s', type=str, default='')
@click.option('--maxdups', '-m', type=int, default=10)
def clear_duplicates(limit: int, offset: str, maxdups: int):
    """Clear duplicate media items

    :param limit: limit number of items to check
    :type limit: int
    """

    if len(offset) == 24:
        offset = F.id < ObjectId(offset)
    else:
        offset = {}

    from .models import Paragraph, MediaItem

    rs = MediaItem.query(~F.dhash.empty(), ~F.whash.empty(), offset).sort(-F.id)
    if limit:
        rs = rs.limit(limit)

    def _around(m, ratio=0.1):
        return (F.width <= m.width * (1+ratio)) & (F.width >= m.width * (1-ratio)) & \
            (F.height <= m.height * (1+ratio)) & (F.height >= m.height * (1-ratio))

    cleared = 0

    try:
        for m in tqdm(rs, total=min(limit, rs.count()) if limit else rs.count()):
            dups = list(MediaItem.query(F.dhash == m.dhash,
                        F.whash == m.whash, F.id != m.id, _around(m)))
            if len(dups) > maxdups:
                print(m.id, m.dhash.hex(), len(dups))
                continue

            if dups:
                Paragraph.merge_by_mediaitems(m, dups)
                cleared += len(dups)
    except KeyboardInterrupt:
        pass
    except Exception as ex:
        print(ex)

    print(cleared, 'duplicates merged.')
    if m:
        print('You may continue with offset', m.id)


@cli.command('sync-terms')
@click.option('--field', default='keywords')
@click.option('--cond', default='')
def sync_terms(field, cond):
    from .models import Term, Paragraph
    from .dbquery import parser
    Term.query(F.field == field).delete()
    agg = Paragraph.aggregator.project(**{field: 1})
    if field == 'keywords': agg = agg.unwind('$' + field)
    if cond:
        agg = agg.match(parser.parse(cond))
    agg = agg.group(_id='$' + field).project(term='$_id', field=field)
    
    batch = []
    for p in tqdm(agg.perform(raw=True)):
        batch.append(p)
        if len(batch) == 100:
            Term.db.insert_many(batch)
            batch.clear()
    
    if batch:
        Term.db.insert_many(batch)
    
    print('OK')


@cli.command('web-service')
@click.option('--port', default=8370, type=int)
@click.option('--deployment', '-D', default=False, flag_value=True)
def web_service(port: int, deployment: bool):
    """Run web service on port

    :param port: port number
    :type port: int
    """
    from .api import app
    prepare_plugins()
    if deployment:
        safe_import('waitress')
        from waitress import serve
        serve(app, host='0.0.0.0', port=port, threads=8)
    else:
        run_service(port=port)

            
@cli.command('storage-serve')
@click.option('--port', '-p', default=8371)
@click.option('--host', '-h', default='0.0.0.0')
@click.option('--debug', '-d', default=None, flag_value=True)
def serve_storage(port: int, host: str, debug: bool):
    """Serve storage
    """
    if debug is None: debug = config.debug
    _init_plugins()
    storage.serve(host, port, debug=debug)


@cli.command('ipython')
def call_ipython():
    from IPython import start_ipython
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    
    import jindai
    from PyMongoWrapper import F, Fn, Var
    from bson import SON, Binary, ObjectId
    import sys
    import glob
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor
    tpe = ThreadPoolExecutor(os.cpu_count())
    init = _init_plugins
    
    def q(query_str, model=''):
        from jindai import parser, Paragraph
        
        if isinstance(model, str):
            model = Paragraph.get_coll(model)
        
        q = parser.parse(query_str)
        if isinstance(q, list):
            return model.aggregate(q)
        
        return model.query(q)
    
    def run(task_name):
        dbo = TaskDBO.first((F.id if re.match('^[0-9a-fA-F]{24}$', task_name) else F.name) == task_name)
        if dbo:
            task = Task.from_dbo(dbo)
            return task.execute()
    
    ns = dict(jindai.__dict__)
    ns.update(**locals())

    start_ipython(argv=[], user_ns=ns)


if __name__ == '__main__':
    print('* loaded config from', config._filename)
    print('* using', config.mongo, config.mongoDbName)
    cli()
