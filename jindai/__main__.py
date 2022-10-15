"""CLI for jindai"""

import base64
from collections import defaultdict
import datetime
import glob
import json
import os
import re
import subprocess
import sys
import zipfile
from io import BytesIO
from itertools import chain
from tempfile import mktemp
from typing import Dict, Iterable

import click
import h5py
import numpy as np
import urllib3
import yaml
from flask import Flask
from PyMongoWrapper import Fn, ObjectId
from PyMongoWrapper.dbo import BatchSave, create_dbo_json_encoder
from tqdm import tqdm

from . import Plugin, PluginManager, Task, config, storage
from .api import prepare_plugins, run_service
from .common import DictObject
from .helpers import get_context, safe_import
from .models import F, MediaItem, Meta, Paragraph, TaskDBO, User, Dataset

MongoJSONEncoder = create_dbo_json_encoder(json.encoder.JSONEncoder)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def _mongodb(coll):
    """Get mongodb cursors"""
    return Meta.db.database[coll]


def _init_plugins():
    """Inititalize plugins"""
    return PluginManager(get_context('plugins', Plugin), Flask(__name__))


def _get_items():
    mediaitems = {m['_id']
                  for m in MediaItem.aggregator.project(_id=1).perform(raw=True)}
    paraitems = {p['_id']: p['images']
                 for p in Paragraph.aggregator.project(_id=1, images=1).perform(raw=True)}
    checkeditems = set(chain(*paraitems.values()))    
    unlinked = mediaitems - checkeditems
    return DictObject(locals())


@click.group()
def cli():
    """Cli group"""
    
    
@cli.command('init')
def first_run():
    if Meta.query().count() != 0:
        print('Already initialized.')
        return
    
    admin = User(name='admin', roles=['admin'])
    admin.set_password('admin')
    admin.save()
    print('Created user: admin, password: admin')
    
    Dataset(display_name='Default').save()
    
    Meta().save()


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
@click.option('-l', '--log', type=str, default='')
@click.option('-n', '--concurrent', type=int, default=0)
@click.option('-v', '--verbose', type=bool, flag_value=True)
@click.option('-e', '--edit', type=bool, flag_value=True)
def run_task(task_id, concurrent, verbose, edit, log):
    """Run task according to id or name"""
    dbo = TaskDBO.first((F.id == task_id) if re.match(
        r'[0-9a-f]{24}', task_id) else (F.name == task_id))
    if not dbo:
        print(f'Task {task_id} not found')
        return

    if edit:
        temp_name = mktemp()
        with open(temp_name, 'w', encoding='utf-8') as fo:
            dat = dbo.as_dict()
            dat.pop('_id', '')
            dat.pop('last_run', '')
            yaml.safe_dump(dat, fo, allow_unicode=True)

        if os.name == 'nt':
            editor = 'notepad.exe'
        elif os.system('which nano') == 0:
            editor = 'nano'
        else:
            editor = 'vi'

        subprocess.Popen([editor, temp_name]).communicate()

        with open(temp_name, encoding='utf-8') as fi:
            param = yaml.safe_load(fi)
            for key, val in param.items():
                dbo[key] = val
            dbo.save()

        os.unlink(temp_name)

    _init_plugins()

    logfile = open(log, 'w', encoding='utf-8') if log else sys.stderr

    task = Task.from_dbo(dbo, verbose=verbose,
                         logger=lambda *x: print(*x, file=logfile))

    if concurrent > 0:
        task.concurrent = concurrent

    result = task.execute()

    print()
    print(result)

    if log:
        print(result, file=logfile)
        logfile.close()


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
        if not user:
            print('User', setrole, 'does not exist.')
            exit()
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
@click.option('--output', '-o', default='tmp')
@click.argument('infiles', nargs=-1)
def storage_merge(infiles, output):
    """Merge h5df storage files"""

    class _QuotaWriter:

        def __init__(self, quota: int, pattern='out') -> None:
            self.quota = quota
            self.subtotal = 0
            self.file_num = 1
            self.file_pattern = pattern + '{}.h5'
            self._file = None

        @property
        def file(self):
            if not self._file:
                self._file = h5py.File(
                    self.file_pattern.format(self.file_num), 'w')
                self.file_num += 1
            return self._file

        def close(self):
            if self._file:
                self._file.flush()
                self._file.close()
                self._file = None
            self.subtotal = 0

        def write(self, key, val):
            if self.subtotal + len(val) > self.quota:
                self.close()

            self.file[f'data/{key}'] = np.frombuffer(val, dtype='uint8')
            self.subtotal += len(val)

    inputs = []
    for file in infiles:
        inputs += glob.glob(file)
    print(len(inputs), 'files')

    items = {str(i['_id'])
             for i in MediaItem.aggregator.project(_id=1).perform(raw=True)}
    items.update({i.thumbnail.split(
        '://')[1] for i in MediaItem.query(F.thumbnail.regex('^hdf5://')) if i.thumbnail})
    print(len(items), 'items')

    output_file = _QuotaWriter(40 << 30, output)
    total = 0
    for h5 in inputs:
        h5 = h5py.File(h5, 'r')
        if 'data' not in h5:
            continue
        for key in tqdm(h5['data']):
            try:
                if key in items:
                    val = h5[f'data/{key}'][:].tobytes()
                    total += len(val)
                    output_file.write(key, val)
                    items.remove(key)
            except Exception as ex:
                print(key, ex)
        h5.close()
    output_file.close()
    print('Total:', total, 'bytes')


@cli.command('storage-convert')
@click.option('--infile', '-i')
@click.option('--output', '-o')
@click.option('--format', '-f')
def stroage_convert(infile, output, format):
    assert format in ('sqlite',)

    if format == 'sqlite':
        import sqlite3
        conn = sqlite3.connect(output)
        outp = conn.cursor()
        outp.execute("""
                     CREATE TABLE IF NOT EXISTS data (id TEXT PRIMARY KEY, bytes BLOB, ctime TIMESTAMP)
                     """)
        inp = h5py.File(infile)['data']

        for k in tqdm(inp):
            buf = inp[k][:].tobytes()
            try:
                outp.execute("""
                            REPLACE INTO data VALUES (?, ?, ?)
                            """, (k, buf, datetime.datetime.utcnow()))
            except Exception as ex:
                print(ex)

        outp.close()
        conn.commit()
        conn.close()


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
        output = f'dump-{config.mongoDbName}-{datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")}'
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

    from .models import Fn, Paragraph, Term
    Paragraph.query(F.keywords == from_, F.author ==
                    from_).update(Fn.set(author=to))
    qs = Paragraph.query(F.keywords == from_)
    qs.update(Fn.addToSet(keywords=to))
    qs.update(Fn.pull(keywords=from_))
    AutoTag.query(F.tag == from_).update(Fn.set(tag=to))
    Term.query(F.term == from_, F.field ==
               'keywords').update(Fn.set(F.term == to))
    print('OK')


@cli.command('keywords-fix')
@click.option('--cond', '-c')
def replace_tag(cond):
    from .dbquery import parser
    from .models import Fn, Paragraph
    rs = Paragraph.query(parser.parse(cond)).update(
        [Fn.set(keywords=Fn.setUnion('$keywords', []))])
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
@click.option('-i', '--infile', default='')
@click.option('--force', type=bool, default=False, flag_value=True)
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
                    records.append(record)
                elif restore_albums and (
                    coll == 'paragraph' and (
                        record['_id'] in restore_albums or
                            restore_albums.intersection(set(record['keywords'])))
                ):
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
                if base == '__pycache__':
                    continue
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


@cli.command('items-dedup')
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

    from .models import MediaItem, Paragraph

    rs = MediaItem.query(~F.dhash.empty(), ~
                         F.whash.empty(), offset).sort(-F.id)
    if limit:
        rs = rs.limit(limit)

    def _around(m, ratio=0.1):
        return (F.width <= m.width * (1+ratio)) & (F.width >= m.width * (1-ratio)) & \
            (F.height <= m.height * (1+ratio)) & (F.height >= m.height * (1-ratio))

    cleared = set()

    try:
        for m in (pbar := tqdm(rs, total=min(limit, rs.count()) if limit else rs.count())):
            if m.id in cleared:
                continue

            dups = [d for d in MediaItem.query(
                    F.item_type == m.item_type,
                    F.dhash == m.dhash, F.whash == m.whash, F.id != m.id, _around(m))]

            if len(dups) > maxdups:
                print(m.id, m.dhash.hex(), len(dups))
                continue

            if dups:
                Paragraph.merge_by_mediaitems(m, dups)
                cleared.update({d.id for d in dups})
                pbar.set_description(f'{len(cleared)} merged')

    except KeyboardInterrupt:
        pass
    except Exception as ex:
        print(type(ex).__name__, ex, locals().get('m', {'id': ''})['id'])

    print(len(cleared), 'duplicates merged.')
    if 'm' in locals():
        print('You may continue with offset', m.id)


@cli.command('items-fix')
@click.option('-q', '--quiet', default=False, flag_value=True)
def fix_integrity(quiet):
    Paragraph.query().update([Fn.set(images=Fn.setUnion('$images', []))])
    
    r = _get_items()
    mediaitems = r.mediaitems
    unlinked = r.unlinked
    checkeditems = r.checkeditems
    paraitems = r.paraitems
    
    print(len(mediaitems), 'items', len(unlinked), 'unlinked')
    
    for p in tqdm(Paragraph.aggregator.match(
            F.images.in_(checkeditems - mediaitems)
        ).project(
            _id=1, images=1
        ).perform(
            raw=True
        ), desc='Checking paragraph items'):
        images = set(p['images']).intersection(mediaitems)
        if len(images) != len(p['images']):
            Paragraph.query(F.id == p['_id']).update(
                Fn.set(images=list(images)))
            paraitems[p['_id']] = list(images)
            
    itemparas = defaultdict(set)
    for pid, images in paraitems.items():
        for i in images:
            itemparas[i].add(pid)
    for iid, paras in tqdm(itemparas.items(), desc='Clearing repeated items'):
        if len(paras) > 1:
            paras = sorted(paras)
            Paragraph.query(F.id.in_(paras[1:])).update(Fn.pull(images=iid))
    
    print(len(unlinked), 'unlinked items')
    if len(unlinked) and (quiet or click.confirm('restore?')):
        batch_name = 'restored:' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        with BatchSave(performer=Paragraph) as batch:
            for m in tqdm(unlinked, desc='Restoring unlinked media items'):
                m = MediaItem.first(F.id == m)
                if m is None:
                    continue
                batch.add(Paragraph(dataset='', lang='auto', images=[m], source=m.source,
                                    keywords=['restored', batch_name]
                                    ))


@cli.command('sync-terms')
@click.option('--field', default='keywords')
@click.option('--cond', default='')
def sync_terms(field, cond):
    from .dbquery import parser
    from .models import Paragraph, Term

    Term.query(F.field == field).delete()
    agg = Paragraph.aggregator.project(**{field: 1})
    if field == 'keywords':
        agg = agg.unwind('$' + field)
    if cond:
        agg = agg.match(parser.parse(cond))
    agg = agg.group(_id='$' + field).project(term='$_id', field=field)

    with BatchSave(performer=Term) as batch:
        for p in tqdm(agg.perform(raw=True)):
            batch.add(p)

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
    if debug is None:
        debug = config.debug
    _init_plugins()
    storage.serve(host, port, debug=debug)


@cli.command('ipython')
def call_ipython():
    from IPython import start_ipython
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

    import glob
    import sys
    from concurrent.futures import ThreadPoolExecutor

    from bson import SON, Binary, ObjectId
    from PyMongoWrapper import F, Fn, Var
    from PyMongoWrapper.dbo import BatchQuery, BatchSave
    from tqdm import tqdm

    import jindai
    tpe = ThreadPoolExecutor(os.cpu_count())
    init = _init_plugins
    get_items = _get_items

    def q(query_str, model=''):
        from jindai import Paragraph, parser

        if isinstance(model, str):
            model = Paragraph.get_coll(model)

        q = parser.parse(query_str)
        if isinstance(q, list):
            return model.aggregate(q)

        return model.query(q)

    def run(task_name):
        dbo = TaskDBO.first(
            (F.id if re.match('^[0-9a-fA-F]{24}$', task_name) else F.name) == task_name)
        if dbo:
            task = Task.from_dbo(dbo)
            return task.execute()

    def deep_delete(rs):
        mediaitems = set()
        for p in rs:
            if isinstance(p, Paragraph):
                for m in p.images:
                    m.delete()
            elif isinstance(p, MediaItem):
                mediaitems.add(m.id)
            p.delete()

        for m in mediaitems:
            Paragraph.query(F.images == m.id).update(Fn.pull(images=m.id))
            
    def read_dump(dump_file, collection):
        with zipfile.ZipFile(dump_file, 'r') as zfile:
            for obj in zfile.open(collection):
                obj = json.loads(obj)
                yield DictObject(obj)
                
    ns = dict(jindai.__dict__)
    ns.update(**locals())

    start_ipython(argv=[], user_ns=ns)


if __name__ == '__main__':
    print('* loaded config from', config._filename)
    print('* using', config.mongo, config.mongoDbName)
    cli()
