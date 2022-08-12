"""API Web Service"""

import datetime
import hashlib
import os
import re
import sys
import shutil
import time
from collections import defaultdict
from io import BytesIO
from typing import IO, Union

import pyotp
from flask import Flask, Response, redirect, request, send_file
from PIL import Image, ImageOps, ImageFile
from PyMongoWrapper import F, Fn, MongoOperand, ObjectId

from .dbquery import DBQuery, parser
from .pipeline import Pipeline
from .plugin import Plugin, PluginManager
from .task import Task
from .config import instance as config
from .helpers import (get_context, logined, rest, serve_file, language_iso639,
                      serve_proxy, JSONEncoder, JSONDecoder, ee)
from .models import (Dataset, History, MediaItem, Meta, Paragraph,
                     TaskDBO, Token, User, Term)
from .storage import expand_path, truncate_path


ImageFile.LOAD_TRUNCATED_IMAGES = True
app = Flask(__name__)
app.config['SECRET_KEY'] = config.secret_key
app.json_encoder = JSONEncoder
app.json_decoder = JSONDecoder


def _task_authorized():
    """Test if task is authorized to current user"""

    if logined('admin'):
        return MongoOperand({})

    return ((F.creator == logined()) | (F.shared == True))


def _expand_results(results):
    """Expand results to serializable dicts"""

    def _patch_mongocollection(r):
        if isinstance(r, Paragraph):
            res = r.as_dict(True)
            if 'mongocollection' not in res:
                res['mongocollection'] = type(r).db.name
        else:
            res = r
        return res

    if not isinstance(results, (str, dict, bytes)) and hasattr(results, '__iter__'):
        return [_patch_mongocollection(r) for r in results]

    return results


def _hashing(msg):
    """Hashing message with SHA-256 and preserve last 9 hexadecimal digits"""
    return hashlib.sha256(
        msg.encode('utf-8')).hexdigest()[-9:]


def _lang(inp):
    """Apply language settings of current client"""
    assert isinstance(inp, (str, list, dict)
                      ), "Input value must be string or dict"
    if isinstance(inp, str):
        result = ''
        default_result = ''
        for line in inp.split('\n'):
            line = line.lstrip()
            if line and not line.startswith('@'):
                default_result += line + ' '
            if request.lang and line.startswith('@' + request.lang + ' '):
                result += line[len(request.lang) + 2:]
        if not result:
            result = default_result
        result = result.strip()

    elif isinstance(inp, dict):
        result = {}
        for key in inp:
            if isinstance(inp[key], (dict, list, str)):
                result[key] = _lang(inp[key])
            else:
                result[key] = inp[key]

    else:  # list
        result = [
            _lang(inpk) if isinstance(inpk, (dict, list, str)) else inpk
            for inpk in inp
        ]

    return result


@app.route('/api/authenticate', methods=['POST'])
@rest(login=False)
def authenticate(username, password, otp='', **_):
    """Authenticate current user"""

    username = User.authenticate(username, password, otp)
    if username:
        Token.query((F.user == username) & (F.expire < time.time())).delete()
        token = User.encrypt_password(str(time.time()), str(time.time_ns()))
        Token(user=username, token=token, expire=time.time() + 86400).save()
        return token
    raise Exception("Unmatched credentials.")


@app.route('/api/authenticate')
@rest()
def whoami():
    """Returns logined user data, without otp_secret and password"""
    if logined():
        user = (User.first(F.username == logined()) or User(
            username=logined(), password='', roles=[])).as_dict()
        del user['password']
        user['otp_secret'] = True if user.get('otp_secret') else False
        return user
    return None


@app.route('/api/authenticate', methods=['DELETE'])
@rest()
def log_out():
    """Log out"""
    Token.uncheck(logined())
    return True


@app.route('/api/users/')
@app.route('/api/users/<username>', methods=['GET', 'POST'])
@rest(role='admin')
def admin_users(username='', password=None, roles=None, datasets=None, **_):
    """Change user profile and permissions"""

    result = None
    if username:
        user = User.first(F.username == username)
        if not user:
            return 'No such user.', 404
        if password is not None:
            user.set_password(password)
        if roles is not None:
            user.roles = roles
        if datasets is not None:
            user.datasets = datasets
        user.save()
        return result
    else:
        return list(User.query({}))


@app.route('/api/users/', methods=['PUT'])
@rest(role='admin')
def admin_users_add(username, password, **_):
    """Add new user"""

    if User.first(F.username == username):
        raise Exception('User already exists: ' + str(username))
    user = User(username=username)
    user.set_password(password)
    user.save()
    user = user.as_dict()
    del user['password']
    return user


@app.route('/api/account/', methods=['POST'])
@rest()
def user_change_password(old_password='', password='', otp=None, **_):
    """Change user passworld or OTP settings"""

    user = User.first(F.username == logined())
    if otp is None:
        assert User.authenticate(logined(), old_password), '原密码错误'
        user.set_password(password)
        user.save()
        user = user.as_dict()
        del user['password']
    else:
        if otp:
            user.otp_secret = pyotp.random_base32()
            user.save()
        else:
            user.otp_secret = ''
            user.save()
        return user.otp_secret

    return user


@app.route('/api/users/<username>', methods=['DELETE'])
@rest(role='admin')
def admin_users_del(username):
    """Delete user"""

    return User.query(F.username == username).delete()


def _file_detail(path):
    """Get detailed status of the file"""

    file_stat = os.stat(path)
    return {
        'name': os.path.basename(path),
        'fullpath': path[len(config.storage):],
        'ctime': file_stat.st_ctime,
        'mtime': file_stat.st_mtime,
        'size': file_stat.st_size,
        'type': 'folder' if os.path.isdir(path) else 'file'
    }


def _file_find(root_path, pattern):
    root_path = expand_path(root_path)
    conds = re.compile('.*'.join([re.escape(cond)
                       for cond in pattern.split() if cond]), flags=re.I)

    for pwd, _, files in os.walk(root_path):
        for f in files:
            if conds.search(f):
                yield os.path.join(pwd, f)


@app.route('/api/storage/<path:path>', methods=['GET', 'POST'])
@app.route('/api/storage/', methods=['GET', 'POST'])
@rest()
def list_storage(path='', search=''):
    """List out files in directory"""

    results = None
    if search:
        # path is a query
        results = list(_file_find(path, search))
    else:
        path = os.path.join(
            config.storage, path) if path and '..' not in path else config.storage
        if os.path.isdir(path):
            results = [os.path.join(path, x) for x in os.listdir(
                path) if not x.startswith(('@', '.'))]
        else:
            results = path

    if isinstance(results, list):
        return sorted(map(_file_detail, results),
                      key=lambda x: x['ctime'], reverse=True)
    else:
        return send_file(results)


@app.route('/api/storage/<path:path>', methods=['PUT'])
@app.route('/api/storage/', methods=['PUT'])
@rest()
def write_storage(path=''):
    """Write to file storage"""

    path = os.path.join(
        config.storage, path) if path and '..' not in path else config.storage
    sfs = []
    for uploaded in request.files.values():
        save_path = os.path.join(path, uploaded.filename)
        uploaded.save(save_path)
        sfs.append(_file_detail(save_path))
    return sfs


@app.route('/api/storage/move', methods=['POST'])
@rest()
def move_storage(source, destination, keep_folder=True):
    """Move/Rename file from source to destination"""
    if keep_folder:
        destination = os.path.basename(destination)
        destination = os.path.join(os.path.dirname(source), destination)
    paragraphs = Paragraph.query(F['source.file'] == source)
    source = expand_path(source)
    destination = expand_path(destination)
    shutil.move(source, destination)
    paragraphs.update(Fn.set({'source.file': truncate_path(destination)}))
    return True


@app.route('/api/collections/<coll>/<pid>', methods=['POST'])
@rest()
def modify_paragraph(coll, pid, **kws):
    """Modify paragraph info"""

    pid = ObjectId(pid)
    para = Paragraph.get_coll(coll).first(F.id == pid)
    flag = False
    if para:
        for field, val in kws.items():
            if field in ('_id', 'matched_content'):
                continue
            if field in ('$push', '$pull'):
                Paragraph.get_coll(coll).query(
                    F.id == pid).update({field: val})
            else:
                flag = True
                if val is None and hasattr(para, field):
                    delattr(para, field)
                else:
                    setattr(para, field, val)
        if flag:
            para.save()
    return True


@app.route('/api/collections/<coll>/<pid>/pagenum', methods=['POST'])
@rest()
def modify_pagenum(coll, pid, sequential, new_pagenum, folio=False, **_):
    """Modify page numbers of the same source"""

    pid = ObjectId(pid)
    para = Paragraph.get_coll(coll).first(F.id == pid)
    if para:
        if sequential == 'solo':
            para.pagenum = new_pagenum
            para.save()
        else:
            folio = 2 if folio else 1
            delta = new_pagenum - para.source['page'] * folio
            source = dict(para.source)
            assert 'page' in source
            if sequential == 'all':
                del source['page']
            else:  # after
                source['page'] = {'$gt': source['page']}
            source = {'source.' + k: w for k, w in source.items()}
            Paragraph.get_coll(coll).query((F.dataset == para.dataset) & MongoOperand(
                source)).update([Fn.set(pagenum=Fn.add(Fn.multiply('$source.page', folio), delta))])
            Paragraph.get_coll(coll).query((F.dataset == para.dataset) & (F.pagenum <= 0)).update([
                Fn.set(pagenum=Fn.concat(
                    "A", Fn.toString(Fn.add(1, "$source.page"))))
            ])
        return True
    return False


@app.route('/api/collections/<coll>/batch', methods=["GET", "POST"])
@rest()
def batch(coll, ids, **kws):
    """Batch edit"""

    paras = list(Paragraph.get_coll(coll).query(
        F.id.in_([ObjectId(_) if len(_) == 24 else _ for _ in ids])))
    for para in paras:
        for field, val in kws.items():
            if field in ('$pull', '$push'):
                for afield, aval in val.items():
                    if para[afield] is None:
                        para[afield] = []
                    for ele in (aval if isinstance(aval, list) else [aval]):
                        ele = ele.strip()
                        if field == '$pull' and ele in para[afield]:
                            para[afield].remove(ele)
                        elif field == '$push' and ele not in para[afield]:
                            para[afield].append(ele)
            elif not field.startswith('$'):
                para[field] = val
        para.save()

    return {
        str(p.id): p.as_dict()
        for p in paras
    }


@app.route('/api/collections/<coll>/group', methods=["GET", "POST"])
@rest()
def grouping(coll, ids, group='', ungroup=False):
    """Grouping selected paragraphs

    Returns:
        Group ID
    """
    paras = list(Paragraph.get_coll(coll).query(
        F.id.in_([ObjectId(_) if len(_) == 24 else _ for _ in ids])))
    if ungroup:
        group_id = ''
        for para in paras:
            para.keywords = [
                _ for _ in para.keywords if not _.startswith('*')]
            para.save()
    else:
        if not paras:
            return True
        gids = []
        for para in paras:
            gids += [_ for _ in para.keywords if _.startswith('*')]
        named = [_ for _ in gids if not _.startswith('*0')]
        if group:
            group_id = '*' + group
        elif named:
            group_id = min(named)
        elif gids:
            group_id = min(gids)
        else:
            group_id = '*0' + _hashing(min(map(lambda p: str(p.id), paras)))
        for para in paras:
            if group_id not in para.keywords:
                para.keywords.append(group_id)
                para.save()

        gids = list(set(gids) - set(named))
        if gids:
            for para in Paragraph.query(F.keywords.in_(gids)):
                for id0 in gids:
                    if id0 in para.keywords:
                        para.keywords.remove(id0)
                if group_id not in para.keywords:
                    para.keywords.append(group_id)
                para.save()

    return group_id


@app.route('/api/collections/<coll>/split', methods=["GET", "POST"])
@app.route('/api/collections/<coll>/merge', methods=["GET", "POST"])
@rest()
def splitting(coll, ids):
    """Split or merge selected items/paragraphs into seperate/single paragraph(s)

    Returns:
        bool: True if succeeded
    """
    paras = list(Paragraph.get_coll(coll).query(
        F.id.in_([ObjectId(_) if len(_) == 24 else _ for _ in ids])))

    if request.path.endswith('/split'):
        for para in paras:
            para_dict = para.as_dict()
            del para_dict['_id']
            del para_dict['images']
            for i in para.images:
                pnew = Paragraph(images=[i], **para_dict)
                pnew.save()
            para.delete()
    else:
        if not paras:
            return False

        para0 = paras[0]
        para0.keywords = list(para0.keywords)
        para0.images = list(para0.images)
        for para in paras[1:]:
            para0.keywords += list(para.keywords)
            para0.images += list(para.images)
        para0.save()

        for para in paras[1:]:
            para.delete()

    return True


@app.route('/api/mediaitem/rating', methods=["GET", "POST"])
@rest()
def set_rating(ids, inc=1, val=0, least=0):
    """Increase or decrease the rating of selected items
    """
    items = list(MediaItem.query(
        F.id.in_([ObjectId(_) if len(_) == 24 else _ for _ in ids])))
    for i in items:
        if i is None:
            continue
        if val:
            i.rating = val
        elif inc:
            i.rating = round(2 * (i.rating)) / 2 + inc

        i.rating = max(least, i.rating)
        i.rating = min(i.rating, 5)
        i.save()
    return {
        str(i.id): i.rating
        for i in items
    }


@app.route('/api/mediaitem/reset_storage', methods=["GET", "POST"])
@rest()
def reset_storage(ids):
    """Reset storage status of selected items
    """

    items = list(MediaItem.query(
        F.id.in_([ObjectId(_) if len(_) == 24 else _ for _ in ids])))
    for i in items:
        if 'file' in i.source:
            del i.source['file']
        else:
            i.source['file'] = 'blocks.h5'
        i.save()
    return {
        str(i.id): i.source.get('file')
        for i in items
    }


@app.route('/api/mediaitem/merge', methods=["POST"])
@rest()
def merge_items(pairs):
    """Process the two specified media items as duplications,
       keeping the first one, deleting the second and merging
       the Paragraphs where they locate
    """
    for rese, dele in pairs:
        dele = MediaItem.first(F.id == dele)
        if not dele:
            continue

        if rese:
            result = Paragraph.first(F.images == ObjectId(rese)) or Paragraph(
                images=[ObjectId(rese)], pdate=None)
            for para in Paragraph.query(F.images == dele.id):
                result.keywords += para.keywords
                if (not result.source.get('url') or 'restored' in result.source['url'])\
                        and para.source.get('url'):
                    result.source = para.source
                if not result.pdate:
                    result.pdate = para.pdate
            if not result.pdate:
                result.pdate = datetime.datetime.utcnow()
            result.save()

        Paragraph.query(F.images == dele.id).update(Fn.pull(images=dele.id))
        dele.delete()

    Paragraph.query(F.images == []).delete()

    return True


@app.route('/api/mediaitem/delete', methods=["POST"])
@rest()
def delete_item(album_items: dict):
    """Remove Media Item from paragraph"""

    del_items = set()
    for pid, items in album_items.items():
        para = Paragraph.first(F.id == pid)
        if para is None:
            continue

        items = list(map(ObjectId, items))
        para.images = [_ for _ in para.images if isinstance(
            _, MediaItem) and _.id not in items]
        para.save()
        del_items.update(items)

    for i in del_items:
        if Paragraph.first(F.images == i):
            continue
        image_item = MediaItem.first(F.id == i)
        if image_item:
            image_item.delete()

    Paragraph.query(F.images == []).delete()

    return True


@app.route('/api/tasks/', methods=['PUT'])
@rest()
def create_task(**task):
    """Create new task"""

    task.pop('shortcut_map', None)
    task = TaskDBO(**task)
    task.creator = logined()
    task.save()
    return task.id


@app.route('/api/tasks/shortcuts', methods=['GET'])
@rest()
def list_tasks_shortcuts():
    """List out quick tasks"""
    return list(TaskDBO.query((F.shortcut_map != {}) & _task_authorized()))


@app.route('/api/tasks/<task_id>', methods=['DELETE'])
@rest()
def delete_task(task_id):
    """Remove task"""

    _id = ObjectId(task_id)
    return TaskDBO.query(F.id == _id).delete()


@app.route('/api/tasks/<task_id>', methods=['POST'])
@rest()
def update_task(task_id, **task):
    """Update task info"""
    task_id = ObjectId(task_id)

    if '_id' in task:
        del task['_id']

    executed = TaskDBO.query(
        (F.id == task_id) & _task_authorized()).update(Fn.set(task))
    return {'acknowledged': executed.acknowledged, 'updated': task}


@app.route('/api/tasks/<task_id>', methods=['GET'])
@app.route('/api/tasks/', methods=['GET'])
@rest()
def list_task(task_id=''):
    """List out tasks"""

    if task_id:
        _id = ObjectId(task_id)
        return TaskDBO.first((F.id == _id) & _task_authorized())
    else:
        return list(TaskDBO.query(_task_authorized()).sort(-F.last_run, -F.id))


@app.route('/api/help/pipelines')
@rest(cache=True)
def help_info():
    """Provide help info for pipelines, with respect to preferred language"""

    ctx = Pipeline.ctx
    result = defaultdict(dict)
    for key, val in ctx.items():
        name = (sys.modules[val.__module__].__doc__ or val.__module__.split(
            '.')[-1] if hasattr(val, '__module__') else key).strip()
        if key in ("DataSourceStage", "MediaItemStage"):
            continue
        name = _lang(name)
        result[name][key] = _lang(val.get_spec())
    return result


@app.route('/api/help/langs')
@rest(cache=True)
def help_langs():
    """Provide supported language codes"""
    return language_iso639


@app.route('/api/help/queryexpr')
@rest(cache=True)
def help_queryexpr():
    """Provide meta data for query expr"""
    return {
        'function_names': list(parser.functions.keys()) + list(ee.implemented_functions),
        'operators': list([str(_) for _ in parser.operators]),
        'defaults': list(map(str, parser.abbrev_prefixes.values()))
    }


@app.route('/api/history')
@rest()
def history():
    """History records for the current user"""

    History.query(F.created_at < datetime.datetime.utcnow() -
                  datetime.timedelta(days=30)).delete()
    return list(
        History.query(F.user == logined()).sort(-F.created_at).limit(100))


@app.route('/api/search', methods=['POST'])
@rest()
def search(q='', req='', sort='', limit=100, offset=0,
           mongocollections=None, groups='none', count=False, **_):
    """Search"""

    if not req:
        if count:
            return 0
        else:
            return {'results': [], 'query': ''}

    datasource = DBQuery((q, req), mongocollections,
                         limit, offset, sort or 'id', False, groups, app.plugins)

    if count:
        return datasource.count()
    
    results = _expand_results(datasource.fetch())

    History(user=logined(), queries=[q, req],
            created_at=datetime.datetime.utcnow()).save()
    
    return {'results': results, 'query': datasource.query}


@app.route('/api/datasets')
@rest()
def get_datasets():
    """Get accessible datasets for the current user"""

    datasets = list(Dataset.query((F.allowed_users == []) | (
        F.allowed_users == logined())).sort(F.order_weight, F.name))
    dataset_patterns = User.first(F.username == logined()).datasets
    if dataset_patterns:
        filtered_datasets = []
        for dataset in datasets:
            for pattern in dataset_patterns:
                if dataset.name.startswith(pattern):
                    filtered_datasets.append(dataset)
                    break
        datasets = filtered_datasets
    return datasets


@app.route('/api/datasets/<action>', methods=['POST'])
@rest(role='admin')
def set_datasets(action, **j):
    """Update dataset info"""

    ds = None
    if '_id' in j:
        ds = Dataset.first(F.id == j['_id'])
        del j['_id']

    if action == 'edit':
        if ds:
            ds.update(Fn.set(**j))
        else:
            Dataset(**j).save()

    elif action == 'batch':
        for dataset in j:
            jset = {k: v for k, v in dataset.items() if k !=
                    '_id' and v is not None}
            if '_id' in dataset:
                Dataset.query(F.id == dataset['_id']).update(Fn.set(**jset))
            else:
                Dataset(**jset).save()

    elif action == 'rename':
        assert ds and 'to' in j, 'must specify valid dataset id and new name'
        ds.rename(j['to'])

    elif action == 'sources':
        assert ds, 'dataset not found'
        ds.update_sources()

    return True


@app.route("/api/term/<field>", methods=['POST'])
@rest()
def query_terms(field, pattern, regex=False):
    """Query terms"""

    if regex:
        pattern = F.term.regex(pattern) | F.aliases.regex(pattern)
    else:
        pattern = (F.term == pattern) | (F.aliases == pattern)

    return list(Term.query(F.field == field, pattern).limit(100))


@app.route("/api/image/<coll>/<storage_id>.<ext>")
@app.route("/api/image")
@rest(cache=True)
def serve_image(coll=None, storage_id=None, ext=None):
    """Serve images"""

    if coll and storage_id and len(storage_id) == 24:
        para = Paragraph.get_coll(coll).first(F.id == storage_id)
        item = MediaItem(para)
        buf = None
        if item:
            buf = item.data
    else:
        item = MediaItem(source=request.args.to_dict())
        filename = item.source.get('file', '')
        for fkey, fmapped in config.file_serve.items():
            if filename.startswith(fkey):
                return redirect(fmapped + filename[len(fkey):])

        buf = item.data
        ext = item.source.get('url', item.source.get(
            'file', '.')).rsplit('.', 1)[-1]

    def _thumb(path_or_io: Union[str, IO], size: int) -> bytes:
        """Thumbnail image

        Args:
            p (Union[str, IO]): image source
            size (int): max size for thumbnail

        Returns:
            bytes: thumbnailed image bytes
        """
        img = Image.open(path_or_io).convert('RGB')
        buf = BytesIO()
        img.thumbnail(size)
        img.save(buf, 'jpeg')
        return buf.getvalue()

    if buf:
        length = len(buf.getvalue()) if hasattr(
            buf, 'getvalue') else getattr(buf, 'st_size', -1)

        if request.args.get('enhance', ''):
            img = Image.open(buf)
            buf = BytesIO()
            ImageOps.autocontrast(img).save(buf, 'jpeg')
            buf.seek(0)
            ext = 'jpg'

        if request.args.get('w', ''):
            width = int(request.args.get('w'))
            size = (width, min(width, 1280))
            buf = BytesIO(_thumb(buf, size))
            ext = 'jpg'

        resp = serve_file(buf, ext, length)
        resp.headers.add("Cache-Control", "public,max-age=86400")
        return resp
    else:
        return Response('Not found.', 404)


@app.route('/api/quicktask', methods=['POST'])
@rest()
def quick_task(query='', pipeline='', raw=False, mongocollection=''):
    """Perform quick tasks"""

    if pipeline:
        if isinstance(pipeline, str):
            pipeline = parser.parse(pipeline)
        assert isinstance(pipeline, (list, tuple)
                          ), f"Unknown format for pipeline: {pipeline}"
        args = pipeline[0]
        if isinstance(args, (list, tuple)):
            args = args[1]
        elif isinstance(args, dict):
            args, = args.values()
        results = Task(stages=pipeline, params=args).execute()
    else:
        results = Task(stages=[
            ('DBQueryDataSource', {}),
            ('AccumulateParagraphs', {}),
        ], params={'query': query, 'raw': raw, 'mongocollections': mongocollection}).execute()

    return _expand_results(results)


@app.route('/api/admin/db', methods=['POST'])
@rest(role='admin')
def dbconsole(mongocollection='', query='', operation='', operation_params='', preview=True):
    """Database console for admin"""

    mongo = Paragraph.db.database[mongocollection]
    query = parser.parse(query)
    operation_params = parser.parse(operation_params)

    if preview:
        return {
            'mongocollection': mongocollection,
            'query': query,
            'operation': operation,
            'operation_params': operation_params
        }
    else:
        result = getattr(mongo, operation)(query, operation_params)
        if operation == 'update_many':
            result = result.modified_count
        elif operation == 'delete_many':
            result = result.deleted_count
        return result


@app.route('/api/admin/db/collections', methods=['GET'])
@rest(role='admin')
def dbconsole_collections():
    """Get all collections in current MongoDB database"""

    return Paragraph.db.database.list_collection_names()


@app.route('/api/meta', methods=['GET'])
@rest(login=False)
def get_meta():
    """Get meta settings"""

    return Meta.first(F.app_title.exists(1)) or Meta()


@app.route('/api/meta', methods=['POST'])
@rest(role='admin')
def set_meta(**vals):
    """Update meta settings"""

    result = Meta.first(F.app_title.exists(1)) or Meta()
    for key, val in vals.items():
        setattr(result, key, val)
    result.save()
    return True


@app.route('/api/plugins', methods=['GET'])
def get_plugins():
    return [type(pl).__name__ for pl in app.plugins]


@app.route('/<path:path>', methods=['GET'])
@app.route('/', methods=['GET'])
def index(path='index.html'):
    """Serve static files"""

    if path.startswith('api/'):
        return Response('', 404)
    path = path or 'index.html'
    for file in [
        path,
        path + '.html',
        os.path.join('ui/dist', path)
    ]:
        if file.startswith('ui/') and config.ui_proxy:
            return serve_proxy(config.ui_proxy, path=path)
        if os.path.exists(file) and os.path.isfile(file):
            return serve_file(file)

    return serve_file('ui/dist/index.html')


def run_service(host='0.0.0.0', port=None):
    """Run API web service

    :param host: Host, defaults to '0.0.0.0'
    :type host: str, optional
    :param port: Port, defaults to None
    :type port: int, optional
    """
    if os.path.exists('restarting'):
        os.unlink('restarting')

    plugin_ctx = get_context('plugins', Plugin)
    app.plugins = PluginManager(plugin_ctx, app)

    # os.environ['FLASK_ENV'] = 'development'
    if port is None:
        port = config.port
    app.run(debug=True, host=host, port=int(port), threaded=True)
