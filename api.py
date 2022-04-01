"""网页界面的 API"""
import datetime
import inspect
import os
import re
import sys
import time
from collections import defaultdict
from io import BytesIO
from typing import IO, Union

import jieba
import numpy as np
from bson import ObjectId
from flask import Flask, Response, json, redirect, request, send_file
from PIL import Image, ImageOps
from PyMongoWrapper import F, Fn, MongoOperand
from PyMongoWrapper.dbo import create_dbo_json_decoder, create_dbo_json_encoder

from jindai import Pipeline, Plugin, PluginManager, Task
from jindai.config import instance as config
from jindai.helpers import get_context, logined, rest, serve_file, serve_proxy
from jindai.models import (Dataset, History, ImageItem, Meta, Paragraph,
                           TaskDBO, Token, User, parser)

app = Flask(__name__)
app.config['SECRET_KEY'] = config.secret_key
JSONEncoderCls = create_dbo_json_encoder(json.JSONEncoder)


class _NumpyEncoder(json.JSONEncoder):
    def __init__(self, **kwargs):
        kwargs['ensure_ascii'] = False
        super().__init__(**kwargs)

    def default(self, o):
        """编码对象"""
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.int32):
            return o.tolist()
        if isinstance(o, Image.Image):
            return str(o)
        if isinstance(o, datetime.datetime):
            return o.isoformat() + "Z"

        return JSONEncoderCls.default(self, o)


app.json_encoder = _NumpyEncoder
app.json_decoder = create_dbo_json_decoder(json.JSONDecoder)


def _task_authorized():
    """是否有权限访问该任务"""

    if logined('admin'):
        return MongoOperand({})

    return ((F.creator == logined()) | (F.shared == True))


def _expand_results(results):
    """将结果扩展为可返回的字典"""

    if not isinstance(results, (str, dict, bytes)) and hasattr(results, '__iter__'):
        return [dict(r.as_dict(True), mongocollection=type(r).db.name)
                if isinstance(r, Paragraph) else r for r in results]

    return results


@app.route('/api/authenticate', methods=['POST'])
@rest(login=False)
def authenticate(username, password, **_):
    """认证"""

    if User.authenticate(username, password):
        Token.query((F.user == username) & (F.expire < time.time())).delete()
        token = User.encrypt_password(str(time.time()), str(time.time_ns()))
        Token(user=username, token=token, expire=time.time() + 86400).save()
        return token
    raise Exception("Wrong user name/password.")


@app.route('/api/authenticate')
@rest()
def whoami():
    """返回登录信息"""
    if logined():
        user = (User.first(F.username == logined()) or User(
            username=logined(), password='', roles=['admin'])).as_dict()
        del user['password']
        return user
    return None


@app.route('/api/authenticate', methods=['DELETE'])
@rest()
def log_out():
    """登出"""
    Token.uncheck(logined())
    return True


@app.route('/api/users/')
@app.route('/api/users/<username>', methods=['GET', 'POST'])
@rest(role='admin')
def admin_users(username='', password=None, roles=None, datasets=None, **_):
    """用户管理修改"""

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
        return True
    else:
        return list(User.query({}))


@app.route('/api/users/', methods=['PUT'])
@rest(role='admin')
def admin_users_add(username, password, **_):
    """添加用户"""

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
def user_change_password(old_password='', password='', **_):
    """修改用户密码"""

    user = User.first(F.username == logined())
    assert User.authenticate(logined(), old_password), '原密码错误'
    user.set_password(password)
    user.save()
    user = user.as_dict()
    del user['password']
    return user


@app.route('/api/users/<username>', methods=['DELETE'])
@rest(role='admin')
def admin_users_del(username):
    """删除用户"""
    return User.query(F.username == username).delete()


def _file_detail(path):
    """获取文件详情"""

    file_stat = os.stat(path)
    return {
        'name': os.path.basename(path),
        'fullpath': path[len(config.storage):],
        'ctime': file_stat.st_ctime,
        'mtime': file_stat.st_mtime,
        'size': file_stat.st_size,
        'type': 'folder' if os.path.isdir(path) else 'file'
    }


@app.route('/api/storage/<path:path>', methods=['GET'])
@app.route('/api/storage/', methods=['GET'])
@rest()
def list_storage(path=''):
    """列出文件夹信息"""

    path = os.path.join(
        config.storage, path) if path and '..' not in path else config.storage
    if os.path.isdir(path):
        return sorted(map(
            _file_detail, [os.path.join(path, x) for x in os.listdir(path)]),
            key=lambda x: x['ctime'], reverse=True)
    else:
        return send_file(path)


@app.route('/api/storage/<path:path>', methods=['PUT'])
@app.route('/api/storage/', methods=['PUT'])
@rest()
def write_storage(path=''):
    """写入存储文件夹"""

    path = os.path.join(
        config.storage, path) if path and '..' not in path else config.storage
    sfs = []
    for uploaded in request.files.values():
        save_path = os.path.join(path, uploaded.filename)
        uploaded.save(save_path)
        sfs.append(_file_detail(save_path))
    return sfs


@app.route('/api/edit/<coll>/<pid>', methods=['POST'])
@rest()
def modify_paragraph(coll, pid, **kws):
    """修改语段"""

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


@app.route('/api/edit/<coll>/<pid>/pagenum', methods=['POST'])
@rest()
def modify_pagenum(coll, pid, sequential, new_pagenum, **_):
    """修改页码"""

    pid = ObjectId(pid)
    para = Paragraph.get_coll(coll).first(F.id == pid)
    delta = new_pagenum - para.source['page']
    if para:
        if sequential == 'solo':
            para.pagenum = new_pagenum
            para.save()
        else:
            source = dict(para.source)
            assert 'page' in source
            if sequential == 'all':
                del source['page']
            else:  # after
                source['page'] = {'$gt': source['page']}
            source = {'source.' + k: w for k, w in source.items()}
            Paragraph.get_coll(coll).query((F.dataset == para.dataset) & MongoOperand(
                source)).update([Fn.set(pagenum=Fn.add('$source.page', delta))])
            Paragraph.get_coll(coll).query((F.dataset == para.dataset) & (F.pagenum <= 0)).update([
                Fn.set(pagenum=Fn.concat(
                    "A", Fn.toString(Fn.add(1, "$source.page"))))
            ])
        return True
    return False


@app.route('/api/edit/<coll>/batch', methods=["GET", "POST"])
@rest()
def batch(coll, ids, **kws):
    """Batch edit
    """

    paragraphs = list(Paragraph.get_coll(coll).query(
        F.id.in_([ObjectId(_) if len(_) == 24 else _ for _ in ids])))
    for para in paragraphs:
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
        for p in paragraphs
    }


@app.route('/api/<coll>/split', methods=["GET", "POST"])
@app.route('/api/<coll>/merge', methods=["GET", "POST"])
@rest()
def splitting(coll, ids):
    """Split or merge selected items/paragraphs into seperate/single paragraph(s)

    Returns:
        bool: True if succeeded
    """
    paragraphs = list(Paragraph.get_coll(coll).query(
        F.id.in_([ObjectId(_) if len(_) == 24 else _ for _ in ids])))

    if request.path.endswith('/split'):
        for para in paragraphs:
            for i in para.images:
                pnew = Paragraph(
                    source={'url': para.source['url']},
                    pdate=para.pdate, keywords=para.keywords, images=[i],
                    dataset=para.dataset)
                pnew.save()
            para.delete()
    else:
        if not paragraphs:
            return False

        first_para = paragraphs[0]
        first_para.keywords = list(first_para.keywords)
        first_para.images = list(first_para.images)
        for para in paragraphs[1:]:
            first_para.keywords += list(para.keywords)
            first_para.images += list(para.images)
        first_para.save()

        for para in paragraphs[1:]:
            para.delete()

    return True


@app.route('/api/imageitem/rating', methods=["GET", "POST"])
@rest()
def set_rating(ids, inc=1, val=0):
    """Increase or decrease the rating of selected items
    """
    items = list(ImageItem.query(
        F.id.in_([ObjectId(_) if len(_) == 24 else _ for _ in ids])))
    for i in items:
        if i is None:
            continue
        old_value = i.rating
        i.rating = val if val else round(2 * (i.rating)) / 2 + inc
        if -1 <= i.rating <= 5:
            i.save()
        else:
            i.rating = old_value
    return {
        str(i.id): i.rating
        for i in items
    }


@app.route('/api/imageitem/reset_storage', methods=["GET", "POST"])
@rest()
def reset_storage(ids):
    """Reset storage status of selected items
    """

    items = list(ImageItem.query(
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


@app.route('/api/imageitem/merge', methods=["POST"])
@rest()
def merge_items(pairs):
    """将两两指定的 ImageItem 作为同一个图像的两个副本处理，保留第一项，删除第二项，并合并所在的 Paragraph
    """
    for rese, dele in pairs:
        dele = ImageItem.first(F.id == dele)
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


@app.route('/api/imageitem/delete', methods=["POST"])
@rest()
def delete_item(album_items: dict):
    """删除图像项目"""
    del_items = set()
    for pid, items in album_items.items():
        para = Paragraph.first(F.id == pid)
        if para is None:
            continue

        items = list(map(ObjectId, items))
        para.images = [_ for _ in para.images if isinstance(
            _, ImageItem) and _.id not in items]
        para.save()
        del_items.update(items)

    for i in del_items:
        if Paragraph.first(F.images == i):
            continue
        image_item = ImageItem.first(F.id == i)
        if image_item:
            image_item.delete()

    Paragraph.query(F.images == []).delete()

    return True


@app.route('/api/tasks/', methods=['PUT'])
@rest()
def create_task(**task):
    """创建任务"""

    task.pop('shortcut_map', None)
    task = TaskDBO(**task)
    task.creator = logined()
    task.save()
    return task.id


@app.route('/api/tasks/shortcuts', methods=['GET'])
@rest()
def list_tasks_shortcuts():
    """列出快捷任务"""
    return list(TaskDBO.query((F.shortcut_map != {}) & _task_authorized()))


@app.route('/api/tasks/<task_id>', methods=['DELETE'])
@rest()
def delete_task(task_id):
    """删除任务"""

    _id = ObjectId(task_id)
    return TaskDBO.query(F.id == _id).delete()


@app.route('/api/tasks/<task_id>', methods=['POST'])
@rest()
def update_task(task_id, **task):
    """更新任务信息"""
    task_id = ObjectId(task_id)

    if '_id' in task:
        del task['_id']

    executed = TaskDBO.query(
        (F.id == task_id) & _task_authorized()).update(Fn.set(task))
    return {'acknowledged': executed.acknowledged, 'updated': task}


@app.route('/api/tasks/<task_id>', methods=['GET'])
@app.route('/api/tasks/<offset>/<limit>', methods=['GET'])
@app.route('/api/tasks/', methods=['GET'])
@rest()
def list_task(task_id='', offset=0, limit=50):
    """列出任务"""
    if task_id:
        _id = ObjectId(task_id)
        return TaskDBO.first((F.id == _id) & _task_authorized())
    else:
        return list(TaskDBO.query(_task_authorized()).sort(-F.last_run, -F.id)
                    .skip(int(offset)).limit(int(limit)))


@app.route('/api/help/pipelines')
@rest(cache=True)
def help_info():
    """提供任务处理帮助信息"""

    def _doc(stage_type):
        args_docs = {}
        cl_doc = stage_type.__doc__ or ''
        cl_name = stage_type.__name__
        stage_type = getattr(stage_type, 'Implementation', stage_type)

        for line in (stage_type.__init__.__doc__ or '').strip().split('\n'):
            match = re.search(r'(\w+)\s\((.+?)\):\s(.*)', line)
            if match:
                matched_groups = match.groups()
                if len(matched_groups) > 2:
                    args_docs[matched_groups[0]] = {'type': matched_groups[1].split(
                        ',')[0], 'description': matched_groups[2]}

        args_spec = inspect.getfullargspec(stage_type.__init__)
        args_defaults = dict(zip(reversed(args_spec.args),
                             reversed(args_spec.defaults or [])))

        for arg in args_spec.args[1:]:
            if arg not in args_docs:
                args_docs[arg] = {}
            if arg in args_defaults:
                args_docs[arg]['default'] = json.dumps(
                    args_defaults[arg], ensure_ascii=False)

        return {
            'name': cl_name,
            'doc': (stage_type.__doc__ or cl_doc).strip(),
            'args': [
                {
                    'name': key,
                    'type': val.get('type'),
                    'description': val.get('description'),
                    'default': val.get('default')
                } for key, val in args_docs.items() if 'type' in val
            ]
        }

    ctx = Pipeline.ctx
    result = defaultdict(dict)
    for key, val in ctx.items():
        name = sys.modules[val.__module__].__doc__ or val.__module__.split(
            '.')[-1] if hasattr(val, '__module__') else key
        result[name][key] = _doc(val)
    return result


@app.route('/api/history')
@rest()
def history():
    """历史记录"""
    History.query(F.created_at < datetime.datetime.utcnow() -
                  datetime.timedelta(days=30)).delete()
    return list(
        History.query(F.user == logined()).sort(-F.created_at).limit(100))


@app.route('/api/search', methods=['POST'])
@rest()
def search(q='', req='', sort='', limit=100, offset=0,
           mongocollections=None, groups='none', count=False, **_):
    """搜索"""

    if mongocollections is None:
        mongocollections = ['']

    def _stringify(obj):
        if obj is None:
            return ''
        if isinstance(obj, dict):
            seq = []
            for key, val in obj.items():
                if key == '$options':
                    continue
                if key.startswith('$'):
                    seq.append(key[1:] + '(' + _stringify(val) + ')')
                if key == '_id':
                    key = 'id'
                seq.append(key + '=' + _stringify(val))
            return '(' + ','.join(s) + ')'
        elif isinstance(obj, str):
            return json.dumps(obj, ensure_ascii=False)
        elif isinstance(obj, (int, float)):
            return str(obj)
        elif isinstance(obj, datetime.datetime):
            return obj.isoformat()+"Z"
        elif isinstance(obj, list):
            return '[' + ','.join([_stringify(e) for e in obj]) + ']'
        elif isinstance(obj, bool):
            return str(bool).lower()
        elif isinstance(obj, ObjectId):
            return 'ObjectId(' + str(obj) + ')'
        else:
            return '_json(`' + json.dumps(obj, ensure_ascii=False) + '`)'

    if not req:
        if count:
            return 0
        else:
            return {'results': [], 'query': ''}

    params = {
        'mongocollections': '\n'.join(mongocollections),
        'groups': groups,
        'sort': sort or '_id',
        'skip': offset or 0,
        'limit': limit,
    }

    expr = False
    if q.startswith('?'):
        q = q[1:]
        expr = True
    elif re.search(r'[,.~=&|()><\'"`@_*\-%]', q):
        expr = True

    if not expr:
        q = '`' + '`,`'.join([_.strip().lower().replace('`', '\\`')
                             for _ in jieba.cut(q) if _.strip()]) + '`'
        if q == '``':
            q = ''

    qparsed = parser.eval(q)
    req = parser.eval(req)

    # merge req into query
    def merge_req(qparsed, req):
        if isinstance(qparsed, dict):
            return (MongoOperand(qparsed) & MongoOperand(req))()
        elif isinstance(qparsed, list) and len(qparsed) > 0:
            first_query = qparsed[0]
            if isinstance(first_query, str):
                first_query = {'$match': parser.eval(first_query)}
            elif isinstance(first_query, dict) and \
                    not [_ for _ in first_query if _.startswith('$')]:
                first_query = {'$match': first_query}

            if isinstance(first_query, dict) and '$match' in first_query:
                return [
                    {'$match':
                     (MongoOperand(first_query['$match']) & MongoOperand(req))()}
                ] + qparsed[1:]
            else:
                return [{'$match': req}] + qparsed[1:]
        else:
            return req

    qparsed = merge_req(qparsed, req)

    # test plugin pages
    def test_plugin_pages(qparsed):
        page_args = []
        if isinstance(qparsed, list) and '$page' in qparsed[-1]:
            qparsed, page_args = qparsed[:-1], qparsed[-1]['$page'].split('/')
        return qparsed, page_args

    qparsed, page_args = test_plugin_pages(qparsed)

    if isinstance(qparsed, list) and len(qparsed) == 1 and '$match' in qparsed[0]:
        qparsed = qparsed[0]['$match']

    qstr = '?'+_stringify(qparsed)

    datasource = Pipeline.ctx['DBQueryDataSource'].Implementation(
        qstr, **params)
    results = None

    if page_args:
        for plugin in app.plugins:
            if page_args[0] in plugin.get_filters():
                if count:
                    return limit
                results = _expand_results(
                    plugin.handle_page(datasource, *page_args[1:]))
                break

    if results is None:
        if count:
            return datasource.count()
        results = _expand_results(datasource.fetch())

    History(user=logined(), querystr=qstr,
            created_at=datetime.datetime.utcnow()).save()
    return {'results': results, 'query': qstr}


@app.route('/api/datasets')
@rest()
def get_datasets():
    """获取用户有权限查看的数据集"""

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


@app.route('/api/datasets', methods=['POST'])
@rest()
def set_datasets(dataset=None, datasets=None, rename=None, sources=None, **j):
    """设置数据集信息"""

    if dataset is not None:
        if dataset.get('_id'):
            dataset = Dataset.query(F.id == dataset['_id'])
            del j['_id']
            dataset.update(Fn.set(**dataset))
        else:
            Dataset(**dataset).save()

    elif datasets is not None:
        for dataset in datasets:
            jset = {k: v for k, v in dataset.items() if k !=
                    '_id' and v is not None}
            if '_id' in dataset:
                Dataset.query(F.id == dataset['_id']).update(Fn.set(**jset))
            else:
                Dataset(**jset).save()

    elif rename is not None:
        coll = Dataset.first(F.name == rename['from'])
        if not coll:
            return False

        results = Paragraph.get_coll(coll.mongocollection)
        results.query(F.dataset == coll.name).update(
            Fn.set(dataset=rename['to']))
        coll.delete()

        new_coll = Dataset.first(F.name == rename['to']) or Dataset(
            name=rename['to'], sources=[], order_weight=coll.order_weight)
        new_coll.sources = sorted(set(new_coll.sources + coll.sources))
        new_coll.save()

    elif sources is not None:
        j = sources
        coll = Dataset.first(F.id == j['_id'])
        if not coll:
            return False

        results = Paragraph.get_coll(coll.mongocollection)
        results = results.aggregator.match(F.dataset == coll.name).group(
            _id='$dataset', sources=Fn.addToSet('$source.file'))
        coll.sources = []
        for result in results.perform(raw=True):
            coll.sources += result['sources']
        coll.save()

    return True


@app.route("/api/image/<coll>/<storage_id>.<ext>")
@app.route("/api/image")
@rest(cache=True)
def serve_image(coll=None, storage_id=None, ext=None):
    """返回图像"""

    if coll and storage_id and len(storage_id) == 24:
        para = Paragraph.get_coll(coll).first(F.id == storage_id)
        item = ImageItem(para)
        buf = None
        if item:
            buf = item.image_raw
    else:
        item = ImageItem(source=request.args.to_dict())
        filename = item.source.get('file', '')
        for fkey, fmapped in config.file_serve.items():
            if filename.startswith(fkey):
                return redirect(fmapped + filename[len(fkey):])

        buf = item.image_raw
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
            # brightness = ImageStat.Stat(img).mean[0]
            # if brightness < 0.2:
            # ImageEnhance.Brightness(img).enhance(1.2).save(p, 'jpeg')
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
    """快速任务"""

    if pipeline:
        pipeline = parser.eval(pipeline)
        args = pipeline[0]
        if isinstance(args, tuple):
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
    """数据库控制台"""

    mongo = Paragraph.db.database[mongocollection]
    query = parser.eval(query)
    operation_params = parser.eval(operation_params)

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
    """数据库控制台获取全部数据库集合"""
    return Paragraph.db.database.list_collection_names()


@app.route('/api/meta', methods=['GET'])
@rest(login=False)
def get_meta():
    """获取元设置"""
    return Meta.first(F.app_title.exists(1)) or Meta()


@app.route('/api/meta', methods=['POST'])
@rest(role='admin')
def set_meta(**vals):
    """更新元设置"""
    result = Meta.first(F.app_title.exists(1)) or Meta()
    for key, val in vals.items():
        setattr(result, key, val)
    result.save()
    return True


@app.route('/<path:path>', methods=['GET'])
@app.route('/', methods=['GET'])
def index(path='index.html'):
    """返回文件"""

    if path.startswith('api/'):
        return Response('', 404)
    path = path or 'index.html'
    for path in [
        path,
        path + '.html',
        os.path.join('ui/dist', path)
    ]:
        if path.startswith('ui/') and config.ui_proxy:
            return serve_proxy(config.ui_proxy, path=path)
        if os.path.exists(path) and os.path.isfile(path):
            return serve_file(path)

    return serve_file('ui/dist/index.html')


if os.path.exists('restarting'):
    os.unlink('restarting')

plugin_ctx = get_context('plugins', Plugin)
app.plugins = PluginManager(plugin_ctx, app)


if __name__ == "__main__":
    os.environ['FLASK_ENV'] = 'development'
    port = os.environ.get('PORT', 8370)
    app.run(debug=True, host='0.0.0.0', port=int(port))
