from dis import dis
from flask import Flask, Response, jsonify, redirect, request, send_file, json
from bson import ObjectId
import datetime
import inspect
from PyMongoWrapper import *
from collections import defaultdict
from models import *
from task import Task
from pipeline import Pipeline
from io import BytesIO
import sys
import config
import base64
from plugin import PluginManager
from pipelines.dbquerydatasource import DBQueryDataSource
from helpers import *


app = Flask(__name__)
app.config['SECRET_KEY'] = config.secret_key
je = dbo.create_dbo_json_encoder(json.JSONEncoder)


class NumpyEncoder(json.JSONEncoder):
    def __init__(self, **kwargs):
        kwargs['ensure_ascii'] = False
        super().__init__(**kwargs)

    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int32):
            return obj.tolist()
        if isinstance(obj, Image.Image):
            return str(obj)
        if isinstance(obj, datetime.datetime):
            return obj.isoformat() + "Z"

        return je.default(self, obj)


app.json_encoder = NumpyEncoder
app.json_decoder = MongoJSONDecoder


def task_authorized():
    if logined('admin'):
        return MongoOperand({})
    else:
        return ((F.creator == logined()) | (F.shared == True))


def expand_rs(rs):
    if not isinstance(rs, (str, dict, bytes)) and hasattr(rs, '__iter__'):
        return [dict(r.as_dict(True), mongocollection=type(r).db.name) if isinstance(r, Paragraph) else r for r in rs]
    else:
        return rs


@app.route('/api/authenticate', methods=['POST'])
@rest(login=False)
def authenticate(username, password, **kws):
    if User.authenticate(username, password):
        Token.query((F.user == username) & (F.expire < time.time())).delete()
        token = User.encrypt_password(str(time.time()), str(time.time_ns()))
        Token(user=username, token=token, expire=time.time() + 86400).save()
        return token
    raise Exception("Wrong user name/password.")


@app.route('/api/authenticate')
@rest()
def whoami():
    if logined():
        u = (User.first(F.username == logined()) or User(
            username=logined(), password='', roles=['admin'])).as_dict()
        del u['password']
        return u
    return None


@app.route('/api/authenticate', methods=['DELETE'])
@rest()
def log_out():
    Token.uncheck(logined())
    return True


@app.route('/api/users/')
@app.route('/api/users/<user>', methods=['GET', 'POST'])
@rest(role='admin')
def admin_users(user='', password=None, roles=None, datasets=None, **kws):
    if user:
        u = User.first(F.username == user)
        if not u:
            return 'No such user.', 404
        if password is not None:
            u.set_password(password)
        if roles is not None:
            u.roles = roles
        if datasets is not None:
            u.datasets = datasets
        u.save()
        return True
    else:
        return list(User.query({}))


@app.route('/api/users/', methods=['PUT'])
@rest(role='admin')
def admin_users_add(username, password, **kws):
    if User.first(F.username == username):
        raise Exception('User already exists: ' + str(username))
    u = User(username=username)
    u.set_password(password)
    u.save()
    u = u.as_dict()
    del u['password']
    return u


@app.route('/api/account/', methods=['POST'])
@rest()
def user_change_password(old_password='', password='', **kws):
    u = User.first(F.username == logined())
    assert User.authenticate(logined(), old_password), '原密码错误'
    u.set_password(password)
    u.save()
    u = u.as_dict()
    del u['password']
    return u


@app.route('/api/users/<uname>', methods=['DELETE'])
@rest(role='admin')
def admin_users_del(uname):
    return User.query(F.username == uname).delete()


def file_detail(path):
    st = os.stat(path)
    return {
        'name': os.path.basename(path),
        'fullpath': path[len(config.storage):],
        'ctime': st.st_ctime,
        'mtime': st.st_mtime,
        'size': st.st_size,
        'type': 'folder' if os.path.isdir(path) else 'file'
    }


@app.route('/api/storage/<path:dir>', methods=['GET'])
@app.route('/api/storage/', methods=['GET'])
@rest()
def list_storage(dir=''):
    dir = os.path.join(
        config.storage, dir) if dir and not '..' in dir else config.storage
    if os.path.isdir(dir):
        return sorted(map(file_detail, [os.path.join(dir, x) for x in os.listdir(dir)]), key=lambda x: x['ctime'], reverse=True)
    else:
        return send_file(dir)


@app.route('/api/storage/<path:dir>', methods=['PUT'])
@app.route('/api/storage/', methods=['PUT'])
@rest()
def write_storage(dir=''):
    dir = os.path.join(
        config.storage, dir) if dir and not '..' in dir else config.storage
    sfs = []
    for f in request.files.values():
        sf = os.path.join(dir, f.filename)
        f.save(sf)
        sfs.append(file_detail(sf))
    return sfs


@app.route('/api/edit/<coll>/<id>', methods=['POST'])
@rest()
def modify_paragraph(coll, id, **kws):
    id = ObjectId(id)
    p = Paragraph.get_coll(coll).first(F.id == id)
    flag = False
    if p:
        for f, v in kws.items():
            if f in ('_id', 'matched_content'):
                continue
            if f in ('$push', '$pull'):
                Paragraph.get_coll(coll).query(F.id == id).update({f: v})
            else:
                flag = True
                if v is None and hasattr(p, f):
                    delattr(p, f)
                else:
                    setattr(p, f, v)
        if flag:
            p.save()
    return True


@app.route('/api/edit/<coll>/<id>/pagenum', methods=['POST'])
@rest()
def modify_pagenum(coll, id, sequential, new_pagenum, **kws):
    id = ObjectId(id)
    p = Paragraph.get_coll(coll).first(F.id == id)
    delta = new_pagenum - p.source['page']
    if p:
        if sequential == 'solo':
            p.pagenum = new_pagenum
            p.save()
        else:
            source = dict(p.source)
            assert 'page' in source
            if sequential == 'all':
                del source['page']
            else:  # after
                source['page'] = {'$gt': source['page']}
            source = {'source.' + k: w for k, w in source.items()}
            Paragraph.get_coll(coll).query((F.dataset == p.dataset) & MongoOperand(
                source)).update([Fn.set(pagenum=Fn.add('$source.page', delta))])
            Paragraph.get_coll(coll).query((F.dataset == p.dataset) & (F.pagenum <= 0)).update([
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

    paras = list(Paragraph.get_coll(coll).query(
        F.id.in_([ObjectId(_) if len(_) == 24 else _ for _ in ids])))
    for p in paras:
        for field, val in kws.items():
            if field in ('$pull', '$push'):
                for afield, aval in val.items():
                    if p[afield] is None:
                        p[afield] = []
                    for t in (aval if isinstance(aval, list) else [aval]):
                        t = t.strip()
                        if field == '$pull' and t in p[afield]:
                            p[afield].remove(t)
                        elif field == '$push' and t not in p[afield]:
                            p[afield].append(t)
            elif not field.startswith('$'):
                p[field] = val
        p.save()

    return {
        str(p.id): p.as_dict()
        for p in paras
    }


@app.route('/api/<coll>/split', methods=["GET", "POST"])
@app.route('/api/<coll>/merge', methods=["GET", "POST"])
@rest()
def splitting(coll, ids):
    """Split or merge selected items/paragraphs into seperate/single paragraph(s)

    Returns:
        bool: True if succeeded
    """
    paras = list(Paragraph.get_coll(coll).query(
        F.id.in_([ObjectId(_) if len(_) == 24 else _ for _ in ids])))

    if request.path.endswith('/split'):
        for p in paras:
            for i in p.images:
                pnew = Paragraph(source={'url': p.source['url']},
                                 pdate=p.pdate, keywords=p.keywords, images=[i], dataset=p.dataset)
                pnew.save()
            p.delete()
    else:
        if not paras:
            return False

        p0 = paras[0]
        p0.keywords = list(p0.keywords)
        p0.images = list(p0.images)
        for p in paras[1:]:
            p0.keywords += list(p.keywords)
            p0.images += list(p.images)
        p0.save()

        for p in paras[1:]:
            p.delete()

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
            pr = Paragraph.first(F.images == ObjectId(rese)) or Paragraph(
                images=[ObjectId(rese)], pdate=None)
            for pd in Paragraph.query(F.images == dele.id):
                pr.keywords += pd.keywords
                if (not pr.source.get('url') or 'restored' in pr.source['url']) and pd.source.get('url'):
                    pr.source = pd.source
                if not pr.pdate:
                    pr.pdate = pd.pdate
            if not pr.pdate:
                pr.pdate = datetime.datetime.utcnow()
            pr.save()

        Paragraph.query(F.images == dele.id).update(Fn.pull(images=dele.id))
        dele.delete()

    Paragraph.query(F.images == []).delete()

    return True


@app.route('/api/imageitem/delete', methods=["POST"])
@rest()
def delete_item(album_items: dict):
    for pid, items in album_items.items():
        p = Paragraph.first(F.id == pid)
        if p is None:
            continue

        items = list(map(ObjectId, items))
        p.images = [_ for _ in p.images if isinstance(
            _, ImageItem) and _.id not in items]
        p.save()

    for i in items:
        if Paragraph.first(F.images == i):
            continue
        ii = ImageItem.first(F.id == i)
        if ii:
            ii.delete()

    Paragraph.query(F.images == []).delete()

    return True


@app.route('/api/tasks/', methods=['PUT'])
@rest()
def create_task(**task):
    task.pop('shortcut_map', None)
    task = TaskDBO(**task)
    task.creator = logined()
    task.save()
    return task.id


@app.route('/api/tasks/shortcuts', methods=['GET'])
@rest()
def list_tasks_shortcuts():
    return list(TaskDBO.query((F.shortcut_map != {}) & task_authorized()))


@app.route('/api/tasks/<id>', methods=['DELETE'])
@rest()
def delete_task(id):
    _id = ObjectId(id)
    return TaskDBO.query(F.id == _id).delete()


@app.route('/api/tasks/<id>', methods=['POST'])
@rest()
def update_task(id, **task):
    _id = ObjectId(id)
    if '_id' in task:
        del task['_id']
    return {'acknowledged': TaskDBO.query((F.id == _id) & task_authorized()).update(Fn.set(task)).acknowledged, 'updated': task}


@app.route('/api/tasks/<id>', methods=['GET'])
@app.route('/api/tasks/<offset>/<limit>', methods=['GET'])
@app.route('/api/tasks/', methods=['GET'])
@rest()
def list_task(id='', offset=0, limit=50):
    if id:
        _id = ObjectId(id)
        return TaskDBO.first((F.id == _id) & task_authorized())
    else:
        return list(TaskDBO.query(task_authorized()).sort(-F.last_run, -F.id).skip(int(offset)).limit(int(limit)))


@app.route('/api/help/<pipe_or_ds>')
@rest(cache=True)
def help(pipe_or_ds):

    def _doc(cl):
        args_docs = {}
        cl_doc = cl.__doc__ or ''
        cl = getattr(cl, '_Implementation', cl)
        
        for l in (cl.__init__.__doc__ or '').strip().split('\n'):
            m = re.search(r'(\w+)\s\((.+?)\):\s(.*)', l)
            if m:
                g = m.groups()
                if len(g) > 2:
                    args_docs[g[0]] = {'type': g[1].split(
                        ',')[0], 'description': g[2]}

        args_spec = inspect.getfullargspec(cl.__init__)
        args_defaults = dict(zip(reversed(args_spec.args),
                             reversed(args_spec.defaults or [])))

        for arg in args_spec.args[1:]:
            if arg not in args_docs:
                args_docs[arg] = {}
            if arg in args_defaults:
                args_docs[arg]['default'] = json.dumps(
                    args_defaults[arg], ensure_ascii=False)

        return {
            'name': cl.__name__,
            'doc': (cl.__doc__ or cl_doc).strip(),
            'args': [
                {'name': k, 'type': v.get('type'), 'description': v.get('description'), 'default': v.get('default')} for k, v in args_docs.items() if 'type' in v
            ]
        }

    ctx = Pipeline.pipeline_ctx
    r = defaultdict(dict)
    for k, v in ctx.items():
        name = sys.modules[v.__module__].__doc__ or v.__module__.split(
            '.')[-1] if hasattr(v, '__module__') else k
        r[name][k] = _doc(v)
    return r


@app.route('/api/history')
@rest()
def history():
    return list(History.query(F.user == logined()).sort(-F.created_at).limit(100))


@app.route('/api/search', methods=['POST'])
@rest()
def search(q='', req='', sort='', limit=100, offset=0, mongocollections=[], groups='none', count=False, **kws):

    def _stringify(r):
        if r is None:
            return ''
        if isinstance(r, dict):
            s = []
            for k, v in r.items():
                if k == '$options':
                    continue
                if k == '_id':
                    k = 'id'
                s.append(k + '=' + _stringify(v))
            return '(' + ','.join(s) + ')'
        elif isinstance(r, str):
            return json.dumps(r, ensure_ascii=False)
        elif isinstance(r, (int, float)):
            return str(r)
        elif isinstance(r, datetime.datetime):
            return r.isoformat()+"Z"
        elif isinstance(r, list):
            return '[' + ','.join([_stringify(e) for e in r]) + ']'
        elif isinstance(r, bool):
            return str(bool).lower()
        elif isinstance(r, ObjectId):
            return 'ObjectId(' + str(r) + ')'
        else:
            return '_json(`' + json.dumps(r, ensure_ascii=False) + '`)'

    if not req and not q:
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

    qparsed = parser.eval(q)
    req = parser.eval(req)

    # merge req into query
    def merge_req(qparsed, req):
        if isinstance(qparsed, dict):
            return (MongoOperand(qparsed) & MongoOperand(req))()
        elif isinstance(qparsed, list) and len(qparsed) > 0:
            q0 = qparsed[0]
            if isinstance(q0, str):
                q0 = {'$match': parser.eval(q0)}
            elif isinstance(q0, dict) and not [_ for _ in q0 if _.startswith('$')]:
                q0 = {'$match': q0}

            if isinstance(q0, dict) and '$match' in q0:
                return [{'$match': (MongoOperand(q0['$match']) & MongoOperand(req))()}] + qparsed[1:]
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

    ds = DBQueryDataSource._Implementation(qstr, **params)
    results = None

    if page_args:
        for pl in app.plugins:
            if page_args[0] in pl.get_pages():
                if count:
                    return limit
                results = expand_rs(pl.handle_page(ds, *page_args[1:]))
                break

    if results is None:
        if count:
            return ds.count()
        results = expand_rs(ds.fetch())

    History(user=logined(), querystr=qstr,
            created_at=datetime.datetime.utcnow()).save()
    return {'results': results, 'query': qstr}


@app.route('/api/datasets')
@rest()
def get_datasets():
    datasets = list(Dataset.query((F.allowed_users == []) | (
        F.allowed_users == logined())).sort(F.order_weight, F.name))
    dataset_patterns = User.first(F.username == logined()).datasets
    if dataset_patterns:
        filtered_datasets = []
        for ds in datasets:
            for p in dataset_patterns:
                if ds.name.startswith(p):
                    filtered_datasets.append(ds)
                    break
        datasets = filtered_datasets
    return datasets


@app.route('/api/datasets', methods=['POST'])
@rest()
def set_datasets(dataset=None, datasets=None, rename=None, sources=None, **j):

    if dataset is not None:
        if dataset.get('_id'):
            c = Dataset.query(F.id == dataset['_id'])
            del j['_id']
            c.update(Fn.set(**dataset))
        else:
            Dataset(**dataset).save()

    elif datasets is not None:
        for jc in datasets:
            jset = {k: v for k, v in jc.items() if k !=
                    '_id' and v is not None}
            if '_id' in jc:
                c = Dataset.query(F.id == jc['_id']).update(Fn.set(**jset))
            else:
                Dataset(**jset).save()

    elif rename is not None:
        coll = Dataset.first(F.name == rename['from'])
        if not coll:
            return False

        rs = Paragraph.get_coll(coll.mongocollection)
        rs.query(F.dataset == coll.name).update(Fn.set(dataset=rename['to']))
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

        rs = Paragraph.get_coll(coll.mongocollection)
        rs = rs.aggregator.match(F.dataset == coll.name).group(
            _id='$dataset', sources=Fn.addToSet('$source.file'))
        coll.sources = []
        for r in rs.perform(raw=True):
            coll.sources += r['sources']
        coll.save()

    return True


@app.route("/api/image/<coll>/<storage_id>.<ext>")
@app.route("/api/image")
@rest(cache=True)
def serve_image(coll=None, storage_id=None, ext=None):
    # from PIL import ImageEnhance, ImageStat
    from PIL import ImageOps

    if coll and storage_id and len(storage_id) == 24:
        p = Paragraph.get_coll(coll).first(F.id == storage_id)
        p = ImageItem(p)
        buf = None
        if p:
            buf = p.image_raw
    else:
        i = ImageItem(source=request.args.to_dict())
        fn = i.source.get('file', '')
        for fkey, fmapped in config.file_serve.items():
            if fn.startswith(fkey):
                return redirect(fmapped + fn[len(fkey):])

        buf = i.image_raw
        ext = i.source.get('url', i.source.get('file', '.')).rsplit('.', 1)[-1]

    def _thumb(p: Union[str, IO], size: int) -> bytes:
        """Thumbnail image

        Args:
            p (Union[str, IO]): image source
            size (int): max size for thumbnail

        Returns:
            bytes: thumbnailed image bytes
        """
        img = Image.open(p).convert('RGB')
        buf = BytesIO()
        img.thumbnail(size)
        img.save(buf, 'jpeg')
        return buf.getvalue()

    if buf:
        length = len(buf.getvalue()) if hasattr(buf, 'getvalue') else len(buf)

        if request.args.get('enhance', ''):
            img = Image.open(buf)
            buf = BytesIO()
            ImageOps.autocontrast(img).save(p, 'jpeg')
            # brightness = ImageStat.Stat(img).mean[0]
            # if brightness < 0.2:
            # ImageEnhance.Brightness(img).enhance(1.2).save(p, 'jpeg')
            buf.seek(0)
            ext = 'jpg'

        if request.args.get('w', ''):
            w = int(request.args.get('w'))
            sz = (w, min(w, 1280))
            buf = BytesIO(_thumb(buf, sz))
            ext = 'jpg'

        resp = serve_file(buf, ext, length)
        resp.headers.add("Cache-Control", "public,max-age=86400")
        return resp
    else:
        return Response('Not found.', 404)


@app.route('/api/put_storage/<key>', methods=['POST'])
@rest()
def put_storage(key):
    with safe_open(f'hdf5://{key}', 'wb') as fo:
        fo.write(base64.b64decode(request.data))
    return True


@app.route('/api/quicktask', methods=['POST'])
@rest()
def quick_task(query='', pipeline='', raw=False, mongocollection=''):
    if pipeline:
        q = parser.eval(pipeline)
        args = q[0]
        if isinstance(args, tuple):
            args = args[1]
        elif isinstance(args, dict):
            args, = args.values()
        r = Task(stages=q, params=args).execute()
    else:
        r = Task(stages=[
            ('DBQueryDataSource', {}),
            ('AccumulateParagraphs', {}),
        ], params={'query': query, 'raw': raw, 'mongocollections': mongocollection}).execute()

    return expand_rs(r)


@app.route('/api/admin/db', methods=['POST'])
@rest(role='admin')
def dbconsole(mongocollection='', query='', operation='', operation_params={}, preview=True):
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
        r = getattr(mongo, operation)(query, operation_params)
        if operation == 'update_many':
            r = r.modified_count
        elif operation == 'delete_many':
            r = r.deleted_count
        return r


@app.route('/api/admin/db/collections', methods=['GET'])
@rest(role='admin')
def dbconsole_collections():
    return Paragraph.db.database.list_collection_names()


@app.route('/api/meta', methods=['GET'])
@rest(login=False)
def get_meta():
    r = Meta.first(F.app_title.exists(1)) or Meta()
    return r


@app.route('/api/meta', methods=['POST'])
@rest(role='admin')
def set_meta(**vals):
    r = Meta.first(F.app_title.exists(1)) or Meta()
    for k, v in vals.items():
        setattr(r, k, v)
    r.save()
    return True


# BLOCKLY UI
@app.route('/api/blockly/<path:p>')
@app.route('/api/blockly/')
def blockly_index(p='index.html'):
    if re.match(r'^[0-9a-f]{24}$', p):
        p = 'index.html'
    fp = os.path.join('blockly', p)
    if os.path.exists(fp) and os.path.isfile(fp):
        return serve_file(fp)
    return '', 404


@app.route('/<path:p>', methods=['GET'])
@app.route('/', methods=['GET'])
def index(p='index.html'):
    if p.startswith('api/'):
        return Response('', 404)
    p = p or 'index.html'
    for path in [
        p,
        p + '.html',
        os.path.join('ui/dist', p)
    ]:
        if path.startswith('ui/') and config.ui_proxy:
            return serve_proxy(config.ui_proxy, path=p)
        if os.path.exists(path) and os.path.isfile(path):
            return serve_file(path)

    return serve_file('ui/dist/index.html')


if os.path.exists('restarting'):
    os.unlink('restarting')

app.plugins = PluginManager(app)


if __name__ == "__main__":
    os.environ['FLASK_ENV'] = 'development'
    port = os.environ.get('PORT', 8370)
    app.run(debug=True, host='0.0.0.0', port=int(port))
