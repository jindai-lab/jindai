from os import name
from flask import Flask, Response, jsonify, request, send_file, json
from bson import ObjectId
import datetime
import inspect
import logging
import traceback
from PyMongoWrapper import *
from collections import defaultdict
from models import *
from task import Task
from pipeline import Pipeline, PipelineStage
from datasource import DataSource
import threading
from collections import deque
from io import BytesIO
import sys
import config
import logging
import base64
from helpers import *


class TasksQueue:

    def __init__(self, n=3):
        self._lock = threading.Lock()
        self.queue = []
        self.results = {}
        self.taskdbos = {}
        self.running = False
        self.n = n

    def start(self):
        self.running = True
        self._workingThread = threading.Thread(target=self.working)
        self._workingThread.start()

    @property
    def status(self):
        return {
            'running': list(self.taskdbos),
            'finished': [{
                'id': k,
                'name': k.split('@')[0],
                'viewable': isinstance(v, list) or (isinstance(v, dict) and 'exception' in v) or (isinstance(v, dict) and 'redirect' in v),
                'isnull': v is None,
                'last_run': datetime.datetime.strptime(k.split('@')[-1], '%Y%m%d %H%M%S').strftime('%Y-%m-%d %H:%M:%S'),
                'file_ext': 'json' if not isinstance(v, dict) else v.get('__file_ext__', 'json')
            } for k, v in self.results.items()],
            'waiting': len(self)
        }

    def working(self):
        while self.running:
            if self.queue and len(self.taskdbos) < self.n: # can run new task
                with self._lock:
                    tkey, t = self.queue.pop(0)
                self.taskdbos[tkey] = t
                try:
                    t._task = Task.from_dbo(t)
                    t._task.run()
                except Exception as ex:
                    self.results[tkey] = {'exception': '初始化任务时出错：' + str(ex), 'tracestack': traceback.format_tb(ex.__traceback__)}
            
            elif not self.queue and not self.taskdbos: # all tasks done
                self.running = False

            else:
                done = []
                for k, v in self.taskdbos.items():
                    if not v._task.alive:
                        done.append(k)
                        self.results[k] = v._task.returned
                for k in done:
                    self.taskdbos.pop(k)
            time.sleep(0.5)

    def enqueue(self, key, val):
        with self._lock:
            self.queue.append((key, val))
        # emit('queue', self.status)

    def stop(self):
        self.running = False

    def find(self, key):
        if key in self.taskdbos:
            return self.taskdbos[key]
        else:
            for k, v in self.queue:
                if k == key:
                    return v
            return None

    def __len__(self):
        return len(self.queue)

    def remove(self, key):
        with self._lock:
            for todel, _ in self.queue:
                if todel == key: break
            else:
                return False
            self.queue.remove(todel)
        
        return True


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

        return je.default(self, obj)


class JsonDecoder(json.JSONDecoder):
    def __init__(self, *args, **kargs):
        _ = kargs.pop('object_hook', None)
        super().__init__(object_hook=self.decoder, *args, **kargs)

    def decoder(self, d):
        import dateutil.parser
        updt = {}
        for k, v in d.items():
            if isinstance(v, str):
                if re.match(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z?$', v):
                    updt[k] = dateutil.parser.isoparse(v)
        if updt: d.update(**updt)
        return d


app.json_encoder = NumpyEncoder
app.json_decoder = JsonDecoder

tasks_queue = TasksQueue()
    

def valid_task(j):

    j = dict(j)
    
    def _valid_args(t, args):
        argnames = inspect.getfullargspec(t.__init__).args[1:]
        toremove = []
        for k in args:
            if k not in argnames or args[k] is None:
                logging.info(k, 'not an arg' if k not in argnames else 'is null')
                toremove.append(k)
        for k in toremove:
            del args[k]

        return args
    
    if 'datasource' not in j:
        j['datasource'] = 'DBQueryDataSource'
    if 'datasource_config' not in j:
        j['datasource_config'] = {}

    _valid_args(Task.datasource_ctx[j['datasource']], j['datasource_config'])

    if 'pipeline' not in j:
        j['pipeline'] = []

    for name, args in j['pipeline']:
        _valid_args(Pipeline.pipeline_ctx[name], args)

    return j


def _get_object(coll=''):
    if coll and coll != 'paragraph':
        class TempParagraph(Paragraph):
            _collection = str(coll)
        rs = TempParagraph
    else:
        rs = Paragraph
    return rs


@app.route('/api/authenticate', methods=['POST'])
@rest(login=False)
def authenticate(username, password, **kws):
    if User.authenticate(username, password):
        Token.query((F.user==username) & (F.expire < time.time())).delete()
        token = User.encrypt_password(str(time.time()), str(time.time_ns()))
        Token(user=username, token=token, expire=time.time() + 86400).save()
        return token
    raise Exception("Wrong user name/password.")


@app.route('/api/authenticate')
@rest()
def whoami():
    if logined():
        u = (User.first(F.username == logined()) or User(username=logined(), password='', roles=['admin'])).as_dict()
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
@rest(user_role='admin')
def admin_users(user='', password=None, roles=None, **kws):
    if user:
        u = User.first(F.username == user)
        if not u: return 'No such user.', 404
        if password is not None:
            u.set_password(password)
        if roles is not None:
            u.roles = roles
        u.save()
    else:
        return list(User.query({}))


@app.route('/api/users/', methods=['PUT'])
@rest(user_role='admin')
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
@rest(user_role='admin')
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
        'folder': os.path.isdir(path)
    }


@app.route('/api/storage/<path:dir>', methods=['GET'])
@app.route('/api/storage/', methods=['GET'])
@rest()
def list_storage(dir=''):
    dir = os.path.join(config.storage, dir) if dir and not '..' in dir else config.storage
    if os.path.isdir(dir):
        return sorted(map(file_detail, [os.path.join(dir, x) for x in os.listdir(dir)]), key=lambda x: x['ctime'], reverse=True)
    else:
        return send_file(dir)


@app.route('/api/storage/<path:dir>', methods=['PUT'])
@app.route('/api/storage/', methods=['PUT'])
@rest()
def write_storage(dir=''):
    dir = os.path.join(config.storage, dir) if dir and not '..' in dir else config.storage
    sfs = []
    for f in request.files.values():
        sf = os.path.join(dir, f.filename)
        f.save(sf)
        sfs.append(file_detail(sf))
    return sfs


@app.route('/api/paragraphs/<id>', methods=['POST'])
@rest()
def modify_paragraph(id, **kws):
    id = ObjectId(id)
    p = Paragraph.first(F.id == id)
    if p:
        for f, v in kws.items():
            if f in ('_id', 'matched_content'): continue
            if v is None and hasattr(p, f):
                delattr(p, f)
            else:
                setattr(p, f, v)
        p.save()
    return True


@app.route('/api/tasks/', methods=['PUT'])
@rest()
def create_task(**task):
    task = valid_task(task)
    task.pop('shortcut_map', None)
    print(task)
    task = TaskDBO(**task)
    task.save()
    return task.id


@app.route('/api/tasks/shortcuts', methods=['GET'])
@rest()
def list_tasks_shortcuts():
    return list(TaskDBO.query(F.shortcut_map != {}))

@app.route('/api/tasks/<id>', methods=['DELETE'])
@rest()
def delete_task(id):
    _id = ObjectId(id)
    return TaskDBO.query(F.id == _id).delete()


@app.route('/api/tasks/<id>', methods=['POST'])
@rest()
def update_task(id, **task):
    _id = ObjectId(id)
    task = valid_task(task)
    if '_id' in task: del task['_id']
    return {'acknowledged': TaskDBO.query(F.id == _id).update(Fn.set(task)).acknowledged, 'updated': task}


@app.route('/api/tasks/<id>', methods=['GET'])
@app.route('/api/tasks/<offset>/<limit>', methods=['GET'])
@app.route('/api/tasks/', methods=['GET'])
@rest()
def list_task(id='', offset=0, limit=10):
    if id:
        _id = ObjectId(id)
        return TaskDBO.first(F.id == _id)
    else:
        return list(TaskDBO.query({}).sort(-F.last_run, -F.id).skip(int(offset)).limit(int(limit)))


@app.route('/api/queue/', methods=['PUT'])
@rest()
def enqueue_task(id=''):
    t = TaskDBO.first(F.id == id)
    assert t, 'No such task.'
    t.last_run = datetime.datetime.now()
    t.save()
    t._task = None
    key = f'{t.name}@{datetime.datetime.now().strftime("%Y%m%d %H%M%S")}'
    tasks_queue.enqueue(key, t)
    if not tasks_queue.running:
        logging.info('start background thread')
        tasks_queue.start()
    return key


@app.route('/api/queue/logs/<path:key>', methods=['GET'])
@rest()
def logs_task(key):
    t = tasks_queue.find(key)
    if t:
        return logs_view(t)
    else:
        return f'No such key: {key}', 404


@app.route('/api/queue/<path:_id>', methods=['DELETE'])
@rest()
def dequeue_task(_id):
    if _id in tasks_queue.results:
        del tasks_queue.results[_id]
        # emit('queue', tasks_queue.status)
        return True
    else:
        return tasks_queue.remove(_id)


@app.route('/api/queue/<path:_id>', methods=['GET'])
@rest(cache=True)
def fetch_task(_id):        
    if _id not in tasks_queue.results:
        return Response('No such id: ' + _id, 404)
    r = tasks_queue.results[_id]

    if isinstance(r, list):
        offset, limit = int(request.args.get('offset', 0)), int(request.args.get('limit', 100))
        return {
            'results': r[offset:offset+limit],
            'total': len(r)
        }
    elif r is None:
        return None
    else:
        if isinstance(r, dict) and '__file_ext__' in r and 'data' in r:
            buf = BytesIO(r['data'])
            buf.seek(0)
            return send_file(buf, 'application/octstream', as_attachment=True, attachment_filename=os.path.basename(_id + '.' + r['__file_ext__']))
        else:
            return jsonify(r)


@app.route('/api/queue/', methods=['GET'])
@rest()
def list_queue():
    return tasks_queue.status


@app.route('/api/help/<pipe_or_ds>')
@rest(cache=True)
def help(pipe_or_ds):

    def _doc(cl):
        args_docs = {}
        for l in (cl.__init__.__doc__ or '').strip().split('\n'):
            m = re.search(r'(\w+)\s\((.+?)\):\s(.*)', l)
            if m:
                g = m.groups()
                if len(g) > 2:
                    args_docs[g[0]] = {'type': g[1].split(',')[0], 'description': g[2]}

        args_spec = inspect.getfullargspec(cl.__init__)
        args_defaults = dict(zip(reversed(args_spec.args), reversed(args_spec.defaults or [])))

        for arg in args_spec.args[1:]:
            if arg not in args_docs:
                args_docs[arg] = {}
            if arg in args_defaults:
                args_docs[arg]['default'] = json.dumps(args_defaults[arg], ensure_ascii=False)

        return {
            'name': cl.__name__,
            'doc': (cl.__doc__ or '').strip(),
            'args': [
                {'name': k, 'type': v.get('type'), 'description': v.get('description'), 'default': v.get('default')} for k, v in args_docs.items() if 'type' in v
            ]
        }

    ctx = Pipeline.pipeline_ctx if pipe_or_ds == 'pipelines' else Task.datasource_ctx
    r = defaultdict(dict)
    for k, v in ctx.items():
        name = sys.modules[v.__module__].__doc__ or v.__module__.split('.')[-1]
        r[name][k] = _doc(v)
    return r


@app.route('/api/refresh_context')
@rest()
def refresh_context():
    Pipeline.pipeline_ctx = get_context('pipelines', PipelineStage)
    Task.datasource_ctx = get_context('datasources', DataSource)
    return True


@app.route('/api/history')
@rest()
def history():
    return list(History.query(F.user == logined()).sort(-F.created_at).limit(100))


@app.route('/api/search', methods=['POST'])
@rest()
def search(q='', req={}, sort='', limit=100, skip=0, dataset='', **kws):

    def _stringify(r):
        if not r: return ''
        if isinstance(r, dict):
            s = []
            for k, v in r.items():
                if k.startswith('$'):
                    s.append(k[1:] + '(' + _stringify(v) + ')')
                else:
                    s.append(k + '=' + _stringify(v))
            if len(s) == 1:
                return s[0]
            else:
                return '(' + ','.join(s) + ')'
        elif isinstance(r, str):
            return '`' + json.dumps(r, ensure_ascii=False)[1:-1].replace('`', '\\`') + '`'
        elif isinstance(r, (int, float)):
            return str(r)
        elif isinstance(r, datetime.datetime):
            return r.isoformat()
        elif isinstance(r, list):
            if len(r) == 0:
                return '[]'
            elif len(r) == 1:
                return '[];' + _stringify(r[0])
            else:
                return ';'.join([_stringify(x) for x in r])
        else:
            return '_json(`' + json.dumps(r, ensure_ascii=False) + '`)'

    qstr = q
    if q and req:
        qstr += ','
    if req:
        qstr += _stringify(req)
    
    History(user=logined(), querystr=qstr, created_at=datetime.datetime.now()).save()

    params = {
        'query': q,
        'mongocollection': dataset
    }
    if limit:
        params['limit'] = limit
    if sort:
        params['sort'] = sort
    if req:
        params['req'] = req
    if skip:
        params['skip'] = skip

    if not req and not q:
        return {
            'results': [], 'query': '', 'total': 0
        }

    task = Task(datasource=('DBQueryDataSource', params), pipeline=[
        ('AccumulateParagraphs', {}),
    ])
    count = task.datasource.count()
    results = [dict(r.as_dict() if isinstance(r, DbObject) else r, dataset=dataset) for r in task.execute()]

    return {'results': results, 'query': task.datasource.querystr, 'total': count}


@app.route('/api/collections')
@rest()
def get_collections():
    return list(Collection.query({}).sort(F.order_weight, F.name))


@app.route('/api/collections', methods=['POST'])
@rest()
def set_collections(collection=None, collections=None, rename=None, **j):

    if collection is not None:
        if collection.get('_id'):
            c = Collection.query(F.id == collection['_id'])
            del j['_id']
            c.update(Fn.set(**collection))
        else:
            Collection(**collection).save()

    elif collections is not None:
        for jc in collections:
            jset = {k: v for k, v in jc.items() if k != '_id' and v is not None}
            if '_id' in jc:
                c = Collection.query(F.id == jc['_id']).update(Fn.set(**jset))
            else:
                Collection(**jset).save()

    elif rename is not None:
        coll = Collection.first(F.name == j['from']).mongocollection
        if not coll:
            return False

        rs = _get_object(coll)
        rs.query(F.collection == coll.name).update(Fn.set(collection=j['to']))
        coll.delete()

        new_coll = Collection.first(F.name == j['to']) or Collection(name=j['to'], sources=[], order_weight=coll.order_weight)
        new_coll.sources = sorted(set(new_coll.sources + coll.sources))
        new_coll.save()

    elif 'sources' in j:
        j = j['sources']
        coll = Collection.first(F.id == j['_id']).mongocollection
        if not coll:
            return False

        rs = _get_object(coll)
        rs = rs.aggregator.match(F.collection == coll.name).group(_id='$collection', sources=Fn.addToSet('$source.file'))
        coll.sources = []
        for r in rs.perform(raw=True):
            coll.sources += r['sources']
        coll.save()

    return True


@app.route("/api/image/<coll>/<storage_id>.<ext>")
@rest(cache=True)
def page_image(coll, storage_id, ext):
    # from PIL import ImageEnhance, ImageStat
    from PIL import ImageOps

    if len(storage_id) == 24:
        p = _get_object(coll).first(F.id==storage_id)
        ext = {'jpg': 'jpeg', 'tiff': 'tiff', 'png': 'png'}.get(ext, 'octstream')
        buf = None
        if p:
            buf = p.image_raw
    else:
        buf = StorageManager().read(storage_id + '.' + ext)

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
        length = len(buf.getvalue())
        
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
    with StorageManager() as mgr:
        mgr.write(base64.b64decode(request.data), key)
    return True


@app.route('/api/quicktask', methods=['POST'])
@rest()
def quick_task(query='', raw=False, mongocollection=''):
    if query.startswith('datasource='):
        q = parser.eval(query)
        r = Task(datasource=q[0]['datasource'], pipeline=q[1:]).execute()
    else:
        r = Task(datasource=('DBQueryDataSource', {'query': query, 'raw': raw, 'mongocollection': mongocollection}), pipeline=[
            ('AccumulateParagraphs', {}),
        ]).execute()
    
    return r


@app.route('/api/admin/db', methods=['POST'])
@rest(user_role='admin')
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
@rest(user_role='admin')
def dbconsole_collections():
    return Paragraph.db.database.list_collection_names()


@app.route('/api/meta', methods=['GET'])
@rest(login=False)
def get_meta():
    r = Meta.first(F.app_title.exists(1)) or Meta()
    return r


@app.route('/api/meta', methods=['POST'])
@rest(user_role='admin')
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
        if os.path.exists(path) and os.path.isfile(path):
            return serve_file(path)

    return serve_file('ui/dist/index.html')


import gallery
gallery.init(app)


if __name__ == "__main__":
    os.environ['FLASK_ENV'] = 'development'
    port = os.environ.get('PORT', 8370)
    app.run(debug=True, host='0.0.0.0', port=int(port))
