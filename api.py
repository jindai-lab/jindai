from functools import wraps
from flask import Flask, Response, render_template, jsonify, request, session, send_file, json
from bson import ObjectId
import datetime
import inspect
import atexit
import traceback
from PyMongoWrapper import *
from collections import defaultdict
from models import *
from task import Task
import threading
from collections import deque
from pdf2image import convert_from_path
from io import BytesIO
import sys
import config
import logging
from articlecompletion import article_completion


class Token(dbo.DbObject):
    
    user = str
    token = str
    expire = float


class TasksQueue:

    def __init__(self):
        self._q = deque() # deque is documented as thread-safe, so no need to use lock.
        self.running = False
        self.running_task = ''
        self.results = {}

    def start(self):
        self.running = True
        self._workingThread = threading.Thread(target=self.working)
        self._workingThread.start()

    def working(self):
        while self.running:
            if self._q:
                self.running_task, task = self._q.popleft()
                try:
                    self.results[self.running_task] = task.execute()
                except Exception as ex:
                    self.results[self.running_task] = {'exception': str(ex), 'tracestack': traceback.format_tb(ex.__traceback__)}
                    
                self.running_task = ''
            else:
                self.running = False

    def enqueue(self, key, val):
        self._q.append((key, val))

    def stop(self):
        self.running = False

    def __len__(self):
        return len(self._q)

    def remove(self, key):
        for todel in self._q:
            if todel[0] == key: break
        else:
            return False
        self._q.remove(todel)
        return True


app = Flask(__name__)
je = dbo.create_dbo_json_encoder(json.JSONEncoder)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return je.default(self, obj)

app.json_encoder = NumpyEncoder
tasks_queue = TasksQueue()


def logined():
    t = Token.first((F.token == request.headers.get('X-Authentication-Token')) & (F.expire > time.time()))
    if t:
        return t.user
    

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
        _valid_args(Task.pipeline_ctx[name], args)

    return j


def rest(login=True, cache=False, user_role=''):
    def do_rest(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            try:
                if (login and not logined()) or \
                    (user_role and not User.first((F.roles == user_role) & (F.username == logined()))):
                    raise Exception('Forbidden.')
                
                f = func(*args, **kwargs)
                if isinstance(f, Response): return f
                resp = jsonify({'result': f})
            except Exception as ex:
                resp = jsonify({'exception': str(ex), 'tracestack': traceback.format_tb(ex.__traceback__)})
            
            resp.headers.add("Access-Control-Allow-Origin", "*")
            if cache:
                resp.headers.add("Cache-Control", "public,max-age=86400")
            return resp
        return wrapped

    return do_rest


@app.route('/api/authenticate', methods=['POST'])
@rest(login=False)
def authenticate():
    j = request.json
    u, p = j['username'], j['password']
    if User.authenticate(u, p):
        Token.query(F.user==u).delete()
        token = User.encrypt_password(str(time.time()), str(time.time_ns()))
        Token(user=u, token=token, expire=time.time() + 86400).save()
        return token
    raise Exception("Wrong user name/password.")


@app.route('/api/authenticate')
@rest()
def whoami():
    if logined():
        u = User.first(F.username == logined()).as_dict()
        del u['password']
        return u
    return None


@app.route('/api/users/')
@app.route('/api/users/<user>', methods=['GET', 'POST'])
@rest(user_role='admin')
def admin_users(user=''):
    if user:
        j = request.json
        u = User.first(F.username == user)
        if not u: return 'No such user.', 404
        if 'password' in j:
            u.set_password(j['password'])
        if 'roles' in j:
            u.roles = j['roles']
        u.save()
    else:
        return list(User.query({}))


@app.route('/api/users/', methods=['PUT'])
@rest(user_role='admin')
def admin_users_add():
    j = request.json
    if User.first(F.username == j['username']):
        raise Exception('User already exists: ' + str(j['username']))
    u = User(username=j['username'])
    u.set_password(j['password'])
    u.save()
    u = u.as_dict()
    del u['password']
    return u


@app.route('/api/account/', methods=['POST'])
@rest()
def user_change_password():
    j = request.json
    u = User.first(F.username == logined())
    assert User.authenticate(logined(), j['old_password']), '原密码错误'
    u.set_password(j['password'])
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
def modify_paragraph(id):
    id = ObjectId(id)
    p = Paragraph.first(F.id == id)
    if p:
        for f, v in request.json.items():
            if f in ('_id', 'matched_content'): continue
            if v is None and hasattr(p, f):
                delattr(p, f)
            else:
                setattr(p, f, v)
        p.save()
    return True


@app.route('/api/tasks/', methods=['PUT'])
@rest()
def create_task():
    j = valid_task(request.json)
    task = TaskDBO(**j)
    task.save()
    return task.id


@app.route('/api/tasks/<id>', methods=['DELETE'])
@rest()
def delete_task(id):
    _id = ObjectId(id)
    return TaskDBO.query(F.id == _id).delete()


@app.route('/api/tasks/<id>', methods=['POST'])
@rest()
def update_task(id):
    _id = ObjectId(id)
    j = valid_task(request.json)
    if '_id' in j: del j['_id']
    return {'acknowledged': TaskDBO.query(F.id == _id).update(Fn.set(j)).acknowledged, 'updated': j}


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
def enqueue_task():
    _id = ObjectId(request.json['id'])
    t = TaskDBO.first(F.id == _id)
    assert t, 'No such task.'
    t.last_run = datetime.datetime.now()
    t.save()
    task = Task(datasource=(t.datasource, t.datasource_config), pipeline=t.pipeline, concurrent=t.concurrent, resume_next=t.resume_next)
    tasks_queue.enqueue(f'{t.id}/{t.name}_{int(time.time())}', task)
    if not tasks_queue.running:
        logging.info('start background thread')
        tasks_queue.start()
    return _id


@app.route('/api/queue/<path:_id>', methods=['DELETE'])
@rest()
def dequeue_task(_id):
    if _id in tasks_queue.results:
        del tasks_queue.results[_id]
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
            return send_file(buf, 'application/octstream')
        else:
            return jsonify(r)


@app.route('/api/queue/', methods=['GET'])
@rest()
def list_queue():
    return {
        'running': tasks_queue.running_task,
        'finished': [{
            'id': k,
            'name': k.split('/', 1)[-1].split('_')[0],
            'viewable': isinstance(v, list) or (isinstance(v, dict) and 'exception' in v),
            'last_run': datetime.datetime.fromtimestamp(int(k.split('_')[-1])).strftime('%Y-%m-%d %H:%M:%S'),
            'file_ext': 'json' if not isinstance(v, dict) else v.get('__file_ext__', 'json')
        } for k, v in tasks_queue.results.items()],
        'waiting': len(tasks_queue)
    }


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

    ctx = Task.pipeline_ctx if pipe_or_ds == 'pipelines' else Task.datasource_ctx
    r = defaultdict(dict)
    for k, v in ctx.items():
        name = sys.modules[v.__module__].__doc__ or v.__module__.split('.')[-1]
        r[name][k] = _doc(v)
    return r


@app.route('/api/history')
@rest()
def history():
    return list(History.query(F.user == logined()).sort(-F.created_at).limit(100))


@app.route('/api/search', methods=['POST'])
@rest()
def search():

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
            return r.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(r, list):
            if len(r) == 0:
                return '[]'
            elif len(r) == 1:
                return '[] => ' + _stringify(r[0])
            else:
                return ' => '.join([_stringify(x) for x in r])
        else:
            return '_json(`' + json.dumps(r, ensure_ascii=False) + '`)'

    j = request.json
    q = j.get('q', '')
    req = j.get('req', {})
    sort = j.get('sort', '')
    limit = j.get('limit', 100)
    skip = j.get('offset', 0)
    dataset = j.get('dataset', '')
    
    History(user=logined(), querystr='?' + q + (',' + _stringify(req) if req else ''), created_at=datetime.datetime.now()).save()

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

    task = Task(datasource=('DBQueryDataSource', params), pipeline=[
        ('AccumulateParagraphs', {}),
    ])
    count = task.datasource.count()
    results = [dict(r.as_dict() if isinstance(r, DbObject) else r, dataset=dataset) for r in task.execute()]

    return {'results': results, 'query': task.datasource.querystr, 'total': count}


@app.route('/api/meta')
@rest()
def get_meta():
    return Meta.first({})


@app.route('/api/meta', methods=['POST'])
@rest()
def update_meta():
    j = request.json
    return Meta.query({}).update(Fn.set(j)).acknowledged


@app.route("/api/pdfimage")
@rest(cache=True)
def page_image():
    pdffile, pdfpage = request.args['pdffile'], int(
        request.args.get('pdfpage', '0'))
    pdfpage += 1
    pdffile = f'sources/{pdffile}'
    if not os.path.exists(pdffile):
        return 'Not found', 404

    img, = convert_from_path(pdffile, 120, first_page=pdfpage,
                            last_page=pdfpage, fmt='png') or [None]
    if img:
        buf = BytesIO()
        img.save(buf, format='png')
        buf.seek(0)
        return Response(buf, mimetype='image/png')
    else:
        return 'Err', 500


@app.route('/api/quicktask', methods=['POST'])
@rest()
def quick_task():
    j = request.json
    q = j.get('query', '')
    raw = j.get('raw', False)
    mongocollection = j.get('mongocollection', '')

    results = Task(datasource=('DBQueryDataSource', {'query': q, 'raw': raw, 'mongocollection': mongocollection}), pipeline=[
        ('AccumulateParagraphs', {}),
    ]).execute()

    return results


@app.route('/api/articlecompletion', methods=['POST'])
@rest()
def articlecompletion():
    n = request.json.get('n', 1)
    topp = request.json.get('topp', 0.95)
    prompt = request.json['prompt']
    return jsonify({
        "config": {
            "n": n, "topp": topp, "prompt": prompt
        },
        "results": [_[len(prompt):] for _ in article_completion.generate(prompt, n, topp)]
    })


if __name__ == "__main__":
    app.debug = True
    app.env = 'development'
    app.run(host='0.0.0.0', port=8370)
