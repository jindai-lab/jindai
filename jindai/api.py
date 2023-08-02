"""API Web Service"""

import datetime
import hashlib
import os
import re
import sys
import itertools
import time
from collections import defaultdict
from PyMongoWrapper.dbo import DbObject

import pyotp
from flask import Flask, Response, redirect, request, send_file, abort
from PyMongoWrapper import F, Fn, MongoOperand, ObjectId

from .dbquery import DBQuery, parser
from .pipeline import Pipeline
from .plugin import Plugin, PluginManager
from .task import Task
from .config import instance as config
from .helpers import (get_context, logined, rest, language_iso639, serve_proxy,
                      JSONEncoder, JSONDecoder, ee, APICrudEndpoint,
                      APIResults, APIUpdate)
from .models import (Dataset, History, MediaItem, Meta, Paragraph, TaskDBO,
                     Token, User, Term)
from .oauthserv import config_oauth
from .storage import instance as storage

app = Flask(__name__)
app.config['SECRET_KEY'] = config.secret_key
app.json_encoder = JSONEncoder
app.json_decoder = JSONDecoder
config_oauth(app)


def _expand_results(results):
    """Expand results to serializable dicts"""

    def _patch_mongocollection(result):
        if isinstance(result, Paragraph):
            res = result.as_dict(True)
            if 'mongocollection' not in res:
                res['mongocollection'] = type(result).db.name
        else:
            res = result

        return res

    if not isinstance(results,
                      (str, dict, bytes)) and hasattr(results, '__iter__'):
        return [_patch_mongocollection(r) for r in results]

    return results


def _hashing(msg):
    """Hashing message with SHA-256 and preserve last 9 hexadecimal digits"""
    return hashlib.sha256(msg.encode('utf-8')).hexdigest()[-9:]


def _lang(inp):
    """Apply language settings of current client"""
    assert isinstance(inp,
                      (str, list, dict)), "Input value must be string or dict"
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


def _select_keys(dictionary: dict, keys: list) -> list:
    if not isinstance(dictionary, dict):
        dictionary = getattr(dictionary, 'as_dict',
                             lambda: dictionary.__dict__)()
    if keys:
        return {key: dictionary[key] for key in keys if key in dictionary}
    else:
        return dictionary


@app.route('/api/authenticate', methods=['POST'])
@rest(login=False)
def authenticate(username, password, otp='', **_):
    """Authenticate current user, return new token string if succeeded"""

    username = User.authenticate(username, password, otp)
    if username:
        Token.query((F.user == username) & (F.expire < time.time())).delete()
        token = User.encrypt_password(str(time.time()), str(time.time_ns()))
        Token(user=username, token=token, expire=time.time() + 86400).save()
        return APIUpdate(bundle={'token': token})
    raise Exception("Unmatched credentials.")


@app.route('/api/authenticate')
@rest()
def whoami():
    """Returns logined user info, or None if not logined"""
    if logined():
        user = (User.first(F.username == logined())
                or User(username=logined(), password='', roles=[])).as_dict()
        del user['password']
        user['otp_secret'] = True if user.get('otp_secret') else False
        return APIUpdate(bundle=_select_keys(user, ['username', 'roles']))
    return APIUpdate(False)


@app.route('/api/authenticate', methods=['DELETE'])
@rest()
def log_out():
    """Log out"""
    Token.uncheck(logined())
    return APIUpdate()


@app.route('/api/account/', methods=['POST'])
@rest()
def user_change_password(old_password='', password='', otp=None, **_):
    """
    Change user passworld or OTP settings
    Returns:
        if opt is set or unset, return user { opt_secret }
        otherwise, return UserInfo
    """

    user = User.first(F.username == logined())
    if otp is None:
        assert User.authenticate(logined(), old_password), '原密码错误'
        user.set_password(password)
        user.save()
        return _select_keys(user, ['username', 'roles'])
    else:
        if otp:
            user.otp_secret = pyotp.random_base32()
            user.save()
        else:
            user.otp_secret = ''
            user.save()
        return _select_keys(user, ['otp_secret'])


# storage and images


@app.route('/api/storage/<path:path>', methods=['GET', 'POST'])
@app.route('/api/storage/', methods=['GET', 'POST'])
@rest()
def list_storage(path='', search='', mkdir=''):
    """
    List out files in directory or get file at that path
    Returns:
        list of file names, or file content
    """

    path = 'file:///' + path

    if path.rsplit('.', 1)[-1].lower() in ('py', 'pyc', 'yaml', 'yml', 'sh'):
        return 'Not supported extension name', 400

    if mkdir:
        storage.mkdir(path, mkdir)

    results = None
    if search:
        results = [
            storage.stat(r) for r in storage.search(path, '*' + search + '*')
        ]
    else:
        results = storage.statdir(path)
        if not results and storage.exists(path):
            results = storage.open(path)

    if isinstance(results, list):
        return APIResults(
            sorted(results, key=lambda x: x['ctime'], reverse=True))
    else:
        return send_file(results, download_name=path.split('/')[-1])


@app.route('/api/storage/<path:path>', methods=['PUT'])
@app.route('/api/storage/', methods=['PUT'])
@rest()
def write_storage(path=''):
    """Write to file storage"""

    path = storage.expand_path(path)
    sfs = []
    for uploaded in request.files.values():
        save_path = os.path.join(path, uploaded.filename)
        with storage.open(save_path, 'wb') as fout:
            uploaded.save(fout)
        sfs.append(storage.stat(save_path))
    return APIUpdate(bundle=sfs)


@app.route('/api/storage/move', methods=['POST'])
@rest()
def move_storage(source, destination, keep_folder=True):
    """Move/Rename file from source to destination"""
    if keep_folder:
        destination = os.path.basename(destination)
        destination = os.path.join(os.path.dirname(source), destination)
    paragraphs = Paragraph.query(F.source.file == source)
    source = 'file:///' + source.replace('\\', '/')
    destination = 'file:///' + destination.replace('\\', '/')
    flag = storage.move(source, destination)
    if flag:
        paragraphs.update(
            Fn.set({'source.file': storage.truncate_path(destination)}))
    return APIUpdate(flag)


@app.route("/api/image/<coll>/<storage_id>.<ext>")
@app.route("/api/image")
@rest(cache=True)
def resolve_media_item(coll=None, storage_id=None, ext=None):
    """Serve media item"""

    if coll and storage_id and len(storage_id) == 24:
        para = Paragraph.get_coll(coll).first(F.id == ObjectId(storage_id))
        item = MediaItem(para)
        if item is None:
            return Response('', 404)
        source = item.source
    else:
        source = request.args.to_dict()

    def _build_image_string(source):
        fpath, url = source.get('file'), source.get('url')
        if fpath:
            if '://' in fpath:
                return fpath.replace('://', '/')
            elif fpath.endswith('.pdf') and 'page' in source:
                return f'file/{fpath}/__hash/pdf/{source["page"]}'
            else:
                return f'file/{fpath}'
        elif url:
            return url

        return ''

    return redirect('/images/' + _build_image_string(source))


@app.route("/images/<scheme>/<path:image_path>")
@rest(cache=True, login=False)
def serve_image(scheme, image_path):
    """Serve images

    Args:
        scheme (str): scheme
        image_path (str): path

    Returns:
        Response: image data
    """
    path, ext = storage.get_schemed_path(scheme, image_path)

    try:
        return storage.serve_file(path, ext)
    except OSError:
        return Response('Not found.', 404)


class APICollectionEndpoint(APICrudEndpoint):

    def __init__(self) -> None:
        super().__init__('api', Paragraph, ['matched_content'])
        self.namespace = '/api/collections/'
        self.bind_endpoint(self.pagenum)
        self.bind_endpoint(self.remove_image)
        self.bind_endpoint(self.merge)
        self.bind_endpoint(self.split)
        self.bind_endpoint(self.group)

    def get_dbobjs(self,
                   id=None,
                   ids=None,
                   query=None,
                   limit=0,
                   offset=0,
                   sort='id',
                   mongocollection='',
                   **data):
        if id and ':' in id:
            mongocollection, id = id.split(':', 1)

        query = self.build_query(id, ids, query, data)
        results = Paragraph.get_coll(mongocollection).query(query)

        if id:
            return results.first()
        else:
            return self.apply_sorting(results, limit, offset, sort)

    def update_object(self, obj, data):
        updated = super().update_object(obj, data)
        if 'keywords' in updated:
            for ele in updated['keywords']:
                Term.write(ele, 'keywords')
        return updated

    def pagenum(self, objs, sequential, new_pagenum, folio, **_):
        para = objs
        new_pagenum = int(new_pagenum)
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
            coll = type(para)
            coll.query((F.dataset == para.dataset)
                       & MongoOperand(source)).update([
                           Fn.set(pagenum=Fn.add(
                               Fn.multiply('$source.page', folio), delta))
                       ])
            coll.query((F.dataset == para.dataset) & (F.pagenum <= 0)).update([
                Fn.set(pagenum=Fn.concat(
                    "A", Fn.toString(Fn.add(1, "$source.page"))))
            ])
        return APIUpdate()

    def group(self, objs, group=None, ungroup=False, **data):
        objs, _ = objs
        proposed_groups = [group]

        if isinstance(proposed_groups, str):
            proposed_groups = [proposed_groups]

        if ungroup:
            objs.update(Fn.pull(F.keywords.regex('^#')))
            groups = []

        else:
            paras = list(objs)
            if paras:
                gids = []
                for para in paras:
                    gids += [_ for _ in para.keywords if _.startswith('#')]
                named = [_ for _ in gids if not _.startswith('#0')]

                if proposed_groups:
                    groups = ['#' + _ for _ in proposed_groups]
                elif named:
                    groups = [min(named)]
                elif gids:
                    groups = [min(gids)]
                else:
                    groups = [
                        '#0' + _hashing(min(map(lambda p: str(p.id), paras)))
                    ]

                gids = list(set(gids) - set(named) - set(groups))
                objs = type(para).query(F.keywords.in_(gids) | objs.mongo_cond)
                for g in groups:
                    objs.update(Fn.addToSet(keywords=g))
                if gids:
                    objs.update(Fn.pull(F.keywords.in_(gids)))

        return APIUpdate(bundle={str(i.id): {'keywords': i.keywords} for i in paras})

    def split(self, objs):
        return self.split_or_merge(objs, False)

    def merge(self, objs, pairs=None):

        if isinstance(pairs, list):
            d_pairs = defaultdict(list)
            for rese, dele in pairs:
                d_pairs[rese].append(dele)
            pairs = d_pairs

        if pairs:
            for rese, dele in pairs.items():
                rese, dele = MediaItem.first(F.id == ObjectId(rese)), list(
                    MediaItem.query(F.id.in_([ObjectId(_) for _ in dele])))
                if rese and dele:
                    Paragraph.merge_by_mediaitems(rese, dele)
            return APIUpdate()

        return self.split_or_merge(objs, True)

    def split_or_merge(self, objs, merge=False, **data):
        """Split or merge selected items/paragraphs into seperate/single paragraph(s)

        Returns:
            bool: True if succeeded
        """
        objs, _ = objs

        retval = []

        if not merge:
            for para in objs:
                para_dict = para.as_dict()
                del para_dict['_id']
                del para_dict['images']
                for i in para.images:
                    pnew = Paragraph(images=[i], **para_dict)
                    pnew.save()
                    retval.append(pnew.as_dict(True))
                para.delete()

        elif objs:
            selected_ids = list(map(ObjectId, itertools.chain(*data.values())))
            selected = MediaItem.query(F.id.in_(selected_ids))
            paras = list(objs)

            para0 = Paragraph(paras[0])
            para0.id = None
            para0.images = selected

            for para in paras:
                para0.keywords += para.keywords
                para.images = [
                    k for k in para.images if k.id not in selected_ids
                ]
                para.save()

            para0.save()
            retval.append(para0.as_dict(True))

        return APIUpdate(bundle={str(p['id']): p for p in retval})

    def remove_image(self, objs, **data):
        del_items = set()

        for para in objs[0]:
            items = [ObjectId(i) for i in data[str(para.id)]]
            para.images.remove(items)
            para.save()
            del_items.update(items)

        for i in del_items:
            if Paragraph.first(F.images == i):
                continue
            image_item = MediaItem.first(F.id == ObjectId(i))
            if image_item:
                image_item.delete()

        return APIUpdate()


APICollectionEndpoint().bind(app)


class APIMediaItemEndpoint(APICrudEndpoint):

    def __init__(self,
                 namespace,
                 db_cls: DbObject,
                 filtered_fields=None,
                 allowed_fields=None) -> None:
        super().__init__(namespace, db_cls, filtered_fields, allowed_fields)
        self.bind_endpoint(self.reset_storage)
        self.bind_endpoint(self.rating)

    def reset_storage(self, objs):
        retval = {}
        for i in objs[0]:
            if 'file' in i.source:
                del i.source['file']
                retval[str(i.id)] = {'source': i.source}
            i.save()
        return APIUpdate(bundle=retval)

    def rating(self, objs, inc=1, val=0, least=0):
        """Increase or decrease the rating of selected items
        """
        retval = {}
        for i in objs[0]:
            if val:
                i.rating = val
            elif inc:
                i.rating = round(2 * (i.rating)) / 2 + inc

            i.rating = max(least, i.rating)
            i.rating = min(i.rating, 5)
            i.save()
            retval[str(i.id)] = {'rating': i.rating}

        return APIUpdate(bundle=retval)


APIMediaItemEndpoint('api', MediaItem).bind(app)


class APITaskEndpoint(APICrudEndpoint):

    def __init__(self) -> None:
        super().__init__('api', TaskDBO, ['last_run'])
        self.namespace = '/api/tasks/'
        self.bind_endpoint(self.shortcuts)

    def create(self, **data):
        data.pop('shortcut_map')
        data['creator'] = logined()
        return super().create(**data)

    def _task_authorized(self):
        """Test if task is authorized to current user"""

        if logined('admin'):
            return MongoOperand({})

        return ((F.creator == logined()) | (F.shared == True))

    def build_query(self, id, ids, query, data):
        query = super().build_query(id, ids, query, data) & self._task_authorized()
        return query

    def shortcuts(self, objs, **_):
        """List out quick tasks"""
        return APIResults(
            TaskDBO.query((F.shortcut_map != {}) & self._task_authorized()))


APITaskEndpoint().bind(app)


class APIHistoryEndpoint(APICrudEndpoint):

    def build_query(self, id, ids, query, data):
        query = super().build_query(id, ids, query, data)
        return query & (F.user == logined())

    def get_dbobjs(self,
                   id=None,
                   ids=None,
                   query=None,
                   limit=0,
                   offset=0,
                   sort='-created_at',
                   **data):
        History.query(F.created_at < datetime.datetime.utcnow() -
                      datetime.timedelta(days=30)).delete()

        return super().get_dbobjs(id, ids, query, limit, offset, sort, **data)


APIHistoryEndpoint('api', History).bind(app)


class APIDatasetEndpoint(APICrudEndpoint):

    def __init__(self,
                 namespace,
                 db_cls: DbObject,
                 filtered_fields=None,
                 allowed_fields=None) -> None:
        super().__init__(namespace, db_cls, filtered_fields, allowed_fields)
        self.bind_endpoint(self.rename)
        self.bind_endpoint(self.sources)

    def test_dataset(self, dataset, dataset_patterns):
        if not dataset_patterns:
            return True

        for pattern in dataset_patterns:
            if dataset.name.startswith(pattern):
                return True

        return False

    def build_query(self, id, ids, query, data):
        query = super().build_query(id, ids, query, data)
        return query & ((F.allowed_users == []) |
                        (F.allowed_users == logined()))

    def apply_sorting(self, results, limit, offset, sort):
        sort = 'order_weight,name'
        return super().apply_sorting(results, limit, offset, sort)

    def get_dbobjs(self,
                   id=None,
                   ids=None,
                   query=None,
                   limit=0,
                   offset=0,
                   sort='id'):
        dataset_patterns = User.first(F.username == logined()).datasets
        result = super().get_dbobjs(id, ids, query, limit, offset, sort)

        if isinstance(result, Dataset):
            if not self.test_dataset(result, dataset_patterns):
                result = None

        else:
            objs, total = result
            filtered_datasets = []
            for dataset in objs:
                if self.test_dataset(dataset, dataset_patterns):
                    filtered_datasets.append(dataset)
            return filtered_datasets, total

    def rename(self, objs, to, **_):
        self.check_role('admin')
        objs.rename(to)

    def sources(self, objs, **_):
        self.check_role('admin')
        objs.update_sources()

    def update(self, objs, **data):
        self.check_role('admin')
        return super().update(objs, **data)

    def delete(self, objs, **data):
        self.check_role('admin')
        return super().delete(objs, **data)

    def create(self, **data):
        self.check_role('admin')
        return super().create(**data)


APIDatasetEndpoint('api', Dataset).bind(app)


class APIUserEndpoint(APICrudEndpoint):

    def update_object(self, obj, data):
        if 'password' in data:
            obj.set_password(data.pop('password'))
        return super().update_object(obj, data)

    def read(self, objs, **data):
        if isinstance(objs, tuple):
            return APIResults([
                self.select_fields(o, ['username', 'roles', 'datasets'])
                for o in objs[0]
            ], objs[1])
        return {}

    def create(self, username='', password='', **_):
        assert username and password
        if User.first(F.username == username):
            raise Exception('User already exists: ' + str(username))
        user = User(username=username)
        user.set_password(password)
        user.save()
        return self.select_fields(user, ['username', 'roles'])

    def build_query(self,
                    id=None,
                    ids=None,
                    query=None,
                    data=None):
        if id:
            return F.username == id
        return super().build_query(id, ids, query, data or {})


APIUserEndpoint('api', User).bind(app, role='admin')

# ui & qx helpers


@app.route('/api/help/pipelines')
@rest(cache=True)
def help_info():
    """Provide help info for pipelines, with respect to preferred language"""

    ctx = Pipeline.ctx
    result = defaultdict(dict)
    for key, val in ctx.items():
        name = (sys.modules[val.__module__].__doc__
                or val.__module__.split('.')[-1]
                if hasattr(val, '__module__') else key).strip()
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


@app.route('/api/help/qx')
@rest(cache=True)
def help_qx():
    """Provide meta data for query expr"""
    return APIResults(
        list(parser.functions.keys()) + list(ee.implemented_functions))


@app.route('/api/qx', methods=['POST'])
@rest()
def qx_parse(query=''):
    return parser.parse(query)


@app.route('/api/search', methods=['POST'])
@rest()
def do_search(q='',
              req='',
              sort='',
              limit=100,
              offset=0,
              mongocollections=None,
              groups='none',
              count=False,
              **_):
    """Search"""

    if not req:
        return APIResults()

    datasource = DBQuery((q, req), mongocollections, limit, offset, sort
                         or 'id', False, groups, app.plugins)

    if count:
        return APIResults(total=datasource.count())

    results = _expand_results(datasource.fetch())

    if datasource.groups != 'none':
        for res in results:
            group = res.get(datasource.groups)
            if isinstance(group, dict):
                group = '(' + ','.join([
                    f'{k}={app.json_encoder().encode(v)}'
                    for k, v in group.items()
                ]) + ')'
            else:
                group = app.json_encoder().encode(group)

    History(user=logined(),
            queries=[q, req],
            created_at=datetime.datetime.utcnow()).save()

    return APIResults(results, -1, datasource.query)


@app.route("/api/term/<field>", methods=['POST'])
@rest()
def query_terms(field, pattern='', regex=False, scope=''):
    """Query terms"""

    if pattern:
        if regex:
            pattern = pattern.replace(' ', '.*')
            pattern = {'term': {'$regex': pattern, '$options': 'i'}}
        else:
            pattern = (F.term == pattern) | (F.aliases == pattern)
        return APIResults(Term.query(F.field == field, pattern).limit(100))
    elif scope:
        return APIResults([
            i['_id']
            for i in Paragraph.aggregator.match(parser.parse(scope)).sort(
                _id=-1).limit(1000).unwind('$' + field).group(
                    _id='$' + field, count=Fn.sum(1)).sort(
                        count=-1).limit(100).perform(raw=True)
        ])


@app.route('/api/quicktask', methods=['POST'])
@rest()
def quick_task(query='', pipeline='', raw=False, mongocollection=''):
    """Perform quick tasks"""

    if pipeline:
        if isinstance(pipeline, str):
            pipeline = parser.parse(pipeline)
        assert isinstance(
            pipeline,
            (list, tuple)), f"Unknown format for pipeline: {pipeline}"
        args = pipeline[0]
        if isinstance(args, (list, tuple)):
            args = args[1]
        elif isinstance(args, dict):
            args, = args.values()
        results = Task(stages=pipeline, params={}).execute()
    else:
        params = {
            'query': query,
            'raw': raw,
            'mongocollections': mongocollection
        }
        results = Task(stages=[
            ('DBQueryDataSource', params),
            ('AccumulateParagraphs', {}),
        ],
                       params={}).execute()

    return APIResults(_expand_results(results))


@app.route('/api/admin/db', methods=['POST'])
@rest(role='admin')
def dbconsole(mongocollection='',
              query='',
              operation='',
              operation_params='',
              preview=True):
    """Database console for admin"""

    mongo = Paragraph.db.database[mongocollection]
    query = parser.parse(query) if query else {}
    operation_params = parser.parse(
        operation_params) if operation_params else None
    if isinstance(operation_params, dict):
        keys = list(operation_params)
        if len(keys) != 1:
            operation_params = {'$set': operation_params}
        elif keys[0] in ['keywords', 'images']:
            operation_params = {'$addToSet': operation_params}
        elif not keys[0].startswith('$'):
            operation_params = {'$set': operation_params}

    if preview:
        return APIUpdate(
            bundle={
                'mongocollection': mongocollection,
                'query': query,
                'operation': operation,
                'operation_params': operation_params
            })
    else:
        result = getattr(mongo, operation)(query, operation_params)
        if operation == 'update_many':
            result = result.modified_count
        elif operation == 'delete_many':
            result = result.deleted_count
        return APIUpdate(bundle=result)


@app.route('/api/admin/db/collections', methods=['GET'])
@rest(role='admin')
def dbconsole_collections():
    """Get all collections in current MongoDB database"""

    return APIResults(Paragraph.db.database.list_collection_names())


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
    return APIUpdate()


@app.route('/api/plugins', methods=['GET'])
@rest()
def get_plugins():
    """Get plugin names
    """
    return APIResults([type(pl).__name__ for pl in app.plugins])


# index


@app.route('/<path:path>', methods=['GET'])
@app.route('/', methods=['GET'])
def index(path='index.html'):
    """Serve static files"""

    if path.startswith('api/'):
        return Response('', 404)
    path = path or 'index.html'
    for file in [path, path + '.html', os.path.join('ui/dist', path)]:
        if file.startswith('ui/') and config.ui_proxy:
            return serve_proxy(config.ui_proxy, path=path)
        if os.path.exists(file) and os.path.isfile(file):
            return storage.serve_file(open(file, 'rb'))

    return storage.serve_file(open('ui/dist/index.html', 'rb'))


def prepare_plugins():
    """Prepare plugins
    """
    if os.path.exists('restarting'):
        os.unlink('restarting')
    plugin_ctx = get_context('plugins', Plugin)
    app.plugins = PluginManager(plugin_ctx, app)


def run_service(host='0.0.0.0', port=None):
    """Run API web service. Must run `prepare_plugins` first.

    :param host: Host, defaults to '0.0.0.0'
    :type host: str, optional
    :param port: Port, defaults to None
    :type port: int, optional
    """
    if port is None:
        port = config.port
    app.run(debug=True, host=host, port=int(port), threaded=True)
