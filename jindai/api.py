"""API Web Service"""

import datetime
import hashlib
import os
import sys
import itertools
import time
from collections import defaultdict
from urllib.parse import quote

import pyotp
from flask import Flask, Response, redirect, request, send_file
from PyMongoWrapper import F, Fn, MongoOperand, ObjectId

from .dbquery import DBQuery, parser
from .pipeline import Pipeline
from .plugin import Plugin, PluginManager
from .task import Task
from .config import instance as config
from .helpers import (get_context, logined, rest, language_iso639,
                      serve_proxy, JSONEncoder, JSONDecoder, ee)
from .models import (Dataset, History, MediaItem, Meta, Paragraph,
                     TaskDBO, Token, User, Term)
from .storage import instance as storage


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

    def _patch_mongocollection(result):
        if isinstance(result, Paragraph):
            res = result.as_dict(True)
            if 'mongocollection' not in res:
                res['mongocollection'] = type(result).db.name
        else:
            res = result

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


def _select_keys(dictionary: dict, keys: list) -> list:
    return {key: dictionary[key] for key in keys if key in dictionary}


@app.route('/api/authenticate', methods=['POST'])
@app.route('/api2/authenticate', methods=['POST'])
@rest(login=False)
def authenticate(username, password, otp='', **_):
    """Authenticate current user, return new token string if succeeded"""

    username = User.authenticate(username, password, otp)
    if username:
        Token.query((F.user == username) & (F.expire < time.time())).delete()
        token = User.encrypt_password(str(time.time()), str(time.time_ns()))
        Token(user=username, token=token, expire=time.time() + 86400).save()
        return token
    raise Exception("Unmatched credentials.")


@app.route('/api/authenticate')
@app.route('/api2/authenticate')
@rest()
def whoami():
    """Returns logined user info, or None if not logined"""
    if logined():
        user = (User.first(F.username == logined()) or User(
            username=logined(), password='', roles=[])).as_dict()
        del user['password']
        user['otp_secret'] = True if user.get('otp_secret') else False
        return _select_keys(user, ['username', 'roles'])
    return None


@app.route('/api/authenticate', methods=['DELETE'])
@app.route('/api2/authenticate', methods=['DELETE'])
@rest()
def log_out():
    """Log out"""
    Token.uncheck(logined())
    return True


@app.route('/api/users/')
@app.route('/api2/users/')
@app.route('/api/users/<username>', methods=['GET', 'POST'])
@app.route('/api2/users/<username>', methods=['GET', 'POST'])
@rest(role='admin')
def admin_users(username='', password=None, roles=None, datasets=None, **_):
    """
    Change user profile and permissions
    Returns:
        a list of AdminUserInfo
    """

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
        return [_select_keys(u, ['username', 'roles', 'datasets']) for u in User.query({})]


@app.route('/api/users/', methods=['PUT'])
@app.route('/api2/users/', methods=['PUT'])
@rest(role='admin')
def admin_users_add(username, password, **_):
    """
    Add new user
    Returns:
        UserInfo
    """

    if User.first(F.username == username):
        raise Exception('User already exists: ' + str(username))
    user = User(username=username)
    user.set_password(password)
    user.save()
    user = user.as_dict()
    return _select_keys(user, ['username', 'roles'])


@app.route('/api/account/', methods=['POST'])
@app.route('/api2/account/', methods=['POST'])
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
        return _select_keys(user.as_dict(), ['username', 'roles'])
    else:
        if otp:
            user.otp_secret = pyotp.random_base32()
            user.save()
        else:
            user.otp_secret = ''
            user.save()
        return _select_keys(user.as_dict(), ['otp_secret'])


@app.route('/api/users/<username>', methods=['DELETE'])
@app.route('/api2/users/<username>', methods=['DELETE'])
@rest(role='admin')
def admin_users_del(username):
    """Delete user"""
    User.query(F.username == username).delete()
    return True


@app.route('/api/storage/<path:path>', methods=['GET', 'POST'])
@app.route('/api2/storage/<path:path>', methods=['GET', 'POST'])
@app.route('/api/storage/', methods=['GET', 'POST'])
@app.route('/api2/storage/', methods=['GET', 'POST'])
@rest()
def list_storage(path='', search='', mkdir=''):
    """
    List out files in directory or get file at that path
    Returns:
        list of file names, or the file itself
    """

    path = 'file://' + path.split('://')[-1]

    if path.rsplit('.', 1)[-1].lower() in ('py', 'pyc', 'yaml', 'yml', 'sh'):
        return 'Not supported extension name', 400

    if mkdir:
        storage.mkdir(path, mkdir)

    results = None
    if search:
        # path is a query
        results = list(storage.search(path, '**' + search))
    else:
        results = storage.statdir(path)
        if not results and storage.exists(path):
            results = storage.open(path)

    if isinstance(results, list):
        return sorted(results,
                      key=lambda x: x['ctime'], reverse=True)
    else:
        return send_file(results, download_name=path.split('/')[-1])


@app.route('/api/storage/<path:path>', methods=['PUT'])
@app.route('/api2/storage/<path:path>', methods=['PUT'])
@app.route('/api/storage/', methods=['PUT'])
@app.route('/api2/storage/', methods=['PUT'])
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
    return sfs


@app.route('/api/storage/move', methods=['POST'])
@app.route('/api2/storage/move', methods=['POST'])
@rest()
def move_storage(source, destination, keep_folder=True):
    """Move/Rename file from source to destination"""
    if keep_folder:
        destination = os.path.basename(destination)
        destination = os.path.join(os.path.dirname(source), destination)
    paragraphs = Paragraph.query(F.source.file == source)
    source = storage.expand_path(source)
    destination = storage.expand_path(destination)
    storage.move(source, destination)
    paragraphs.update(
        Fn.set({'source.file': storage.truncate_path(destination)}))
    return True


@app.route('/api2/parser/parse', methods=['POST'])
@rest(login=True)
def parser_parse(query, literal=False, tokens=False):
    if literal:
        return parser.parse_literal(query)
    elif tokens:
        return [{'text': token.text, 'type': parser.get_symbol(token.type)} for token in parser.tokenize(query)]
    return parser.parse(query)


@app.route('/api2/collections/<coll>/<pid>', methods=['POST'])
@app.route('/api/collections/<coll>/<pid>', methods=['POST'])
@rest()
def modify_paragraph(coll, pid, **kws):
    """
    Modify paragraph info
    Returns: Paragraph with changed fields only
    """

    pid = ObjectId(pid)
    para = Paragraph.get_coll(coll).first(F.id == pid)
    flag = False
    changed = set()
    if para:
        for field, val in kws.items():
            if field in ('_id', 'matched_content'):
                continue
            if field in ('$push', '$pull'):
                Paragraph.get_coll(coll).query(
                    F.id == pid).update({field: val})
                changed.update(val.keys())
            else:
                flag = True
                if val is None and hasattr(para, field):
                    delattr(para, field)
                else:
                    setattr(para, field, val)
                changed.add(field)
        if flag:
            para.save()
    return {'paragraphs': {str(para.id): _select_keys(para.as_dict(), changed)}}


@app.route('/api2/collections/<coll>/<pid>/pagenum', methods=['POST'])
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


@app.route('/api2/collections/<coll>/batch', methods=["GET", "POST"])
@app.route('/api/collections/<coll>/batch', methods=["GET", "POST"])
@rest()
def batch(coll, ids, **kws):
    """Batch edit"""

    paras = list(Paragraph.get_coll(coll).query(
        F.id.in_([ObjectId(_) if len(_) == 24 else _ for _ in ids])))
    changed = set()
    for field, val in kws.items():
        if field.startswith('$'):
            changed.update(val.keys())
        else:
            changed.add(field)
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
                            if afield == 'keywords':
                                Term.write(ele, 'keywords')
            elif not field.startswith('$'):
                para[field] = val
        para.save()

    if 'api2' in request.path:
        return {
            'paragraphs': {
                str(p.id): _select_keys(p.as_dict(), changed)
                for p in paras
            }
        }
    else:
        return {
            str(p.id): p.as_dict()
            for p in paras
        }


@app.route('/api/collections/<coll>/group', methods=["GET", "POST"])
@app.route('/api2/collections/<coll>/group', methods=["GET", "POST"])
@rest()
def grouping(coll, ids, group='', values=None, ungroup=False, limit=100):
    """Grouping selected paragraphs

    Returns:
        Group ID
    """
    para_query = Paragraph.get_coll(coll).query(
        F.id.in_([ObjectId(_) if len(_) == 24 else _ for _ in ids]))

    proposed_groups = group or values
    if isinstance(proposed_groups, str):
        proposed_groups = [proposed_groups]

    if ungroup:
        para_query.update(Fn.pull(F.keywords.regex('^#')))
        groups = []

    else:
        paras = list(para_query)
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
                    '#0' + _hashing(min(map(lambda p: str(p.id), paras)))]

            gids = list(set(gids) - set(named) - set(groups))
            para_query = Paragraph.get_coll(coll).query(F.id.in_(
                [ObjectId(_) if len(_) == 24 else _ for _ in ids]) | F.keywords.in_(gids))
            for g in groups:
                para_query.update(Fn.addToSet(keywords=g))
            if gids:
                para_query.update(Fn.pull(F.keywords.in_(gids)))

    return {
        'group_ids': groups,
        'paragraph_ids': ids,
        # required by v2
        'values': groups,
        'paragraphs': {str(para.id): {} for para in para_query.limit(limit)}
    }


@app.route('/api/collections/<coll>/split', methods=["GET", "POST"])
@app.route('/api/collections/<coll>/merge', methods=["GET", "POST"])
@app.route('/api2/collections/<coll>/split', methods=["GET", "POST"])
@app.route('/api2/collections/<coll>/merge', methods=["GET", "POST"])
@rest()
def splitting(coll, paragraphs):
    """Split or merge selected items/paragraphs into seperate/single paragraph(s)

    Returns:
        bool: True if succeeded
    """
    paras = list(Paragraph.get_coll(coll).query(
        F.id.in_([ObjectId(_) for _ in paragraphs])))

    retval = []

    if request.path.endswith('/split'):
        for para in paras:
            para_dict = para.as_dict()
            del para_dict['_id']
            del para_dict['images']
            for i in para.images:
                pnew = Paragraph(images=[i], **para_dict)
                pnew.save()
                retval.append(pnew.as_dict(True))
            para.delete()
    else:
        if not paras:
            return False

        selected_ids = list(
            map(ObjectId, itertools.chain(*paragraphs.values())))
        selected = MediaItem.query(F.id.in_(selected_ids))

        para0 = Paragraph(paras[0])
        para0.id = None
        para0.images = selected

        for para in paras:
            para0.keywords += para.keywords
            para.images = [k for k in para.images if k.id not in selected_ids]
            para.save()

        para0.save()
        retval.append(para0.as_dict(True))

    return {
        'paragraphs': retval
    }


@app.route('/api/mediaitem/rating', methods=["GET", "POST"])
@app.route('/api2/mediaitem/rating', methods=["GET", "POST"])
@rest()
def set_rating(ids=None, inc=1, val=0, least=0):
    """Increase or decrease the rating of selected items
    """

    if not ids:
        return False

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

    if 'api2' in request.path:
        return {
            'items': {str(i.id): {'rating': i.rating}
                      for i in items}
        }
    else:
        return {str(i.id): i.rating
                for i in items}


@app.route('/api/mediaitem/reset_storage', methods=["GET", "POST"])
@app.route('/api2/mediaitem/reset_storage', methods=["GET", "POST"])
@rest()
def reset_storage(ids):
    """Reset storage status of selected items
    """

    items = list(MediaItem.query(
        F.id.in_([ObjectId(_) if len(_) == 24 else _ for _ in ids])))
    for i in items:
        if 'file' in i.source:
            del i.source['file']
        i.save()
    if 'api2' in request.path:
        return {
            'items': {
                str(i.id): {'source': {'file': None}}
                for i in items
            }
        }
    else:
        return {
            str(i.id): i.source.get('file')
            for i in items
        }


@app.route('/api/mediaitem/merge', methods=["POST"])
@app.route('/api2/mediaitem/merge', methods=["POST"])
@rest()
def merge_items(pairs):
    """Process the two specified media items as duplications,
       keeping the first one, deleting the second and merging
       the Paragraphs where they locate
    """
    if isinstance(pairs, list):
        d_pairs = defaultdict(list)
        for rese, dele in pairs:
            d_pairs[rese].append(dele)
        pairs = d_pairs

    for rese, dele in pairs.items():
        rese, dele = MediaItem.first(
            F.id == ObjectId(rese)), list(MediaItem.query(F.id.in_([ObjectId(_) for _ in dele])))
        if rese and dele:
            Paragraph.merge_by_mediaitems(rese, dele)

    return True


@app.route('/api/mediaitem/delete', methods=["POST"])
@app.route('/api2/mediaitem/delete', methods=["POST"])
@rest()
def delete_item(para_items: dict):
    """Remove Media Item from paragraph"""

    del_items = set()
    for pid, items in para_items.items():
        para = Paragraph.first(F.id == ObjectId(pid))
        if para is None:
            continue

        items = [ObjectId(i) for i in items]
        para.images.remove(items)
        para.save()
        del_items.update(items)

    for i in del_items:
        if Paragraph.first(F.images == i):
            continue
        image_item = MediaItem.first(F.id == ObjectId(i))
        if image_item:
            image_item.delete()

    return True


@app.route('/api/tasks/', methods=['PUT'])
@app.route('/api2/tasks/', methods=['PUT'])
@rest()
def create_task(**task):
    """Create new task"""

    task.pop('shortcut_map', None)
    task = TaskDBO(**task)
    task.creator = logined()
    task.save()
    return str(task.id)


@app.route('/api/tasks/shortcuts', methods=['GET'])
@app.route('/api2/tasks/shortcuts', methods=['GET'])
@rest()
def list_tasks_shortcuts():
    """List out quick tasks"""
    return list(TaskDBO.query((F.shortcut_map != {}) & _task_authorized()))


@app.route('/api/tasks/<task_id>', methods=['DELETE'])
@app.route('/api2/tasks/<task_id>', methods=['DELETE'])
@rest()
def delete_task(task_id):
    """Remove task"""

    _id = ObjectId(task_id)
    return TaskDBO.query(F.id == _id).delete().acknowledged


@app.route('/api/tasks/<task_id>', methods=['POST'])
@app.route('/api2/tasks/<task_id>', methods=['POST'])
@rest()
def update_task(task_id, **task):
    """Update task info"""
    task_id = ObjectId(task_id)

    if '_id' in task:
        del task['_id']

    executed = TaskDBO.query(
        (F.id == task_id) & _task_authorized()).update(Fn.set(task))

    return {'tasks': {
        str(task_id): task
    },
        'updated': task
    }


@app.route('/api/tasks/<task_id>', methods=['GET'])
@app.route('/api/tasks/', methods=['GET'])
@app.route('/api2/tasks/', methods=['GET'])
@app.route('/api2/tasks/<task_id>', methods=['GET'])
@rest()
def list_task(task_id=''):
    """List out tasks"""

    if task_id:
        _id = ObjectId(task_id)
        return TaskDBO.first((F.id == _id) & _task_authorized())
    else:
        return list(TaskDBO.query(_task_authorized()).sort(-F.last_run, -F.id))


@app.route('/api/help/pipelines')
@app.route('/api2/help/pipelines')
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
@app.route('/api2/help/langs')
@rest(cache=True)
def help_langs():
    """Provide supported language codes"""
    return language_iso639


@app.route('/api/help/queryexpr')
@app.route('/api2/help/queryexpr')
@rest(cache=True)
def help_queryexpr():
    """Provide meta data for query expr"""
    return {
        'function_names': list(parser.functions.keys()) + list(ee.implemented_functions),
    }


@app.route('/api/history')
@app.route('/api2/history')
@rest()
def history():
    """History records for the current user"""

    History.query(F.created_at < datetime.datetime.utcnow() -
                  datetime.timedelta(days=30)).delete()
    return list(
        History.query(F.user == logined()).sort(-F.created_at).limit(100))


@app.route('/api/search', methods=['POST'])
@app.route('/api2/search', methods=['POST'])
@rest()
def do_search(q='', req='', sort='', limit=100, offset=0,
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

    if datasource.groups != 'none':
        for res in results:
            group = res.get(datasource.groups)
            if isinstance(group, dict):
                group = '(' + ','.join(
                    [f'{k}={app.json_encoder().encode(v)}' for k, v in group.items()]) + ')'
            else:
                group = app.json_encoder().encode(group)

    History(user=logined(), queries=[q, req],
            created_at=datetime.datetime.utcnow()).save()

    return {'results': results, 'query': datasource.query}


@app.route('/api2/datasets')
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
@app.route('/api2/datasets/<action>', methods=['POST'])
@rest(role='admin')
def set_datasets(action, **j):
    """Update dataset info"""

    dataset = None
    if '_id' in j:
        dataset = Dataset.first(F.id == ObjectId(j['_id']))
        del j['_id']

    if action == 'edit':
        if dataset:
            for k, v in j.items():
                dataset[k] = v
            dataset.save()
        else:
            Dataset(**j).save()

    elif action == 'batch':
        for dataset in j:
            jset = {k: v for k, v in dataset.items() if k !=
                    '_id' and v is not None}
            if '_id' in dataset:
                Dataset.query(F.id == ObjectId(dataset['_id'])).update(Fn.set(**jset))
            else:
                Dataset(**jset).save()

    elif action == 'rename':
        assert dataset and 'to' in j, 'must specify valid dataset id and new name'
        dataset.rename(j['to'])

    elif action == 'sources':
        assert dataset, 'dataset not found'
        dataset.update_sources()

    return True


@app.route("/api/term/<field>", methods=['POST'])
@app.route("/api2/term/<field>", methods=['POST'])
@rest()
def query_terms(field, pattern='', regex=False, scope=''):
    """Query terms"""

    if pattern:
        if regex:
            pattern = pattern.replace(' ', '.*')
            pattern = {'term': {'$regex': pattern, '$options': 'i'}}
        else:
            pattern = (F.term == pattern) | (F.aliases == pattern)
        return list(Term.query(F.field == field, pattern).limit(100))
    elif scope:
        return [i['_id'] for i in Paragraph.aggregator.match(
                parser.parse(scope)
                ).sort(_id=-1).limit(1000).unwind('$' + field).group(
                _id='$' + field, count=Fn.sum(1)
                ).sort(count=-1).limit(100).perform(raw=True)]


@app.route("/api/image/<coll>/<storage_id>.<ext>")
@app.route("/api2/image/<coll>/<storage_id>.<ext>")
@app.route("/api/image")
@app.route("/api2/image")
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
@rest(cache=True)
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


@app.route('/api/quicktask', methods=['POST'])
@app.route('/api2/quicktask', methods=['POST'])
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
        results = Task(stages=pipeline, params={}).execute()
    else:
        params = {'query': query, 'raw': raw,
                  'mongocollections': mongocollection}
        results = Task(stages=[
            ('DBQueryDataSource', params),
            ('AccumulateParagraphs', {}),
        ], params={}).execute()

    return _expand_results(results)


@app.route('/api/admin/db', methods=['POST'])
@app.route('/api2/admin/db', methods=['POST'])
@rest(role='admin')
def dbconsole(mongocollection='', query='', operation='', operation_params='', preview=True):
    """Database console for admin"""

    mongo = Paragraph.db.database[mongocollection]
    query = parser.parse(query) if query else {}
    operation_params = parser.parse(
        operation_params) if operation_params else None
    if isinstance(operation_params, dict):
        keys = list(operation_params)
        if len(keys) != 1:
            operation_params = {
                '$set': operation_params
            }
        elif keys[0] in ['keywords', 'images']:
            operation_params = {'$addToSet': operation_params}
        elif not keys[0].startswith('$'):
            operation_params = {
                '$set': operation_params
            }

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
@app.route('/api2/admin/db/collections', methods=['GET'])
@rest(role='admin')
def dbconsole_collections():
    """Get all collections in current MongoDB database"""

    return Paragraph.db.database.list_collection_names()


@app.route('/api/meta', methods=['GET'])
@app.route('/api2/meta', methods=['GET'])
@rest(login=False)
def get_meta():
    """Get meta settings"""

    return Meta.first(F.app_title.exists(1)) or Meta()


@app.route('/api/meta', methods=['POST'])
@app.route('/api2/meta', methods=['POST'])
@rest(role='admin')
def set_meta(**vals):
    """Update meta settings"""

    result = Meta.first(F.app_title.exists(1)) or Meta()
    for key, val in vals.items():
        setattr(result, key, val)
    result.save()
    return True


@app.route('/api/plugins', methods=['GET'])
@app.route('/api2/plugins', methods=['GET'])
@rest()
def get_plugins():
    """Get plugin names
    """
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
