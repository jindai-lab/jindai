import re
from models import try_download
import glob
import os
import re
import zipfile
import pickle
from io import BytesIO
from typing import IO, Tuple
import config

RE_DIGITS = re.compile(r'[\+\-]?\d+')


def _opr(k):
    oprname = {
        'lte': 'le',
        'gte': 'ge',
        '': 'eq'
    }.get(k, k)
    return '__' + oprname + '__'


def _getattr(o, k, default=None):
    if '.' in k:
        for k_ in k.split('.'):
            o = _getattr(o, k_, default)
        return o

    if isinstance(o, dict):
        return o.get(k, default)
    elif isinstance(o, list) and RE_DIGITS.match(k):
        return o[int(k)] if 0 <= int(k) < len(o) else default
    else:
        return getattr(o, k, default)

    
def _hasattr(o, k):
    if '.' in k:
        for k_ in k.split('.')[:-1]:
            o = _getattr(o, k_)
        k = k.split('.')[-1]

    if isinstance(o, dict):
        return k in o
    elif isinstance(o, list) and RE_DIGITS.match(k):
        return 0 <= int(k) < len(o)
    else:
        return hasattr(o, k)


def _test_inputs(inputs, v, k='eq'):
    oprname = _opr(k)
    if oprname == '__in__':
        return inputs in v
    elif oprname == '__size__':
        return len(inputs) == v
    
    if isinstance(inputs, list):
        rr = False
        for v_ in inputs:
            rr = rr or _getattr(v_, oprname)(v)
            if rr:
                break
    else:
        rr = _getattr(inputs, oprname)(v)
    return rr is True


def execute_query_expr(parsed, inputs):
    r = True
    assert isinstance(parsed, dict), 'QueryExpr should be parsed first and well-formed.'
    for k, v in parsed.items():
        if k.startswith('$'):
            if k == '$and':
                rr = True
                for v_ in v:
                    rr = rr and execute_query_expr(v_, inputs)
            elif k == '$or':
                rr = False
                for v_ in v:
                    rr = rr or execute_query_expr(v_, inputs)
                    if rr:
                        break
            elif k == '$not':
                rr = not execute_query_expr(v, inputs)
            elif k == '$regex':
                rr = re.search(v, inputs) is not None
            elif k == '$options':
                continue
            else:
                rr = _test_inputs(inputs, v, k[1:])
            r = r and rr
        elif not isinstance(v, dict) or not [1 for v_ in v if v_.startswith('$')]:
            r = r and _test_inputs(_getattr(inputs, k), v)
        else:
            r = r and execute_query_expr(v, _getattr(inputs, k) if _hasattr(inputs, k) else None)
    return r


def expand_file_patterns(patterns : list, names_only = False) -> Tuple[IO, str]:
    """
    读取文件（包括压缩包内容）或网址，其中文件名可以使用 */? 通配符，网址可以使用 {num1-num2} 形式给定迭代范围
    Returns:
        Tuple[IO, str]: IO 为内容，str 为文件名或网址
    """
            
    for pattern in patterns:
        if pattern.startswith('https://') or pattern.startswith('http://'):
            urls = []
            iterate = re.search(r'\{(\d+\-\d+)\}', pattern)
            if iterate:
                start, end=map(int,iterate.group(1).split('-'))
                for i in range(start, end+1):
                    urls.append(pattern.replace(iterate.group(0), str(i)))
            else:
                urls = [pattern]
            for url in urls:
                if names_only:
                    yield url
                else:
                    yield BytesIO(try_download(url, '/'.join(url.split('/')[:-1]))), url
        else:
            if not pattern.startswith('sources/'):
                pattern = os.path.join(config.storage, pattern)
            for f in glob.glob(pattern):
                if not names_only and f.endswith('.zip') or f.endswith('.epub'):
                    with zipfile.ZipFile(f) as z:                    
                        for f_ in z.filelist:
                            yield z.open(f_), f + '#' + f_.filename
                elif os.path.isdir(f):
                    patterns.append(f + '/*')
                else:
                    if names_only:
                        yield f
                    else:
                        yield open(f, 'rb'), f


with open(os.path.join(config.rootpath, 'models_data', 'language_iso639'), 'rb') as flang:
    language_iso639 = pickle.load(flang)
