import re

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
