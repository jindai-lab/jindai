"""Database access functionality"""
import hashlib
import json
import re
import jieba
from bson import SON
from PyMongoWrapper import MongoOperand, F, Fn, Var, QueryExprInterpreter, QueryExpressionError, MongoConcating

from .models import Paragraph, Term


parser = QueryExprInterpreter(
    'keywords', '='
)


def _groupby(_id='', images=None, count=None, **params):
    field_name = re.sub(r'[{}"\']', '_', str(_id).strip('$'))

    join_images = False
    if images == 1:
        params['images'] = Fn.push('$images')
        join_images = True
    elif images is not None:
        params['images'] = images
    if count is None:
        params['count'] = Fn.sum(Fn.size('$images'))

    stages = [
        Fn.group(orig=Fn.first('$$ROOT'), _id=_id, **params),
        Fn.replaceRoot(newRoot=Fn.mergeObjects(
            '$orig', {'group_id': '$_id'}, {k: f'${k}' for k in params if k != '_id'})),
    ]

    if join_images:
        stages.append(Fn.addFields(
            group_field=field_name,
            images=Fn.reduce(
                input='$images',
                initialValue=[],
                in_=Fn.concatArrays('$$value', '$$this')
            )
        ))
    else:
        stages.append(
            Fn.addFields(group_field=field_name)
        )

    return MongoConcating(stages)


def _auto(param):
    
    param = param.strip()

    def _judge_type(query):
        """Judge query mode: keywords or expression"""

        is_expr = False
        if query.startswith('?'):
            is_expr = True
        elif re.search(r'[,.~=&|()><\'"`@_#:\-%]', query):
            is_expr = True

        return is_expr

    is_expr = _judge_type(param)
    if is_expr:
        param = param.lstrip('?')
    else:
        param = '`' + '`,`'.join([_.strip().lower().replace('`', '\\`')
                                  for _ in jieba.cut(param) if _.strip()]) + '`'
        if param == '``':
            param = ''

    if not param:
        return {}

    try:
        return parser.parse(param) or {}
    except QueryExpressionError as qe:
        try:
            return parser.parse(param + ';')
        except QueryExpressionError:
            raise qe
        except:
            pass


def _term(term):
    terms = []
    for r in Term.query(F.term == term, F.aliases == term, logic='or'):
        terms += [r.term] + r.aliases
    if terms and len(terms) > 1:
        return {'$in': terms}
    return term


def _set_author(name):
    return Fn.addFields(author=name, keywords=Fn.setUnion(Var.keywords, [name]))


parser.functions.update({
    'expand': lambda: MongoConcating([
        Fn.unwind('$images'),
        Fn.addFields(originals='$images'),
        Fn.lookup(from_='mediaitem', localField='images',
                  foreignField='_id', as_='images'),
        Fn.addFields(images=Fn.cond(Fn.size('$images')
                     == 0, ['$originals'], '$images'))
    ]),
    'gid': lambda: MongoConcating([
        Fn.addFields(
            gid=Fn.filter(input=Var.keywords, as_='t', cond=Fn.regexMatch(
                input="$$t", regex=r'^#'))
        ),
        Fn.unwind(path=Var.gid, preserveNullAndEmptyArrays=True),
        Fn.addFields(gid=Fn.ifNull(
            Var.gid, Fn.concat('id=', Fn.toString('$_id')))),
    ]),
    'begin': lambda prefix: F.keywords.regex(f'^{re.escape(prefix)}'),
    'groupby': _groupby,
    'auto': _auto,
    'term': _term,
    'source': lambda url: (F.source.url == url) | (F.source.file == url),
    'c': lambda text: F.content.regex(text.strip()) | F.caption.regex(text.strip()),
    'setAuthor': _set_author,
})

parser.functions['s'] = parser.functions['source']
parser.functions['seta'] = parser.functions['setAuthor']


class DBQuery:
    """Database query class"""

    @staticmethod
    def _parse(query, wordcutter=None):
        """Parse query (keywords or expression) and convert it to aggregation query"""

        if wordcutter is None:
            wordcutter = jieba.cut

        if not query:
            return []

        if isinstance(query, (tuple, list)):
            query, *limitations = query
        else:
            limitations = []

        # parse limitations
        if limitations and limitations[0]:
            limitations = {
                '$and': [parser.parse(expr) for expr in limitations if expr]}

        if isinstance(query, str):
            # judge type of major query and formulate
            query = _auto(query) or []

        if not isinstance(query, list):
            query = [query]

        if not query:
            return [{'$match': limitations}] if limitations else []

        for i in range(len(query)):
            stage = query[i]
            if isinstance(stage, str):
                query[i] = {'$match': parser.parse(stage)}
            elif isinstance(stage, dict) and \
                    not [_ for _ in stage if _.startswith('$') and _ not in ('$not', '$and', '$or')]:
                query[i] = {'$match': stage}

        query = DBQuery._merge_req(query, limitations)
        return query

    @staticmethod
    def _merge_req(qparsed, req):
        """Merge to parsed expressions"""
        if not req:
            return qparsed

        first_query = qparsed[0]
        if isinstance(first_query, dict) and '$match' in first_query:
            return [
                {'$match':
                    (MongoOperand(first_query['$match']) & MongoOperand(req))()}
            ] + qparsed[1:]

        return qparsed + [{'$match': req}]
    
    @staticmethod
    def _find_query(query, keyname):
        for v in query:
            if isinstance(v, dict) and keyname in v:
                return v[keyname]

    @staticmethod
    def _pop_query(query, keyname):
        val = None
        for i, v in enumerate(query):
            if isinstance(v, dict) and keyname in v:
                val = v.pop(keyname)
                if not v:
                    query.remove(v)
                return val
            
    def __init__(self, query, mongocollections='', limit=0, skip=0, sort='',
                 raw=False, groups='none', pmanager=None, wordcutter=None):

        self.query = DBQuery._parse(query, wordcutter)
        self.raw = raw

        # test plugin pages
        if pmanager and len(self.query) > 0 and '$plugin' in self.query[-1]:
            self.query, plugin_args = self.query[:-1], \
                self.query[-1]['$plugin'].split('/')
        else:
            plugin_args = []

        self.handler = None
        if plugin_args:
            self.handler = pmanager.filters.get(
                plugin_args[0]), plugin_args[1:]

        if not mongocollections:
            mongocollections = ''
        self.mongocollections = mongocollections.split('\n') if isinstance(
            mongocollections, str) else mongocollections

        if sort == 'id':
            sort = ''

        if len(self.query) > 1 and MongoOperand.get_key(self.query[0]) == '$from':
            self.mongocollections = self.query[0]['$from']
            if not isinstance(self.mongocollections, list):
                self.mongocollections = self.mongocollections.split(',')
            self.query = self.query[1:]

        if len(self.query) > 1 and '$raw' in self.query[-1]:
            self.raw = self.query[-1]['$raw']
            self.query = self.query[:-1]
            
        sort = self._pop_query(self.query, '$sort') or sort
        if isinstance(sort, SON):
            sort = ','.join([('-' if v < 0 else '') + k for k, v in sort.items()])

        self.groups = groups

        groupping = ''
        if groups in ('', 'none'):
            if not sort:
                sort = '-pdate,-id'
        elif groups == 'group':
            groupping = ''';gid();group_id: $gid;'''
            if not sort:
                sort = 'group_id,-pdate'
        elif groups == 'source':
            groupping = ';addFields(group_id=ifNull($source.url,$source.file))'
            if not sort:
                sort = 'source'
        else:
            groupping = f';addFields(group_id=${groups})'
            if not sort:
                sort = '-group_id'

        if groupping:
            if '.' not in sort and ',' not in sort:
                if sort.startswith('-'):
                    sorting = f',sorting_field=max(${sort[1:]})'
                else:
                    sorting = f',sorting_field=min(${sort})'
                sorting = sorting.replace('($id)', '($_id)')
                sort = ('-' if sort.startswith('-') else '') + 'sorting_field'
            else:
                sorting = ''
            groupping += f''';
                gid: ifNull($group_id, concat("id=o'", toString($_id), "'"));
                images: ifNull($images,[]);
                groupby(id=$gid{sorting},images=1);
                groupby(id=$_id,count=sum($count),images=1);
            '''
            groupping = parser.parse(groupping)

            self.query += groupping

        self.limit = limit
        self.sort = sort if sort != '$' else None
        self.skips = {}
        self.skip = skip

    @property
    def query_hash(self):
        return hashlib.sha1(json.dumps(self.query).encode('utf-8')).hexdigest()

    def fetch_rs(self, mongocollection, sort=None, limit=-1, skip=-1):
        """Fetch result set for single mongo collection"""

        rs = Paragraph.get_coll(mongocollection)

        if sort is None:
            sort = self.sort
        if skip < 0:
            skip = self.skips.get(mongocollection, 0)
        if limit < 0:
            limit = self.limit

        agg = self.query

        if sort:
            sort = parser.parse_sort(sort)

        if sort == [('random', 1)]:
            agg.append({'$sample': {'size': limit}})
            limit = 0
            skip = 0
        elif sort:
            agg.append(
                {'$sort': SON(sort)})
        if skip > 0:
            agg.append({'$skip': skip})
        if limit > 0:
            agg.append({'$limit': limit})
        rs = rs.aggregate(agg, raw=self.raw, allowDiskUse=True)

        return rs

    def fetch_all_rs(self):
        """Fetch all result sets"""

        if self.skip is not None and self.skip > 0:
            skip = self.skip
            for coll in self.mongocollections:
                count = self.fetch_rs(coll, sort='id', limit=0, skip=0).count()
                if count <= skip:
                    skip -= count
                    self.skips[coll] = -1
                else:
                    self.skips[coll] = skip
                    break

        for coll in self.mongocollections:
            if self.skips.get(coll, 0) >= 0:
                yield from self.fetch_rs(coll)

    def fetch(self):
        """Fetch results"""

        if self.handler:
            handler, args = self.handler
            yield from handler['handler'](self, *args)
        else:
            yield from self.fetch_all_rs()

    def count(self):
        """Count documents, -1 if err"""
        try:
            return sum([self.fetch_rs(r, sort='id', limit=0, skip=0).count() for r in self.mongocollections])
        except Exception:
            return -1
