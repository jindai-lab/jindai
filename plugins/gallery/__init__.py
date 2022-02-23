import hashlib
import os
import random
from typing import List
from bson import ObjectId
from datasources.gallerydatasource import GalleryAlbumDataSource
from flask import Response, jsonify, request
from helpers import *
from models import Paragraph
from plugin import Plugin
from PyMongoWrapper import F, Fn


# HELPER FUNCS
def single_item(pid: str, iid: str) -> List[Paragraph]:
    """Return a single-item paragraph object with id = `pid` and item id = `iid`

    Args:
        pid (str): Paragraph ID
        iid (str): ImageItem ID

    Returns:
        List[Paragraph]: a list with at most one element, i.e., the single-item paragraph object
    """
    if pid:
        pid = ObjectId(pid)
        p = Paragraph.first(F.id == pid)
    elif iid:
        iid = ObjectId(iid)
        p = Paragraph.first(F.images == iid)

    if iid and p:
        p.images = [i for i in p.images if i.id == iid]
        p.group_id = f"id={p['_id']}"
        return [p]
    else:
        return []


class Gallery(Plugin):

    def __init__(self, app):
        super().__init__(app)
           
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        # ALBUM OPERATIONS

        @app.route('/api/gallery/grouping', methods=["GET", "POST", "PUT"])
        @rest()
        def grouping(ids, group='', delete=False):
            """Grouping selected paragraphs

            Returns:
                Response: 'OK' if succeeded
            """
            def gh(x): return hashlib.sha256(x.encode('utf-8')).hexdigest()[-9:]
            
            paras = list(Paragraph.query(F.id.in_([ObjectId(_) if len(_) == 24 else _ for _ in ids])))

            if delete:
                group_id = ''
                for p in paras:
                    p.keywords = [_ for _ in p.keywords if not _.startswith('*')]
                    p.save()

            else:
                if not paras:
                    return True
                gids = []
                for p in paras:
                    gids += [_ for _ in p.keywords if _.startswith('*')]
                named = [_ for _ in gids if not _.startswith('*0')]

                if group:
                    group_id = '*' + group
                elif named:
                    group_id = min(named)
                elif gids:
                    group_id = min(gids)
                else:
                    group_id = '*0' + gh(min(map(lambda p: str(p.id), paras)))

                for p in paras:
                    if group_id not in p.keywords:
                        p.keywords.append(group_id)
                        p.save()

                gids = list(set(gids) - set(named))
                if gids:
                    for p in Paragraph.query(F.keywords.in_(gids)):
                        for id0 in gids:
                            if id0 in p.keywords:
                                p.keywords.remove(id0)
                        if group_id not in p.keywords:
                            p.keywords.append(group_id)
                        p.save()

            return group_id

        @app.route('/api/gallery/get', methods=["GET", "POST"])
        @rest()
        def get(query='', flag=0, post='', limit=20, offset=0, order={'keys':['random']}, direction='next', groups=False, archive=False, count=False):
            """Get records

            Returns:
                Response: json document of records
            """
            ds = GalleryAlbumDataSource(query, limit, offset, groups, archive, True, '', direction, order)
            
            if count:
                return (list(ds.aggregator.group(_id='', count=Fn.sum(1)).perform(raw=True)) + [{'count': 0}])[0]['count']

            prev_order, next_order = {}, {}

            post_args = post.split('/')
            special_page = special_pages.get(post_args[0])
            if special_page:
                ret = special_page.handle_special_page(ds, post_args) or ([], None, None)
                if isinstance(ret, tuple) and len(ret) == 3:
                    rs, prev_order, next_order = ret
                else:
                    return ret
            else:
                rs = ds.fetch()

            r = []
            res = {}
            try:
                for res in rs:
                    if isinstance(res, Paragraph):
                        res = res.as_dict(expand=True)

                    if 'count' not in res:
                        res['count'] = ''

                    if '_id' not in res or not res['images']:
                        continue

                    res['images'] = [_ for _ in res['images'] if isinstance(
                        _, dict) and _.get('flag', 0) == flag]
                    if not res['images']:
                        continue

                    if ds.random:
                        res['images'] = random.sample(res['images'], 1)
                    elif archive or groups or 'counts' in res:
                        cnt = res.get('counts', len(res['images']))
                        if cnt > 1:
                            res['count'] = '(+{})'.format(cnt)

                    if direction != 'next':
                        r.insert(0, res)
                    else:
                        r.append(res)

            except Exception as ex:
                import traceback
                return jsonify({
                    'exception': repr(ex),
                    'trackstack': traceback.format_exc(),
                    'results': [res],
                    '_filters': ds.aggregator.aggregators
                }), 500

            def mkorder(rk):
                o = dict(order)
                for k in o['keys']:
                    k = k[1:] if k.startswith('-') else k
                    if '.' in k:
                        o[k] = rk
                        for k_ in k.split('.'):
                            o[k] = o[k].get(k_, {})
                            if isinstance(o[k], list):
                                o[k] = o[k][0] if o[k] else {}
                        if o[k] == {}:
                            o[k] = 0
                    else:
                        o[k] = rk.get(k, 0)
                return o

            if r:
                if not prev_order:
                    prev_order = mkorder(r[0])
                if not next_order:
                    next_order = mkorder(r[-1])

            return jsonify({
                'total_count': len(r),
                'params': request.json,
                'prev': prev_order,
                'next': next_order,
                'results': r,
                '_filters': ds.aggregator.aggregators
            })

        @app.route('/api/gallery/styles.css')
        def plugins_style():
            """Returns css from all enabled plugins

            Returns:
                Response: css document
            """
            css = '\n'.join([p.run_callback('css')
                            for p in callbacks['css']])
            return Response(css, mimetype='text/css')

        @app.route('/api/gallery/plugin_pages', methods=["GET", "POST"])
        @rest()
        def plugin_pages():
            """Returns names for special pages in every plugins
            """
            return list(special_pages.keys())

        