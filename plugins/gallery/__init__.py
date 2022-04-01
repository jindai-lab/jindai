"""相册和图像相关"""

import hashlib
from typing import List
from PyMongoWrapper import F, ObjectId
from jindai.helpers import *
from jindai.models import Paragraph
from jindai import Plugin

from .imageproc import *
from .ocr import *


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
        para = Paragraph.first(F.id == pid)
    elif iid:
        iid = ObjectId(iid)
        para = Paragraph.first(F.images == iid)
    if iid and para:
        para.images = [i for i in para.images if i.id == iid]
        para.group_id = f"id={para['_id']}"
        return [para]

    return []


class Gallery(Plugin):
    """相册插件"""

    def __init__(self, app):
        super().__init__(app)
        self.register_pipelines(globals())

        # ALBUM OPERATIONS

        @app.route('/api/gallery/grouping', methods=["GET", "POST", "PUT"])
        @rest()
        def grouping(ids, group='', delete=False):
            """Grouping selected paragraphs

            Returns:
                Response: 'OK' if succeeded
            """
            def hashing(msg):
                return hashlib.sha256(
                    msg.encode('utf-8')).hexdigest()[-9:]

            paras = list(Paragraph.query(
                F.id.in_([ObjectId(_) if len(_) == 24 else _ for _ in ids])))
            if delete:
                group_id = ''
                for para in paras:
                    para.keywords = [
                        _ for _ in para.keywords if not _.startswith('*')]
                    para.save()
            else:
                if not paras:
                    return True
                gids = []
                for para in paras:
                    gids += [_ for _ in para.keywords if _.startswith('*')]
                named = [_ for _ in gids if not _.startswith('*0')]
                if group:
                    group_id = '*' + group
                elif named:
                    group_id = min(named)
                elif gids:
                    group_id = min(gids)
                else:
                    group_id = '*0' + hashing(min(map(lambda p: str(p.id), paras)))
                for para in paras:
                    if group_id not in para.keywords:
                        para.keywords.append(group_id)
                        para.save()

                gids = list(set(gids) - set(named))
                if gids:
                    for para in Paragraph.query(F.keywords.in_(gids)):
                        for id0 in gids:
                            if id0 in para.keywords:
                                para.keywords.remove(id0)
                        if group_id not in para.keywords:
                            para.keywords.append(group_id)
                        para.save()
            return group_id
