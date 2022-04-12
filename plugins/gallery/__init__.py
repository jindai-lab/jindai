"""相册和图像相关"""

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

    def __init__(self, pmanager):
        super().__init__(pmanager)
        self.register_pipelines(globals())
