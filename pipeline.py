from collections import defaultdict
from typing import Dict, Iterable, List, Tuple, Any
from PyMongoWrapper import MongoResultSet, F
from numpy.lib.arraysetops import isin
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import threading
from models import Paragraph, get_context
import importlib
import os
import glob


class PipelineStage:
    """
    流程各阶段。返回应同样为一个段落记录（可以不与数据库中的一致）。
    注意，针对段落的处理可能是并发进行的。
    """

    def resolve(self, p : Paragraph) -> Paragraph:
        return p

    def summarize(self, r) -> Any:
        pass


class Pipeline:
    
    def __init__(self, stages : List[PipelineStage], concurrent, resume_next):
        self.stages = stages
        self.concurrent = concurrent
        self.resume_next = resume_next

    def apply(self, p : Paragraph):
        if not self.stages:
            return p
        
        for stage in self.stages:
            try:
                p = stage.resolve(p)
                if not p: return
            except Exception as ex:
                if not self.resume_next:
                    raise ex

        return p

    def applyParagraphs(self, rs : MongoResultSet, total=0):

        if not self.stages:
            return rs

        if self.concurrent > 1:
            def _update_and_do(p):
                p = self.apply(p)
                return p

            with ThreadPoolExecutor(max_workers=self.concurrent) as te:
                r = te.map(_update_and_do, rs)
                return [_ for _ in r if _]

        else:
            def _seq():
                for p in rs:
                    p = self.apply(p)
                    if p: yield p
            
            return _seq()

    def summarize(self):
        returned = None
        for stage in self.stages:
            returned = stage.summarize(returned)

        return returned
