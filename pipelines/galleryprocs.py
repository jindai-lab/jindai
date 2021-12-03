"""图集相关工具
"""
from pipeline import PipelineStage
from models import Album, AutoTag
import re


class ApplyAutoTags(PipelineStage):
    """应用自动标签设置
    """
    
    def __init__(self) -> None:
        self.ats = list(AutoTag.query({}))
    
    def resolve(self, p : Album) -> Album:
        for i in self.ats:
            pattern, from_tag, tag = i.pattern, i.from_tag, i.tag
            if (from_tag and from_tag in p.tags) or (pattern and re.search(pattern, p.source['url'])):
                if tag not in p.tags:
                    p.tags.append(tag)
                if tag.startswith('@'):
                    p.author = tag
        p.save()
        return p

