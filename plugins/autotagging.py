"""
Auto tagging
@zhs 自动标签
"""
from PyMongoWrapper.dbo import DbObject
from PyMongoWrapper import F, Fn, ObjectId, QExprError, MongoOperand
from flask import request
from jindai import PipelineStage, Plugin, parser
from jindai.helpers import rest, evaluateqx, APICrudEndpoint, APIUpdate
from jindai.models import db, Paragraph
import json


class AutoTag(db.DbObject):
    """Auto Tagging Object"""

    cond = str
    tag = str

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.parsed_state = '{"cond": "", "parsed": {}}'
        self._parsed = {}
        self._parsed_cond = ''

    @property
    def parsed(self):
        """Get parsed condition

        :return: parsed condition
        :rtype: dict
        """

        if self.parsed_state:
            state = json.loads(self.parsed_state)
            self._parsed_cond = state['cond']
            self._parsed = state['parsed']

        if self._parsed_cond != self.cond:
            self._parsed_cond = self.cond
            try:
                self._parsed = parser.parse(self.cond)
            except QExprError:
                self._parsed = {'__FALSE__': True}
                print('Error while parsing', self.cond)
            
            self.parsed_state = json.dumps({'cond': self._parsed_cond, 'parsed': self._parsed})
            self.save()

        return self._parsed
    
    
def apply_tag(parsed, tag, paragraph):
    modified = False
    try:
        if evaluateqx(parsed, paragraph):
            if tag.startswith('~') and tag[1:] in paragraph.keywords:
                paragraph.keywords.remove(tag[1:])
                modified = True
            else:
                if tag.startswith('@') and paragraph.author != tag:
                    paragraph.author = tag
                    modified = True
                if tag not in paragraph.keywords:
                    paragraph.keywords.append(tag)
                    modified = True
    except TypeError:
        pass
    return modified
    

class ApplyAutoTags(PipelineStage):
    """
    Apply auto tagging settings
    @zhs 应用自动标签设置
    """

    def __init__(self) -> None:
        super().__init__()
        self.ats = [(a.parsed, a.tag) for a in AutoTag.query({})]

    def resolve(self, paragraph):
        for _ in range(10):
            flag = False
            for parsed, tag in self.ats:
                matched = apply_tag(parsed, tag, paragraph)
                if matched:
                    self.log(paragraph.id, 'matches', parsed, ', tagging', tag)
                flag = flag or matched
            if not flag:
                break
        paragraph.save()
        return paragraph
    
    
class AddAutoTag(PipelineStage):
    """
    Create new auto tagging rule
    @zhs 创建新的自动标签规则
    """
    
    def __init__(self, cond='false', tag='') -> None:
        """
        Args:
            cond (QUERY, optional):
                Condition, n.b. should return a string
                @zhs 条件表达式，注意应返回字符串
            tag (QUERY, optional): 
                Tag
                @zhs 标签
        """        
        super().__init__()
        self.cond = parser.parse(cond)
        self.tag = parser.parse(tag)

    def resolve(self, paragraph):
        cond = evaluateqx(self.cond, paragraph)
        tag = evaluateqx(self.tag, paragraph)
        if not AutoTag.first(F.cond == cond, F.tag == tag):
            AutoTag(cond, tag=tag).save()


class AutoTaggingEndpoint(APICrudEndpoint):

    def __init__(self) -> None:
        super().__init__('/api/plugins/', AutoTag, {'apply': ['POST']})

    def apply(self, objs, coll='', **_):
        cond = objs.parsed
        if objs.tag.startswith('~'):
            Paragraph.get_coll(coll).query(cond).update(Fn.pull(keywords=objs.tag[1:]))
        else:
            if objs.tag.startswith('#'):
                cond = MongoOperand(cond) & (~F.keywords.regex(r'^#'))
            Paragraph.get_coll(coll).query(cond).update(Fn.addToSet(keywords=objs.tag))
        return APIUpdate()


class AutoTaggingPlugin(Plugin):
    """Auto tagging plugin"""

    def __init__(self, pmanager, **_) -> None:
        super().__init__(pmanager)
        self.register_pipelines(globals())

        app = self.pmanager.app
        AutoTaggingEndpoint().bind(app)
