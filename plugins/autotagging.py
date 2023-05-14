"""
Auto tagging
@zhs 自动标签
"""
from PyMongoWrapper import F, Fn, ObjectId, QueryExpressionError, MongoOperand
from jindai import PipelineStage, Plugin, parser
from jindai.helpers import rest, execute_query_expr
from jindai.models import db, Paragraph


class AutoTag(db.DbObject):
    """Auto Tagging Object"""

    cond = str
    tag = str

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._parsed = {}
        self._parsed_cond = ''

    @property
    def parsed(self):
        """Get parsed condition

        :return: parsed condition
        :rtype: dict
        """
        if self._parsed_cond != self.cond:
            self._parsed_cond = self.cond
            try:
                self._parsed = parser.parse(self.cond)
            except QueryExpressionError:
                self._parsed = {'__FALSE__': True}
        return self._parsed
    
    
def apply_tag(parsed, tag, paragraph):
    modified = False
    try:
        if execute_query_expr(parsed, paragraph):
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
                flag = flag or apply_tag(parsed, tag, paragraph)
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
            cond (str, optional):
                Condition
                @zhs 条件
            tag (str, optional): 
                Tag
                @zhs 标签
        """        
        super().__init__()
        self.cond = cond
        self.parsed = parser.parse(cond)
        self.tag = tag
        if not AutoTag.first(F.cond == cond, F.tag == tag):
            AutoTag(cond=cond, tag=tag).save()


class AutoTaggingPlugin(Plugin):
    """Auto tagging plugin"""

    def __init__(self, pmanager, **_) -> None:
        super().__init__(pmanager)
        self.register_pipelines(globals())

        app = self.pmanager.app

        @app.route('/api/plugins/autotags', methods=['POST', 'PUT', 'GET'])
        @rest()
        def autotags_list(tag='', cond='', delete=False, ids=None, apply=None):
            if ids is None:
                ids = []
            if tag:
                AutoTag(tag=tag, cond=cond).save()
            elif delete:
                AutoTag.query(F.id.in_([ObjectId(i) for i in ids])).delete()
            elif apply:
                a = AutoTag.first(F.id == apply)
                if not a:
                    return '', 404
                cond = a.parsed
                if a.tag.startswith('~'):
                    Paragraph.query(cond).update(Fn.pull(keywords=a.tag[1:]))
                else:
                    if a.tag.startswith('#'):
                        cond = MongoOperand(cond) & (~F.keywords.regex(r'^#'))
                    Paragraph.query(cond).update(Fn.addToSet(keywords=a.tag))
            else:
                return list(AutoTag.query({}).sort('-_id'))
            return True
