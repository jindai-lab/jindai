"""
Auto tagging
@chs 自动标签
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

    def parse(self):
        """Get parsed condition

        :return: parsed condition
        :rtype: dict
        """
        if not self._parsed:
            try:
                self._parsed = parser.parse(self.cond)
            except QueryExpressionError:
                self._parsed = 'False'
        return self._parsed


class ApplyAutoTags(PipelineStage):
    """
    Apply auto tagging settings
    @chs 应用自动标签设置
    """

    def __init__(self) -> None:
        super().__init__()
        self.ats = list(AutoTag.query({}))

    def resolve(self, paragraph):
        for i in self.ats:
            parsed, tag = i.parse(), i.tag
            try:
                if execute_query_expr(parsed, paragraph):
                    if tag not in paragraph.keywords:
                        paragraph.keywords.append(tag)
                    if tag.startswith('@'):
                        paragraph.author = tag
            except TypeError:
                pass
        paragraph.save()
        return paragraph


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
                cond = a.parse()
                if a.tag.startswith('*'):
                    cond = MongoOperand(cond) & (~F.keywords.regex(r'^\*'))
                Paragraph.query(cond).update(Fn.addToSet(keywords=a.tag))
            else:
                return list(AutoTag.query({}).sort('-_id'))
            return True
