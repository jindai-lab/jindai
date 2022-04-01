"""自动标签"""
import re

from jindai import PipelineStage, Plugin
from jindai.helpers import rest
from jindai.models import F, db, ObjectId


class AutoTag(db.DbObject):
    """Auto Tagging Object"""

    from_tag = str
    pattern = str
    tag = str


class ApplyAutoTags(PipelineStage):
    """应用自动标签设置
    """

    def __init__(self) -> None:
        super().__init__()
        self.ats = list(AutoTag.query({}))

    def resolve(self, paragraph):
        for i in self.ats:
            pattern, from_tag, tag = i.pattern, i.from_tag, i.tag
            if (from_tag and from_tag in paragraph.keywords) or \
            (pattern and re.search(pattern, paragraph.source['url'])):
                if tag not in paragraph.keywords:
                    paragraph.keywords.append(tag)
                if tag.startswith('@'):
                    paragraph.author = tag
        paragraph.save()
        return paragraph


class AutoTaggingPlugin(Plugin):
    """自动标签"""

    def __init__(self, app) -> None:
        super().__init__(app)
        self.register_pipelines(globals())

        @app.route('/api/plugins/autotags', methods=['POST', 'PUT', 'GET'])
        @rest()
        def autotags_list(tag='', from_tag='', pattern='', delete=False, ids=None):
            if ids is None:
                ids = []
            if tag:
                AutoTag(tag=tag, from_tag=from_tag, pattern=pattern).save()
            elif delete:
                AutoTag.query(F.id.in_([ObjectId(i) for i in ids])).delete()
            else:
                return list(AutoTag.query({}).sort('-_id'))
            return True
