import re
from jindai.helpers import rest
from bson import ObjectId

from jindai.models import db, F
from jindai import Plugin
from jindai import  PipelineStage


class AutoTag(db.DbObject):
    """Auto Tagging Object"""
    
    from_tag = str
    pattern = str
    tag = str


class ApplyAutoTags(PipelineStage):
    """应用自动标签设置
    """

    def __init__(self) -> None:
        self.ats = list(AutoTag.query({}))

    def resolve(self, p):
        for i in self.ats:
            pattern, from_tag, tag = i.pattern, i.from_tag, i.tag
            if (from_tag and from_tag in p.keywords) or (pattern and re.search(pattern, p.source['url'])):
                if tag not in p.keywords:
                    p.keywords.append(tag)
                if tag.startswith('@'):
                    p.author = tag
        p.save()
        return p


class AutoTaggingPlugin(Plugin):

    def __init__(self, app) -> None:
        super().__init__(app)
        self.register_pipelines(globals())

        @app.route('/api/plugins/autotags', methods=['POST', 'PUT', 'GET'])
        @rest()
        def autotags_list(tag='', from_tag='', pattern='', delete=False, ids=[]):
            if tag:
                AutoTag(tag=tag, from_tag=from_tag, pattern=pattern).save()
            elif delete:
                AutoTag.query(F.id.in_([ObjectId(i) for i in ids])).delete()
            else:
                return list(AutoTag.query({}).sort('-_id'))
