"""
Database for Digital Arts Backend
"""
import os
import requests
import markdown
import datetime
from PyMongoWrapper import QExprInterpreter

from jindai.models import Paragraph, MediaItem
from jindai.plugin import Plugin
from jindai.helpers import APICrudEndpoint, APIResults, APIUpdate
from jindai.task import Task

from plugins.datasources.htmldatasource import WebPageListingDataSource, ExtractHTMLParagraphs
from plugins.pipelines.basics import AccumulateParagraphs, LanguageDetect, WordCut, FieldAssignment, FilterStopWords


class Dada(Paragraph):

    tags = list

    def save(self):
        self.tags = list(set(self.tags))
        return super().save()

    def as_dict(self, expand: bool = False) -> dict:
        result = super().as_dict(expand)
        if isinstance(self.pdate, datetime.datetime):
            result['pdate'] = self.pdate.strftime('%Y-%m-%d')
        return result


parser = QExprInterpreter('tags')


class DadaEndpoints(APICrudEndpoint):

    def __init__(self, llm_endpoint=''):
        super().__init__('/api/', Dada)
        self.namespace = '/api/dada/'
        self.llm_endpoint = llm_endpoint
        self.bind_endpoint(self.llm)
        self.bind_endpoint(self.fetch)
        self.bind_endpoint(self.keywords)

    def fetch(self, objs, url, depth=1, assignments=None, selector='body', scopes='', **params):
        if url:
            wlds = WebPageListingDataSource(base_cls=Dada, dataset='dada', scopes=scopes,
                                            content=url, mongocollection='dada', list_depth=int(depth), **params)
            results = Task({'content': url}, [
                wlds,
                ExtractHTMLParagraphs(
                    paragraph_selector=selector, assignments=assignments),
                LanguageDetect(),
                FieldAssignment('tags', '[]'),
                WordCut(True),
                FilterStopWords(),
                AccumulateParagraphs()
            ], resume_next=True).execute()
        else:
            results = []
        return APIResults(results)

    def llm(self, objs, messages=None):
        # TEST
        enabled = False
        response = f'''测试文本测试文本---你提交了{len(messages)}条消息，其中第一条的文本长度为{len(messages[0]['content'])}。---\n'''

        if enabled:
            if messages:
                resp = requests.post(self.llm_endpoint, json={'thread': messages}, headers={
                                     'Content-Type': 'application/json', 'Agent': 'jindai-mt/1.0'})
                try:
                    resp = resp.json()
                except requests.JSONDecodeError:
                    raise ValueError(resp.content)
                assert resp and resp['success'], f'Failed with response: {resp.content}'
                response = resp['choices'][0]['message']['content']
            else:
                response = ''

        return APIResults([Dada(content=response, lang='auto', dataset='dada')])

    def keywords(self, objs):
        wc = WordCut(for_search=True, field='keywords')
        for obj in objs:
            wc.resolve(obj)
            obj.save()
        return APIUpdate()

    def update_object(self, obj, data):
        if 'images' in data:
            images = data['images']
            images = [MediaItem.get(i['src']) if 'src' in i and '_id' not in i else i.get(
                '_id') for i in images]
            data['images'] = images
        return super().update_object(obj, data)


class DadaBackendPlugin(Plugin):

    def __init__(self, pm, llm_endpoint, **conf):
        super().__init__(pm, **conf)
        DadaEndpoints(llm_endpoint).bind(pm.app, login=False)

        @pm.app.route('/api/dada/doc')
        def dada_doc():
            doc_file = os.path.join(os.path.dirname(__file__), 'dada.md')
            with open(doc_file, 'r', encoding='utf-8') as fi:
                return markdown.markdown(fi.read(), extensions=['tables'])
