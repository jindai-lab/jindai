"""
Database for Digital Arts Backend
"""
import os
from bson import ObjectId
import requests
import markdown
import datetime
from PyMongoWrapper import QExprInterpreter, F, Fn

from jindai.models import Paragraph, Meta
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
        self.bind_endpoint(self.prompts)
        self.bind_endpoint(self.keywords)
        self.bind_endpoint(self.counts)

    def build_query(self, id, ids, query, data):
        if query:
            query = parser.parse(query)
        return super().build_query(id, ids, query, data)

    def fetch(self, objs, url, depth=1, assignments=None, selector='body', scopes='', with_chrome=False, **params):
        if url:
            wlds = WebPageListingDataSource(base_cls=Dada, dataset='dada', scopes=scopes, with_chrome=with_chrome,
                                            content=url, mongocollection='dada', list_depth=int(depth), **params)
            results = Task({'content': url}, [
                wlds,
                ExtractHTMLParagraphs(
                    paragraph_selector=selector, assignments=assignments,
                    autoextract=True),
                LanguageDetect(),
                FieldAssignment('tags', '[]'),
                WordCut(True),
                FilterStopWords(),
                AccumulateParagraphs()
            ], resume_next=False).execute()
        else:
            results = []
        return APIResults(results)

    def llm(self, objs, messages=None):

        if messages:
            resp = requests.post(self.llm_endpoint, json={'thread': messages}, headers={
                'Content-Type': 'application/json', 'Agent': 'jindai-mt/1.0'})
            try:
                resp = resp.json()
                assert resp and resp['success'], f'Failed with response: {resp.content}'
                response = resp['choices'][0]['message']['content']
                return APIResults([Dada(content=response, lang='auto', dataset='dada')])
            except requests.JSONDecodeError:
                raise ValueError(resp.content)
        else:
            response = ''

    def prompts(self, objs, action='', prompt=''):
        prompts_obj = Meta.first(F.prompts.exists(1)) or Meta(prompts=[])
        if action == 'create' and prompt:
            prompts_obj.prompts = list(prompts_obj.prompts)
            prompts_obj.prompts.append(prompt)
            prompts_obj.save()
        elif action == 'delete' and prompt:
            if prompt in prompts_obj.prompts:
                prompts_obj.prompts = list(prompts_obj.prompts)
                prompts_obj.prompts.remove(prompt)
                prompts_obj.save()
        return APIResults(prompts_obj.prompts)

    def counts(self, objs, field):
        return APIResults(Dada.aggregator.project(
            {field: 1}
        ).unwind('$' + field).group(
            id='$' + field, count=Fn.sum(1)
        ).sort({'count': -1}).limit(200).perform(raw=True))

    def keywords(self, objs):
        wc = WordCut(for_search=True, field='keywords')
        for obj in objs:
            wc.resolve(obj)
            obj.save()
        return APIUpdate()


class DadaBackendPlugin(Plugin):

    def __init__(self, pm, llm_endpoint, **conf):
        super().__init__(pm, **conf)
        DadaEndpoints(llm_endpoint).bind(pm.app, login=False)

        @pm.app.route('/api/dada/doc')
        def dada_doc():
            doc_file = os.path.join(os.path.dirname(__file__), 'dada.md')
            with open(doc_file, 'r', encoding='utf-8') as fi:
                return markdown.markdown(fi.read(), extensions=['tables'])
