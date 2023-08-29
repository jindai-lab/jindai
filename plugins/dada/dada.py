"""
Database for Digital Arts Backend
"""
import os
import requests
import markdown
from PyMongoWrapper import QExprInterpreter

from jindai.models import Paragraph
from jindai.plugin import Plugin
from jindai.helpers import APICrudEndpoint, APIResults, APIUpdate
from jindai.task import Task

from plugins.datasources.htmldatasource import WebPageListingDataSource, ExtractHTMLParagraphs
from plugins.pipelines.basics import AccumulateParagraphs, LanguageDetect, WordCut, FieldAssignment


class Dada(Paragraph):
    pass


parser = QExprInterpreter('tags')


class DadaEndpoints(APICrudEndpoint):

    def __init__(self, llm_endpoint=''):
        super().__init__('/api/', Dada)
        self.namespace = '/api/dada/'
        self.llm_endpoint = llm_endpoint
        self.bind_endpoint(self.llm)
        self.bind_endpoint(self.fetch)
        self.bind_endpoint(self.keywords)

    def fetch(self, objs, url, depth=1, assignments=None, scopes='', **params):
        if url:
            wlds = WebPageListingDataSource()
            wlds.apply_params('dada', url, scopes, mongocollection='dada', list_depth=depth, **params)
            results = Task({'content': url}, [
                wlds,
                LanguageDetect(),
                ExtractHTMLParagraphs(assignments=assignments),
                FieldAssignment('tags', '$keywords'),
                WordCut(True),
                AccumulateParagraphs()
            ], resume_next=True).execute()
        else:
            results = []
        return APIResults(results)

    def llm(self, objs, messages=None):
        if messages:
            resp = requests.post(self.llm_endpoint, json={'thread': messages}, headers={'Content-Type': 'application/json', 'Agent': 'jindai-mt/1.0'})
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


class DadaBackendPlugin(Plugin):

    def __init__(self, pm, llm_endpoint, **conf):
        super().__init__(pm, **conf)
        DadaEndpoints(llm_endpoint).bind(pm.app, login=False)

        @pm.app.route('/api/dada/doc')
        def dada_doc():
            doc_file = os.path.join(os.path.dirname(__file__), 'dada.md')
            with open(doc_file, 'r', encoding='utf-8') as fi:
                return markdown.markdown(fi.read(), extensions=['tables'])
