"""Book Searcher"""

import requests
import json
from urllib.parse import quote

from jindai.plugin import Plugin
from jindai.pipeline import DataSourceStage
from jindai.models import Paragraph


class BookSearcherDataSource(DataSourceStage):
    
    """Book Searcher Data Source"""
    
    def apply_params(self, server, ipfs, title='', author='', 
                     publisher='', extension='', language='', isbn='',
                     sort=''):
        """
        Args:
            server (str): Server URL
            ipfs (str): IPFS gateway
            title (str): Book title
            author (str): Author
            publisher (str): Publisher
            extension (|pdf|epub|azw3|mobi): Extension name, e.g. pdf
            language (Chinese|English|Other:): Language name, e.g. English
            isbn (str): ISBN
            sort (Default:|Year:year|Title:title|Author:author): Sort by
        """
        query = ''
        if isbn:
            isbn = isbn.replace('-', '').strip()

        for pname, val in zip(('', 'author', 'publisher', 'extension', 'language', 'isbn'),
                              (title, author, publisher, extension, language, isbn)):
            if val:
                if query:
                    query += ' '
                if pname:
                    query += pname + ':'
                query += val

        self.query = query

        self.server = server
        self.ipfs = ipfs
        self.sort = sort
        
    def fetch(self):
        self.logger(self.server, self.query)
        resp = requests.get(self.server + 'search?limit=1000&query=' + quote(self.query))
        books = json.loads(resp.content)['books']
        for j in sorted(books, key=lambda x: id(x) if not self.sort else x[self.sort]):
            yield Paragraph(content=f'''
                            Title: {j["title"]}<br/>
                            Author: {j["author"]}<br/>
                            Format: {j["extension"]}<br/>
                            Year: {j["year"]}<br/>
                            Publisher: {j["publisher"]}<br/>
                            Size: {j["filesize"]/1024/1024:.2f} MB<br/>
                            <a target="_blank"
                               href="{self.ipfs}/ipfs/{j["ipfs_cid"]}?filename={j["title"]}.{j["extension"]}">
                               Download</a>
                            ''',
                            pdate=j['year'])


class ZBookSearcherPlugin(Plugin):

    def __init__(self, pm, **config) -> None:
        super().__init__(pm, **config)
        self.register_pipelines(globals())
        