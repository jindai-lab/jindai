"""Book Searcher"""

from typing import Iterable
import requests
import json
import sqlite3
import os
from urllib.parse import quote
from bs4 import BeautifulSoup as B
from urllib.parse import urljoin

from jindai.models import Paragraph
from jindai.plugin import Plugin
from jindai.pipeline import DataSourceStage
from jindai.helpers import safe_import
from jindai.models import Paragraph


class BookSearcherDataSource(DataSourceStage):
    
    """Book Searcher Data Source"""
    
    def apply_params(self, server, ipfs, content='', title='', author='', 
                     publisher='', extension='', language='', isbn='',
                     sort=''):
        """
        Args:
            server (str): Server URL
            ipfs (str): IPFS gateway
            content (str): Query expression
            title (str): Book title
            author (str): Author
            publisher (str): Publisher
            extension (|pdf|epub|azw3|mobi): Extension name, e.g. pdf
            language (Chinese|English|German|French|Spanish|Japanese|Other:): Language name, e.g. English
            isbn (str): ISBN
            sort (Default:|Year:year|Title:title|Author:author): Sort by
        """
        query = content
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
        resp = requests.get(self.server + '/search?limit=1000&query=' + quote(self.query))
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
                            pdate=j['year'],
                            title=j['title'],
                            author=j['author'])
            

class ArchiveOrgSearcherDataSource(DataSourceStage):
    
    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.ia = safe_import('internetarchive')
        
    def apply_params(self, content: str = ''):
        """
        Args:
            content (str): Query string
                @zhs 查询字符串
        """
        self.query = content
        
    def fetch(self) -> Iterable[Paragraph]:
        s = self.ia.get_session()
        for _, i in zip(range(10), s.search_items(self.query)):
            ident = i.get('identifier')
            if ident:
                item = s.get_item(ident)
                yield Paragraph(identifier=item.identifier, files=item.files, 
                                pdate=item.metadata.get('date'),
                                author=item.metadata.get('creator'),
                                content=f'''
                                Title: {item.metadata['title']}<br />
                                {item.metadata.get('description', '')}<br /><br />
                                ''' + '<br />'.join([
                                    f'<a target="blank" href="https://archive.org/download/{item.identifier}/{filedata["name"]}">{filedata["name"]}</a>'
                                    for filedata in item.files
                                ]))
                

class VufindDataSource(DataSourceStage):
    
    def apply_params(self, server: str, content: str = ''):
        """
        Args:
            content (str): Query string
                @zhs 查询字符串
            server (str): Server
        """
        self.query = content
        self.server = server.rstrip('/')
        
    def fetch(self):
        
        headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7,de-DE;q=0.6,de;q=0.5',
            'Connection': 'keep-alive',
            'Referer': self.server + '/',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'same-origin',
            'Sec-Fetch-User': '?1',
            'Upgrade-Insecure-Requests': '1',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
            'sec-ch-ua': '"Not/A)Brand";v="99", "Google Chrome";v="115", "Chromium";v="115"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
        }

        params = {
            'searchtype': 'vague',
            'lookfor': self.query,
            'type': 'AllFields',
            'limit': '20',
        }

        try:
            resp = requests.get(self.server + '/Search/Results', params=params, headers=headers).content
            b = B(resp, 'lxml')
            for res in b.select('.media'):
                for a in res.select('a'):
                    href = a.attrs.get('href')
                    if href:
                        a.attrs['href'] = urljoin(self.server, href)
                yield Paragraph(title=res.select_one('.title').text.strip(),
                                author=res.select_one('.author-data').text.strip(),
                                content=str(res.select_one('.result-body')))
        except Exception as ex:
            self.log_exception(f'Error while fetching from {self.server}', ex)
            
            
class LibgenDataSource(DataSourceStage):
    
    db_file = os.path.join(os.path.dirname(__file__), 'libgen.db')
    
    @staticmethod
    def create_connection():
        def dict_factory(cursor, row):
            d = {}
            for idx, col in enumerate(cursor.description):
                d[col[0].lower()] = row[idx]
            return d
        
        conn = sqlite3.connect('file:' + LibgenDataSource.db_file + '?mode=ro')
        conn.row_factory = dict_factory
        return conn
    
    def apply_params(self, content: str = ''):
        """
        Args:
            content (str): Query string
                @zhs 查询
        """
        self.query = content.strip()
    
    def fetch(self) -> Iterable[Paragraph]:
        if not self.query:
            return []
        
        with self.create_connection() as conn:
            cursor = conn.cursor()
            conds, args = [], []
            
            for word in self.query.split():
                word = f"%{word}%"
                conds.append('(title like ? or authors like ?)')
                args.extend([word, word])
                        
            sql = f'select * from non_fiction where {" and ".join(conds)} limit 1000'
                        
            for row in cursor.execute(sql, args):
                row['link'] = f'<a href="http://library.lol/main/{row["md5hash"]}" target="_blank">Download</a>'
                p = Paragraph().fill_dict(row)
                p.content = '<br/>'.join((f'{key}: {val}' for key, val in row.items() if val))
                yield p


class BookSearcherPlugin(Plugin):

    def __init__(self, pm, libgen_db='', **config) -> None:
        super().__init__(pm, **config)
        if libgen_db:
            LibgenDataSource.db_file = libgen_db
        self.register_pipelines(globals())
        