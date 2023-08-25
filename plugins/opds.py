"""Read Calibre database"""

import urllib
import dateutil.parser
from flask import request, Response, send_file
import jieba
from jindai import Plugin, storage
from jindai.pipeline import DataSourceStage
from jindai.models import Paragraph
from jindai.helpers import safe_import, rest

from lxml import etree
import requests


class OPDSDataSource(DataSourceStage):
    """Search from OPDS catalog"""
    
    def apply_params(self, entrypoint, content='') -> None:
        """
        Args:
            entrypoint (str): OPDS entrypoint URL
            content (str): Query
        """
        self.entrypoint = entrypoint
        self.search = self.get_search()
        self.query = ' '.join(set(jieba.cut(content.lower())))
        
    def get_search(self):
        xr = etree.fromstring(requests.get(self.entrypoint).content).getroottree()
        return urllib.parse.urljoin(self.entrypoint, xr.find('//{*}link[@rel="search"][@type="application/atom+xml"]').attrib['href'])
        
    def fetch(self):
        url = self.search.format(searchTerms=urllib.parse.quote(self.query))
        xr = etree.fromstring(requests.get(url).content).getroottree()
        for entry in xr.findall('//{*}entry'):
            info = {}
            content = '<h3>{title}</h3><p>Author: {author}</p><a href="{href}">Download ({filesize} MB)<a>'
            for child in entry.iterchildren():
                tag = child.tag.split('}')[-1]
                attrib = child.attrib
                if tag == 'title':
                    info['title'] = child.text
                elif tag == 'updated':
                    info['pdate'] = dateutil.parser.isoparse(child.text)
                elif tag == 'author':
                    info['author'] = ' '.join([c.text for c in child.iterchildren()])
                elif tag == 'link':
                    if attrib['rel'].startswith("http://opds-spec.org/image"):
                        info['images'] = [{'source': {'url': urllib.parse.urljoin(self.entrypoint, attrib['href'])}}]
                    elif attrib['rel'] == 'http://opds-spec.org/acquisition':
                        info['href'] = attrib['href']
                        info['filesize'] = round(float(attrib['length']) / 1024. / 1024., 2)
                else:
                    info[tag] = child.text
            content = content.format(**info)
            yield Paragraph(content=content, **info)


class OPDSPlugin(Plugin):
    """Auto tagging plugin"""

    def __init__(self, pmanager, **_) -> None:
        super().__init__(pmanager)
        self.register_pipelines(globals())
        
        LIST_PAGE = '/api/plugins/calibre/list'
        
        @pmanager.app.route(LIST_PAGE)
        @rest()
        def calibre_list():
            path = request.args.get('path', '')
            assert path, 'Path should not be empty'
            stat = storage.stat(path)['type']
            if stat == 'file':
                return send_file(storage.open(path), 'application/' + path.rsplit('.', 1)[-1], download_name=path.rsplit('/', 1)[-1])
            else:
                files = storage.statdir(path)
                content = '<html><ul>'
                for file in files:
                    content += '<li><a href="{list_page}?path={path}">{file}</a></li>'.format(file=file['name'], path=urllib.parse.quote(file['fullpath']), list_page=LIST_PAGE)
                content += '</ul></html>'
                
                return Response(content, mimetype='text/html; encoding=utf-8')
