from models import Paragraph
from datasource import DataSource
import glob
import fitz
import re


class MinjianGushiJicheng(DataSource):

    def fetch(self):
        d, field, results = {'title': ''}, 'title', []
        for filename in glob.glob('sources/中国民间故事集成/*.pdf'):
            pdf = fitz.open(filename)
            for p in pdf:
                for l in p.get_text().split('\n'):
                    l = re.sub(r'\s', '', l)
                    if '....' in l: continue
                    if l.startswith('讲述'):
                        field = 'narrator'
                    elif l.startswith('采录'):
                        field = 'collector'
                    elif l.startswith('附记'):
                        field = 'note'
                    elif re.match(r'^\d{3}[^\d]+$', l):
                        if 'content' in d and d['title']:
                            yield Paragraph(lang='chs', **d)    
                        d = {'source': {'page': p.number, 'file': filename}, 'title': l}
                        d['title'] = l
                        field = 'content'
                        continue
                    if field in d:
                        d[field] += l
                    else:
                        d[field] = l
