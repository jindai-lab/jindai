from jindai.pipeline import DataSourceStage
from jindai.helpers import safe_import
from jindai.plugin import Plugin
from jindai.models import Paragraph, F
from plugins.pipelines.basics import WordCut
from PyMongoWrapper.dbo import BatchSave

import click
from bs4 import BeautifulSoup as B
from tqdm import tqdm


MDictEntry = Paragraph.get_coll('mdict')
safe_import('mdict_utils')
wc = WordCut(True)


@click.command()
@click.argument('lang')
@click.argument('input_file')
def import_mdict(lang, input_file):
    import mdict_utils.reader
    mdx = mdict_utils.reader.MDX(input_file)
    with BatchSave(performer=MDictEntry) as batch:
        for key, val in tqdm(mdx.items()):
            val = f'<h1>{key}</h1><p>{val}</p>'
            plain_text = B(val, 'lxml').text
            pa = MDictEntry(keywords=[key], content=plain_text, html=val, 
                            source={'file': input_file}, lang=lang, dataset='mdict')
            wc.resolve(pa)
            batch.add(pa)
            
            
class MDictDataSource(DataSourceStage):

    def apply_params(self, content, source=''):
        '''
        Args:
            content (str): Word to look up
            source (str): Dictionary to look up in
        '''
        self.query = F.keywords.regex(content)
        if source:
            self.query &= F.source.file.regex(source)
        
    def fetch(self):
        return MDictEntry.query(self.query)


class MDictPlugin(Plugin):
    """Plugin for machin translations
    """

    def __init__(self, pmanager, **config):
        super().__init__(pmanager, **config)
        self.register_pipelines(globals())


if __name__ == '__main__':
    import_mdict()
