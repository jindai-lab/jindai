from jindai.pipeline import DataSourceStage
from jindai.helpers import safe_import, rest, APIResults, storage
from jindai.plugin import Plugin
from jindai.models import Paragraph, F
from plugins.pipelines.basics import WordCut
from PyMongoWrapper.dbo import BatchSave

from flask import request, render_template_string, Response
from urllib.parse import urljoin
import click
import re
from bs4 import BeautifulSoup as B
from tqdm import tqdm


MDictEntry = Paragraph.get_coll('mdict')
MDictEntry.ensure_index('entry')
safe_import('mdict_utils')
wc = WordCut(True)


@click.command()
@click.argument('lang')
@click.argument('input_files', nargs=-1)
def import_mdict(lang, input_files):
    import mdict_utils.reader
    for input_file in input_files:
        mdx = mdict_utils.reader.MDX(input_file)
        with BatchSave(performer=MDictEntry) as batch:
            for key, val in tqdm(mdx.items()):
                key, val = map(lambda x: x.decode(
                    'utf-8', errors='ignore'), [key, val])
                val = f'<h1>{key}</h1><p>{val}</p>'
                plain_text = B(val, 'lxml').text
                pa = MDictEntry(keywords=[key], entry=key, content=plain_text, html=val,
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
        self.word = content
        self.query = F.keywords.regex(content)
        if source:
            self.query &= F.source.file.regex(source)

    def fetch(self):
        for rs in [MDictEntry.query(self.query & (F.entry == self.word)), MDictEntry.query(self.query)]:
            for p in rs.limit(10):
                p.html = re.sub(r'@@@LINK=(\w+)',
                                '<a href="entry://\\1">\\1</a>', p.html)
                b = B(p.html)
                for a in b.select('a[href]'):
                    href = a['href']
                    if a['href'].startswith('entry://'):
                        entry = a['href'].split('://')[1]
                        a['href'] = f'/api/plugins/mdict?entry={entry}&source={p.source.file}'
                for attr in ('src', 'href'):
                    for ele in b.select(f'[{attr}]'):
                        ele[attr] = '/images/file/' + urljoin(p.source.file, ele[attr]).lstrip('/')
                p.content = str(b)
                yield p


class MDictPlugin(Plugin):
    """Plugin for machin translations
    """

    def __init__(self, pmanager, **config):
        super().__init__(pmanager, **config)
        self.register_pipelines(globals())

        safe_import('flask_bootstrap', 'bootstrap-flask')
        safe_import('flask_wtf')
        safe_import('wtforms')

        from flask_bootstrap import Bootstrap5
        from wtforms.fields import SearchField, SelectField, SubmitField
        from flask_wtf import FlaskForm

        class SearchEntryForm(FlaskForm):
            entry = SearchField('Entry')
            source = SelectField('Source', choices=[''])
            submit = SubmitField('Search')

        bootstrap = Bootstrap5(pmanager.app)

        @pmanager.app.route('/api/plugins/mdict/', methods=['GET'])
        @rest()
        def mdict_lookup(entry, source='', **_):
            ds = MDictDataSource()
            ds.apply_params(entry, source)
            return Response(render_template_string('''
{% from 'bootstrap5/form.html' import render_form %}
<html>
<head>
{{ bootstrap.load_css() }}
<title>{{ entry }} - MDict</title>
</head>
<body>
<script src="https://cdn.bootcdn.net/ajax/libs/jquery/3.7.1/jquery.js"></script>
                                                   
{{ render_form(form, action="./", method="GET") }}

{% for entry in entries %}
<div>
    {{ entry.content|safe }}
</div>
{% endfor %}
<script>
const selected_source = "{{ selected_source }}"
$.get('sources').then(data => {
    const sources = $('select')[0];
    for (let source of data.results) {
    var opt = new Option(); 
    opt.value = source; opt.text = source.split('/').pop();
    if (opt.value == selected_source) opt.selected = true;
    sources.appendChild(opt);                  
    } 
})
</script>
<style>body { max-width: 1200px; margin: 0 auto; }</style>
{{ bootstrap.load_js() }}
</body>
</html>''',
                bootstrap=bootstrap,
                entry=entry,
                selected_source=source,
                form=SearchEntryForm(entry=entry, source=source),
                entries=ds.fetch()))

        @pmanager.app.route('/api/plugins/mdict/sources', methods=['GET'])
        @rest(cache=True)
        def mdict_sources():
            return APIResults([
                _['_id']
                for _ in MDictEntry.aggregator.project(
                    F.source.file == 1).group(_id='$source.file').perform(raw=True)])


if __name__ == '__main__':
    import_mdict()
