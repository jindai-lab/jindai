import click
from task import Task


rules = [
    r'^([I\|](.{,3}年)+[I\|])?鲁迅.{,8}第[〇一二三四五六七八九十]+卷', 
    r'^[\w\s]+[I丨\|]\s{,2}[〇一二三四五六七八九]+年\s{,2}[I丨\|]\s{,2}',
    r'^I',
    r'[〇oO0\s]+',
    r'沈..全集◎',
    r'^...全集第.{,2}卷[一二三四五六七八九〇O0]*',
    r'..全集（.?卷[一二三四五六七八九十]*）\d+$',
    r'\d+..全集（.?卷[一二三四五六七八九十]*）$',
    r'[^\w]{,3}\d+[^\w]{,3}$'
]


@click.group()
def cli():
    pass


@cli.command('pmi')
@click.option('--output')
@click.option('--query')
def pmi(output, query):

    task = Task(datasource=('DBQueryDataSource', {'query': query}), pipeline=[
        ('FilterPunctuations', {}),
        ('TranToSimpChinese', {}),
        ('PMILREntropyWordFetcher', {'word_length': 4}),
        ('Export', {'format': 'csv', 'inp': 'return'})
    ])

    csv = task.execute()

    with open(output, 'w', encoding='utf-8') as fo:
        fo.write(csv)


if __name__ == '__main__':
    cli()
