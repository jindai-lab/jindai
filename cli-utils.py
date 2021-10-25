import click
from PyMongoWrapper import F
from models import User
from task import Task
import requests


@click.group()
def cli():
    pass


@cli.command('export')
@click.option('--output')
@click.option('--query')
def export(query, output):

    task = Task(datasource=('DBQueryDataSource', {'query': query}), pipeline=[
        ('AccumulateParagraphs', {}),
        ('Export', {'format': 'xlsx', 'inp': 'return'})
    ])

    xlsx = task.execute()

    with open(output, 'wb') as fo:
        fo.write(xlsx)


# @cli.command('task_test')
def task():
    j = requests.put('http://localhost:8370/api/tasks/', json={
        'name': 'test russian task',
        'datasource': 'DBQueryDataSource',
        'datasource_config': {'query': '', 'mongocollection': 'slg', 'limit': 100},
        'pipeline': [
            ('AccumulateParagraphs', {}),
            ('Export', {'format': 'xlsx', 'inp': 'return'})
        ]
    })
    print(j.json)


@cli.command('enqueue')
@click.option('--id')
def task_enqueue(id):
    j = requests.put('http://localhost:8370/api/queue/', json={'id': id}, headers={'Accept-ContentType': 'text/plain'})
    if j.status_code == 200:
        print(j.json())
    else:
        print(j.content)


@cli.command('user')
@click.option('--add', default='')
@click.option('--setrole', default='')
@click.option('--delete', default='')
@click.argument('roles', nargs=-1)
def user(add, delete, setrole, roles):
    if add:
        print('Password: ', end='')
        password = input()
        if not User.first(F.username == add):
            u = User(username=add)
            u.set_password(password)
            u.save()
        else:
            print('User already exists.')
    elif delete:
        User.query(F.username == delete).delete()
    elif setrole:
        u = User.first(F.username == setrole)
        u.roles = roles
        u.save()


if __name__ == '__main__':
    cli()
    