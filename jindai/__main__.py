"""CLI for jindai"""

import glob
import json
import os
import re
import subprocess
import sys
import zipfile
import tempfile
from typing import Dict, Iterable

import click
import dateutil.parser
import numpy as np
import urllib3
import yaml
from flask import Flask
from tqdm import tqdm

from . import Plugin, PluginManager, Task, config, storage
from .api import prepare_plugins, run_service
from .helpers import get_context, safe_import
from .models import Dataset, Paragraph, TaskDBO, UserInfo, db_session

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def _init_plugins(*paths):
    """Inititalize plugins"""
    return PluginManager(get_context("plugins", Plugin, *paths), Flask(__name__))


@click.group()
def cli():
    """Cli group"""


@cli.command("init")
def first_run():
    u = db_session.query(UserInfo).first()
    if not u:
        u = UserInfo(username="admin", roles=["admin"])
        db_session.add(u)
        db_session.commit()
        print("Created user: admin; dataset: Default")


@cli.command("export")
@click.option("--output")
@click.option("--query")
def export(query, output_file):
    """Export query results to json"""
    _init_plugins()
    task_obj = Task(
        stages=[
            ("DBQueryDataSource", {"query": query}),
            ("AccumulateParagraphs", {}),
            ("Export", {"output_format": "xlsx", "inp": "return"}),
        ],
        params={"query": query},
    )

    xlsx = task_obj.execute()

    with open(output_file, "wb") as output_file:
        output_file.write(xlsx)


@cli.command("worker")
def run_worker():
    from .worker import worker
    worker()


@cli.command("task")
@click.argument("task_id")
@click.option("-l", "--log", type=str, default="")
@click.option("-n", "--concurrent", type=int, default=0)
@click.option("-v", "--verbose", type=bool, flag_value=True)
@click.option("-e", "--edit", type=bool, flag_value=True)
@click.option("-o", "--output", type=click.File("w", "utf-8", "ignore"), default=None)
def run_task(task_id, concurrent, verbose, edit, log, output):
    """Run task according to id or name"""
    dbo = db_session.query(TaskDBO).filter((TaskDBO.id == task_id) | (TaskDBO.name == task_id)).first()
    
    if not dbo:
        print(f"Task {task_id} not found")
        return

    if edit:
        temp_name = tempfile.mkstemp()
        with open(temp_name, "w", encoding="utf-8") as fo:
            dat = dbo.as_dict()
            dat.pop("_id", "")
            dat.pop("last_run", "")
            yaml.safe_dump(dat, fo, allow_unicode=True)

        if os.name == "nt":
            editor = "notepad.exe"
        elif os.system("which nano") == 0:
            editor = "nano"
        else:
            editor = "vi"

        subprocess.Popen([editor, temp_name]).communicate()

        with open(temp_name, encoding="utf-8") as fi:
            param = yaml.safe_load(fi)
            for key, val in param.items():
                dbo[key] = val
            dbo.save()

        os.unlink(temp_name)

    _init_plugins()

    logfile = open(log, "w", encoding="utf-8") if log else sys.stderr

    task = Task.from_dbo(dbo, verbose=verbose, log=lambda *x: print(*x, file=logfile))

    if concurrent > 0:
        task.concurrent = concurrent

    result = task.execute()

    print()
    print(result)

    if output:
        output.write(json.dumps(result, ensure_ascii=False, indent=2))
        output.close()

    if log:
        print(result, file=logfile)
        logfile.close()


@cli.command("user")
@click.option("--add", "-a", default="")
@click.option("--setrole", "-g", default="")
@click.option("--delete", "-d", default="")
@click.argument("roles", nargs=-1)
def user_manage(add, delete, setrole, roles):
    """User management"""
    if add:
        print("Password: ", end="")
        password = input()
        if not User.first(F.username == add):
            user = User(username=add)
            user.set_password(password)
            user.save()
        else:
            print("User already exists.")
    elif delete:
        User.query(F.username == delete).delete()
    elif setrole:
        user = User.first(F.username == setrole)
        if not user:
            print("User", setrole, "does not exist.")
            exit()
        user.roles = roles
        user.save()


@cli.command("plugin-install")
@click.argument("url")
def plugin_install(url: str):
    """Install plugin

    :param url: install from
    :type url: str
    """
    pmanager = _init_plugins()
    pmanager.install(url)


@cli.command("plugin-export")
@click.option("--output", "-o")
@click.argument("infiles", nargs=-1)
def plugin_export(output: str, infiles):
    """Export plugin

    :param output: output file name
    :type output: str
    :param infiles: includes path
    :type infiles: path
    """

    def _all_files(path):
        if os.path.isfile(path):
            yield path
        else:
            for base, _, files in os.walk(path):
                if base == "__pycache__":
                    continue
                for f in files:
                    yield os.path.join(base, f)

    def _export_one(outputzip, filelist):
        if not outputzip.startswith("jindai.plugins."):
            outputzip = f"jindai.plugins.{outputzip}"
        if outputzip.endswith(".zip"):
            outputzip = outputzip[:-4]

        print("output to", outputzip)
        with zipfile.ZipFile(outputzip + ".zip", "w", zipfile.ZIP_DEFLATED) as zout:
            for filepath in filelist:
                for filename in _all_files(filepath):
                    print(" ...", filename)
                    zout.write(filename, filename)

    if len(infiles) > 0:
        _export_one(output, infiles)
    else:
        for p in glob.glob("plugins/*"):
            pname = os.path.basename(p)
            if (
                pname.startswith(("_", "temp_"))
                or ("." in p and not p.endswith(".py"))
                or ("." not in p and os.path.isfile(p))
            ):
                continue
            if pname in (
                "datasources",
                "hashing",
                "imageproc",
                "pipelines",
                "shortcuts",
                "taskqueue.py",
                "onedrive.py",
                "scheduler.py",
                "autotagging.py",
            ):
                continue

            _export_one(os.path.basename(p).split(".")[0], [p])


@cli.command("web-service")
@click.option("--port", default=8370, type=int)
@click.option("--deployment", "-D", default=False, flag_value=True)
def web_service(port: int, deployment: bool):
    """Run web service on port

    :param port: port number
    :type port: int
    """
    from .api import app

    prepare_plugins()
    if deployment:
        safe_import("waitress")
        from waitress import serve

        serve(app, host="0.0.0.0", port=port, threads=8)
    else:
        run_service(port=port)


@cli.command("ipython")
def call_ipython():
    from IPython import start_ipython

    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

    import glob
    import sys
    from concurrent.futures import ThreadPoolExecutor
    from uuid import UUID

    from tqdm import tqdm

    import jindai

    tpe = ThreadPoolExecutor(os.cpu_count())
    init = _init_plugins
    from .app import app
    app.app_context().push()

    def q(query_str, model=""):
        from jindai.dbquery import DBQuery

        return DBQuery("? " + query_str, model, raw=True).fetch()

    def run(task_name):
        dbo = TaskDBO.first(
            (F.id if re.match("^[0-9a-fA-F]{24}$", task_name) else F.name) == task_name
        )
        if dbo:
            task = Task.from_dbo(dbo)
            return task.execute()

    ns = dict(jindai.__dict__)
    ns.update(**locals())

    start_ipython(argv=[], user_ns=ns)


if __name__ == "__main__":
    print("* loaded config from", config._filename)
    print("* using", config.database)
    cli()
