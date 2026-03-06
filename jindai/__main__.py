"""Command-line interface for Jindai application.

This module provides CLI commands for:
- Initializing the application
- Exporting query results
- Running tasks
- User management
- Starting the web service
- Launching an IPython shell with application context
"""

import glob
import json
import os
import subprocess
import sys
import tempfile
import zipfile

import click
import asyncio
import regex as re
from sqlalchemy import select
import urllib3
import yaml
import logging

from . import Plugin, PluginManager, Task, config, storage
from .app import app, run_service
from .helpers import get_context, safe_import
from .models import (Dataset, Paragraph, QueryFilters, TaskDBO, UserInfo, get_db_session)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def _init_plugins(*paths) -> "PluginManager":
    """Initialize the plugin manager with plugins from specified paths.
    
    Args:
        *paths: Variable length path arguments to plugin directories.
        
    Returns:
        PluginManager: Initialized plugin manager instance.
    """
    return PluginManager(get_context("plugins", Plugin, *paths))


def asyncio_run(func):
    """Decorator to run async functions synchronously.
    
    Args:
        func: Async function to wrap.
        
    Returns:
        Wrapped function that runs the async function in an event loop.
    """
    def wrapped(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))
    return wrapped


@click.group()
def cli() -> None:
    """CLI group for Jindai application commands."""


@cli.command("init")
@asyncio_run
async def first_run() -> None:
    """Initialize the application by creating the default admin user.
    
    Checks if a user exists and creates one with username 'admin' and
    admin role if no users are found.
    """
    async with get_db_session() as session:
        u = (await session.execute(select(UserInfo).limit(1))).first()
        if u:
            print("Already inited.")
        else:
            u = UserInfo(username="admin", roles=["admin"])
            session.add(u)
            await session.flush()
            print("Created user: admin; dataset: Default")


@cli.command("export")
@click.option("--output")
@click.option("--query")
def export(query: str, output_file: str) -> None:
    """Export query results to an Excel file.
    
    Args:
        query: Database query string to filter records.
        output_file: Path to the output Excel file.
    """
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


@cli.command("task")
@click.argument("task_id")
@click.option("-l", "--log", type=str, default="")
@click.option("-n", "--concurrent", type=int, default=0)
@click.option("-v", "--verbose", type=bool, flag_value=True)
@click.option("-e", "--edit", type=bool, flag_value=True)
@click.option("-o", "--output", type=click.File("w", "utf-8", "ignore"), default=None)
@asyncio_run
async def run_task(
    task_id: str,
    concurrent: int,
    verbose: bool,
    edit: bool,
    log: str,
    output: click.File
) -> None:
    """Run a task by ID or name.
    
    Args:
        task_id: Task ID or name to execute.
        concurrent: Number of concurrent workers (0 uses default).
        verbose: Enable verbose logging if True.
        edit: Edit task parameters before running if True.
        log: Path to log file (defaults to stderr).
        output: Output file for JSON result.
    """
    dbo = await TaskDBO.get(task_id)
    
    if not dbo:
        print(f"Task {task_id} not found")
        return

    if edit:
        _, temp_name = tempfile.mkstemp()
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
            async with get_db_session() as session:
                for key, val in param.items():
                    setattr(dbo, key, val)
                await session.merge(dbo)
            
        os.unlink(temp_name)

    _init_plugins()

    logfile = open(log, "w", encoding="utf-8") if log else sys.stderr

    task = Task.from_dbo(dbo, verbose=verbose, log=lambda *x: print(*x, file=logfile))

    if concurrent > 0:
        task.concurrent = concurrent

    result = await task.execute_async()

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
@asyncio_run
async def user_manage(
    add: str,
    delete: str,
    setrole: str,
    roles: tuple
) -> None:
    """Manage users - add, delete, or set roles.
    
    Args:
        add: Username to add.
        delete: Username to delete.
        setrole: Username to modify roles for.
        roles: List of roles to add to the user.
    """
    async with get_db_session() as session:
        if add:
            user = (await session.execute(select(UserInfo).filter(UserInfo.username == add))).first()
            if not user:
                user = UserInfo(username=add)
                session.add(user)
                await session.flush()
            else:
                print("User already exists.")
        elif delete:
            user = (await session.execute(select(UserInfo).filter(UserInfo.username == delete))).first()
            if user:
                await session.delete(user)
        elif setrole:
            user = (await session.execute(select(UserInfo).filter(UserInfo.username == setrole))).first()
            if user:
                user.roles.extend(roles)
                await session.flush()


@cli.command("web-service")
@click.option("--port", default=8370, type=int)
def web_service(port: int) -> None:
    """Run the web service on the specified port.
    
    Args:
        port: Port number to run the web service on.
    """
    run_service(port=port)


@cli.command("ipython")
def call_ipython() -> None:
    """Launch an IPython shell with Jindai application context.
    
    Sets up an interactive Python shell with all Jindai modules and
    components pre-imported for easy experimentation and debugging.
    """
    from IPython import start_ipython

    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

    import glob
    import sys
    from concurrent.futures import ThreadPoolExecutor
    from uuid import UUID

    from tqdm import tqdm

    import jindai
    import jindai.models
    import jindai.resources
    import palitra

    tpe = ThreadPoolExecutor(os.cpu_count())
    init = _init_plugins
    from jindai import app
    def q(query_str):
        return Paragraph.build_query(QueryFilters(q=query_str))

    async def run(task_name, **kwargs):
        dbo = await TaskDBO.get(task_name)
        if dbo:
            task = Task.from_dbo(dbo, log=print, **kwargs)
            return await task.execute_async()

    ns = dict(jindai.__dict__)
    ns.update(**locals())
    for obj in [jindai, jindai.models, jindai.resources]:
        for k in dir(obj):
            if k.startswith('_'): continue
            ns[k] = getattr(obj, k)

    start_ipython(argv=[], user_ns=ns)


if __name__ == "__main__":
    logging.debug("* db connection:", re.sub(r'://.+?@', '://***@', config.database), 'redis:', config.redis)
    cli()
