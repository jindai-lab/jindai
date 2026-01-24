"""API Web Service"""

import os
import sys
from collections import defaultdict

from flask import Response, send_file, jsonify

from .app import app, config, api
from .helpers import get_context
from .pipeline import Pipeline
from .plugin import Plugin, PluginManager
from .resources import apply_resources


apply_resources(api)


@app.route("/api/v2/pipelines")
def help_info():
    """Provide help info for pipelines, with respect to preferred language"""

    ctx = Pipeline.ctx
    result = defaultdict(dict)
    for key, val in ctx.items():
        name = (
            sys.modules[val.__module__].__doc__ or val.__module__.split(".")[-1]
            if hasattr(val, "__module__")
            else key
        ).strip()
        if key in ("DataSourceStage", "MediaItemStage"):
            continue
        result[name][key] = val.get_spec()
    return jsonify(result)


@app.route("/<path:path>", methods=["GET"])
@app.route("/", methods=["GET"])
def index(path="index.html"):
    """Serve static files"""

    if path.startswith("api/"):
        return Response("", 404)
    path = path or "index.html"
    for file in [path, path + ".html", (config.ui_dist or './dist/') + path]:
        if os.path.exists(file) and os.path.isfile(file):
            return send_file(file)

    return "Not found for " + path, 404


def prepare_plugins():
    """Prepare plugins"""
    if os.path.exists("restarting"):
        os.unlink("restarting")
    plugin_ctx = get_context("plugins", Plugin)
    return PluginManager(plugin_ctx, app)


@app.route("/api/v2/plugins", methods=["GET"])
def get_plugins():
    """Get plugin names"""
    return jsonify([type(pl).__name__ for pl in plugins])


plugins = prepare_plugins()


def run_service(host="0.0.0.0", port=8370):
    """Run API web service. Must run `prepare_plugins` first.

    :param host: Host, defaults to '0.0.0.0'
    :type host: str, optional
    :param port: Port, defaults to None
    :type port: int, optional
    """
    if port is None:
        port = config.port
    app.run(debug=True, host=host, port=int(port), threaded=True)
