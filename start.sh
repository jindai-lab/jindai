#!/bin/sh
# supervisord -n -c /app/supervisor/supervisord.conf
gunicorn -w 1 --threads 4 --chdir /app -b 0.0.0.0:8370 --reload --timeout 120 api:app