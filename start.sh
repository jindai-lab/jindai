#!/bin/bash
rm -rf /tmp/tmp*
while [ 0 ]; do
    # gunicorn -w 1 --threads 16 --chdir /app -b 0.0.0.0:8370 --reload --timeout 600 jindai.api:app
    cd /app; python3 -m jindai web-service
    sleep 10
done
