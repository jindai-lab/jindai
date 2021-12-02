#!/bin/sh
# supervisord -n -c /app/supervisor/supervisord.conf
which python3 || (sed -i s@security.ubuntu@mirrors.aliyun@ /etc/apt/sources.list && \
    sed -i s@archive.ubuntu@mirrors.aliyun@ /etc/apt/sources.list && apt-get update && apt-get install -yqq python3 python3-pip && \
    pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple && pip3 install gunicorn && pip3 install -r /app/requirements.txt)
while [ 0 ]; do
    #gunicorn -w 1 --chdir /app -b 0.0.0.0:8370 -k geventwebsocket.gunicorn.workers.GeventWebSocketWorker --reload --timeout 120 api:app
    cd /app
    python3 api.py
    sleep 10
done