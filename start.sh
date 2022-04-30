#!/bin/bash
rm /tmp/tmp*
which python3 || (sed -i s@security.ubuntu@mirrors.aliyun@ /etc/apt/sources.list && \
    sed -i s@archive.ubuntu@mirrors.aliyun@ /etc/apt/sources.list && apt-get update && apt-get install -yqq python3 python3-pip && \
    pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple && pip3 install gunicorn && pip3 install -r /app/requirements.txt)
while [ 0 ]; do
    # gunicorn -w 1 --threads 16 --chdir /app -b 0.0.0.0:8370 --reload --timeout 600 jindai.api:app
    cd /app; python3 -m jindai web-service
    sleep 10
done