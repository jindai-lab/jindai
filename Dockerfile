FROM python:3.11-bookworm
WORKDIR /app
ADD . /app
EXPOSE 8370

RUN apt-get update && apt-get install -yqq python3 python3-pip && \
    pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple && pip3 install gunicorn && \
    pip3 install -r /app/requirements.txt

CMD ["bash", "start.sh"]
