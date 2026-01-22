FROM python:3.13

WORKDIR /app

ENV LANG=C.UTF-8
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update && apt install -yqq nano tesseract-ocr ghostscript

# 国内源加速
ENV PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ENV PIP_TRUSTED_HOST=mirrors.aliyun.com
RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./

RUN uv pip install --system --no-cache --disable-pip-version-check pyproject.toml

COPY . .

EXPOSE 8370

# 启动命令（uv标准运行方式），按需替换
CMD ["uv", "run", "-m", "jindai", "web-service"]
