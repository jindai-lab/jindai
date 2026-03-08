# Jindai - 近现代中外文语料检索与管理平台

## 项目简介

`jindai` 是一款专注于近现代中外文语料的检索与管理平台，核心提供**跨语种语义检索**能力，依托 `baai/bge-m3` 模型实现 embedding 向量计算，打破语言壁垒，实现中文、外文语料的精准语义匹配；同时支持语料数据集管理、文件上传/解析/存储等配套功能，集成基于 OpenID 协议的 Authentik 身份认证，采用本地挂载方式存储文件，为近现代文献研究、语料分析等场景提供安全高效的工具支持。

## 核心功能

1.  **跨语种语义检索**：基于 `baai/bge-m3` 模型实现 embedding 向量计算与语义相似度检索，支持模糊匹配、精准筛选，突破传统关键词检索局限
2.  **数据集管理**：支持语料数据集的创建、编辑、删除、批量导入/导出
3.  **文件管理**：支持多种格式（txt、docx、pdf、csv 等）语料文件的上传、解析、预览、下载，采用本地挂载方式存储文件
4.  **身份认证**：基于 OpenID 协议集成 Authentik 认证，保障平台访问安全与用户身份管理
5.  **检索结果处理**：检索结果支持排序（相关性、时间戳）、筛选（语种、来源、时间范围）、导出（Excel/CSV）
6.  **异步任务处理**：批量语料 embedding 生成、大文件解析等耗时操作异步执行，提升系统响应速度

## 技术栈

### 后端

- **框架**：FastAPI + Flask（双框架架构，FastAPI 处理异步 API，Flask 处理传统 Web 服务）
- **数据库 ORM**：SQLAlchemy (async) + asyncpg + pgvector（PostgreSQL 向量扩展）
- **任务队列**：Celery + Redis + taskiq-redis（多任务队列支持）
- **身份认证**：Flask-oidc / Authlib（OpenID Connect 协议实现）
- **虚拟环境管理**：uv（快速、简洁的 Python 包管理器与虚拟环境工具）
- **Embedding 模型**：sentence-transformers (baai/bge-m3) + torch（多语言通用向量模型）
- **文件解析**：pymupdf (pdf)、python-docx (docx)、pandas (csv)、ocrmypdf (OCR)
- **NLP 处理**：hanlp、nltk、spacy、jieba3、lingua-language-detector（多语言文本处理）
- **Web 服务器**：uvicorn (FastAPI) + waitress (Flask)

### 数据存储

- **关系型数据库**：PostgreSQL 14+（存储用户信息、数据集元数据、语料索引等）
- **向量存储**：pgvector（PostgreSQL 向量扩展，支持语义检索）
- **缓存/任务队列**：Redis 6+（缓存热点数据、存储 Celery/Taskiq 任务队列）
- **文件存储**：本地挂载目录（存储上传的语料文件，支持磁盘挂载扩容）

### 前端

- **框架**：React 19.2.0
- **路由**：React Router DOM 7.12.0
- **UI 组件库**：Ant Design 6.1.4
- **网络请求**：Axios 1.13.2
- **构建工具**：Vite 7.2.4（高效打包构建，支持 CSS 压缩优化）
- **状态管理**：原生 React 状态管理方案 + Context API
- **国际化**：i18next + react-i18next
- **身份认证**：oidc-client-ts（OpenID Connect 客户端实现）

## 安装方式（Docker 部署）

### 前置准备

1.  **安装 Docker 和 Docker Compose**：参考 [Docker 官方文档](https://docs.docker.com/get-docker/) 完成安装
2.  **准备配置文件**：复制 `config.yaml.sample` 为 `config.yaml`，并根据实际情况修改配置项
3.  **准备数据库**：确保 PostgreSQL 14+ 和 Redis 6+ 服务可访问
4.  **准备文件存储目录**：创建用于存储上传文件的本地目录（如 `/data/jindai/uploads`）

### 部署步骤

#### 1. 构建镜像

```bash
cd jindai
docker build -t jindai:latest .
```

#### 2. 运行容器

```bash
docker run -d \
  --name jindai \
  -p 8370:8370 \
  -v /path/to/config.yaml:/app/config.yaml \
  -v /path/to/uploads:/app/uploads \
  -e TZ=Asia/Shanghai \
  jindai:latest
```

#### 3. 访问服务

服务启动后，访问 `http://localhost:8370` 即可使用平台。

## 配置说明

### config.yaml 配置文件

项目使用 YAML 格式的配置文件 [`config.yaml`](config.yaml.sample) 进行配置管理。部署前需复制 `config.yaml.sample` 为 `config.yaml` 并根据实际情况修改配置项。

#### 主要配置项

| 配置项 | 说明 | 示例 |
|--------|------|------|
| `secret_key` | 应用加密密钥 | `random-secret-key` |
| `concurrent` | 并发处理线程数 | `3` |
| `storage` | 文件存储路径 | `/storage` |
| `database` | PostgreSQL 数据库连接字符串 | `postgresql+psycopg2://user:pass@host:5432/db` |
| `plugins` | 启用的插件列表 | `['*']`（启用所有插件） |
| `oidc_secrets` | OIDC 认证密钥文件路径 | `oidc-secrets.json` |
| `redis.host` | Redis 主机地址 | `localhost` |
| `redis.port` | Redis 端口号 | `6379` |
| `redis.db` | Redis 数据库编号 | `0` |
| `embedding_model` | Embedding 模型名称 | `BAAI/bge-m3` |
| `embdding_dims` | Embedding 向量维度 | `1024` |
| `ui_dist` | 前端构建产物路径 | `./dist/` |
| `paddle_remote` | PaddleOCR 远程服务地址 | `http://paddle-ocr:8080/` |

#### 完整配置示例

```yaml
secret_key: 'your-secure-secret-key'
concurrent: 3
storage: /app/uploads
database: postgresql+psycopg2://jindai_user:your_password@db:5432/jindai_db

plugins:
  - '*'

oidc_secrets: 'oidc-secrets.json'

redis:
  host: redis
  port: 6379
  db: 0

embedding_model: 'BAAI/bge-m3'
embdding_dims: 1024

ui_dist: ./dist/
paddle_remote: http://paddle-ocr:8080/
```

#### Docker 部署配置

在 Docker 部署时，需将配置文件挂载到容器内：

```bash
docker run -d \
  --name jindai \
  -p 8370:8370 \
  -v /path/to/config.yaml:/app/config.yaml \
  -v /path/to/uploads:/app/uploads \
  -v /path/to/oidc-secrets.json:/app/oidc-secrets.json \
  -e TZ=Asia/Shanghai \
  jindai:latest
```

### oidc-secrets.json 认证配置

项目使用 OpenID Connect (OIDC) 协议进行身份认证，需配置 [`oidc-secrets.json`](oidc-secrets.json.sample) 文件。该文件包含与 Authentik 身份认证服务的连接信息。

#### 配置文件格式

```json
{
  "web": {
    "auth_uri": "https://your-authentik-host/application/o/jindai/",
    "client_id": "your-client-id",
    "client_secret": "your-client-secret",
    "token_uri": "https://your-authentik-host/application/o/token/",
    "userinfo_uri": "https://your-authentik-host/application/o/userinfo/",
    "issuer": "https://your-authentik-host/application/o/jindai/",
    "redirect_uris": [
      "http://localhost:8370/login/authorized/"
    ],
    "javascript_origins": [
      "http://localhost:8370"
    ]
  }
}
```

#### 配置项说明

| 配置项 | 说明 | 来源 |
|--------|------|------|
| `auth_uri` | Authentik 授权端点 URL | Authentik 应用配置 |
| `client_id` | OAuth2 客户端 ID | Authentik 应用配置 |
| `client_secret` | OAuth2 客户端密钥 | Authentik 应用配置 |
| `token_uri` | Authentik Token 端点 URL | Authentik 应用配置 |
| `userinfo_uri` | Authentik 用户信息端点 URL | Authentik 应用配置 |
| `issuer` | Authentik 发行者 URL | Authentik 应用配置 |
| `redirect_uris` | 登录回调 URL | 需与应用配置一致 |
| `javascript_origins` | 允许的前端域名 | 需与前端域名一致 |

#### Authentik 配置步骤

1.  **创建 OAuth2 应用**：在 Authentik 管理后台创建新的 OAuth2 应用
2.  **配置回调 URL**：设置 `Redirect URIs` 为 `http://your-host/login/authorized/`
3.  **生成密钥**：记录生成的 `Client ID` 和 `Client Secret`
4.  **创建 oidc-secrets.json**：将上述信息填入配置文件
5.  **挂载到容器**：将 `oidc-secrets.json` 文件挂载到容器内（路径需与 `config.yaml` 中的 `oidc_secrets` 配置一致）

#### 完整部署示例

```bash
docker run -d \
  --name jindai \
  -p 8370:8370 \
  -v /path/to/config.yaml:/app/config.yaml \
  -v /path/to/uploads:/app/uploads \
  -v /path/to/oidc-secrets.json:/app/oidc-secrets.json \
  -e TZ=Asia/Shanghai \
  jindai:latest
```

### PaddleOCR 远程服务配置

项目使用 [`ocrmypdf-paddleocr-remote`](https://github.com/zhuth/ocrmypdf-paddleocr-remote) 插件实现 PDF 文件的 OCR 处理，通过远程 PaddleOCR 服务提供 OCR 能力。

#### 服务说明

`ocrmypdf-paddleocr-remote` 是 OCRmyPDF 的插件，允许使用远程 PaddleOCR 服务作为 OCR 引擎。该服务基于 PaddleX 的 PaddleOCR 模型，支持自动语言检测，无需指定语言参数。

#### Docker 部署 PaddleOCR 服务

使用以下 Docker 命令启动 PaddleOCR 服务：

```bash
docker run -d \
  --name paddle-ocr \
  --gpus=all \
  -p 8080:8080 \
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/paddlex:3.3.4-paddlepaddle3.2.0-gpu-cuda11.8-cudnn8.9-trt8.6
```

服务启动后，PaddleOCR 服务将在 `http://localhost:8080` 提供 OCR 服务。

#### 配置说明

在 `config.yaml` 中配置 `paddle_remote` 项，指向 PaddleOCR 服务地址：

```yaml
paddle_remote: http://paddle-ocr:8080/
```

如果 PaddleOCR 服务与 jindai 服务在同一 Docker 网络中，使用服务名 `paddle-ocr`；如果在不同网络，使用主机 IP 地址。

#### OCR 功能说明

- **自动语言检测**：PaddleOCR 可自动识别 PDF 中的文本语言（中文、英文等）
- **文本定位**：提供精确的文本位置信息（bounding boxes）
- **hOCR 输出**：生成符合 OCRmyPDF 标准的 hOCR 格式
- **优化的词级定位**：插件优化了词级边界框计算，消除行尾间隙，提高文本选择精度

#### 完整部署示例（含 PaddleOCR）

```bash
# 启动 PaddleOCR 服务
docker run -d \
  --name paddle-ocr \
  --gpus=all \
  -p 8080:8080 \
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/paddlex:3.3.4-paddlepaddle3.2.0-gpu-cuda11.8-cudnn8.9-trt8.6

# 启动 jindai 服务
docker run -d \
  --name jindai \
  --network host \
  -p 8370:8370 \
  -v /path/to/config.yaml:/app/config.yaml \
  -v /path/to/uploads:/app/uploads \
  -v /path/to/oidc-secrets.json:/app/oidc-secrets.json \
  -e TZ=Asia/Shanghai \
  jindai:latest
```

## 开发部署（非 Docker）

### 前置要求

- Python 3.13+
- uv（Python 包管理器）
- PostgreSQL 14+
- Redis 6+

### 后端部署

```bash
# 克隆代码
git clone <项目仓库地址>
cd jindai

# 创建虚拟环境
uv venv

# 激活虚拟环境
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 安装依赖
uv sync

# 运行服务
uv run -m jindai web-service
```

### 前端部署

```bash
cd jindai-ui

# 安装依赖
npm install
# 或 pnpm install

# 运行开发服务器
npm run dev
# 或 pnpm dev

# 构建生产版本
npm run build
# 或 pnpm build
```

## 依赖库（核心依赖）

### 后端核心依赖

```txt
fastapi>=0.128.0
flask>=3.1.2
sqlalchemy[asyncio]>=2.0.45
asyncpg>=0.31.0
pgvector>=0.4.2
redis>=7.1.0
celery>=5.6.2
taskiq-redis>=1.2.2
sentence-transformers>=5.2.0
torch>=2.9.1
hanlp>=2.1.3
pymupdf>=1.26.7
python-docx>=1.1.0
pandas>=2.3.3
flask-oidc>=2.4.0
authlib>=1.6.6
uvicorn[standard]>=0.40.0
waitress>=3.0.2
```

### 前端核心依赖

```txt
react>=19.2.0
react-dom>=19.2.0
react-router-dom>=7.12.0
antd>=6.1.4
axios>=1.13.2
vite>=7.2.4
i18next>=25.8.14
oidc-client-ts>=3.4.1
```

## 许可证

本项目采用 [MIT 许可证](LICENSE)。
