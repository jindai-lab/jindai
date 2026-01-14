# Jindai
# jindai - 近现代中外文语料检索与管理平台
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
### 前端
- 框架：React 18+
- 路由：React Router v6+
- UI 组件库：Ant Design
- 网络请求：Axios
- 构建工具：**Vite**（高效打包构建，提升开发与生产环境性能）
- 状态管理：原生 React 状态管理方案（未使用第三方状态管理库）

### 后端
- 框架：Flask + Flask RESTful
- 跨域：Flask-CORS
- 数据库 ORM：SQLAlchemy
- 身份认证：OpenID Connect（基于 Authentik 实现）
- 虚拟环境管理：**uv**（快速、简洁的 Python 包管理器与虚拟环境工具）
- 异步任务队列：Celery + Redis
- Embedding 模型：**baai/bge-m3**（多语言通用向量模型，适配跨语种检索需求）
- 文件解析：python-docx（docx）、PyPDF2（pdf）、pandas（csv）

### 数据存储
- 关系型数据库：PostgreSQL 14+（存储用户信息、数据集元数据、语料索引等）
- 缓存/任务队列：Redis 6+（缓存热点语料、存储 Celery 任务队列）
- 文件存储：**本地挂载目录**（存储上传的语料文件，支持磁盘挂载扩容）

## 环境要求
### 基础环境
- Python 3.8+
- uv（Python 虚拟环境与包管理工具）
- Node.js 16+
- npm / yarn
- PostgreSQL 14+
- Redis 6+
- Authentik 部署完成（用于身份认证）
- 本地挂载磁盘（用于文件存储，需提前规划挂载路径）

### 依赖库（后端核心依赖）
```txt
flask==2.3.3
flask-restful==0.3.10
flask-cors==4.0.0
flask-sqlalchemy==3.1.1
psycopg2-binary==2.9.9
redis==5.0.1
celery==5.3.6
sentence-transformers==2.7.0
python-docx==1.1.0
pypdf2==3.0.1
pandas==2.1.4
numpy==1.26.3
python-openidc==3.0.0
```

### 依赖库（前端核心依赖）
```txt
react==18.2.0
react-dom==18.2.0
react-router-dom==6.22.0
antd==5.14.0
axios==1.6.7
vite==5.0.10
```

## 配置与部署步骤
### 1. 前置准备
#### 1.1 基础软件安装
- 安装 PostgreSQL、Redis、Node.js，参考对应官方文档完成部署
- 安装 **uv**：参考 [uv 官方文档](https://github.com/astral-sh/uv) 完成安装（如 `curl -LsSf https://astral.sh/uv/install.sh | sh`）
- 完成 **Authentik 部署**，并创建对应的 OpenID 应用，记录以下信息：
  - `AUTHENTIK_ISSUER_URL`（Authentik 服务地址）
  - `CLIENT_ID`（应用客户端 ID）
  - `CLIENT_SECRET`（应用客户端密钥）
- 配置**本地磁盘挂载**：将用于存储语料文件的磁盘挂载到服务器指定路径（如 `/data/jindai/uploads`），确保该路径有读写权限

#### 1.2 数据库初始化
1. 启动 PostgreSQL 服务，创建项目数据库及用户：
   ```sql
   CREATE DATABASE jindai_db;
   CREATE USER jindai_user WITH PASSWORD 'your_secure_password';
   GRANT ALL PRIVILEGES ON DATABASE jindai_db TO jindai_user;
   ALTER ROLE jindai_user SET client_encoding TO 'utf8';
   ALTER ROLE jindai_user SET default_transaction_isolation TO 'read committed';
   ALTER ROLE jindai_user SET timezone TO 'UTC';
   ```
2. 测试数据库连接，确保 `jindai_user` 可正常访问 `jindai_db`

### 2. 后端部署（基于 uv 虚拟环境）
#### 2.1 拉取代码并进入目录
```bash
git clone <项目仓库地址>
cd jindai/backend
```

#### 2.2 基于 uv 创建并激活虚拟环境
```bash
# 创建虚拟环境
uv venv

# 激活虚拟环境
# Windows
uv venv\Scripts\activate
# Mac/Linux
source uv venv/bin/activate
```

#### 2.3 安装后端依赖
```bash
uv sync
```

