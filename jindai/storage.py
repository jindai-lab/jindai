from io import BytesIO
import os
import shutil
import mimetypes
from werkzeug.utils import secure_filename
from flask import Response, send_file, abort, request
from flask_restful import Resource, reqparse
from datetime import datetime

from .app import ResponseTuple, api, config, assert_admin
from .models import Paragraph, db_session


# 文件存储根目录（必须配置，确保有读写权限）
FILE_STORAGE_ROOT: str = config.storage
# 允许上传的文件大小上限（MB）
MAX_FILE_SIZE = 1024
# 允许上传的文件类型（空列表表示不限）
ALLOWED_EXTENSIONS = [
    "txt",
    "pdf",
    "doc",
    "docx",
    "jpg",
    "png",
    "json",
    "csv",
    "xlsx",
]
# 路径白名单，防止路径穿越攻击
ALLOWED_ROOT_PATHS = [os.path.abspath(FILE_STORAGE_ROOT)]

# 初始化存储目录
os.makedirs(FILE_STORAGE_ROOT, exist_ok=True)


def safe_join(base, *paths):
    """安全拼接路径，防止路径穿越攻击"""
    try:
        # 拼接并规范化路径
        joined_path = os.path.abspath(os.path.join(base, *paths))
        # 检查路径是否在允许的根目录下
        if not any(
            joined_path.startswith(allowed_root) for allowed_root in ALLOWED_ROOT_PATHS
        ):
            raise ValueError("访问的路径不在允许范围内")
        return joined_path
    except Exception as e:
        abort(403, description=f"路径错误：{str(e)}")


def allowed_file(filename):
    """检查文件类型是否允许"""
    if not ALLOWED_EXTENSIONS:
        return True
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_file_info(file_path):
    """获取文件/文件夹的详细信息"""
    if not os.path.exists(file_path):
        return {}

    is_dir = os.path.isdir(file_path)
    stats = os.stat(file_path)

    return {
        "name": os.path.basename(file_path),
        "path": file_path,
        "relative_path": os.path.relpath(file_path, FILE_STORAGE_ROOT),
        "is_directory": is_dir,
        "size": stats.st_size,  # 字节
        "modified_at": datetime.fromtimestamp(stats.st_mtime).isoformat(),
        "created_at": datetime.fromtimestamp(stats.st_ctime).isoformat(),
        "mime_type": (
            "directory"
            if is_dir
            else mimetypes.guess_type(file_path)[0] or "application/octet-stream"
        ),
    }


# ------------------------------
# 文件管理 RESTful API 资源
# ------------------------------
class FileManagerResource(Resource):
    def get(self, file_path: str = "") -> ResponseTuple | Response:
        """
        GET /api/files/[file_path]
        - 无file_path：列出根目录文件
        - file_path是目录：列出该目录下的文件/文件夹
        - file_path是文件：下载该文件
        - metadata参数：仅返回文件/目录元信息，不下载文件
        - page参数：下载文件指定页码（仅对PDF有效），从0开始
        """
        parser = reqparse.RequestParser()
        parser.add_argument("metadata", type=bool, default=False, location="args")
        parser.add_argument("page", type=int, location="args")
        args = parser.parse_args()

        # 拼接安全路径
        target_path = safe_join(FILE_STORAGE_ROOT, file_path)

        # 路径不存在
        if not os.path.exists(target_path):
            return {"error": "文件/目录不存在"}, 404

        # 是目录：返回目录列表
        if os.path.isdir(target_path):
            items = []
            for item in os.listdir(target_path):
                if item.startswith('.'): continue
                item_path = os.path.join(target_path, item)
                if args["metadata"]:
                    items.append(get_file_info(item_path))
                else:
                    items.append(
                        {
                            "name": item,
                            "is_directory": os.path.isdir(item_path),
                            "relative_path": os.path.relpath(
                                item_path, FILE_STORAGE_ROOT
                            ),
                        }
                    )
            return {
                "directory": target_path,
                "relative_directory": os.path.relpath(target_path, FILE_STORAGE_ROOT),
                "items": items,
            }, 200

        # 是文件：要返回元数据吗？
        mime_type, _ = mimetypes.guess_type(target_path)

        if args["metadata"]:
            info = get_file_info(target_path)
            if mime_type == "application/pdf":
                from PyPDF2 import PdfReader

                try:
                    with open(target_path, "rb") as f:
                        reader = PdfReader(f)
                        info["page_count"] = len(reader.pages)
                except Exception:
                    info["page_count"] = None
            return info, 200

        # 是文件：下载文件
        try:
            # 自动识别文件MIME类型
            if mime_type == "application/pdf" and args["page"] is not None:
                # PDF文件按页下载
                from PyPDF2 import PdfReader, PdfWriter

                reader = PdfReader(target_path)
                if args["page"] < 0 or args["page"] >= len(reader.pages):
                    return {"error": "页码超出范围"}, 400
                writer = PdfWriter()
                writer.add_page(reader.pages[args["page"]])
                temp_pdf_path = target_path + f".page{args['page']}.pdf"
                buf = BytesIO()
                writer.write(buf)
                buf.seek(0)
                return send_file(
                    buf,
                    mimetype="application/pdf",
                    as_attachment=False,
                    download_name=os.path.basename(temp_pdf_path),
                )
            # 普通文件下载
            return send_file(
                target_path,
                mimetype=mime_type,
                as_attachment=False,  # True 强制下载，False 浏览器预览
                download_name=os.path.basename(target_path),
            )
        except Exception as e:
            return {"error": f"文件下载失败：{str(e)}"}, 500

    def post(self, file_path="") -> ResponseTuple:
        """
        POST /api/files/[file_path]
        - file_path为空：上传文件到根目录
        - file_path是目录：上传文件到该目录
        - 支持创建空目录（通过参数 is_directory=true）
        """
        # 解析请求参数
        args = {"is_directory": False, "name": ""}
        if request.is_json:
            args.update(request.json)

        # 创建空目录
        if args["is_directory"] and args["name"]:
            dir_path = safe_join(FILE_STORAGE_ROOT, file_path, args["name"])
            if os.path.exists(dir_path):
                return {"error": "目录已存在"}, 409
            try:
                os.makedirs(dir_path, exist_ok=True)
                return get_file_info(dir_path), 201
            except Exception as e:
                return {"error": f"创建目录失败：{str(e)}"}, 500

        # 文件上传
        if "file" not in request.files:
            return {"error": "未找到上传的文件"}, 400

        file = request.files["file"]
        if not file.filename:
            return {"error": "文件名不能为空"}, 400

        # 安全校验
        if not allowed_file(file.filename):
            return {
                "error": f"不允许上传该类型文件，支持的类型：{ALLOWED_EXTENSIONS}"
            }, 403

        # 拼接存储路径
        filename = secure_filename(file.filename)
        save_path = safe_join(FILE_STORAGE_ROOT, file_path, filename)

        # 检查文件大小（流式检查，避免大文件占用内存）
        max_size = MAX_FILE_SIZE * 1024 * 1024  # 转换为字节
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        if file_size > max_size:
            return {"error": f"文件大小超过限制（最大{MAX_FILE_SIZE}MB）"}, 413

        # 保存文件
        try:
            file.save(save_path)
            return get_file_info(save_path), 201
        except Exception as e:
            return {"error": f"文件上传失败：{str(e)}"}, 500

    def put(self, file_path) -> ResponseTuple:
        """
        PUT /api/files/[file_path]
        - 重命名文件/目录
        - 移动文件/目录
        """
        data = request.get_json()
        if not data or not data.get("name") and not data.get("path"):
            return {"error": "需要提供 name 或 path 参数"}, 400

        # 原文件/目录路径
        old_path = safe_join(FILE_STORAGE_ROOT, file_path)
        if not os.path.exists(old_path):
            return {"error": "文件/目录不存在"}, 404

        # 计算新路径
        if data.get("name"):
            # 重命名：保留目录，修改文件名
            new_name = secure_filename(data["name"])
            new_path = os.path.join(os.path.dirname(old_path), new_name)
        else:
            # 移动：修改完整路径
            new_path = safe_join(FILE_STORAGE_ROOT, data["path"])

        # 检查新路径是否已存在
        if os.path.exists(new_path):
            return {"error": "目标路径已存在"}, 409
        
        try:
            # 执行移动/重命名
            shutil.move(old_path, new_path)
            file_info = get_file_info(new_path)
        except Exception as e:
            return {"error": f"操作失败：{str(e)}"}, 500
        
        # 数据库更改
        if os.path.isfile(old_path):
            pattern = '%' + os.path.relpath(old_path, FILE_STORAGE_ROOT)
        else:
            pattern = os.path.relpath(old_path, FILE_STORAGE_ROOT) + '/%'
        
        db_session.query(Paragraph).filter(Paragraph.source_url.like(pattern)).update(
            {
                Paragraph.source_url: db.func.replace(
                    Paragraph.source_url,
                    os.path.relpath(old_path, FILE_STORAGE_ROOT),
                    os.path.relpath(new_path, FILE_STORAGE_ROOT)
                )
            },
            synchronize_session=False
        )
        db_session.commit()
        
        return file_info, 200

    def delete(self, file_path: str) -> ResponseTuple:
        """
        DELETE /api/files/[file_path]
        - 删除文件/空目录
        - 如需删除非空目录，需额外确认
        """
        assert_admin()
        
        target_path = safe_join(FILE_STORAGE_ROOT, file_path)
        if not os.path.exists(target_path):
            return {"error": "文件/目录不存在"}, 404

        try:
            if os.path.isfile(target_path):
                # 删除文件
                os.remove(target_path)
            elif os.path.isdir(target_path):
                # 删除目录（仅空目录）
                if len(os.listdir(target_path)) > 0:
                    return {"error": "无法删除非空目录"}, 400
                os.rmdir(target_path)
            return {"message": "文件/目录删除成功"}, 200
        except Exception as e:
            return {"error": f"删除失败：{str(e)}"}, 500


api.add_resource(FileManagerResource, "/api/files", "/api/files/<path:file_path>")
instance = FileManagerResource()
