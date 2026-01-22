import glob
import mimetypes
import os
import shutil
from datetime import datetime
from io import BytesIO

from werkzeug.utils import secure_filename


class Storage:
    __instance = None
    __initialized = False

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(self, storage_root: str):
        # 防止重复初始化
        if Storage.__initialized:
            return
        # 核心配置
        self.FILE_STORAGE_ROOT: str = os.path.abspath(storage_root)
        self.MAX_FILE_SIZE = 1024  # MB
        self.ALLOWED_EXTENSIONS = [
            "txt", "pdf", "doc", "docx", "jpg", "png", "json", "csv", "xlsx",
        ]
        # 初始化存储目录
        os.makedirs(self.FILE_STORAGE_ROOT, exist_ok=True)
        Storage.__initialized = True

    def safe_join(self, *segs):
        """安全拼接路径，防止路径穿越攻击，核心安全校验"""
        segs = '/'.join(segs).lstrip('/').replace('../', '/')
        if segs.startswith(self.FILE_STORAGE_ROOT.lstrip('/')):
            segs = segs[len(self.FILE_STORAGE_ROOT)]
        joined_path = os.path.abspath(os.path.join(self.FILE_STORAGE_ROOT, segs))
        if not joined_path.startswith(self.FILE_STORAGE_ROOT):
            raise ValueError(f"访问的路径不在允许范围内，疑似路径穿越攻击 {segs} -> {joined_path}")
        return joined_path

    def allowed_file(self, filename):
        """检查文件后缀是否在允许列表内"""
        if not self.ALLOWED_EXTENSIONS:
            return True
        return "." in filename and filename.rsplit(".", 1)[1].lower() in self.ALLOWED_EXTENSIONS

    # ------------------------------ 文件元信息核心方法 ------------------------------
    def get_file_info(self, file_path):
        """获取文件/文件夹的完整元信息"""
        if not os.path.exists(file_path):
            return {}

        is_dir = os.path.isdir(file_path)
        stats = os.stat(file_path)
        file_info = {
            "name": os.path.basename(file_path),
            "path": file_path,
            "relative_path": os.path.relpath(file_path, self.FILE_STORAGE_ROOT),
            "is_directory": is_dir,
            "size": stats.st_size,
            "modified_at": datetime.fromtimestamp(stats.st_mtime).isoformat(),
            "created_at": datetime.fromtimestamp(stats.st_ctime).isoformat(),
            "mime_type": "directory" if is_dir else mimetypes.guess_type(file_path)[0] or "application/octet-stream"
        }
        # 额外追加PDF页数信息
        if not is_dir and file_info["mime_type"] == "application/pdf":
            file_info["page_count"] = self._get_pdf_page_count(file_path)
        return file_info

    def _get_pdf_page_count(self, pdf_path):
        """私有方法：获取PDF文件页数"""
        try:
            from PyPDF2 import PdfReader
            with open(pdf_path, "rb") as f:
                return len(PdfReader(f).pages)
        except Exception:
            return None

    # ------------------------------ 目录操作 ------------------------------
    def list_directory(self, target_path, only_basic=False):
        """列出目录下所有文件/文件夹，过滤隐藏文件"""
        items = []
        for item in os.listdir(target_path):
            if item.startswith('.'): continue
            item_path = os.path.join(target_path, item)
            if only_basic:
                items.append({
                    "name": item,
                    "is_directory": os.path.isdir(item_path),
                    "relative_path": os.path.relpath(item_path, self.FILE_STORAGE_ROOT),
                })
            else:
                items.append(self.get_file_info(item_path))
        return {
            "directory": target_path,
            "relative_directory": os.path.relpath(target_path, self.FILE_STORAGE_ROOT),
            "items": items
        }

    # ------------------------------ 文件上传 ------------------------------
    def save_file(self, file_obj, save_rel_path):
        """保存上传文件，流式校验文件大小"""
        # 安全校验文件名和路径
        filename = secure_filename(file_obj.filename)
        save_path = self.safe_join(save_rel_path, filename)
        # 校验文件类型
        if not self.allowed_file(filename):
            raise ValueError(f"不允许上传该类型文件，支持的类型：{self.ALLOWED_EXTENSIONS}")
        # 流式校验文件大小（不加载全文件到内存）
        max_size_bytes = self.MAX_FILE_SIZE * 1024 * 1024
        file_obj.seek(0, os.SEEK_END)
        file_size = file_obj.tell()
        file_obj.seek(0)
        if file_size > max_size_bytes:
            raise ValueError(f"文件大小超过限制（最大{self.MAX_FILE_SIZE}MB）")
        # 保存文件
        file_obj.save(save_path)
        return self.get_file_info(save_path)

    # ------------------------------ 创建目录 ------------------------------
    def create_directory(self, dir_rel_path, dir_name):
        """创建空目录"""
        dir_path = self.safe_join(dir_rel_path, dir_name)
        if os.path.exists(dir_path):
            raise ValueError("目录已存在")
        os.makedirs(dir_path, exist_ok=True)
        return self.get_file_info(dir_path)

    # ------------------------------ 文件/目录 移动/重命名 ------------------------------
    def move_or_rename(self, old_rel_path, new_name=None, new_rel_path=None):
        """
        移动或重命名文件/目录
        :param old_rel_path: 原文件相对路径
        :param new_name: 新文件名（重命名用）
        :param new_rel_path: 新完整相对路径（移动用）
        :return: 新文件信息
        """
        old_path = self.safe_join(old_rel_path)
        if not os.path.exists(old_path):
            raise ValueError("文件/目录不存在")
        
        # 计算新路径
        if new_name:
            new_name = secure_filename(new_name)
            new_path = os.path.join(os.path.dirname(old_path), new_name)
        else:
            new_path = self.safe_join(new_rel_path)

        if os.path.exists(new_path):
            raise ValueError("目标路径已存在")
        
        # 执行移动/重命名
        shutil.move(old_path, new_path)
        return {
            "old_relative_path": os.path.relpath(old_path, self.FILE_STORAGE_ROOT),
            "new_info": self.get_file_info(new_path)
        }

    # ------------------------------ 文件/目录 删除 ------------------------------
    def delete(self, target_rel_path):
        """删除文件或空目录"""
        target_path = self.safe_join(target_rel_path)
        if not os.path.exists(target_path):
            raise ValueError("文件/目录不存在")
        
        if os.path.isfile(target_path):
            os.remove(target_path)
        elif os.path.isdir(target_path):
            if len(os.listdir(target_path)) > 0:
                raise ValueError("无法删除非空目录")
            os.rmdir(target_path)
        return True

    # ------------------------------ 文件下载核心 ------------------------------
    def read_file(self, target_rel_path, page=None):
        """
        读取文件内容，支持PDF分页下载
        :return: (BytesIO流, MIME类型, 下载文件名)
        """
        target_path = self.safe_join(target_rel_path)
        if not os.path.exists(target_path):
            raise ValueError("文件不存在")
        
        mime_type, _ = mimetypes.guess_type(target_path) or ("application/octet-stream", None)
        file_name = os.path.basename(target_path)

        # 普通文件下载
        if mime_type != "application/pdf" or page is None:
            with open(target_path, "rb") as f:
                buf = BytesIO(f.read())
            buf.seek(0)
            return buf, mime_type, file_name

        # PDF按页码下载
        from PyPDF2 import PdfReader, PdfWriter
        reader = PdfReader(target_path)
        if page < 0 or page >= len(reader.pages):
            raise ValueError("页码超出范围")
        writer = PdfWriter()
        writer.add_page(reader.pages[page])
        buf = BytesIO()
        writer.write(buf)
        buf.seek(0)
        return buf, "application/pdf", f"{file_name}.page{page}.pdf"
    
    def relative_path(self, p):
        return os.path.relpath(p, self.FILE_STORAGE_ROOT)
    
    def globs(self, pattern):
        return [self.relative_path(p) for p in glob.glob(self.safe_join(pattern))]
    
    def open(self, relpath, mode='rb'):
        return open(self.safe_join(relpath), mode)

# 初始化单例实例 (项目全局唯一)
# 导入方式：from .storage import instance as storage
from .app import config

instance = Storage(config.storage)
