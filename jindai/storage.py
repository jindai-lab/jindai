"""File storage management for Jindai application.

This module provides:
- Storage: Singleton class for secure file operations
- Path traversal protection
- File type validation
- Size limits
- Comprehensive file management capabilities
"""

import glob
import mimetypes
import os
import shutil
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Self

from PIL import Image
from werkzeug.utils import secure_filename

from .pdfutils import get_pdf_page_count, render_pdf_with_fitz


class Storage:
    """Singleton file storage manager for Jindai application.

    Provides secure file operations with path traversal protection,
    file type validation, size limits, and comprehensive
    file management capabilities including upload, download,
    directory operations, and format conversion.

    Attributes:
        FILE_STORAGE_ROOT: Absolute path to the storage root directory
        MAX_FILE_SIZE: Maximum file size in MB (default: 1024)
        ALLOWED_EXTENSIONS: List of allowed file extensions
    """

    __instance: "Storage | None" = None
    __initialized: bool = False

    def __new__(cls, *args, **kwargs) -> Self:
        """Implement singleton pattern for Storage.

        Args:
            *args: Variable length arguments.
            **kwargs: Keyword arguments.

        Returns:
            Singleton instance.
        """
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(self, storage_root: str) -> None:
        """Initialize storage with root directory.

        Args:
            storage_root: Root directory for file storage.
        """
        # Prevent duplicate initialization
        if Storage.__initialized:
            return
        # Core configuration
        self.FILE_STORAGE_ROOT: str = os.path.abspath(storage_root)
        self.MAX_FILE_SIZE: int = 1024  # MB
        self.ALLOWED_EXTENSIONS: List[str] = [
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
        # Initialize storage directory
        os.makedirs(self.FILE_STORAGE_ROOT, exist_ok=True)
        Storage.__initialized = True

    def safe_join(self, *segs: str) -> str:
        """Securely join path segments to prevent directory traversal attacks.

        Args:
            *segs: Path segments to join.

        Returns:
            Safe joined absolute path.

        Raises:
            ValueError: If path would escape storage root directory.
        """
        segs = "/".join(segs).lstrip("/").replace("../", "/")
        if segs.startswith(self.FILE_STORAGE_ROOT.lstrip("/")):
            segs = segs[len(self.FILE_STORAGE_ROOT) :]
        joined_path = os.path.abspath(os.path.join(self.FILE_STORAGE_ROOT, segs))
        if not joined_path.startswith(self.FILE_STORAGE_ROOT):
            raise ValueError(
                f"Access path is not within allowed range, possible path traversal attack {segs} -> {joined_path}"
            )
        return joined_path

    def allowed_file(self, filename: str) -> bool:
        """Check if file extension is in allowed list.

        Args:
            filename: Name of file to check.

        Returns:
            True if file extension is allowed.
        """
        if not self.ALLOWED_EXTENSIONS:
            return True
        return (
            "." in filename
            and filename.rsplit(".", 1)[1].lower() in self.ALLOWED_EXTENSIONS
        )

    def fileinfo(self, file_path: str) -> Dict[str, Any]:
        """Get comprehensive metadata for file or directory.

        Args:
            file_path: Path to file or directory.

        Returns:
            Dictionary containing file metadata.
        """
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
            "mime_type": "directory"
            if is_dir
            else mimetypes.guess_type(file_path)[0] or "application/octet-stream",
        }
        # Append PDF page count information
        if not is_dir and file_info["mime_type"] == "application/pdf":
            file_info["page_count"] = get_pdf_page_count(file_path)
        return file_info

    def _dir_files(self, basedir: str, filenames: List[str], detailed: bool = True) -> Dict[str, Any]:
        """Process files in directory and return file information.

        Args:
            basedir: Base directory path.
            filenames: List of filenames to process.
            detailed: Whether to include detailed file info.

        Returns:
            Directory information with items.
        """
        items = []
        for item in filenames:
            if item.startswith('.'):
                continue
            item_path = os.path.join(basedir, item)
            if detailed:
                items.append(self.fileinfo(item_path))
            else:
                items.append({
                    "name": item,
                    "is_directory": os.path.isdir(item_path),
                    "relative_path": os.path.relpath(item_path, self.FILE_STORAGE_ROOT),
                })

        return {
            "directory": basedir,
            "relative_directory": os.path.relpath(basedir, self.FILE_STORAGE_ROOT),
            "items": items
        }

    def ls(self, basedir: str, detailed: bool = True) -> Dict[str, Any]:
        """List all files/directories in directory, filtering hidden files.

        Args:
            basedir: Directory path to list.
            detailed: Whether to include detailed file info.

        Returns:
            Directory information with items.
        """
        return self._dir_files(basedir, os.listdir(basedir), detailed)

    def glob(self, pattern: str, recursive: bool = False) -> List[str]:
        """Get relative paths matching glob pattern.

        Args:
            pattern: Glob pattern to match files.
            recursive: Whether to search recursively.

        Returns:
            List of relative paths matching pattern.
        """
        results = glob.glob(self.safe_join(pattern), recursive=recursive)
        return [self.relative_path(p) for p in results]

    def search(self, base_dir: str, search: str, detailed: bool = True) -> Dict[str, Any]:
        """Search for files/directories matching pattern in directory tree.

        Args:
            base_dir: Base directory to search in.
            search: Search pattern (space-separated terms).
            detailed: Whether to include detailed file info.

        Returns:
            Search results with matching items.
        """
        pattern = search.split()

        def _match_pattern(fn: str) -> bool:
            return all(pat in fn for pat in pattern)

        items = []
        for pwd, ds, fs in os.walk(self.safe_join(base_dir)):
            fs = [f for f in fs if _match_pattern(f)]
            ds = [d for d in ds if _match_pattern(d)]
            items.extend(self._dir_files(pwd, fs, detailed)['items'] + self._dir_files(pwd, ds, detailed)['items'])

        return {
            "directory": base_dir,
            "relative_directory": os.path.relpath(base_dir, self.FILE_STORAGE_ROOT),
            "items": items
        }

    def save(self, file_obj, save_rel_path: str) -> Dict[str, Any]:
        """Save uploaded file with streaming size validation.

        Args:
            file_obj: File object.
            save_rel_path: Relative path where to save the file.

        Returns:
            File information dictionary.

        Raises:
            ValueError: If file type not allowed or size exceeds limit.
        """
        # Securely validate filename and path
        filename = secure_filename(file_obj.filename)
        save_path = self.safe_join(save_rel_path, filename)
        # Validate file type
        if not self.allowed_file(filename):
            raise ValueError(
                f"File type not allowed, supported types: {self.ALLOWED_EXTENSIONS}"
            )
        # Stream file size validation (doesn't load full file into memory)
        max_size_bytes = self.MAX_FILE_SIZE * 1024 * 1024
        file_obj.seek(0, os.SEEK_END)
        file_size = file_obj.tell()
        file_obj.seek(0)
        if file_size > max_size_bytes:
            raise ValueError(f"File size exceeds limit (max {self.MAX_FILE_SIZE}MB)")
        # Save file
        file_obj.save(save_path)
        return self.fileinfo(save_path)

    def mkdir(self, dir_rel_path: str, dir_name: str) -> Dict[str, Any]:
        """Create empty directory.

        Args:
            dir_rel_path: Parent directory path relative to storage root.
            dir_name: Name of directory to create.

        Returns:
            Directory information dictionary.

        Raises:
            ValueError: If directory already exists.
        """
        dir_path = self.safe_join(dir_rel_path, dir_name)
        if os.path.exists(dir_path):
            raise ValueError("Directory already exists")
        os.makedirs(dir_path, exist_ok=True)
        return self.fileinfo(dir_path)

    def mv(
        self, old_rel_path, new_name=None, new_rel_path=None
    ) -> dict[str, bytes | str | dict]:
        """Move or rename file/directory.

        Args:
            old_rel_path: Original relative path.
            new_name: New filename (for rename).
            new_rel_path: New full relative path (for move).

        Returns:
            Dictionary with old path info and new file info.
        """
        old_path = self.safe_join(old_rel_path)
        if not os.path.exists(old_path):
            raise ValueError("File/directory does not exist")

        # Calculate new path
        if new_name:
            new_name = secure_filename(new_name)
            new_path = os.path.join(os.path.dirname(old_path), new_name)
        else:
            new_path = self.safe_join(new_rel_path)

        if os.path.exists(new_path):
            raise ValueError("Target path already exists")

        # Execute move/rename
        shutil.move(old_path, new_path)
        return {
            "old_relative_path": os.path.relpath(old_path, self.FILE_STORAGE_ROOT),
            "new_info": self.fileinfo(new_path),
        }

    def delete(self, target_rel_path) -> bool:
        """Delete file or empty directory.

        Args:
            target_rel_path: Relative path to delete.

        Returns:
            True if deletion succeeded.

        Raises:
            ValueError: If file/directory doesn't exist or directory not empty.
        """
        target_path = self.safe_join(target_rel_path)
        if not os.path.exists(target_path):
            raise ValueError("File/directory does not exist")

        if os.path.isfile(target_path):
            os.remove(target_path)
        elif os.path.isdir(target_path):
            if len(os.listdir(target_path)) > 0:
                raise ValueError("Cannot delete non-empty directory")
            os.rmdir(target_path)
        return True

    def read_file(self, target_rel_path, page=None, format=None) -> tuple:
        """Read file content, supporting PDF page download.

        Args:
            target_rel_path: Relative path to file.
            page: Page number for PDF (optional).
            format: Output format for conversion.

        Returns:
            Tuple of (BytesIO stream, MIME type, download filename).
        """
        target_path = self.safe_join(target_rel_path)
        if not os.path.exists(target_path):
            raise ValueError("File does not exist")

        mime_type, _ = mimetypes.guess_type(target_path) or (
            "application/octet-stream",
            None,
        )
        file_name = os.path.basename(target_path)

        buf = None

        # Handle PDF pages
        if mime_type == "application/pdf":
            if page:
                # PDF page download
                from PyPDF2 import PdfReader, PdfWriter

                reader = PdfReader(target_path)
                if page < 0 or page >= len(reader.pages):
                    raise ValueError("Page number out of range")
                writer = PdfWriter()
                writer.add_page(reader.pages[page])
                buf = BytesIO()
                writer.write(buf)
                buf.seek(0)
                file_name = f"{file_name}.page{page}.pdf"

        if buf is None:
            buf = open(target_path, "rb")

        # Check format parameter
        if format:
            file_name = file_name.rsplit(".", 1)[0] + "." + format
            if mime_type.startswith("image/"):
                im = Image.open(buf)
                buf = BytesIO()
                im.save(buf, format)
                mime_type = f"image/{format}"
            elif mime_type == "application/pdf":
                buf = render_pdf_with_fitz(buf, format=format)
                mime_type = f"image/{format}"

        buf.seek(0)
        return buf, mime_type, file_name

    def relative_path(self, p) -> bytes | str:
        """Get path relative to storage root.

        Args:
            p: Full path.

        Returns:
            Relative path.
        """
        return os.path.relpath(p, self.FILE_STORAGE_ROOT)

    def open(self, relpath: str, mode: str = "rb") -> open:
        """Open file with safe path joining.

        Args:
            relpath: Relative path from storage root.
            mode: File open mode.

        Returns:
            File object.
        """
        return open(self.safe_join(relpath), mode)


# Initialize singleton instance (global unique)
# Import方式: from .storage import instance as storage
from .config import config

storage = Storage(config.storage)
