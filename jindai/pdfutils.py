import os
from io import BytesIO

import fitz
from flask import request, send_file
from flask_restful.reqparse import RequestParser
from PIL import Image
from PyPDF2 import PdfReader, PdfWriter


def get_pdf_page_count(pdf_path):
    """私有方法：获取PDF文件页数"""
    try:
        from PyPDF2 import PdfReader

        with open(pdf_path, "rb") as f:
            return len(PdfReader(f).pages)
    except Exception:
        return None


def render_pdf_with_fitz(pdf_path, page_num=0, format="png"):
    # 打开 PDF
    if isinstance(pdf_path, str):
        doc = fitz.open(pdf_path)
    else:
        doc = fitz.open(stream=pdf_path)
    # 获取指定页（注意：fitz 从 0 开始计数）
    page = doc.load_page(page_num)

    # 将页面渲染为像素图 (PixMap)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2倍缩放相当于提高清晰度

    # 转换为字节流
    img_data = pix.tobytes(format)
    return BytesIO(img_data)


def convert_pdf_to_tiff_group4(pdf, outp):
    """
    Converts a PDF file to a multi-page Group 4 compressed TIFF file.

    Args:
        pdf_path (str): The path to the input PDF file.
        output_path (str): The path where the output TIFF file will be saved.
    """
    doc = fitz.open(pdf)
    image_list = []

    # Process each page
    for page in doc:
        # Render page to a pixmap (image representation)
        # Zoom factor increases resolution (e.g., 2 for 150-200 DPI approx, 4 for 300+ DPI)
        zoom = 4
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        # Convert pixmap to a Pillow Image
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

        # Convert to black and white (1-bit pixels) for efficient Group 4 compression
        # This is crucial for Group 4 compression to work effectively
        img = img.convert("1")
        image_list.append(img)

    doc.close()

    if image_list:
        # Save all pages as a single multi-page TIFF with Group 4 compression
        image_list[0].save(
            outp,
            format="pdf",
            save_all=True,
            append_images=image_list[1:],
            compression="group4",
            dpi=(300, 300),  # Set desired DPI
        )


def merge_images_from_folder(folderpath, outp):
    image_list = []
    for f in sorted(os.listdir(folderpath)):
        if f.endswith((".jpg", ".png", ".jpeg", ".tif", ".tiff")):
            im = Image.open(os.path.join(folderpath, f))
            image_list.append(im)
    if image_list:
        image_list[0].save(
            outp,
            format="pdf",
            save_all=True,
            append_images=image_list[1:],
            compression="group4",
            dpi=(300, 300),  # Set desired DPI
        )


def read_pdf_pages(path, reverse=False):
    reader = PdfReader(path)
    return list(reversed(reader.pages) if reverse else reader.pages)


def cross_merge_pdf(outp, pdf1, pdf2, reversed1, reversed2):
    """合并PDF文件，返回合并后的临时文件路径"""
    try:
        pdf1_pages, pdf2_pages = read_pdf_pages(pdf1, reversed1), read_pdf_pages(
            pdf2, reversed2
        )

        # 检查页数是否匹配
        if len(pdf1_pages) != len(pdf2_pages):
            raise ValueError("两个PDF文件的页数不匹配，无法合并")

        # 创建输出PDF
        pdf_writer = PdfWriter()

        # 交叉合并页面
        for page1, page2 in zip(pdf1_pages, pdf2_pages):
            pdf_writer.add_page(page1)
            pdf_writer.add_page(page2)

        # 创建临时文件保存合并结果
        pdf_writer.write(outp)

    except Exception as e:
        raise e


def sequential_merge_pdf(outp, pdf1, pdf2, reversed1, reversed2):
    """顺序合并PDF文件，返回合并后的临时文件路径"""
    try:
        pdf1_pages, pdf2_pages = read_pdf_pages(pdf1, reversed1), read_pdf_pages(
            pdf2, reversed2
        )

        # 创建输出PDF
        pdf_writer = PdfWriter()

        # 交叉合并页面
        for page in pdf1_pages + pdf2_pages:
            pdf_writer.add_page(page)

        # 创建临时文件保存合并结果
        pdf_writer.write(outp)

    except Exception as e:
        raise e


def extract_pdf_texts(filename, since=0):
    doc = fitz.open(filename)

    for page in range(since, doc.page_count):
        try:
            label = doc[page].get_label()
        except (RuntimeError, TypeError):
            label = ""

        try:
            content = doc[page].get_text()
        except Exception as ex:
            content = ""

        yield page, label, content


def requestio(func, **kwargs):
    files = {}
    for key, file in request.files.items():
        inp = BytesIO()
        file.save(inp)
        inp.seek(0)
        files[key] = inp

    outp = BytesIO()
    func(**files, **kwargs, outp=outp)
    outp.seek(0)
    return send_file(outp, as_attachment=True, mimetype="")


def inject_endpoints():
    from .app import app

    @app.route("/api/v2/pdfutils/convert_monochrome", methods=["POST"])
    def api_convert_monochrome():
        return requestio(convert_pdf_to_tiff_group4)

    @app.route("/api/v2/pdfutils/cross_merge_pdf", methods=["POST"])
    def api_cross_merge_pdf():
        parser = RequestParser()
        parser.add_argument("cross", type=bool)
        parser.add_argument("reversed1", type=bool)
        parser.add_argument("reversed2", type=bool)
        args = parser.parse_args()

        return requestio(
            cross_merge_pdf if args["cross"] else sequential_merge_pdf,
            reversed1=args["reversed1"],
            reversed2=args["reversed2"],
        )
