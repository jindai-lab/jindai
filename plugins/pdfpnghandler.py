import tempfile
from pdf2image import convert_from_path as _pdf_convert
import os
from io import BytesIO
from jindai import storage
from jindai.helpers import safe_import

def handle_pdf(buf, page, *args):
    """Get PNG data from PDF

    :param file: PDF path
    :type file: str
    :param page: page index, starting form 0
    :type page: int
    :return: PNG bytes in a BytesIO
    :rtype: BytesIO
    """

    image_format = args[-1].rsplit('.', 1)[-1]
    if image_format == 'jpg':
        image_format = 'jpeg'
    elif image_format == 'pdf':
        pass
    else:
        image_format = 'png'

    def _pdf_image(file: str, page: int, **_) -> BytesIO:
        buf = BytesIO()
        page = int(page)

        img, = _pdf_convert(file, 120, first_page=page+1,
                            last_page=page+1, fmt=image_format) or [None]
        if img:
            img.save(buf, format=image_format)
            buf.seek(0)

        return buf

    page = int(page)
    if image_format != 'pdf':        
        filename = getattr(buf, 'name', '')
        temp = not not filename
        if temp:
            filename = tempfile.mktemp(suffix='.pdf')
            with open(filename, 'wb') as fout:
                fout.write(buf.read())
        output = _pdf_image(filename, page)
        if temp:
            os.unlink(filename)
    else:
        safe_import('PyPDF2')
        from PyPDF2 import PdfReader, PdfWriter
        pdf_reader = PdfReader(buf)
        pdf_writer = PdfWriter()
        pdf_writer.add_page(pdf_reader.pages[page])
        output = BytesIO()
        pdf_writer.write(output)
        output.seek(0)

    return output


storage.register_fragment_handler('pdf', handle_pdf)