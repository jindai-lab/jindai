"""Default fragment handlers"""
import tempfile
import zipfile
from PIL import Image
from pdf2image import convert_from_path as _pdf_convert
import os
from io import BytesIO


def _get_fragment_handlers():

    def handle_zip(buf, *inner_path):
        """Handle zip file"""
        zpath = '/'.join(inner_path)
        with zipfile.ZipFile(buf, 'r') as zip_file:
            return zip_file.open(zpath)

    def handle_pdf(buf, page, *_):
        """Get PNG data from PDF

        :param file: PDF path
        :type file: str
        :param page: page index, starting form 0
        :type page: int
        :return: PNG bytes in a BytesIO
        :rtype: BytesIO
        """

        def _pdf_image(file: str, page: int, **_) -> BytesIO:
            buf = BytesIO()
            page = int(page)

            img, = _pdf_convert(file, 120, first_page=page+1,
                                last_page=page+1, fmt='png') or [None]
            if img:
                img.save(buf, format='png')
                buf.seek(0)

            return buf

        filename = getattr(buf, 'name', '')
        temp = not not filename
        if temp:
            filename = tempfile.mktemp(suffix='.pdf')
            with open(filename, 'wb') as fout:
                fout.write(buf.read())

        buf = _pdf_image(filename, int(page))

        if temp:
            os.unlink(filename)

        return buf

    def handle_thumbnail(buf, width, height='', *_):
        """Get thumbnail for image"""

        if not height:
            height = width
        width, height = int(width), int(height)

        image = Image.open(buf)
        image.thumbnail((width, height))

        buf = BytesIO()
        image.save(buf, 'JPEG')
        buf.seek(0)

        return buf

    return locals().items()


fragment_handlers = _get_fragment_handlers()
