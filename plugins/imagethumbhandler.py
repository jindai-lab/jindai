"""Storage fragment handler for image thumbnails"""


from PIL import Image
from io import BytesIO

from jindai import storage


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


storage.register_fragment_handler('thumbnail', handle_thumbnail)
