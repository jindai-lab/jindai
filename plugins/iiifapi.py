from PIL import Image, ImageOps
from io import BytesIO
from flask import abort, send_file, request
from PyMongoWrapper import F

from jindai import Plugin
from jindai.models import Paragraph, MediaItem
from jindai.helpers import rest


class IIIFEndPoints(Plugin):

    def __init__(self, pm):

        app = pm.app

        def _resolve_identifer(identifier: str):
            classname, identifier = identifier.split('-', 1)
            classname = classname.lower()
            if identifier == 'mediaitem':
                return MediaItem.first(F.id == identifier)
            else:
                coll = Paragraph.get_coll(classname)
                r = coll.first(F.id == identifier)
                if r is not None:
                    return MediaItem(source=r.source)

        def _resolve_tuple(s):
            return tuple(int(i or 0) for i in s.split(','))

        @app.route('/api/plugins/iiif/<identifier>/info.json')
        @rest(login=False)
        def fetch_image_data(identifier):
            item = _resolve_identifer(identifier)
            return {
                "@context": "http://iiif.io/api/image/3/context.json",
                "id": f"{request.host}/api/plugins/iiif/{identifier}/info.json",
                "type": "ImageService3",
                "protocol": "http://iiif.io/api/image",
                "profile": "level2",
                "width": item.width,
                "height": item.height,
                "maxHeight": item.height,
                "maxWidth": item.width,
                "maxArea": item.width * item.height,
                "preferredFormats": ['jpg', 'png']
            }

        @app.route('/api/plugins/iiif/<identifier>/<region>/<size>/<rotation>/<quality>.<format>')
        @rest(login=False)
        def fetch_image(identifier, region, size, rotation, quality, format):
            item = _resolve_identifer(identifier)
            if item is None:
                abort(404)

            im = item.image
            ow, oh = im.size
            aspect = float(ow) / oh

            # handling region
            if region == 'square':
                sq = min(ow, oh)
                region = ((ow - sq) // 2, (oh - sq) // 2, sq, sq)
            elif region.startswith('pct:'):
                region = _resolve_tuple(region[4:])
                region = (int(region[0]*ow/100), int(region[1]*oh/100),
                          int(region[2]*ow/100), int(region[3]*oh/100))
            elif ',' in region:
                region = _resolve_tuple(region)
            else:
                region = ''

            if len(region) == 4:
                im = im.crop(*region)

            # handling size
            size = size.strip('^')  # we ignore maxWidth, maxHeight, or maxArea
            if size.startswith('pct:'):
                pct = float(size[4]) / 100
                size = (ow * pct, oh * pct)
            elif size.startswith('!') and ',' in size:
                size = _resolve_tuple(size)
                im = im.thumbnail(size)
                size = ''
            elif ',' in size:
                size = _resolve_tuple(size)
            else:
                size = ''

            if size:
                if size[0] == 0:
                    nw = int(aspect * size[1])
                    size = (nw, size[1])
                elif size[1] == 0:
                    nh = int(size[0] / aspect)
                    size = (size[0], nh)
                im = im.resize(size)

            # handling mirror and rotation
            mirror = rotation.startswith('!')
            rotation = int(rotation.strip('!'))

            if mirror:
                im = ImageOps.mirror(im)
            if rotation:
                im = im.rotate(rotation)

            # handling quality
            qualities = {
                'gray': 'L',
                'color': 'RGB',
                'bitonal': '1'
            }
            if quality in qualities:
                im = im.convert(qualities[quality])

            buf = BytesIO()
            im.save(buf, format=format)
            buf.seek(0)

            return send_file(buf, attachment_filename=f'{quality}.{format}', as_attachment=True)
