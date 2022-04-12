"""图像哈希"""
import os
import tempfile
from io import BytesIO
from queue import deque
from typing import Union

import imagehash
from bson import binary
from flask import Response, request
from PIL import Image
from PyMongoWrapper import F, ObjectId
from jindai import *
from jindai.helpers import serve_file
from jindai.models import ImageItem, Paragraph
from plugins.gallery import ImageOrAlbumStage, single_item


def dhash(image: Union[Image.Image, BytesIO]) -> bytes:
    """Generate d-hash for image

    Args:
        image (Image | BytesIO): PIL Image

    Returns:
        bytes: hash
    """
    if isinstance(image, BytesIO):
        image.seek(0)
        image = Image.open(image)
    hash_val = imagehash.dhash(image)
    hash_val = bytes.fromhex(str(hash_val))
    return hash_val


def whash(image: Union[Image.Image, BytesIO]) -> bytes:
    """Generate w-hash for image

    Args:
        image (Image): PIL Image

    Returns:
        bytes: hash
    """
    if isinstance(image, BytesIO):
        image.seek(0)
        image = Image.open(image)
    hash_val = imagehash.whash(image)
    hash_val = bytes.fromhex(str(hash_val))
    return hash_val


def bitcount(val):
    """Count 1's in a 64-bit integer"""
    return bin(val).count('1')


def to_int(val):
    """Get int value of bytes"""
    return int(val.hex(), 16)


def to_binary(val):
    """To binary

    Args:
        val (int): the value

    Returns:
        _type_: _description_
    """
    return binary.Binary(bytes.fromhex(f'{val:016x}'))


def flip(val, bit_position):
    """Flip the i-th bit in x"""
    val ^= 1 << bit_position
    return val


def flips(val, bit_num, least_position=0):
    """Flip at most n bits for x"""
    for i in range(least_position, 64):
        new_val = flip(val, i)
        if bit_num == 1:
            yield new_val
        else:
            for new_val in flips(new_val, bit_num - 1, i + 1):
                yield new_val


def resolve_dups(tmp_file_name, slimit):
    """Resolve duplications from temp file, with scores < slimit"""

    def _parse_compare_results():
        with safe_open(tmp_file_name, 'r') as input_file:
            for line in input_file:
                row = line.strip().split('\t')
                if len(row) < 3:
                    continue
                id1, id2, score = row
                if id1 == id2:
                    continue
                yield id1, id2, int(score)

    def _get_items():
        ids = set()
        for id1, id2, _ in _parse_compare_results():
            ids.add(id1)
            ids.add(id2)
        items = {}
        for i in ImageItem.query(F.flag.eq(0) & F.id.in_([ObjectId(_) for _ in ids])):
            items[str(i.id)] = i
        return items

    items = _get_items()
    for id1, id2, score in sorted(_parse_compare_results(),
                                  key=lambda x: x[2]):
        if score > slimit:
            continue
        if id1 in items and id2 in items:
            yield items[id1], items[id2], score


class ImageHash(ImageOrAlbumStage):
    """建立图像哈希检索
    """

    def resolve_image(self, i: ImageItem, _):
        try:
            i_dhash, i_whash = i.dhash, i.whash
            if i_dhash and i_whash:
                return i

            image_raw = i.image_raw
            if not image_raw:
                return None

            if not i_dhash:
                i_dhash = dhash(image_raw) or ''
            if not i_whash:
                i_whash = whash(image_raw) or ''

            i.dhash, i.whash = i_dhash, i_whash
        except (IOError, AssertionError):
            pass
        i.save()


class ImageHashDuplications(ImageOrAlbumStage):
    """进行图像哈希去重
    """

    def __init__(self) -> None:
        super().__init__()
        self.results = deque()
        self.result_pairs = set()

    def resolve_image(self, i: ImageItem, _):
        """处理图像哈希
        """
        if not i.dhash:
            return

        if not isinstance(i.dhash, bytes):
            self.logger(type(i).__name__, i.as_dict())
        d_hash = to_int(i.dhash)
        w_hash = to_int(i.whash)
        image_height, image_width = i.height, i.width

        for j in ImageItem.query(F.dhash.in_(
                [to_binary(x)
                 for x in [d_hash] + list(flips(d_hash, 1)) + list(flips(d_hash, 2))])):
            target_id = j.id
            if target_id == i.id or f'{i.id}-{target_id}' in self.result_pairs \
                    or f'{target_id}-{i.id}' in self.result_pairs:
                continue

            self.result_pairs.add(f'{target_id}-{i.id}')
            id_a, id_b = i.id, target_id
            if j.width * j.height < image_width * image_height:
                id_b, id_a = id_a, id_b

            result_line = f'{id_a}\t{id_b}\t' + \
                f'{bitcount(to_int(i.dhash) ^ d_hash) + bitcount(to_int(j.whash) ^ w_hash)}'
            self.logger(result_line)
            self.results.append(result_line + '\n')

        return i

    def summarize(self, _):
        k = tempfile.mktemp()
        output_file = safe_open(k + '.tsv', 'w')
        for line in self.results:
            output_file.write(line)
        output_file.close()
        return {'redirect': '/api/plugins/compare?' + k}


class Hashing(Plugin):
    """哈希插件"""

    def __init__(self, pmanager):
        super().__init__(pmanager)
        app = self.pmanager.app
        ImageItem.set_field('dhash', bytes)
        ImageItem.set_field('whash', bytes)
        self.register_pipelines([ImageHashDuplications, ImageHash])
        self.register_filter(
            'sim', keybind='s', format_string='sim/{imageitem._id}', icon='mdi-image', handler=self.handle_page)

        @app.route('/api/plugins/compare.tsv')
        def _compare_tsv():
            file_path = request.args.get('key', '') + '.tsv'
            if not os.path.exists(file_path):
                return Response('')

            buf = ''
            for item1, item2, score in resolve_dups(file_path, int(request.args.get('q', 10))):
                buf += f'{item1.id} {item2.id} {score}\n'
            return Response(buf)

        @app.route('/api/plugins/compare')
        def _compare_html():
            return serve_file(os.path.join(os.path.dirname(__file__), 'compare.html'))

        @app.route('/api/plugins/hashing-jquery.min.js')
        def _jquery_js():
            return serve_file(os.path.join(os.path.dirname(__file__), 'jquery.min.js'))

    def handle_page(self, datasource_impl, iid):
        """Handle page"""
        limit = datasource_impl.limit
        offset = datasource_impl.skip
        datasource_impl.limit = 0
        datasource_impl.raw = False

        groups = datasource_impl.groups in ('both', 'group')
        archive = datasource_impl.groups in ('both', 'source')

        if groups:
            return single_item('', iid)
        else:
            image_item = ImageItem.first(F.id == iid)
            if image_item.dhash is None:
                return
            pgroups = [g
                       for g in (Paragraph.first(F.images == ObjectId(iid)) or Paragraph()).keywords
                       if g.startswith('*')
                       ] or [(Paragraph.first(F.images == ObjectId(iid))
                              or Paragraph()).source.get('url', '')]
            dha, dhb = to_int(image_item.dhash), to_int(image_item.whash)
            results = []
            groupped = {}

            for paragraph in datasource_impl.fetch():
                for i in paragraph.images:
                    if i.id == image_item.id:
                        continue
                    if i.flag != 0 or i.dhash is None or i.dhash == b'':
                        continue
                    dha1, dhb1 = to_int(i.dhash), to_int(i.whash)
                    i.score = bitcount(dha ^ dha1) + bitcount(dhb ^ dhb1)
                    new_paragraph = Paragraph(**paragraph.as_dict())
                    new_paragraph.images = [i]
                    new_paragraph.score = i.score
                    if archive:
                        groups = [
                            g for g in paragraph.keywords if g.startswith('*')]
                        for group in groups or [new_paragraph.source['url']]:
                            if group not in pgroups and \
                                (group not in groupped or groupped[group].score >
                                    new_paragraph.score):
                                groupped[group] = new_paragraph
                    else:
                        results.append(new_paragraph)

            if archive:
                results = list(groupped.values())

            results = sorted(results, key=lambda x: x.score)[
                offset:offset + limit]
            return single_item('', iid) + [{'spacer': 'spacer'}] + results
