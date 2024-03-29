"""图像哈希"""
from locale import format_string
import os
from re import S
import tempfile
from io import BytesIO, IOBase
from queue import deque
from threading import Lock
from typing import Union

import imagehash
from bson import binary
from flask import Response, request
from PIL import Image, ImageFile
from collections import namedtuple

from PyMongoWrapper import F, ObjectId
from jindai import *
from jindai.models import MediaItem, Paragraph
from jindai.common import DictObject, CacheDict
from jindai.helpers import evaluateqx
from plugins.imageproc import MediaItemStage

ImageFile.LOAD_TRUNCATED_IMAGES = True

# HELPER FUNCS


def single_item(pid: str, iid: str):
    """Return a single-item paragraph object with id = `pid` and item id = `iid`
    Args:
        pid (str): Paragraph ID
        iid (str): MediaItem ID
    Returns:
        List[Paragraph]: a list with at most one element, i.e., the single-item paragraph object
    """
    if pid:
        pid = ObjectId(pid)
        para = Paragraph.first(F.id == pid)
    elif iid:
        iid = ObjectId(iid)
        para = Paragraph.first(F.images == iid)
    if iid and para:
        para.images = [i for i in para.images if i.id == iid]
        para.group_id = f"id=o'{para['_id']}'"
        return [para]

    return []


def _hash_image(image, method):
    if isinstance(image, IOBase):
        if image.seekable():
            image.seek(0)
        image = Image.open(image)
    hash_val = getattr(imagehash, method)(image)
    return bytes.fromhex(str(hash_val))


def dhash(image: Union[Image.Image, BytesIO]) -> bytes:
    """Generate d-hash for image

    Args:
        image (Image | BytesIO): PIL Image

    Returns:
        bytes: hash
    """
    return _hash_image(image, 'dhash')


def whash(image: Union[Image.Image, BytesIO]) -> bytes:
    """Generate w-hash for image

    Args:
        image (Image): PIL Image

    Returns:
        bytes: hash
    """
    return _hash_image(image, 'whash')


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


def resolve_dups(input_file, slimit):
    """Resolve duplications from temp file, with scores <= slimit"""

    input_file = open(input_file, 'r') if isinstance(
        input_file, str) else input_file
    ids = set()

    def _parse_compare_results():
        with input_file:
            for line in input_file:
                row = line.strip().split('\t')
                if len(row) < 3:
                    continue
                id1, id2, score = row
                id1, id2, score = ObjectId(id1), ObjectId(id2), int(score)
                ids.add(id1)
                ids.add(id2)
                if id1 == id2 or score > slimit:
                    continue
                yield id1, id2, score

    results = list(_parse_compare_results())
    items = {
        i.id: i for i in MediaItem.query(F.id.in_(list(ids)))
    }

    results = [
        (items[id1], items[id2], score)
        for id1, id2, score in results
        if id1 in items and id2 in items
    ]

    return sorted(results, key=lambda x: x[2])


class ImageHash(MediaItemStage):
    """Hash images
    @zhs 建立图像哈希检索
    """

    def resolve_image(self, i: MediaItem, _):
        try:
            i_dhash, i_whash = i.dhash, i.whash
            if i_dhash and i_whash:
                return i

            if not i.data:
                return None

            if not i_dhash:
                i_dhash = dhash(i.image) or ''
            if not i_whash:
                i_whash = whash(i.image) or ''

            i.dhash, i.whash = i_dhash, i_whash
        except (IOError, AssertionError) as ex:
            self.log_exception(f'Error while hashing image {i.id}', ex)

        i.save()
        return i


class ImageHashDuplications(MediaItemStage):
    """Deduplication with image hash
    @zhs 进行图像哈希去重
    """

    def __init__(self, auto_remove=-1) -> None:
        """
        Args:
            auto_remove (int, optional):
                Auto remove duplicates with max diff. score
                @zhs 自动删除差异值小于等于该数值的项目，默认为 -1 即不删除
        """
        super().__init__()
        self.results = deque()
        self.result_pairs = set()
        self.auto_remove = auto_remove

        self._tmpfile = tempfile.mktemp('.tsv')
        self._writable_file = open(self._tmpfile, 'w', encoding='utf-8')
        self._write_lock = Lock()
        self._deleted = set()

    def resolve_image(self, i: MediaItem, _):
        if not i.dhash:
            return

        if i.id in self._deleted:
            return

        if not isinstance(i.dhash, bytes):
            self.log(type(i).__name__, i.as_dict())
        d_hash = to_int(i.dhash)
        w_hash = to_int(i.whash)
        image_height, image_width = i.height, i.width

        def _score(j):
            return bitcount(to_int(j.dhash) ^ d_hash) + \
                bitcount(to_int(j.whash) ^ w_hash)

        candidates = [
            (j, _score(j))
            for j in MediaItem.query(
                F.item_type == 'image',
                F.dhash.in_([
                    to_binary(x)
                    for x in [d_hash] + list(flips(d_hash, 1)) + list(flips(d_hash, 2))
                ])
            )
            if (i.width > i.height) == (j.width > j.height)
        ]

        dups, sims = [j for j, sc in candidates if sc <= self.auto_remove], [
            (j, sc) for j, sc in candidates if sc > self.auto_remove]

        if len(dups) == 1:
            return

        best = i
        for j in dups:
            if j.width * j.height > best.width * best.height:
                best = j
            elif j.width * j.height < best.width * best.height:
                continue
            elif j.id < best.id:
                best = j
        dups = [j for j in dups if j.id != best.id]

        if dups:
            Paragraph.merge_by_mediaitems(best, dups)
            self._deleted.update({d.id for d in dups})
            self.log(
                f'delete {" ".join([str(d.id) for d in dups])}, preserving {best.id}')

        for j, score in sims:
            target_id = j.id
            if target_id == i.id or f'{i.id}-{target_id}' in self.result_pairs \
                    or f'{target_id}-{i.id}' in self.result_pairs:
                continue
            self.result_pairs.add(f'{target_id}-{i.id}')

            if j.width * j.height < image_width * image_height or (j.width * j.height == image_width * image_height and i.id < j.id):
                i, j = j, i

            result_line = f'{i.id}\t{j.id}\t{score}'
            self.log(result_line)
            self.results.append(result_line + '\n')

        return i

    def flush_results(self):
        with self._write_lock:
            for line in self.results:
                self._writable_file.write(line)
            self.results.clear()

    def summarize(self, _):
        self.flush_results()
        self._writable_file.close()
        return PipelineStage.return_redirect(f'/api/plugins/compare?{self._tmpfile[:-4]}')
    

class HashingBase(Plugin):
    """Hashing plugin base; also used for facedet"""
    
    def __init__(self, pmanager) -> None:
        super().__init__(pmanager)
        
        self._cache = CacheDict()
        
    def handle_filter_check(self, context):
        return False
    
    def handle_filter_item(self, context):
        return False
        
    def handle_filter(self, dbq, iid, *args):
        """Handle page"""
        query_hash = f'{dbq.query_hash}:{iid}'
        
        limit = dbq.limit
        offset = dbq.skip
        dbq.limit = 0
        dbq.raw = False

        groupby = dbq.groups
        dbq.groups = 'none'
        
        def _groups(paragraph):
            if groupby == 'group':
                groups = [
                    g for g in paragraph.keywords if g.startswith('#')] or [paragraph.source.get('url')]
            elif groupby == 'source':
                groups = [paragraph.source.get(
                    'url', paragraph.source.get('file')) or str(paragraph.id)]
            elif groupby == 'none':
                groups = [str(paragraph.id)]
            else:
                groups = [paragraph[groupby] or str(paragraph.id)]
            return groups


        pgroups = _groups(Paragraph.first(F.images == ObjectId(iid)) or Paragraph())
        groupped = {}
        
        context = DictObject(dict(sticky_paragraphs=single_item('', iid), iid=ObjectId(iid), args=args))
        
        if not self.handle_filter_check(context):
            return context.sticky_paragraphs
        
        all_results = self._cache[query_hash]
        
        if not all_results:        
            for paragraph in dbq.fetch_all_rs():
                for i in paragraph.images:
                    if i.id == iid:
                        continue
                    
                    i = self.handle_filter_item(i, context)
                    if i is None:
                        continue
                    
                    new_paragraph = Paragraph(**paragraph.as_dict())
                    new_paragraph.images = [i]
                    new_paragraph.score = i.score
                    
                    for group in _groups(paragraph):
                        if group not in pgroups and \
                            (group not in groupped or groupped[group].score >
                                new_paragraph.score):
                            groupped[group] = new_paragraph
        
            all_results = sorted(groupped.values(), key=lambda x: x.score)
            self._cache[query_hash] = all_results
            
        results = all_results[offset:offset+limit]

        return context.sticky_paragraphs + [{'spacer': 'spacer'}] + results


class Hashing(HashingBase):
    """Hashing plugin"""

    def __init__(self, pmanager):
        super().__init__(pmanager)
        
        MediaItem.set_field('dhash', bytes)
        MediaItem.set_field('whash', bytes)
        self.register_pipelines([ImageHashDuplications, ImageHash])
        self.register_filter(
            'sim', keybind='s', format_string='sim,o"{mediaitem._id}"', icon='mdi-image',
            handler=self.handle_filter)
        self.register_filter(
            'around', handler=self.handle_filter_around
        )
        self.register_filter(
            'combine', handler=self.handle_filter_combine
        )
        
        self._around_cache = CacheDict()

        app = self.pmanager.app
        
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
            return storage.serve_file(open(os.path.join(os.path.dirname(__file__), 'compare.html'), 'rb'))

        @app.route('/api/plugins/hashing-jquery.min.js')
        def _jquery_js():
            return storage.serve_file(open(os.path.join(os.path.dirname(__file__), 'jquery.min.js'), 'rb'))

    def handle_filter_check(self, context):
        image_item = MediaItem.first(F.id == ObjectId(context.iid))
        
        if not image_item.dhash:
            ImageHash().resolve_image(image_item, None)

        if not image_item.dhash:
            return False
        
        context.dha, context.dhb = to_int(image_item.dhash), to_int(image_item.whash)
        return True

    def handle_filter_item(self, i, context):
        if i.dhash is None or i.dhash == b'':
            return
        
        dha1, dhb1 = to_int(i.dhash), to_int(i.whash)
        i.score = bitcount(context.dha ^ dha1) + bitcount(context.dhb ^ dhb1)
        return i
    
    def handle_filter_combine(self, dbq, field):
        last_val = None
        last_paragraph = None
        dbq.limit = 1000
        for paragraph in dbq.fetch_all_rs():
            if paragraph[field] == last_val:
                last_paragraph.images += paragraph.images
            else:
                if last_paragraph: yield last_paragraph
                last_val = paragraph[field]
                last_paragraph = paragraph
        if last_paragraph: yield last_paragraph
        
    def handle_filter_around(self, dbq, cond, count=1):
        cond = parser.parse(cond)
        prevs = []
        nexts_count = count
        
        limit, skip = dbq.limit, dbq.skip
        dbq.limit = 0
        dbq.skip = self._around_cache[f'{cond};{skip-limit}'] or 0
        
        total = 0
        fetched = 0
        
        for paragraph in dbq.fetch_all_rs():
            fetched += 1
            if evaluateqx(cond, paragraph):
                yield from prevs
                yield paragraph
                total += len(prevs) + 1
                prevs = []
                nexts_count = count
            elif nexts_count > 0:
                nexts_count -= 1
                total += 1
                yield paragraph
            else:
                prevs = prevs[:count - 1] + [paragraph]
            if total >= limit:
                break
                
        self._around_cache[f'{cond};{skip}'] = dbq.skip + fetched
