"""图像相关的简单处理
"""

import os
import re
from queue import deque
from typing import Union

import numpy as np
from models import Album, F, ImageItem, parser, try_download, AutoTag
from PIL import Image, ImageOps
from pipeline import PipelineStage
from plugins.hashing import dhash, whash
from storage import StorageManager
import traceback


class ImageOrAlbumStage(PipelineStage):
    
    def resolve(self, p : Union[Album, ImageItem]) -> Union[Album, ImageItem]:
        if isinstance(p, Album):
            items = p.items
        else:
            items = [p]
        for i in items:
            try:
                self.resolve_image(i)
            except Exception as ex:
                self.logger(i.id, self.__class__.__name__, ex)
                self.logger(traceback.format_tb(ex.__traceback__))
                p = None
        return p

    def resolve_image(self, i : ImageItem):
        return i


class CheckImage(ImageOrAlbumStage):
    """获取图像基本信息
    """

    def resolve_image(self, p : ImageItem):
        try:
            im = p.image
            p.width, p.height = im.width, im.height
            im.verify()
            p.save()
            return p
        except OSError:
            p.flag = 10
            p.save()
        except Exception as ex:
            self.logger(p.id, ex)


class ImageHash(ImageOrAlbumStage):
    """建立图像哈希检索
    """

    def resolve_image(self, i : ImageItem):
        try:
            dh, wh = i.dhash, i.whash
            if dh and wh: return i

            try: f = i.image_raw
            except: f = None
            if not f: return

            if not dh:
                dh = dhash(f) or ''
            if not wh:
                wh = whash(f) or ''

            i.dhash, i.whash = dh, wh
        except (IOError, AssertionError):
            pass
        i.save()


class ImageHashDuplications(ImageOrAlbumStage):
    """进行图像哈希去重
    """
    
    def __init__(self) -> None:
        self.results = deque()
        self.result_pairs = set()

    def _unv(self, x):
        from bson import binary
        return binary.Binary(bytes.fromhex(f'{x:016x}'))

    def resolve_image(self, i: ImageItem):
        from plugins.hashing import bitcount, flips, v        
        if not i.dhash: return
        dh2 = v(i.dhash)
        dhb2 = v(i.whash)
        h2, w2 = i.height, i.width
        for j in ImageItem.query(F.dhash.in_(
                [self._unv(x) for x in [dh2] + list(flips(dh2, 1)) + list(flips(dh2, 2))])):
            id1 = j.id
            if id1 == i.id or f'{i.id}-{id1}' in self.result_pairs or f'{id1}-{i.id}' in self.result_pairs: continue
            self.result_pairs.add(f'{id1}-{i.id}')
            a, b = i.id, id1
            if j.width * j.height < w2 * h2: b, a = a, b
            r = '{}\t{}\t{}'.format(a, b, bitcount(v(i.dhash) ^ dh2) + bitcount(v(j.whash) ^ dhb2))
            self.logger(r)
            self.results.append(r + '\n')
        return i

    def summarize(self, r):
        self.fo = open('compare.tsv', 'w')
        for l in self.results:
            self.fo.write(l)
        self.fo.close()
        return {'redirect': '/api/gallery/compare'}
        

class ImageGrayScale(ImageOrAlbumStage):
    """图像灰度化
    """

    def resolve_image(self, p):
        p.image = p.image.convert('L')
        return p


class ImageBW(ImageOrAlbumStage):
    """图像简单二值化
    """

    def __init__(self, threshold=128):
        """
        Args:
            threshold (int): 二值化的阈值
        """
        assert 0 < threshold < 256
        self.table = [0] * int(threshold) + [1] * (256-int(threshold))

    def resolve_image(self, p):
        p.image = p.image.point(self.table, '1')
        return p


class ImageBWAdaptive(ImageOrAlbumStage):
    """图像自适应二值化
    """

    def __init__(self, block_size=35, offset=10):
        """
        Args:
            block_size (int): 区域大小
            offset (int): 偏差值
        """
        self.block_size = block_size
        self.offset = offset

    def resolve_image(self, p):
        from skimage.filters import threshold_local
        img = np.array(p.image.convert('L'))
        thr = threshold_local(img, self.block_size, offset=self.offset)
        img = img > thr
        p.image = Image.fromarray(img)
        return p


class ImageEnhance(ImageOrAlbumStage):
    """图像增强
    """
    
    def __init__(self, method, args):
        """
        Args:
            method (autocontrast|colorize|crop|deform|equalize|exif_transpose|expand|fit|flip|grayscale|invert|mirror|operator|pad|posterize|scale|solarize): 要调用的方法
            args (str): 调用的参数，以回车分隔，类型将自动转换
        """
        self.method = method
        self.args = [parser.expand_literal(l) for l in args.split('\n')] if args else []

    def resolve_image(self, p):
        p.image = getattr(ImageOps, self.method)(p.image, *self.args)
        return p


class ImageResize(ImageOrAlbumStage):
    """调整大小
    """

    def __init__(self, max_width, max_height, proportional=True):
        """
        Args:
            max_width (float): 最大宽度或0-1之间的比例，下同
            max_height (float): 最大高度
            proportional (bool): 是否等比例放大缩小
        """
        self.proportional = proportional
        self.percentage = 0 <= max_height < 1 and 0 <= max_width < 1
        self.max_width, self.max_height = max_width, max_height

    def resolve_image(self, p):
        w, h = self.max_width, self.max_height
        if self.percentage:
            w, h = int(w * p.image.width), int(h * p.image.height)
        if self.proportional:
            p.image.thumbnail((w, h))
        else:
            p.image = p.image.resize((w, h))
        return p


class ImageCrop(ImageOrAlbumStage):
    """图像切割
    """

    def __init__(self, top, left, right, bottom):
        """
        Args:
            top (float): 上端距离（像素或0-1之间的比例，下同）
            bottom (float): 下端距离
            left (float): 左端距离
            right (float): 右端距离
        """
        self.box = (top, left, right, bottom)
        self.percentage = 0 <= top < 1 and 0 <= left < 1 and 0 <= right < 1 and 0 <= bottom < 1

    def resolve_image(self, p):
        width, height = p.image.size
        top, left, right, bottom = self.box
        if self.percentage:
            top, left, right, bottom = int(top * height), int(left * width), int(right * width), int(bottom * height)
        p.image = p.image.crop((left, top, width - right, height - bottom))
        return p

    
class ImageRotate(ImageOrAlbumStage):
    """图像旋转
    """

    def __init__(self, degree=90):
        """
        Args:
            degree (int): 旋转的角度，应为90、180、270中的一个
        """
        assert degree in (90, 180, 270, 0), "Degree must in (0, 90, 180, 270)"
        self.degree = degree

    def resolve_image(self, p):
        p.image = p.image.rotate(90, Image.NEAREST, expand=1)
        return p


class DumpImages(ImageOrAlbumStage):
    """保存图像
    """

    def __init__(self, folder):
        """
        Args:
            folder (str): 文件夹名称
        """
        self.folder = folder
        if not os.path.exists(self.folder):
            os.makedirs(self.folder, exist_ok=True)

    def resolve_image(self, p):
        f = os.path.basename(p.source.get('url', p.source.get('file', ''))).lower()
        if f.endswith('.pdf'):
            f = f[:-4] + f'_{p.pdfpage:04d}.jpg'
        f = os.path.join(self.folder, f)
        p.image.save(f)
        delattr(p, 'image')
        return p
    
    
class DownloadImages(PipelineStage):
    """根据 ImageItem 的 source['url'] 下载图像
    """
    
    def __init__(self, proxy='') -> None:
        """
        Args:
            proxy (str): 代理服务器
        """
        self.mgr = StorageManager()
        self.proxies = {
            'http': proxy, 'https': proxy
        } if proxy else {}
    
    def resolve(self, p):
        items = p.items if isinstance(p, Album) else [p]

        for i in items:
            if not i.id:
                i.save()
                
            content = try_download(i.source['url'], p.source.get('url', ''), proxies=self.proxies)
            if not content: return
            with self.mgr:
                self.mgr.write(content, str(i.id))
                self.logger(i.id, len(content))
            i.source['file'] = 'blocks.h5'
            i.save()

        return p


class ApplyAutoTags(PipelineStage):
    """应用自动标签设置
    """
    
    def __init__(self) -> None:
        self.ats = list(AutoTag.query({}))
    
    def resolve(self, p : Album) -> Album:
        for i in self.ats:
            pattern, from_tag, tag = i.pattern, i.from_tag, i.tag
            if (from_tag and from_tag in p.keywords) or (pattern and re.search(pattern, p.source['url'])):
                if tag not in p.keywords:
                    p.keywords.append(tag)
                if tag.startswith('@'):
                    p.author = tag
        p.save()
        return p

