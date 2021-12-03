"""图像相关的简单处理
"""

from storage import StorageManager
from pipeline import PipelineStage
import numpy as np
from PIL import Image, ImageOps
from typing import Union
import os
from models import ImageItem, Album, F, parser, try_download
from plugins.hashing import dhash, whash
from collections import defaultdict
from tqdm import tqdm


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
                self.logger(i.id, ex)
                p = None
        return p

    def resolve_image(self, i : ImageItem):
        return i


class CheckImage(ImageOrAlbumStage):
    """获取图像基本信息
    """

    def resolve_image(self, p : ImageItem):
        try:
            buf = p.read_image()
            im = Image.open(buf)
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
        if not i.storage: return
        try:
            dh, wh = i.dhash, i.whash
            if dh and wh: return i

            try: f = i.read_image()
            except: f = None
            if not f: return

            if not dh:
                f.seek(0)
                im = Image.open(f)
                dh = dhash(im) or ''
            if not wh:
                f.seek(0)
                im = Image.open(f)
                wh = whash(im) or ''

            i.dhash, i.whash = dh, wh
            i.save()
        except IOError:
            pass
        except AssertionError:
            i.storage = None
            i.save()


class ImageHashDuplications(ImageOrAlbumStage):
    """进行图像哈希去重
    """
    
    def __init__(self) -> None:
        from plugins.hashing import v
        self.d2 = defaultdict(list)
    
        for i in tqdm(ImageItem.query(F.dhash.exists(1) & ~F.dhash.empty() & F.flag.eq(0) & (F.width > 0) & F.url.regex(r'\.(jpe?g|gif|png|tiff)$'))):
            if not i.dhash: continue
            dha = v(i.dhash)
            dhb = v(i.whash)
            self.d2[dha].append((i.id, i.width, i.height, dhb))
        
        self.fo = open('compare.tsv', 'w')

    def resolve_image(self, i: ImageItem):
        from plugins.hashing import v, flips, bitcount
        if not i.dhash: return
        dh2 = v(i.dhash)
        if dh2 not in self.d2: return
        ls2 = self.d2[dh2]
        for id2, w2, h2, dhb2 in ls2:
            for dh1, sc in [(dh2, 0)] + list(zip(flips(dh2, 1), [1] * 64)) + list(zip(flips(dh2, 2), [2] * 2080)):
                if dh1 not in self.d2: continue
                for id1, w1, h1, dhb1 in self.d2[dh1]:
                    if id1 >= id2 or w1 == 0: continue
                    a, b = id2, id1
                    if w1 * h1 < w2 * h2: b, a = a, b
                    r = '{}\t{}\t{}'.format(a, b, sc + bitcount(dhb1 ^ dhb2))
                    self.logger(r)
                    self.fo.write(r + '\n')
        return i

    def summarize(self, r):
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
    
    def __init__(self) -> None:
        self.mgr = StorageManager()
    
    def resolve(self, p):
        items = [p] if isinstance(p, ImageItem) else p.items if isinstance(p, Album) else []

        for i in items:
            if not i.id:
                i.save()
                
            content = try_download(i.source['url'], p.source.get('url', ''))
            if not content: return
            with self.mgr:
                self.mgr.write(content, str(i.id))
                self.logger(i.id, len(content))
            i.storage = True
            i.save()

        return p
