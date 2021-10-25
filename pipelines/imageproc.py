"""图像相关的简单处理
"""

from storage import StorageManager
from pipeline import PipelineStage
import numpy as np
from PIL import Image, ImageOps
import os
from models import parser, try_download

class ImageGrayScale(PipelineStage):
    """图像灰度化
    """

    def resolve(self, p):
        p.image = p.image.convert('L')
        return p


class ImageBW(PipelineStage):
    """图像简单二值化
    """

    def __init__(self, threshold=128):
        """
        Args:
            threshold (int): 二值化的阈值
        """
        assert 0 < threshold < 256
        self.table = [0] * int(threshold) + [1] * (256-int(threshold))

    def resolve(self, p):
        p.image = p.image.point(self.table, '1')
        return p


class ImageBWAdaptive(PipelineStage):
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

    def resolve(self, p):
        from skimage.filters import threshold_local
        img = np.array(p.image.convert('L'))
        thr = threshold_local(img, self.block_size, offset=self.offset)
        img = img > thr
        p.image = Image.fromarray(img)
        return p


class ImageEnhance(PipelineStage):
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

    def resolve(self, p):
        p.image = getattr(ImageOps, self.method)(p.image, *self.args)
        return p


class ImageResize(PipelineStage):
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

    def resolve(self, p):
        w, h = self.max_width, self.max_height
        if self.percentage:
            w, h = int(w * p.image.width), int(h * p.image.height)
        if self.proportional:
            p.image.thumbnail((w, h))
        else:
            p.image = p.image.resize((w, h))
        return p


class ImageCrop(PipelineStage):
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

    def resolve(self, p):
        width, height = p.image.size
        top, left, right, bottom = self.box
        if self.percentage:
            top, left, right, bottom = int(top * height), int(left * width), int(right * width), int(bottom * height)
        p.image = p.image.crop((left, top, width - right, height - bottom))
        return p

    
class ImageRotate(PipelineStage):
    """图像旋转
    """

    def __init__(self, degree=90):
        """
        Args:
            degree (int): 旋转的角度，应为90、180、270中的一个
        """
        assert degree in (90, 180, 270, 0), "Degree must in (0, 90, 180, 270)"
        self.degree = degree

    def resolve(self, p):
        p.image = image.rotate(90, Image.NEAREST, expand=1)
        return p


class DumpImages(PipelineStage):
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

    def resolve(self, p):
        f = os.path.basename(p.pdffile).lower()
        if f.endswith('.pdf'):
            f = f[:-4] + f'_{p.pdfpage:04d}.jpg'
        f = os.path.join(self.folder, f)
        p.image.save(f)
        delattr(p, 'image')
        return p
    
    
class DownloadImages(PipelineStage):
    """根据 image_storage.url 下载图像
    """
    
    def __init__(self) -> None:
        self.mgr = StorageManager()
    
    def resolve(self, p):
        if not p.id:
            p.save()
        print(p.id, p.image_storage)
        if not p.image_storage or 'url' not in p.image_storage:
            return
        content = try_download(p.image_storage['url'], p.source.get('url', ''))
        if not content: return    
        with self.mgr:
            self.mgr.write(content, str(self.id))
            print(self.id, len(content))
        p.image_storage['blocks'] = True
        p.save()
        return p
