"""
Basic processing for multimedia materials
@zhs 多媒体相关的简单处理
"""
import os
import tempfile
import traceback
import requests
from io import BytesIO
from typing import Union

import numpy as np
from jindai import PipelineStage, parser, storage
from jindai.helpers import safe_import
from jindai.models import MediaItem, Paragraph
from PIL import Image, ImageOps
from PyMongoWrapper import ObjectId


class VideoItemImageDelegate:
    
    def __init__(self, item) -> None:
        self._i = item
    
    @property
    def source(self):
        return {'file': self._i.thumbnail}
    
    @property
    def data_path(self):
        return self._i.thumbnail
    
    @property
    def data(self):
        return storage.open(self._i.thumbnail)
    
    def save(self):
        self._i.save()
        
    def __getattribute__(self, __name: str):
        if __name not in ('_i', 'source', 'save', 'data', 'data_path'):
            return getattr(self._i, __name)
        
        return object.__getattribute__(self, __name)
    
    def __setattr__(self, __name: str, __value) -> None:
        if __name != '_i':
            setattr(self._i, __name, __value)
        elif __name not in ('source', 'save', 'data', 'data_path'):
            object.__setattr__(self, __name, __value)


class MediaItemStage(PipelineStage):
    """Base class for pipeline stages relating to media processing"""

    def resolve(self, paragraph: Paragraph) -> Union[Paragraph, MediaItem]:
        """Resolve media item or paragraph"""
        items = paragraph.images if paragraph.images else [
            MediaItem(paragraph)]
        for i in items:
            try:
                self.resolve_item(i, paragraph)
                
                if i.item_type == 'video' and i.thumbnail:
                    i_img = VideoItemImageDelegate(i)
                    self.resolve_image(i_img, paragraph)
                    i_img.save()

                getattr(self, f'resolve_{i.item_type or "image"}')(
                    i, paragraph)
                i.save()
            except Exception as ex:
                self.logger(i.id, self.__class__.__name__, ex)
                self.logger(traceback.format_tb(ex.__traceback__))
        return paragraph

    def resolve_item(self, i: MediaItem, paragraph: Paragraph):
        """Resolve item, regardless of its type"""

    def resolve_image(self, i: MediaItem, paragraph: Paragraph):
        """Resolve image item"""

    def resolve_video(self, i: MediaItem, paragraph: Paragraph):
        """Resolve video item"""

    def resolve_audio(self, i: MediaItem, paragraph: Paragraph):
        """Resolve audio item"""


class ImagesFromSource(PipelineStage):
    """Use paragraph as images from PDF
    @zhs 将来自 PDF 等语段的页面作为图片"""

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        i = MediaItem(**paragraph.as_dict())
        i._id = None
        paragraph.images = [i]
        return paragraph


class CheckImage(MediaItemStage):
    """
    Get Basic Info for Images
    @zhs 获取图像基本信息
    """

    def resolve_image(self, i: MediaItem, _):
        if i.source.get('url', '').split('.')[-1].lower() in ('mp4', 'avi'):
            return None
        try:
            image = i.image
            i.width, i.height = image.width, image.height
            image.verify()
            i.save()
            return i
        except Exception as ex:
            self.logger(i.id, type(ex).__name__, ex)
        return i


class ImageGrayScale(MediaItemStage):
    """
    Image Grayscalize
    @zhs 图像灰度化
    """

    def resolve_image(self, i: MediaItem, _):
        i.image = i.image.convert('L')
        return i


class ImageBW(MediaItemStage):
    """
    Image Binaryzation with Threshold
    @zhs 图像简单二值化
    """

    def __init__(self, threshold=128):
        """
        Args:
            threshold (int): 二值化的阈值
        """
        super().__init__()
        assert 0 < threshold < 256
        self.table = [0] * int(threshold) + [1] * (256-int(threshold))

    def resolve_image(self, i: MediaItem, _):
        i.image = i.image.point(self.table, '1')
        return i


class ImageBWAdaptive(MediaItemStage):
    """
    Adapative Binaryzation for Images
    @zhs 图像自适应二值化
    """

    def __init__(self, block_size=35, offset=10):
        """
        Args:
            block_size (int): Block size
                @zhs 区域大小
            offset (int): Offset
                @zhs 偏差值
        """
        super().__init__()
        self.block_size = block_size
        self.offset = offset

    def resolve_image(self, i: MediaItem, _):
        safe_import('skimage', 'scikit-image')
        threshold_local = safe_import('skimage.filter').threshold_local
        img = np.array(i.image.convert('L'))
        thr = threshold_local(img, self.block_size, offset=self.offset)
        img = img > thr
        i.image = Image.fromarray(img)
        return i


class ImageEnhance(MediaItemStage):
    """
    Enhance images
    @zhs 图像增强
    """

    def __init__(self, method, args):
        """
        Args:
            method (autocontrast|colorize|crop|deform|equalize|exif_transpose|expand|fit|flip|grayscale|invert|mirror|operator|pad|posterize|scale|solarize): 
                Method to call
                @zhs 要调用的方法
            args (str): Arguments, split by lines
                @zhs 调用的参数，以回车分隔，类型将自动转换
        """
        super().__init__()
        self.method = method
        self.args = [parser.parse(
            l) for l in args.split('\n')] if args else []

    def resolve_image(self, i: MediaItem, _):
        i.image = getattr(ImageOps, self.method)(i.image, *self.args)
        return i


class ImageResize(MediaItemStage):
    """Resize image
    @zhs 调整大小
    """

    def __init__(self, max_width, max_height, proportional=True):
        """
        Args:
            max_width (float): Max width, in pixels (>=1) or ratio (0-1)
                @zhs 最大宽度或0-1之间的比例，下同
            max_height (float): Max height
                @zhs 最大高度
            proportional (bool): Keep proportion
                @zhs 是否等比例放大缩小
        """
        super().__init__()
        self.proportional = proportional
        self.percentage = 0 <= max_height < 1 and 0 <= max_width < 1
        self.max_width, self.max_height = max_width, max_height

    def resolve_image(self, i: MediaItem, _):
        width, height = self.max_width, self.max_height
        if self.percentage:
            width, height = int(
                width * i.image.width), int(height * i.image.height)
        if self.proportional:
            i.image.thumbnail((width, height))
        else:
            i.image = i.image.resize((width, height))
        return i


class ImageCrop(MediaItemStage):
    """Crop images
    @zhs 图像切割
    """

    def __init__(self, top, left, right, bottom):
        """
        Args:
            top (float): Top, in pixels (>=1) or ratio (0-1)
                @zhs 上端距离（像素或0-1之间的比例，下同）
            bottom (float): Bottom
                @zhs 下端距离
            left (float): Left
                @zhs 左端距离
            right (float): Right
                @zhs 右端距离
        """
        super().__init__()
        self.box = (top, left, right, bottom)
        self.percentage = 0 <= top < 1 and 0 <= left < 1 and 0 <= right < 1 and 0 <= bottom < 1

    def resolve_image(self, i: MediaItem, _):
        width, height = i.image.size
        top, left, right, bottom = self.box
        if self.percentage:
            top, left, right, bottom = int(
                top * height), int(left * width), int(right * width), int(bottom * height)
        i.image = i.image.crop((left, top, width - right, height - bottom))
        return i


class ImageRotate(MediaItemStage):
    """Rotate Image
    @zhs 图像旋转
    """

    def __init__(self, degree=90):
        """
        Args:
            degree (int): Rotate by degree, one of 0, 90, 180, and 270.
                @zhs 旋转的角度，应为90、180、270中的一个
        """
        super().__init__()
        assert degree in (90, 180, 270, 0), "Degree must in (0, 90, 180, 270)"
        self.degree = degree

    def resolve_image(self, i: MediaItem, _):
        i.image = i.image.rotate(90, Image.NEAREST, expand=1)
        return i


class DumpImages(MediaItemStage):
    """Save Images to Folder
    @zhs 保存图像
    """

    def __init__(self, folder):
        """
        Args:
            folder (str): Folder name
                @zhs 文件夹名称
        """
        super().__init__()
        self.folder = folder
        if not os.path.exists(self.folder):
            os.makedirs(self.folder, exist_ok=True)

    def resolve_image(self, i: MediaItem, _):
        path = os.path.basename(i.source.get(
            'url', i.source.get('file', ''))).lower()
        if path.endswith('.pdf'):
            path = path[:-4] + f'_{i.pdfpage:04d}.jpg'
        path = os.path.join(self.folder, path)
        i.image.save(path)
        delattr(i, 'image')
        return i


class DownloadMedia(MediaItemStage):
    """Download media items
    @zhs 下载媒体内容
    """

    def __init__(self, proxy='') -> None:
        """
        Args:
            proxy (str): Proxy server
                @zhs 代理服务器
        """
        super().__init__()
        self.proxies = {
            'http': proxy, 'https': proxy
        } if proxy else {}

    def resolve_item(self, i: MediaItem, post):
        if not i.id:
            i.save()
            
        if i.no_download:
            return
        
        if 'file' in i.source:
            return

        try:
            resp = requests.get(i.source['url'],
                headers={
                    'Referer': post.source.get('url', '')
                },
                verify=False,
                proxies=self.proxies)
            
            if resp.status_code != 200:
                raise OSError(f'HTTP {resp.status_code}')
            if not resp.content:
                raise ValueError(f'Empty response')
            
            content = resp.content
        except Exception as ex:
            if isinstance(ex, OSError):
                i.no_download = True
                i.save()
            self.log_exception(
                f'Error while downloading item {i.source.url}',
                ex)
            return

        path = storage.default_path(i.id)
        with storage.open(path, 'wb') as output:
            output.write(content)
            self.logger(i.id, len(content))

        i.source = {'file': path, 'url': i.source['url']}
        i.data = BytesIO(content)
        i.save()
        
        
class QRCodeScanner(PipelineStage):
    """Read QR-Code info from image
    @zhs 获取图像中的二维码信息
    """

    def __init__(self) -> None:
        super().__init__()
        cv2 = safe_import('cv2', 'opencv-python-headless')
        self.qr_detector = cv2.QRCodeDetector()

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        paragraph.qrcodes = []
        for i in paragraph.images:
            try:
                image = np.asarray(i.image)
            except Exception:
                continue
            data, vertices_array, _ = self.qr_detector.detectAndDecode(image)
            if vertices_array is not None:
                try:
                    data = data.decode('utf-8')
                except Exception:
                    data = 'bindata:' + data.hex()
                paragraph.qrcodes.append(data)
        if paragraph.qrcodes:
            paragraph.save()
        else:
            del paragraph.qrcodes


class VideoFrame(MediaItemStage):
    """
    Get a frame from video
    @zhs 获取视频中的某一帧
    """

    def __init__(self, frame_num=0.5, field='thumbnail') -> None:
        """
        Args:
            frame_num (float): Frame #, or ratio
                @zhs 大于等于一的帧数，或表示时长占比的0-1之间的浮点数
            field (str): Write storage ID to field
                @zhs 写入到字段名
        """
        super().__init__()
        self.frame_num = frame_num
        self.field = field
        self.cv2 = safe_import('cv2', 'opencv-python-headless')

    def get_video_frame(self, buf, frame=0.5):
        cv2 = self.cv2
        use_temp = False
        read_from = tempfile.mktemp('.mp4')
        pic = b''

        # read video data
        if isinstance(buf, str):
            read_from = buf
        elif hasattr(buf, 'name'):
            read_from = buf.name
        elif hasattr(buf, 'read'):
            use_temp = True
            with open(read_from, 'wb') as fo:
                fo.write(buf.read())

        if not isinstance(read_from, str) or not os.path.exists(read_from):
            self.logger(f'{read_from} not found')
            return
        
        try:
            self.logger(f'generate frame image from {read_from}')
            cap = cv2.VideoCapture(read_from)
            frame = float(frame)
            frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) *
                            frame if frame < 1 else frame)

            if cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                rval, frame = cap.read()
                cap.release()
                if rval:
                    rval, npa = cv2.imencode('.jpg', frame)
                    pic = npa.tobytes()

        except Exception as ex:
            self.log_exception('Error while generating thumbnail from video', ex)

        if use_temp and os.path.exists(read_from):
            os.unlink(read_from)

        return BytesIO(pic)

    def resolve_video(self, i: MediaItem, _):
        thumb = storage.default_path(f'{i.id}.thumb.jpg')
        
        # generate video thumbnail
        read_from = i.data_path + f'#videoframe/{self.frame_num}'
        pic = storage.open(read_from, 'rb').read()
        if pic:
            with storage.open(thumb, 'wb') as output:
                output.write(pic)
            setattr(i, self.field, thumb)
            i.save()
            self.logger(
                f'wrote {i.id} frame#{self.frame_num} to {thumb}')
        else:
            self.logger('cannot read from', read_from)

        return i


storage.register_fragment_handler('videoframe', VideoFrame().get_video_frame)


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
