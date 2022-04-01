"""图像相关的简单处理
"""
import os
import tempfile
import traceback
from typing import Union
import numpy as np
from bson import ObjectId
from PIL import Image, ImageOps

from jindai import PipelineStage, expand_path, safe_open
from jindai.helpers import safe_import
from jindai.models import ImageItem, Paragraph, parser


class ImageOrAlbumStage(PipelineStage):
    """图像或相册相关的基类"""

    def resolve(self, paragraph: Paragraph) -> Union[Paragraph, ImageItem]:
        """处理图像或段落"""
        items = paragraph.images if paragraph.images else [
            ImageItem(paragraph)]
        for i in items:
            try:
                self.resolve_image(i, paragraph)
                i.save()
            except Exception as ex:
                self.logger(i.id, self.__class__.__name__, ex)
                self.logger(traceback.format_tb(ex.__traceback__))
        return paragraph

    def resolve_image(self, i: ImageItem, _):
        """处理图像"""
        return i


class ImagesFromSource(PipelineStage):
    """将来自 PDF 等语段的页面作为图片"""

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        i = ImageItem(**paragraph.as_dict())
        i._id = None
        paragraph.images = [i]
        return paragraph


class CheckImage(ImageOrAlbumStage):
    """获取图像基本信息
    """

    def resolve_image(self, i: ImageItem, _):
        if i.source.get('url', '').split('.')[-1].lower() in ('mp4', 'avi'):
            return None
        try:
            image = i.image
            i.width, i.height = image.width, image.height
            image.verify()
            i.save()
            return i
        except OSError:
            i.flag = 10
            i.save()
        except Exception as ex:
            self.logger(i.id, ex)
        return i


class ImageGrayScale(ImageOrAlbumStage):
    """图像灰度化
    """

    def resolve_image(self, i: ImageItem, _):
        i.image = i.image.convert('L')
        return i


class ImageBW(ImageOrAlbumStage):
    """图像简单二值化
    """

    def __init__(self, threshold=128):
        """
        Args:
            threshold (int): 二值化的阈值
        """
        super().__init__()
        assert 0 < threshold < 256
        self.table = [0] * int(threshold) + [1] * (256-int(threshold))

    def resolve_image(self, i: ImageItem, _):
        i.image = i.image.point(self.table, '1')
        return i


class ImageBWAdaptive(ImageOrAlbumStage):
    """图像自适应二值化
    """

    def __init__(self, block_size=35, offset=10):
        """
        Args:
            block_size (int): 区域大小
            offset (int): 偏差值
        """
        super().__init__()
        self.block_size = block_size
        self.offset = offset

    def resolve_image(self, i: ImageItem, _):
        safe_import('skimage', 'scikit-image')
        threshold_local = safe_import('skimage.filter').threshold_local
        img = np.array(i.image.convert('L'))
        thr = threshold_local(img, self.block_size, offset=self.offset)
        img = img > thr
        i.image = Image.fromarray(img)
        return i


class ImageEnhance(ImageOrAlbumStage):
    """图像增强
    """

    def __init__(self, method, args):
        """
        Args:
            method (autocontrast|colorize|crop|deform|equalize|exif_transpose|expand|fit|flip|grayscale|invert|mirror|operator|pad|posterize|scale|solarize): 要调用的方法
            args (str): 调用的参数，以回车分隔，类型将自动转换
        """
        super().__init__()
        self.method = method
        self.args = [parser.parse_literal(
            l) for l in args.split('\n')] if args else []

    def resolve_image(self, i: ImageItem, _):
        i.image = getattr(ImageOps, self.method)(i.image, *self.args)
        return i


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
        super().__init__()
        self.proportional = proportional
        self.percentage = 0 <= max_height < 1 and 0 <= max_width < 1
        self.max_width, self.max_height = max_width, max_height

    def resolve_image(self, i: ImageItem, _):
        width, height = self.max_width, self.max_height
        if self.percentage:
            width, height = int(
                width * i.image.width), int(height * i.image.height)
        if self.proportional:
            i.image.thumbnail((width, height))
        else:
            i.image = i.image.resize((width, height))
        return i


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
        super().__init__()
        self.box = (top, left, right, bottom)
        self.percentage = 0 <= top < 1 and 0 <= left < 1 and 0 <= right < 1 and 0 <= bottom < 1

    def resolve_image(self, i: ImageItem, _):
        width, height = i.image.size
        top, left, right, bottom = self.box
        if self.percentage:
            top, left, right, bottom = int(
                top * height), int(left * width), int(right * width), int(bottom * height)
        i.image = i.image.crop((left, top, width - right, height - bottom))
        return i


class ImageRotate(ImageOrAlbumStage):
    """图像旋转
    """

    def __init__(self, degree=90):
        """
        Args:
            degree (int): 旋转的角度，应为90、180、270中的一个
        """
        super().__init__()
        assert degree in (90, 180, 270, 0), "Degree must in (0, 90, 180, 270)"
        self.degree = degree

    def resolve_image(self, i: ImageItem, _):
        i.image = i.image.rotate(90, Image.NEAREST, expand=1)
        return i


class DumpImages(ImageOrAlbumStage):
    """保存图像
    """

    def __init__(self, folder):
        """
        Args:
            folder (str): 文件夹名称
        """
        super().__init__()
        self.folder = folder
        if not os.path.exists(self.folder):
            os.makedirs(self.folder, exist_ok=True)

    def resolve_image(self, i: ImageItem, _):
        path = os.path.basename(i.source.get(
            'url', i.source.get('file', ''))).lower()
        if path.endswith('.pdf'):
            path = path[:-4] + f'_{i.pdfpage:04d}.jpg'
        path = os.path.join(self.folder, path)
        i.image.save(path)
        delattr(i, 'image')
        return i


class DownloadImages(ImageOrAlbumStage):
    """下载图像
    """

    def __init__(self, proxy='') -> None:
        """
        Args:
            proxy (str): 代理服务器
        """
        super().__init__()
        self.proxies = {
            'http': proxy, 'https': proxy
        } if proxy else {}

    def resolve_image(self, i: ImageItem, post):
        if not i.id:
            i.save()

        try:
            content = safe_open(i.source['url'], referer=post.source.get(
                'url', ''), proxies=self.proxies).read()
            assert content
        except Exception:
            return

        with safe_open(f'hdf5://{i.id}', 'wb') as output:
            output.write(content)
            self.logger(i.id, len(content))

        i.source = {'file': 'blocks.h5', 'url': i.source['url']}
        i.flag = 0


class QRCodeScanner(PipelineStage):
    """获取图像中的二维码信息
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


class VideoFrame(ImageOrAlbumStage):
    """获取视频中的某一帧
    """

    def __init__(self, frame_num=0.5, field='thumbnail') -> None:
        """
        Args:
            frame_num (float): 大于等于一的帧数，或表示时长占比的0-1之间的浮点数
            field (str): 写入到字段名
        """
        super().__init__()
        self.frame_num = frame_num
        self.field = field
        self.cv2 = safe_import('cv2', 'opencv-python-headless')

    def resolve_image(self, i: ImageItem, _):
        cv2 = self.cv2
        thumb = f'{ObjectId()}.thumb.jpg'
        temp_file = tempfile.mktemp()
        read_from = ''

        # check file type by extension name
        ext = (i.source.get('url') or i.source.get(
            'file') or '.').rsplit('.', 1)[-1]
        if ext.lower() not in ('mp4', 'avi'):
            self.logger(f'skip {ext} data')
            return

        try:
            # read video data
            filename = i.source.get('file')
            if filename == 'blocks.h5':
                with safe_open(temp_file, 'wb') as output:
                    blen = output.write(i.image_raw.read())
                if not blen:
                    self.logger(f'unable to fetch data from blocks.h5: {i.id}')
                    os.unlink(temp_file)
                    return
                read_from = temp_file
            else:
                read_from = expand_path(filename)

            if not os.path.exists(read_from):
                self.logger(f'{read_from} not found')
                return

            # generate video thumbnail
            cap = cv2.VideoCapture(read_from)
            frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) *
                            self.frame_num) if self.frame_num < 1 else int(self.frame_num)

            if cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                rval, frame = cap.read()
                cap.release()
                if rval:
                    # write to hdf5
                    rval, npa = cv2.imencode('.jpg', frame)
                    pic = npa.tobytes()
                    with safe_open(f'hdf5://{thumb}', 'wb') as output:
                        output.write(pic)

                    setattr(i, self.field, thumb)
                    i.save()
                    self.logger(
                        f'wrote {temp_file} frame#{frame_num} to {thumb}')

        except Exception as ex:
            self.logger(ex)

        # clear up temp file
        if os.path.exists(temp_file):
            os.unlink(temp_file)

        return i
