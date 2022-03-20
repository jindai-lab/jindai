"""若干工具函数
"""
import glob
import os
import re
import zipfile
from io import BytesIO
from typing import IO, Tuple

from models import try_download


def expand_file_patterns(patterns : list, names_only = False) -> Tuple[IO, str]:
    """
    读取文件（包括压缩包内容）或网址，其中文件名可以使用 */? 通配符，网址可以使用 {num1-num2} 形式给定迭代范围
    Returns:
        Tuple[IO, str]: IO 为内容，str 为文件名或网址
    """
            
    for pattern in patterns:
        if pattern.startswith('https://') or pattern.startswith('http://'):
            urls = []
            iterate = re.search(r'\{(\d+\-\d+)\}', pattern)
            if iterate:
                start, end=map(int,iterate.group(1).split('-'))
                for i in range(start, end+1):
                    urls.append(pattern.replace(iterate.group(0), str(i)))
            else:
                urls = [pattern]
            for url in urls:
                if names_only:
                    yield url
                else:
                    yield BytesIO(try_download(url, '/'.join(url.split('/')[:-1]))), url
        else:
            if not pattern.startswith('sources/'):
                pattern = 'sources/' + pattern
            for f in glob.glob(pattern):
                if not names_only and f.endswith('.zip') or f.endswith('.epub'):
                    with zipfile.ZipFile(f) as z:                    
                        for f_ in z.filelist:
                            yield z.open(f_), f + '#' + f_.filename
                elif os.path.isdir(f):
                    patterns.append(f + '/*')
                else:
                    if names_only:
                        yield f
                    else:
                        yield open(f, 'rb'), f
