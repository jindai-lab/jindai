"""若干工具函数
"""
import glob
import os
import re
import statistics
import zipfile
from io import BytesIO
from typing import IO, Tuple

import hanzidentifier
import langdetect
from models import try_download


def lang_detect(s):
    """
    简易的语言检测，使用正则表达式和 hanzidentifier 弥补 langdetect 在检测中日韩文字时准确率低的问题，返回 ISO 两字母代码或 chs 或 cht。
    """
    s = re.sub('[0-9]', '', s).strip()
    
    if re.search(r"[\uac00-\ud7ff]+", s):
        return 'ko'

    if re.search(r"[\u30a0-\u30ff\u3040-\u309f]+", s):
        return 'ja'
    
    if hanzidentifier.has_chinese(s):
        if hanzidentifier.is_simplified(s):
            return 'chs'
        else:
            return 'cht'
    
    return langdetect.detect(s)


def merge_lines(lines, lang):
    """
    根据语言标识将多行文本重新分段
    """    
    
    def paragraph_finished(t):
        return t.endswith(tuple('.!?…\"。！？…—：”）'))

    lens = [len(_) for _ in lines]
    if len(lens) < 3:
        yield ('' if lang[:2] == 'ch' else ' ').join(lines)
        return

    std = abs(statistics.stdev(lens))
    maxl = max(lens)
    t = ''
    last_line = '1'
    for l in lines:
        l = l.strip()
        if not l:
            continue
        if re.search(r'^[①-⑩]', l):
            break

        if lang[:2] != 'ch':
            t += ' '
        t += l
        if len(l) < maxl - std:
            if paragraph_finished(t) or not last_line:
                yield t
                t = ''
        last_line = l.strip()

    if t:
        yield t


def expand_file_patterns(patterns : list) -> Tuple[IO, str]:
    """
    读取文件（包括压缩包内容）或网址，其中文件名可以使用 */? 通配符，网址可以使用 {num1-num2} 形式给定迭代范围
    Returns:
        Tuple[IO, str]: IO 为内容，str 为文件名或网址
    """
    for pattern in patterns:
        if pattern.startswith('https://') or pattern.startswith('http://'):
            urls = []
            iterate = re.match(r'\{(\d+\-\d+)\}', pattern)
            if iterate:
                start, end=map(int,iterate.group(1).split('-'))
                for i in range(start, end+1):
                    urls.append(pattern.replace(iterate.group(0), str(i)))
            else:
                urls = [pattern]
            for url in urls:
                yield BytesIO(try_download(url, '/'.join(url.split('/')[:-1]))), url
        else:
            if not pattern.startswith('sources/'):
                pattern = 'sources/' + pattern
            for f in glob.glob(pattern):
                if f.endswith('.zip') or f.endswith('.epub'):
                    with zipfile.ZipFile(f) as z:                    
                        for f_ in z.filelist:
                            yield z.open(f_), f + '#' + f_.filename
                elif os.path.isdir(f):
                    patterns.append(f + '/*')
                else:
                    yield open(f, 'rb'), f
