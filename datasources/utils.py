"""若干工具函数
"""
import statistics
from typing import Union
import zipfile
import os
import re
import glob
import time
import requests
from io import BytesIO
from models import Paragraph, try_download


def paragraph_finished(t):

    def _endswith(heap, needles):
        for _ in needles:
            if heap.endswith(_):
                return True
        return False

    return _endswith(t.strip(), '.!?…\"。！？…—：”')


def merge_lines(lines, lang):
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


def expand_file_patterns(patterns):
    for pattern in patterns:
        if pattern.startswith('https://') or pattern.startswith('http://'):
            yield BytesIO(try_download(pattern, '/'.join(pattern.split('/')[:-1]))), pattern
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
