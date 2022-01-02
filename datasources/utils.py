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

def paragraph_finished(t):
    return t.endswith(tuple('.!?…\"。！？…—：”）'))


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


def expand_file_patterns(patterns : list) -> Tuple[IO, str]:
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
