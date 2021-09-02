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
from models import Paragraph


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


def try_download(url: str, referer: str = '', attempts: int = 3, proxies = {}) -> Union[bytes, None]:
    """Try download from url

    Args:
        url (str): url
        referer (str, optional): referer url. Defaults to ''.
        attempts (int, optional): max attempts. Defaults to 3.
        ctx (PluginContext, optional): plugin context. Defaults to None.

    Returns:
        Union[bytes, None]: response content or None if failed
    """

    buf = None
    for itry in range(attempts):
        try:
            if '://' not in url and os.path.exists(url):
                buf = open(url, 'rb').read()
            else:
                code = -1
                if isinstance(url, tuple):
                    url, referer = url
                headers = {
                    "user-agent": "Mozilla/5.1 (Windows NT 6.0) Gecko/20180101 Firefox/23.5.1", "referer": referer.encode('utf-8')}
                try:
                    r = requests.get(url, headers=headers, cookies={},
                                     proxies=proxies, verify=False, timeout=60)
                    buf = r.content
                    code = r.status_code
                except requests.exceptions.ProxyError:
                    buf = None
            if code != -1:
                break
        except Exception as ex:
            time.sleep(1)
    return buf
