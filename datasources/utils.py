"""若干工具函数
"""
import statistics
import zipfile
import os
import re
import glob

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


def merge_paras(name):
    last_pdf = ''
    last_page = ''
    accumulate_content = ''
    page = ''

    for p in Paragraph.query(F.collection == name).sort(F.lang, F.pdffile, F.pdfpage, F._id):
        content = re.sub(r'\d*(.+全集|部.|篇.)(（卷.）)?\s*\d*',
                         '', p.content.strip())
        if re.search(r'^[①-⑩]', content) or not content:
            p.delete()
            continue

        if last_page != p.pdfpage:
            page = ''
        if '①' in content:
            if '①' in page or content.find('①') != content.rfind('①'):
                content = content[:content.rfind('①')]
        page += content

        if p.lang not in ('cht', 'chs'):
            accumulate_content += ' '
        accumulate_content += content
        accumulate_content = accumulate_content.strip()

        if last_pdf != p.pdffile or paragraph_finished(accumulate_content):
            if accumulate_content != content and last_p != p:
                last_p.content = accumulate_content
                last_p.save()
                Paragraph.query((F._id > last_p.id) & (
                    F._id <= p.id) & (F.pdffile == last_pdf)).delete()
                print('merge paragraphs', last_p.id, 'through', p.id)
            accumulate_content = ''
        else:
            last_p = p

        last_pdf, last_page = p.pdffile, p.pdfpage

    
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
