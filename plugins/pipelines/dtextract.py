from datetime import datetime, timedelta

import chardet
import regex as re

from jindai.models import Paragraph
from jindai.pipeline import PipelineStage

# adapted from https://www.cnblogs.com/i-love-python/p/12763063.html

# 匹配正则表达式
matchs = {
    1: (r'\d{4}%s\d{1,2}%s\d{1,2}%s \d{1,2}%s\d{1,2}%s\d{1,2}%s', '%%Y%s%%m%s%%d%s %%H%s%%M%s%%S%s'),
    2: (r'\d{4}%s\d{1,2}%s\d{1,2}%s \d{1,2}%s\d{1,2}%s', '%%Y%s%%m%s%%d%s %%H%s%%M%s'),
    3: (r'\d{4}%s\d{1,2}%s\d{1,2}%s', '%%Y%s%%m%s%%d%s'),
    4: (r'\d{2}%s\d{1,2}%s\d{1,2}%s', '%%y%s%%m%s%%d%s'),

    # 没有年份
    5: (r'\d{1,2}%s\d{1,2}%s \d{1,2}%s\d{1,2}%s\d{1,2}%s', '%%m%s%%d%s %%H%s%%M%s%%S%s'),
    6: (r'\d{1,2}%s\d{1,2}%s \d{1,2}%s\d{1,2}%s', '%%m%s%%d%s %%H%s%%M%s'),
    7: (r'\d{1,2}%s\d{1,2}%s', '%%m%s%%d%s'),


    # 没有年月日
    8: (r'\d{1,2}%s\d{1,2}%s\d{1,2}%s', '%%H%s%%M%s%%S%s'),
    9: (r'\d{1,2}%s\d{1,2}%s', '%%H%s%%M%s'),
}

# 正则中的%s分割
splits = [
    {1: [('年', '月', '日', '点', '分', '秒'), ('-', '-', '', ':', ':', ''),
         (r'\/', r'\/', '', ':', ':', ''), (r'\.', r'\.', '', ':', ':', '')]},
    {2: [('年', '月', '日', '点', '分'), ('-', '-', '', ':', ''),
         (r'\/', r'\/', '', ':', ''), (r'\.', r'\.', '', ':', '')]},
    {3: [('年', '月', '日'), ('-', '-', ''), (r'\/', r'\/', ''), (r'\.', r'\.', '')]},
    {4: [('年', '月', '日'), ('-', '-', ''), (r'\/', r'\/', ''), (r'\.', r'\.', '')]},

    {5: [('月', '日', '点', '分', '秒'), ('-', '', ':', ':', ''),
         (r'\/', '', ':', ':', ''), (r'\.', '', ':', ':', '')]},
    {6: [('月', '日', '点', '分'), ('-', '', ':', ''),
         (r'\/', '', ':', ''), (r'\.', '', ':', '')]},
    {7: [('月', '日'), ('-', ''), (r'\/', ''), (r'\.', '')]},

    {8: [('点', '分', '秒'), (':', ':', '')]},
    {9: [('点', '分'), (':', '')]},
]


def func(parten, tp) -> None:
    re.search(parten, parten)


parten_other = r'\d+天前|\d+分钟前|\d+小时前|\d+秒前'


from _pydatetime import datetime


class TimeExtractor(PipelineStage):
    '''
    Extact Date/Time from text
    @zhs 从文本中提取日期时间
    '''

    def __init__(self, base_date=None) -> None:
        self.base_date = base_date or datetime.now()
        self.match_item = []

        if self.base_date and not isinstance(self.base_date, datetime):
            try:
                self.base_date = datetime.strptime(
                    self.base_date, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                raise 'type of base_date must be str of %Y-%m-%d %H:%M:%S or datetime'

        for item in splits:
            for num, value in item.items():
                match = matchs[num]
                for sp in value:
                    tmp = []
                    for m in match:
                        tmp.append(m % sp)
                    self.match_item.append(tuple(tmp))

    def get_time_other(self, text) -> None:
        m = re.search(r'\d+', text)
        if not m:
            return None
        num = int(m.group())
        if '天' in text:
            return self.base_date - timedelta(days=num)
        elif '小时' in text:
            return self.base_date - timedelta(hours=num)
        elif '分钟' in text:
            return self.base_date - timedelta(minutes=num)
        elif '秒' in text:
            return self.base_date - timedelta(seconds=num)

        return None

    def find_time(self, text) -> list | list[datetime] | None:
        # 格式化text为str类型
        if isinstance(text, bytes):
            encoding = chardet.detect(text)['encoding']
            text = text.decode(encoding)

        res = []
        parten = '|'.join([x[0] for x in self.match_item])

        parten = parten + '|' + parten_other
        match_list = re.findall(parten, text)
        if not match_list:
            return None
        for match in match_list:
            for item in self.match_item:
                try:
                    date = datetime.strptime(match, item[1].replace('\\', ''))
                    if date.year == 1900:
                        date = date.replace(year=self.base_date.year)
                        if date.month == 1:
                            date = date.replace(month=self.base_date.month)
                            if date.day == 1:
                                date = date.replace(day=self.base_date.day)
                    res.append(date)
                    break
                except Exception as e:
                    date = self.get_time_other(match)
                    if date:
                        res.append(date)
                        break
        if not res:
            return [self.base_date]
        return res

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        paragraph.pdate = self.find_time(paragraph.content)[0]
        return paragraph
