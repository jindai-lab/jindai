import requests
import re
from concurrent.futures import ThreadPoolExecutor

from jindai.pipeline import DataSourceStage
from jindai.models import Paragraph, MediaItem


headers = {
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    'accept-encoding': 'gzip, deflate, br',
    'accept-language': 'zh-CN,zh;q=0.9',
    'cookie': 'PHPSESSID=36c8n4lsbb8u63glevh1ksc9a1; webp_enabled=1; _ga=GA1.2.1167535880.1534758916; _gid=GA1.2.1330668796.1534758916; weilisessionid=aa3bf69b4f35c91ca4866315f1f300b1; wluuid=WLGEUST-02ADBA37-4B6C-DE33-2769-8697C4B575BB; wlsource=tc_pc_home; webp_enabled=0; _ga=GA1.3.1167535880.1534758916; _gid=GA1.3.1330668796.1534758916; _ba=BA0.2-20180820-51751-eyUyUL4rqUHUI1lh6uRM; qimo_seosource_e7dfc0b0-b3b6-11e7-b58e-df773034efe4=%E5%85%B6%E4%BB%96%E7%BD%91%E7%AB%99; qimo_seokeywords_e7dfc0b0-b3b6-11e7-b58e-df773034efe4=%E6%9C%AA%E7%9F%A5; accessId=e7dfc0b0-b3b6-11e7-b58e-df773034efe4; pageViewNum=1; bad_ide7dfc0b0-b3b6-11e7-b58e-df773034efe4=3c85f321-a45f-11e8-92ed-072415955da9; nice_ide7dfc0b0-b3b6-11e7-b58e-df773034efe4=3c85f322-a45f-11e8-92ed-072415955da9',
    'dnt': '1',
    'upgrade-insecure-requests': '1',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'
}


class TuchongDataSource(DataSourceStage):
    '''
    Tuchong Data Source
    @chs 从图虫导入图像
    '''

    def apply_params(self, dataset, term, pages='1-1', proxy='') -> None:
        '''
        Args:
            term (str): Keyword to search
            pages (str): Page range
            dataset (str): Dataset name
            proxy (str): Proxy
        '''
        self.term = term
        st, ed = [int(_) for _ in pages.split('-', 1)]
        self.pages = range(st, ed+1)
        self.dataset = dataset
        self.proxy = {'http': proxy, 'https': proxy} if proxy else {}

    def fetch(self):
        for page in self.pages:
            yield from self.download(self.term, page)

    def download(self, term, page):
        self.logger(term, page)
        try:
            entry = f'https://tuchong.com/rest/tags/{term}/posts?page={page}&count=20'
            json = requests.get(entry, headers=headers, proxies=self.proxy).json()['postList']
        except:
            return

        paragraphs = [Paragraph.get(item['url'], keywords=['@tuchong'], dataset=self.dataset, content=item['title'])
                      for item in json
                      if re.match(r"https://tuchong.com/(\d)*?/(\d)*?/", item["url"])]

        def _parse_img(paragraph):
            url = paragraph.source.url
            self.logger('downlaod from', url)
            html = requests.get(url, headers=headers, proxies=self.proxy).content
            images = re.findall(r'\"img_id\"\:(\d+)\,\"user_id\"\:(\d+)', html.decode('utf-8'))
            paragraph.images = [
                MediaItem.get(f'https://photo.tuchong.com/{uid}/f/{img}.jpg') for (img, uid) in images
            ]
            for i in paragraph.images:
                if not i.id:
                    i.save()
            return paragraph

        yield from ThreadPoolExecutor(10).map(_parse_img, paragraphs)
