from models import Paragraph, parser
from datasource import DataSource
import datetime
import time
import os
import re
from collections import defaultdict
from typing import List, Union
import twitter


def twitter_id_from_timestamp(ts: float) -> int:
    """Get twitter id from timestamp

    Args:
        ts (float): time stamp in seconds

    Returns:
        int: twitter id representing the timestamp
    """
    return (int(ts * 1000) - 1288834974657) << 22


class TwitterDataSource(DataSource):
    """导入社交网络信息
    """    
    
    def __init__(self, allow_video=False, allow_images=True, allow_text=True, allow_retweet=True, consumer_key='', consumer_secret='', access_token_key='', access_token_secret='',
                 import_username='', import_timeline=False,
                 time_after='', time_before=''
                 ) -> None:
        """

        Args:
            allow_video (bool, optional): 导入视频. Defaults to False.
            allow_images (bool, optional): 导入图片. Defaults to True.
            allow_text (bool, optional): 导入文本. Defaults to True.
            allow_retweet (bool, optional): 导入转发. Defaults to True.
            consumer_key (str, optional): API CONSUMER KEY. Defaults to ''.
            consumer_secret (str, optional): API CONSUMER SECRET. Defaults to ''.
            access_token_key (str, optional): API ACCESS TOKEN KEY. Defaults to ''.
            access_token_secret (str, optional): API ACCESS TOKEN SECRET. Defaults to ''.
            import_username (str, optional): 要导入用户名. Defaults to ''.
            import_timeline (bool, optional): 是否导入时间线，用户名和时间线必有且只有一个有值. Defaults to False.
            time_after (str): 时间上限
            time_before (str): 时间下限
        """        
        super().__init__()
        self.allow_video = allow_video
        self.allow_images = allow_images
        self.allow_text = allow_text
        self.allow_retweet = allow_retweet
        self.api = twitter.Api(consumer_key=consumer_key, consumer_secret=consumer_secret, access_token_key=access_token_key, access_token_secret=access_token_secret)
        self.import_username = import_username
        self.import_timeline = import_timeline
        self.time_after = parser.parse_dt_span(time_after)
        self.time_before = parser.parse_dt_span(time_before)
        if self.time_before == 0:
            self.time_before = time.time()
        assert self.import_timeline or self.import_username, '导入时间线和用户名必有一个有值'
        assert not (self.import_timeline and self.import_username), '导入时间线和用户名必有且只有一个有值'
        
    def fetch(self) -> Paragraph:
        if self.import_timeline:
            yield from self._get_timeline()
        else:
            yield from self._get_tweets(self.import_username, self.last_id)

    def _parse_status(self, st) -> Paragraph:
        l = f'https://twitter.com/{st.user.screen_name}/status/{st.id}'
        text = re.sub(r'https?://[^\s]+', '', st.text).strip()

        p = Paragraph(pdate=datetime.datetime.fromtimestamp(st.created_at_in_seconds), source={'url': l}, content=text)
        
        if st.text.startswith('RT '):
            if not self.allow_retweet:
                return
            p.retweet = True
            
        p.people = re.findall(r'(@[a-z_A-Z0-9]+)', text)
        p.groups = []
        p.keywords = [t.strip(':').strip('#')
                                    for t in text.split()]
        p = p.as_dict()
        del p['image_storage']
        
        if not st.media and self.allow_text:
            yield Paragraph(**p)
            
        if (self.allow_video or self.allow_images) and st.media:
            for m in st.media:
                if m.video_info:
                    if not self.allow_video:
                        continue  # skip videos
                    url = m.video_info['variants'][-1]['url'].split('?')[0]
                    yield Paragraph(image_storage={'url': url}, **p)
                elif self.allow_images:
                    yield Paragraph(image_storage={'url': m.media_url_https}, **p)

    def _get_tweets(self) -> Paragraph:
        o = self.time_before
        while o > self.time_after:
            tl = self.api.GetUserTimeline(
                screen_name=self.import_username, count=100, max_id=twitter_id_from_timestamp(o))
            for st in tl:
                if st.created_at_in_seconds < self.time_after:
                    break
                yield from self._parse_status(st)
                o = min(o, st.created_at_in_seconds)
    
    def _get_timeline(self) -> Paragraph:
        o = self.time_before
        
        for _i in range(100):
            time.sleep(0.5)
            tl = self.api.GetHomeTimeline(count=100, max_id=twitter_id_from_timestamp(o))
            for st in tl:
                if st.created_at_in_seconds < self.time_after:
                    break
                yield from self._parse_status(st)
                o = min(o, st.created_at_in_seconds)
           