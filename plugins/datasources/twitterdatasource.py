import datetime
import glob
import os
import re
import time
from collections import defaultdict
from typing import List

import twitter
from jindai.models import MediaItem, Paragraph
from jindai.pipeline import DataSourceStage
from jindai.dbquery import parser, ObjectId, F, Fn

from .dbquerydatasource import ImageImportDataSource


def twitter_id_from_timestamp(stamp: float) -> int:
    """Get twitter id from timestamp

    Args:
        stamp (float): time stamp in seconds

    Returns:
        int: twitter id representing the timestamp
    """
    return (int(stamp * 1000) - 1288834974657) << 22


def create_albums(posts_imported: List[Paragraph]):
    """Create posts from imported files

    Args:
        posts_imported (List[Paragraph]): posts representing imported files
    """
    def _expand_twi_url(zipfn):
        a = zipfn.split('-', 2)
        if len(a) != 3:
            return zipfn
        a, b, _ = a
        return f'https://twitter.com/{a}/status/{b}'

    items = []
    impids = []
    for para in posts_imported:
        items += list(para.images)
        impids.append(ObjectId(para.id))

    albums = defaultdict(Paragraph)

    for i in items:
        filename = i.source['url'].split('/')[-1].split('#')[-1][:-5]
        page_url = _expand_twi_url(filename)
        para = albums[page_url]
        if para.source.get('url') != page_url:
            author = '@' + filename.split('-')[0]
            para.keywords.append(author)
            para.source = {'url': page_url}
            dtstr = ('--' + filename).rsplit('-', 2)[1].replace('_', '')
            if len(dtstr) == 14:
                para.pdate = datetime.datetime(int(dtstr[:4]), *[int(_1 + _2) for _1, _2 in
                                                                 zip(dtstr[4::2], dtstr[5::2])])
        para.images.append(i)

    for para in albums.values():
        yield para

    Paragraph.query(F.id.in_(impids)).delete()


def _stamp(dtval):
    if isinstance(dtval, str):
        dtval = parser.parse_literal(dtval)
    if isinstance(dtval, datetime.datetime):
        return dtval.timestamp()
    elif isinstance(dtval, (int, float)):
        return dtval
    return None


class TwitterDataSource(DataSourceStage):
    """
    Load from social network
    @chs 导入社交网络信息
    """

    class Implementation(DataSourceStage.Implementation):
        """implementing datasource"""

        def __init__(self,
                     allow_video=False,
                     allow_retweet=True,
                     media_only=True,
                     consumer_key='', consumer_secret='', access_token_key='', access_token_secret='',
                     import_username='',
                     time_after='', time_before='',
                     skip_existent=True,
                     proxy=''
                     ) -> None:
            """

            Args:
                allow_video (bool, optional):
                    Allow video
                    @chs 允许导入视频
                allow_retweet (bool, optional):
                    Allow retweet
                    @chs 允许导入转发
                media_only (bool, optional):
                    Media only
                    @chs 只导入包含媒体内容的 Tweets
                consumer_key (str, optional): API CONSUMER KEY
                consumer_secret (str, optional): API CONSUMER SECRET
                access_token_key (str, optional): API ACCESS TOKEN KEY
                access_token_secret (str, optional): API ACCESS TOKEN SECRET
                import_username (str, optional):
                    Import source, blank for timeline
                    @chs 导入的用户名或留空以导入 Timeline
                time_after (str):
                    Time after
                    @chs 时间上限
                time_before (str):
                    Time before
                    @chs 时间下限
                skip_existent (bool):
                    Skip existent tweets
                    @chs 跳过已经导入的 URL
                proxy (str):
                    Proxy settings
                    @chs 代理服务器
            """
            super().__init__()
            self.allow_video = allow_video
            self.allow_retweet = allow_retweet
            self.media_only = media_only
            self.import_username = import_username
            self.time_after = _stamp(time_after) or 0
            self.time_before = _stamp(time_before) or time.time()
            self.proxies = {'http': proxy, 'https': proxy} if proxy else {}
            self.api = twitter.Api(consumer_key=consumer_key, consumer_secret=consumer_secret,
                                   access_token_key=access_token_key, access_token_secret=access_token_secret, proxies=self.proxies)
            self.skip_existent = skip_existent
            self.imported = set()

        def parse_tweet(self, tweet, skip_existent=None) -> Paragraph:
            """Parse twitter status
            Args:
                st (twitter.status): status
            Returns:
                Post: post
            """
            if skip_existent is None: skip_existent = self.skip_existent
            
            tweet_url = f'https://twitter.com/{tweet.user.screen_name}/status/{tweet.id}'

            author = '@' + tweet.user.screen_name
            if tweet.id in self.imported:
                return
            self.imported.add(tweet.id)
            para = Paragraph.first(F.tweet_id == f'{tweet.id}')

            self.logger(tweet_url, 'skip' if para is not None and skip_existent else '')
            if not skip_existent or not para:
                para = para or Paragraph(dataset='twitter', pdate=datetime.datetime.utcfromtimestamp(
                    tweet.created_at_in_seconds), source={'url': tweet_url}, tweet_id=f'{tweet.id}')
                for media in tweet.media or []:
                    if media.video_info:
                        if not self.allow_video:
                            continue  # skip videos
                        url = media.video_info['variants'][-1]['url'].split('?')[
                            0]
                        if url.endswith('.m3u8'):
                            self.logger('found m3u8, pass', url)
                            continue
                    else:
                        url = media.media_url_https
                    if url:
                        item = MediaItem.first(F['source.url'] == url)
                        if item:
                            Paragraph.query(F.images == item.id, F.id != para.id).update(Fn.pull(images=item.id))
                        else:
                            item = MediaItem(
                                source={'url': url},
                                item_type='video' if media.video_info else 'image')
                            item.save()
                        para.images.append(item)
                if tweet.text.startswith('RT '):
                    author = re.match(r'^RT (@[\w_-]*)', tweet.text)
                    if author:
                        author = author.group(1)
                    else:
                        author = ''
                text = re.sub(r'https?://[^\s]+', '', tweet.text).strip()
                para.keywords += [t.strip().strip('#') for t in re.findall(
                    r'@[a-z_A-Z0-9]+', text) + re.findall(r'[#\s][^\s@]{,10}', text)] + [author]
                para.keywords = [_ for _ in para.keywords if _]
                para.content = text
                para.author = author

            return para

        def import_twiimg(self, urls: List[str]):
            """Import twitter posts from url strings
            Args:
                ls (List[str]): urls
            """
            for url in urls:
                if 'twitter.com' not in url or '/status/' not in url:
                    continue
                self.logger(url)

                tweet_id = url.split('/')
                tweet_id = tweet_id[tweet_id.index('status') + 1]

                try:
                    tweet = self.api.GetStatus(tweet_id)
                except Exception:
                    continue

                para = self.parse_tweet(tweet, False)
                if para and (not self.media_only or para.images) and not para.id:
                    yield para

        def import_timeline(self, user=''):
            """Import posts of a twitter user, or timeline if blank"""

            if user:
                def source(max_id):
                    return self.api.GetUserTimeline(
                        screen_name=user, count=100, max_id=max_id-1)
            else:
                def source(max_id):
                    return self.api.GetHomeTimeline(
                        count=100, max_id=max_id-1, exclude_replies=True)

            self.logger('twiuser', self.time_before, self.time_after)

            if self.time_before < self.time_after:
                self.time_before, self.time_after = self.time_after, self.time_before

            max_id = twitter_id_from_timestamp(self.time_before)+1
            before = self.time_before

            try:
                pages = 0
                while before > self.time_after and pages < 50:
                    pages += 1
                    yielded = False
                    self.logger(max_id, datetime.datetime.fromtimestamp(before), self.time_after)

                    timeline = source(max_id)
                    for status in timeline:
                        before = min(before, status.created_at_in_seconds)
                        max_id = min(max_id, status.id)
                        para = self.parse_tweet(status)
                        if not para or para.id:
                            continue
                        if para.author != '@' + status.user.screen_name and not self.allow_retweet:
                            continue
                        if status.created_at_in_seconds > self.time_after:
                            if (not self.media_only or para.images) and not para.id:
                                yield para
                                yielded = True

                    if not yielded:
                        break
            except twitter.TwitterError as ex:
                self.logger('twitter exception', ex.__class__.__name__, ex)

        def fetch(self):
            args = self.import_username.split('\n')
            arg = args[0]
            if arg.startswith('@'):
                if arg == '@' or arg.startswith('@%'):
                    unames = sorted(
                        map(lambda x: x.screen_name, self.api.GetFriends()))
                    if arg.startswith('@%'):
                        unames = [_ for _ in unames if re.search(arg[2:], _)]
                    for u in unames:
                        if self.time_after == 0:
                            last_updated = Paragraph.query(F['source.url'].regex(
                                f'^https://twitter.com/{u}/')).sort(-F.pdate).first()
                            if last_updated:
                                self.time_after = last_updated.pdate.timestamp()
                        self.logger(u)
                        yield from self.import_timeline(u)
                else:
                    for u in args:
                        yield from self.import_timeline(u)
            elif arg.startswith('http://') or arg.startswith('https://'):
                yield from self.import_twiimg(args)
            elif os.path.exists(arg) or glob.glob(arg):
                yield from create_albums(list(ImageImportDataSource.Implementation('\n'.join(args)).fetch()))
            else:
                yield from self.import_timeline()
