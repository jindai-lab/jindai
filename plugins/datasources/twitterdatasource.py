import base64
import datetime
import re
import time
from typing import List

import tweepy
from jindai.common import DictObject
from jindai.models import MediaItem, Paragraph
from jindai.pipeline import DataSourceStage
from jindai.dbquery import parser, ObjectId, F, Fn


def twitter_id_from_timestamp(stamp: float) -> int:
    """Get twitter id from timestamp

    Args:
        stamp (float): time stamp in seconds

    Returns:
        int: twitter id representing the timestamp
    """
    return (int(stamp * 1000) - 1288834974657) << 22


def timestamp_from_twitter_id(tweet_id: int) -> float:
    """Get timestamp from tweet id

    Args:
        tweet_id (int): tweet id

    Returns:
        float: timestamp in UTC
    """
    return ((tweet_id >> 22) + 1288834974657) / 1000


def tweet_id_from_media_url(url: str) -> int:
    url = url.split('/')[-1].split('.')[0]
    tweet_id = int.from_bytes(base64.urlsafe_b64decode(url[:12])[:8], 'big')
    return tweet_id


def timestamp_from_media_url(url: str) -> float:
    """Get timestamp from Base64-encoded media url

    Args:
        url (str): _description_

    Returns:
        float: _description_
    """
    return timestamp_from_twitter_id(tweet_id_from_media_url(url))


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

    def apply_params(self,
                     dataset_name='',
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
            dataset_name (DATASET):
                Dataset name
                @chs 数据集名称
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
            import_username (LINES, optional):
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
        self.dataset_name = dataset_name
        self.allow_video = allow_video
        self.allow_retweet = allow_retweet
        self.media_only = media_only
        self.import_username = import_username
        self.time_after = _stamp(time_after) or 0
        self.time_before = _stamp(time_before) or time.time()
        self.api = tweepy.API(tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token_key, access_token_secret),
                              proxy=proxy)
        self.skip_existent = skip_existent
        self.imported = set()

    def parse_tweet(self, tweet, skip_existent=None) -> Paragraph:
        """Parse twitter status
        Args:
            st (twitter.status): status
        Returns:
            Post: post
        """
        if skip_existent is None:
            skip_existent = self.skip_existent

        if tweet.id in self.imported:
            return
        self.imported.add(tweet.id)

        tweet_url = f'https://twitter.com/{tweet.user.screen_name}/status/{tweet.id}'

        # get author info
        author = '@' + tweet.user.screen_name
        if not tweet.text:
            tweet.text = tweet.full_text or ''
        if tweet.text.startswith('RT '):
            author = re.match(r'^RT (@[\w_-]*)', tweet.text)
            if author:
                author = author.group(1)
            else:
                author = ''

        # get media entities
        media_entities = [
            DictObject(media)
            for media in getattr(tweet, 'extended_entities', tweet.entities).get('media', [])
        ]
        if not media_entities and self.media_only:
            return

        para = Paragraph.get(
            F.tweet_id == f'{tweet.id}', tweet_id=f'{tweet.id}', author=author)
        self.logger(tweet_url, 'existent' if para.id else '')

        if skip_existent and para.id:
            return

        para.dataset = self.dataset_name
        para.pdate = datetime.datetime.utcfromtimestamp(
            timestamp_from_twitter_id(tweet.id))
        para.source = {'url': tweet_url}
        para.tweet_id = f'{tweet.id}'
        para.images = []

        if para.id:
            self.logger('... matched existent paragraph', para.id)

        for media in media_entities:
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
                item = MediaItem.get(
                    url, item_type='video' if media.video_info else 'image')

                if item.id and self.skip_existent:
                    continue

                if not item.id:
                    item.save()
                    self.logger('... add new item', url)
                para.images.append(item)

        text = re.sub(r'https?://[^\s]+', '', tweet.text).strip()
        para.keywords += [t.strip().strip('#') for t in re.findall(
            r'@[a-z_A-Z0-9]+', text) + re.findall(r'[#\s][^\s@]{,10}', text)] + [author]
        para.keywords = [_ for _ in para.keywords if _]
        para.content = text
        para.save()

        assert len(para.images) == len(media_entities) or len(para.images) == 0
        self.logger(len(para.images), 'media items')

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
                tweet = self.api.get_status(tweet_id)
            except Exception:
                continue

            para = self.parse_tweet(tweet, False)
            if para:
                yield para

    def import_timeline(self, user=''):
        """Import posts of a twitter user, or timeline if blank"""

        params = dict(count=100, include_rts=self.allow_retweet,
                      exclude_replies=True)
        if user:
            def source(max_id):
                return self.api.user_timeline(
                    screen_name=user, max_id=max_id-1,
                    **params)
        else:
            def source(max_id):
                return self.api.home_timeline(
                    max_id=max_id-1,
                    **params)

        if self.time_before < self.time_after:
            self.time_before, self.time_after = self.time_after, self.time_before

        max_id = twitter_id_from_timestamp(self.time_before)+1
        before = self.time_before

        self.logger('import timeline', user,
                    self.time_before, self.time_after)

        try:
            pages = 0
            min_id = max(0, twitter_id_from_timestamp(self.time_after))
            while before >= self.time_after and pages < 50:
                pages += 1
                yielded = False
                has_data = False
                self.logger(max_id, datetime.datetime.fromtimestamp(
                    before), self.time_after)

                timeline = source(max_id)
                for status in timeline:

                    if min_id > status.id:
                        break

                    if max_id > status.id:
                        has_data = True
                        before = min(
                            before, timestamp_from_twitter_id(status.id))
                        max_id = min(max_id, status.id)

                    try:
                        para = self.parse_tweet(status)
                    except Exception as exc:
                        self.log_exception('parse tweet error', exc)
                        para = None

                    if para:
                        yield para
                        yielded = True

                if (not user and not yielded) or not has_data:
                    break

                time.sleep(1)

            if pages >= 50:
                self.logger(f'Reached max pages count, interrupted. {user}')

        except tweepy.TwitterServerError as ex:
            self.log_exception('twitter exception', ex)

    def fetch(self):
        args = self.import_username.split('\n')
        arg = args[0]
        if arg.startswith('@'):
            if arg == '@' or arg.startswith('@%'):
                unames = sorted(
                    map(lambda x: x.screen_name, self.api.get_friends()))
                if arg.startswith('@%'):
                    unames = [_ for _ in unames if re.search(arg[2:], _)]
                for u in unames:
                    if self.time_after == 0:
                        last_updated = Paragraph.query(F.source.url.regex(
                            f'^https://twitter.com/{u}/')).sort(-F.pdate).first()
                        if last_updated:
                            self.time_after = _stamp(last_updated.pdate)
                    self.logger(u)
                    yield from self.import_timeline(u)
            else:
                for u in args:
                    yield from self.import_timeline(u)
        elif arg.startswith('http://') or arg.startswith('https://'):
            yield from self.import_twiimg(args)
        else:
            yield from self.import_timeline()
