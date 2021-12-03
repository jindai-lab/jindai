import datetime
import glob
import os
import re
import time
from collections import defaultdict
from typing import List, Union
import twitter
from datasource import DataSource
from models import Album, ImageItem, ObjectId, parser
from PyMongoWrapper import F

from datasources.gallerydatasource import ImageImportDataSource, queryparser


def find_post(url: str) -> Union[Album, None]:
    """Find post with twitter id in the url

    Args:
        url (str): twitter url

    Returns:
        Union[Album, None]: Album
    """
    return Album.first(F['source.url'].regex(r'https://twitter.com/.*/status/' + url.split('/')[-1]))


def twitter_id_from_timestamp(ts: float) -> int:
    """Get twitter id from timestamp

    Args:
        ts (float): time stamp in seconds

    Returns:
        int: twitter id representing the timestamp
    """
    return (int(ts * 1000) - 1288834974657) << 22


def create_albums(posts_imported: List[Album]):
    """Create posts from imported files

    Args:
        posts_imported (List[Album]): posts representing imported files
    """
    def __expand_twi_url(zipfn):
        a = zipfn.split('-', 2)
        if len(a) != 3:
            return zipfn
        a, b, _ = a
        return f'https://twitter.com/{a}/status/{b}'

    items = []
    impids = []
    for p in posts_imported:
        items += p.items.a
        impids.append(ObjectId(p.id))

    albums = defaultdict(Album)

    for i in items:
        fn = i.url.split('/')[-1].split('#')[-1][:-5]
        pu = __expand_twi_url(fn)
        p = albums[pu]
        if p.source['url'] != pu:
            author = '@' + fn.split('-')[0]
            p.tags.append(author)
            p.source = {'url': pu}
            dtstr = ('--' + fn).rsplit('-', 2)[1].replace('_', '')
            if len(dtstr) == 14:
                p.pdate = datetime.datetime(int(dtstr[:4]), *[int(_1 + _2) for _1, _2 in
                          zip(dtstr[4::2], dtstr[5::2])])
        p.items.append(i)
    
    for p in albums.values():
        yield p

    Album.query(F.id.in_(impids)).delete()


class TwitterDataSource(DataSource):
    """导入社交网络信息
    """    
    
    def __init__(self, allow_video=False, allow_images=True, allow_text=True, allow_retweet=True, consumer_key='', consumer_secret='', access_token_key='', access_token_secret='',
                 import_username='',
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
        self.time_after = parser.parse_dt_span(time_after)
        self.time_before = parser.parse_dt_span(time_before)
        if self.time_before == 0:
            self.time_before = time.time()
        
    def parse_status(self, st, allow_video=None) -> Album:
        """Parse twitter status
        Args:
            st (twitter.status): status
        Returns:
            Post: post
        """
        l = f'https://twitter.com/{st.user.screen_name}/status/{st.id}'
        author = '@' + st.user.screen_name
        p = find_post(l)
        if allow_video is None:
            allow_video = self.allow_video
        
        if not p:
            p = Album(pdate=datetime.datetime.fromtimestamp(st.created_at_in_seconds), source={'url': l})
            for m in st.media or []:
                if m.video_info:
                    if not allow_video:
                        continue  # skip videos
                    url = m.video_info['variants'][-1]['url'].split('?')[0]
                    if url.endswith('.m3u8'):
                        self.logger('found m3u8, pass', url)
                        continue
                    p.items.append(ImageItem(source={'url': url}))
                else:
                    p.items.append(ImageItem(source={'url': m.media_url_https}))
            if st.text.startswith('RT '):
                author = re.match(r'^RT (@.*?):', st.text)
                if author:
                    author = author.group(1)
                else:
                    author = ''
            text = re.sub(r'https?://[^\s]+', '', st.text).strip()
            p.tags = [author, text] + [t.strip(':').strip('#')
                                       for t in text.split()] + re.findall(r'(@[a-z_A-Z0-9]+)', text)
            p.author = author
        return p
    
    def import_twiimg(self, ls: List[str]):
        """Import twitter posts from url strings
        Args:
            ls (List[str]): urls
        """
        albums = []
        for l in ls:
            if 'twitter.com' not in l or '/status/' not in l:
                continue
            self.logger(l)

            stid = l.split('/')
            stid = stid[stid.index('status') + 1]

            try:
                st = self.api.GetStatus(stid)
            except:
                continue

            p = self.parse_status(st, allow_video=True)
            if p.items and not p.id:
                albums.append(p)

        yield from albums

    def import_twiuser(self, user):
        """Import posts of a twitter user"""

        before = self.time_before
        after = self.time_after
        
        if before < after:
            before, after = after, before

        max_id = twitter_id_from_timestamp(before)

        while before > after:
            albums = []
            tl = self.api.GetUserTimeline(
                screen_name=user, count=100, max_id=max_id)
            for st in tl:
                p = self.parse_status(st, allow_video=self.allow_video)
                before = min(before, p.pdate.timestamp())
                max_id = min(max_id, st.id)
                if p.author != '@' + st.user.screen_name and not self.allow_retweet:
                    continue
                if p.items and not p.id and p.pdate.timestamp() > after:
                    albums.append(p)
            
            yield from albums

    def import_twitl(self):
        after, before = self.time_after, self.time_before
        if after == 0:
            after = Album.query(F.source_url.regex(r'twitter\.com')).sort(-F.pdate).limit(1).first().pdate.timestamp()
        
        albums = []
        o = twitter_id_from_timestamp(before or time.time())
        p = None

        self.logger('twitl', o, after)

        for _i in range(100):
            time.sleep(0.5)
            try:
                tl = self.api.GetHomeTimeline(count=100, max_id=o)
                for st in tl:
                    p = self.parse_status(st)
                    if not p.author and not self.allow_retweet:
                        continue
                    o = min(st.id, o)
                    if p.pdate.timestamp() < after:
                        break
                    if p.items and not p.id:
                        albums.append(p)

                if p and p.pdate.timestamp() < after:
                    break
            except Exception as ex:
                self.logger('exception', ex.__class__.__name__, ex)
                break

        yield from albums

    def fetch(self):
        args = self.import_username.split('\n')
        arg = args[0]
        if arg.startswith('@'):
            if arg == '@' or arg.startswith('@%'):
                unames = sorted(map(lambda x: x.screen_name, self.api.GetFriends()))
                if arg.startswith('@%'):
                    unames = [_ for _ in unames if re.search(arg[2:], _)]
                before, after = self.time_before, self.time_after
                for u in unames:
                    if after == 0:
                        last_updated = Album.query(F['source.url'].regex(
                            f'^https://twitter.com/{u}/')).sort(-F.pdate).first()
                        if last_updated:
                            after = last_updated.pdate.timestamp()
                    self.logger(u)
                    yield from self.import_twiuser(u, after=after, before=before)
            else:
                for u in args:
                    yield from self.import_twiuser(u)
        elif arg.startswith('http://') or arg.startswith('https://'):
            yield from self.import_twiimg(args)
        elif os.path.exists(arg) or glob.glob(arg):
            yield from create_albums(list(ImageImportDataSource('\n'.join(self.locs)).fetch()))
        else:
            yield from self.import_twitl()
