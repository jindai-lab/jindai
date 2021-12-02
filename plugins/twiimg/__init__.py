import datetime
import time
import os, glob
import re
from collections import defaultdict
from typing import List, Union
import twitter

from gallery import Post, Item, F, Fn, Var, ObjectId, Plugin, PluginContext, apply_auto_tags, DefaultTools, queryparser


def find_post(url: str) -> Union[Post, None]:
    """Find post with twitter id in the url

    Args:
        url (str): twitter url

    Returns:
        Union[Post, None]: Post
    """
    return Post.first(F.source_url.regex(r'https://twitter.com/.*/status/' + url.split('/')[-1]))


def twitter_id_from_timestamp(ts: float) -> int:
    """Get twitter id from timestamp

    Args:
        ts (float): time stamp in seconds

    Returns:
        int: twitter id representing the timestamp
    """
    return (int(ts * 1000) - 1288834974657) << 22


def create_posts(posts_imported: List[Post]):
    """Create posts from imported files

    Args:
        posts_imported (List[Post]): posts representing imported files
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

    posts = defaultdict(Post)

    for i in items:
        fn = i.url.split('/')[-1].split('#')[-1][:-5]
        pu = __expand_twi_url(fn)
        p = posts[pu]
        if p.source_url != pu:
            author = '@' + fn.split('-')[0]
            p.tags.append(author)
            p.source_url = pu
            dtstr = ('--' + fn).rsplit('-', 2)[1].replace('_', '')
            if len(dtstr) == 14:
                p.created_at = datetime.datetime(int(dtstr[:4]), *[int(_1 + _2) for _1, _2 in
                                                                   zip(dtstr[4::2], dtstr[5::2])]).timestamp()
        p.items.append(i)

    apply_auto_tags(posts.values())

    for p in posts.values():
        p.save()

    Post.query(F.id.in_(impids)).delete()


class TwiImage(Plugin):
    """TwiImage plugin"""

    def __init__(self, app, allow_video=False, **settings):
        """Init

        Args:
            app (Flask): flask app
            allow_video (bool, optional): import videos. Defaults to False.
        """
        self.api = twitter.Api(**settings)
        self.allow_video = allow_video

    def parse_status(self, st, allow_video=None) -> Post:
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
            p = Post(created_at=st.created_at_in_seconds, source_url=l)
            for m in st.media or []:
                if m.video_info:
                    if not allow_video:
                        continue  # skip videos
                    url = m.video_info['variants'][-1]['url'].split('?')[0]
                    if url.endswith('.m3u8'):
                        ctx.log('found m3u8, pass', url)
                        continue
                    p.items.append(Item(url=url))
                else:
                    p.items.append(Item(url=m.media_url_https))
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

    def import_twiimg(self, ctx: PluginContext, ls: List[str]):
        """Import twitter posts from url strings

        Args:
            ctx (PluginContext): plugin context
            ls (List[str]): urls
        """
        posts = []
        for l in ls:
            if 'twitter.com' not in l or '/status/' not in l:
                continue
            ctx.log(l)

            stid = l.split('/')
            stid = stid[stid.index('status') + 1]

            try:
                st = self.api.GetStatus(stid)
            except:
                continue

            p = self.parse_status(st, allow_video=True)
            if p.items and not p.id:
                posts.append(p)

        DefaultTools.instance.import_posts(ctx, posts)

    def import_twiuser(self, ctx: PluginContext, l: str, after: Union[str, int] = 0, before: Union[str, int] = 0, include_rt: bool = False, allow_video = False):
        """Import posts of a twitter user

        Args:
            ctx (PluginContext): plugin context
            l (str): twitter user name
            after (Union[str, int], optional): after timestamp or str representing a timestamp. Defaults to 0.
            before (Union[str,int], optional): before timestamp or str representing a timestamp. Defaults to 0.
            include_rt (bool, optional): include retweets. Defaults to False.
        """

        before = time.time() if before == 0 else queryparser.parse_dt_span(before)
        after = queryparser.parse_dt_span(after) if after > 0 else 0
        if before < after:
            before, after = after, before

        max_id = twitter_id_from_timestamp(before)

        while before > after:
            ctx.log(max_id, before, after)
            posts = []
            tl = self.api.GetUserTimeline(
                screen_name=l, count=100, max_id=max_id)
            for st in tl:
                p = self.parse_status(st, allow_video=allow_video)
                before = min(before, p.created_at)
                max_id = min(max_id, st.id)
                if p.author != '@' + st.user.screen_name and not include_rt:
                    continue
                if p.items and not p.id and p.created_at > after:
                    posts.append(p)
            if posts:
                DefaultTools.instance.import_posts(ctx, posts)
            else:
                break

    def import_twitl(self, ctx: PluginContext, after: Union[str, int] = 0, before: Union[str, int] = 0, include_rt: bool = False):
        """Import twitter timeline

        Args:
            ctx (PluginContext): plugin context
            after (Union[str, int], optional): after timestamp or str representing a timestamp. Defaults to 0.
            before (Union[str,int], optional): before timestamp or str representing a timestamp. Defaults to 0.
            include_rt (bool, optional): include retweets. Defaults to True.
        """
        if not isinstance(after, int):
            after = queryparser.parse_dt_span(after)
        if not isinstance(before, int):
            before = queryparser.parse_dt_span(before) + 86400
        if isinstance(include_rt, str):
            include_rt = include_rt.lower() == 'true'

        if after == 0:
            after = Post.query(F.source_url.regex(r'twitter\.com') & ~F.source_url.regex(
                r'/i/invalid')).sort(-F.created_at).limit(1).first().created_at
        
        ctx.log(after)

        posts = []
        o = twitter_id_from_timestamp(before or time.time())
        p = None

        for _i in range(100):
            time.sleep(0.5)
            try:
                tl = self.api.GetHomeTimeline(count=100, max_id=o)
                for st in tl:
                    p = self.parse_status(st)
                    if not p.author and not include_rt:
                        continue
                    o = min(st.id, o)
                    if p.created_at < after:
                        break
                    if p.items and not p.id:
                        posts.append(p)

                if p and p.created_at < after:
                    break
            except Exception as ex:
                ctx.log('exception', ex.__class__.__name__, ex)
                break

        DefaultTools.instance.import_posts(ctx, posts)

    def special_page(self, aggregate, params, orders_params, **vars):
        """Dealing special pages"""

        post1 = params['post']
        groups = int(params['groups'])
        archive = int(params['archive']) == 1

        offset = int(params.get('offset', 0))
        limit = int(params.get('limit', 50))

        if post1.startswith('twi-authors/'):

            if groups:
                return [], {}, {}

            aggregate.group(
                _id=Var.author,
                id=Fn.first(Var._id),
                liked_at=Fn.first(Var.liked_at),
                created_at=Fn.first(Var.created_at),
                source_url=Fn.first(Var.source_url),
                items=Fn.first(Var.items),
                counts=Fn.sum(Fn.size(Var.items))
            ).addFields(
                tags=[Fn.concat('@', Var._id)],
                _id=Var.id
            ).sort(counts=-1).skip(offset).limit(limit)

            return aggregate.perform(), \
                {'offset': max(
                    0, offset-limit), 'limit': limit}, {'offset': offset + limit, 'limit': limit}

    def import_twi(self, ctx: PluginContext, *args):
        """Import twitter-related items

        Args:
            ctx (PluginContext): plugin context
            args: ['@'] for all followed twitter users; '@' + username for import_twiuser; urls for import_twiimg; file path for import_local; otherwise for import_twitl
        """
        arg = args[0] if args else ''
        if arg.startswith('@'):
            if arg == '@' or arg.startswith('@%'):
                unames = sorted(map(lambda x: x.screen_name, self.api.GetFriends()))
                if arg.startswith('@%'):
                    unames = [_ for _ in unames if re.search(arg[2:], _)]

                before = time.time() if len(args) < 3 else queryparser.parse_dt_span(args[2])
                after = 0 if len(args) < 2 else queryparser.parse_dt_span(args[1])

                for u in unames:
                    if after == 0:
                        last_updated = Post.query(F.source_url.regex(
                            f'^https://twitter.com/{u}/')).sort(-F.created_at).first()
                        if last_updated:
                            after = last_updated.created_at
                            
                    ctx.log(u)
                    self.import_twiuser(ctx, u, after=after, before=before)
            else:
                self.import_twiuser(ctx, *args)
        elif arg.startswith('http://') or arg.startswith('https://'):
            self.import_twiimg(ctx, args)
        elif os.path.exists(arg) or glob.glob(arg):
            create_posts(DefaultTools.instance.import_local(ctx, '', *args))
        else:
            self.import_twitl(ctx, *args)

    def get_special_pages(self):
        """Get special pages

        Returns:
            List[str]: special page prefixes
        """        
        return ['twi-authors']

    def get_tools(self):
        """Get tools

        Returns:
            List[str]: tool function names
        """        
        return ['import-twi']
