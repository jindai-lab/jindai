"""Machine Translation
@zhs 机器翻译
"""
import hashlib
import json
import random
import time
import urllib
import uuid
from hashlib import md5
from typing import Iterator

import regex as re
import httpx
from opencc import OpenCC

from jindai.pipeline import PipelineStage
from jindai.plugin import Plugin
from jindai.helpers import safe_import
from jindai.models import Paragraph


def split_chunks(content, max_len=5000) -> Iterator:
    """Split content into chunks of no more than `max_len` characters"""
    while len(content) > max_len:
        res = re.match(r'.*[\.\?!。？！”’\'\"]', content)
        if res:
            yield res.group()
            content = content[:len(res.group())]
        else:
            yield content[:max_len]
            content = content[max_len:]
    yield content


class RemoteTranslation(PipelineStage):
    """Machine Translation with Remote API Calls
    @zhs 调用远程 API 进行机器翻译
    """

    def __init__(self, server='http://localhost/translate', to_lang='zhs') -> str:
        """
        :param to_lang:
            Target language
            @zhs 目标语言标识
        :type to_lang: LANG
        :param server:
            Entrypoint for remote calls
            @zhs 远程调用的入口点
        :type server: str
        :return: Translated text
        :rtype: str
        """
        super().__init__()
        self.server = server
        self.to_lang = to_lang if to_lang not in ('zhs', 'zht') else 'zh'
        self.convert = OpenCC('s2t' if to_lang == 'zht' else 't2s').convert

    async def resolve(self, paragraph) -> None:
        """Translate the paragraph
        """
        result = ''
        try:
            for chunk in split_chunks(paragraph.content):
                chunk = chunk.strip()
                if not chunk:
                    continue
                
                async with httpx.AsyncClient() as client:
                    resp = await client.post(self.server, json={
                        'text': chunk,
                        'source_lang': paragraph.lang.upper() if paragraph.lang != 'auto' else 'auto',
                        'target_lang': self.to_lang.upper()
                    })
                    resp = resp.json()
                if resp.get('code') != 200:
                    self.log('Error while translating:',
                             resp.get('msg', 'null message'))
                    return

                content = resp.json()['data']
                if self.to_lang == 'zh':
                    content = self.convert(content)
                if not self.to_lang in ('zh', 'jp', 'kr'):
                    result += ' '
                result += content

                time.sleep(1 + random.random())

        except ValueError:
            self.log('Error while reading from remote server')
            return

        if result:
            paragraph.content = result.strip()

        return paragraph


class YoudaoTranslation(PipelineStage):
    """Machine Translation via Youdao
    @zhs 有道云机器翻译（付费）
    """

    def __init__(self, api_id, api_key, to_lang='zhs') -> None:
        """
        :param api_id: Youdao API ID
        :type api_id: str
        :param api_key: Youdao API Key
        :type api_key: str
        :param to_lang: Target language code
            @zhs 目标语言
        :type to_lang: LANG
        """
        super().__init__()
        self.api_id, self.api_key = api_id, api_key
        self.to_lang = to_lang

    async def resolve(self, paragraph: Paragraph):

        def _regulate_lang(lang):
            if lang in ('zhs', 'zht'):
                return 'zh-' + lang.upper()
            else:
                return lang

        translate_text = paragraph.content.strip()

        if not translate_text:
            return

        youdao_url = 'https://openapi.youdao.com/api'
        input_text = ""

        if (len(translate_text) <= 20):
            input_text = translate_text
        elif (len(translate_text) > 20):
            input_text = translate_text[:10] + \
                str(len(translate_text)) + translate_text[-10:]

        time_curtime = int(time.time())
        uu_id = uuid.uuid4()
        sign = hashlib.sha256(
            f'{self.api_id}{input_text}{uu_id}{time_curtime}{self.api_key}'.encode('utf-8')).hexdigest()
        data = {
            'q': translate_text,
            'from': _regulate_lang(paragraph.lang),   # 源语言
            'to': _regulate_lang(self.to_lang),
            'appKey': self.api_id,
            'salt': uu_id,
            'sign': sign,
            'signType': "v3",
            'curtime': time_curtime,
        }

        async with httpx.AsyncClient() as client:
            resp = (await client.get(youdao_url, params=data)).json()
        paragraph.content = resp['translation'][0]
        return paragraph


class BaiduTranslation(PipelineStage):
    """Machine Translation via Baidu API
    @zhs 百度云机器翻译
    """

    def __init__(self, to_lang='zhs', api_key='', api_id='') -> None:
        """
        :param to_lang: Target language
            @zhs 目标语言
        :type to_lang: LANG
        :param api_key: API Key
        :type api_key: str, optional
        :param api_id: API ID
        :type api_id: str, optional
        """
        super().__init__()
        self.to_lang = to_lang
        self.api_id, self.api_key = api_id, api_key

    async def resolve(self, paragraph: Paragraph):
        api_endpoint = 'https://fanyi-api.baidu.com/api/trans/vip/translate'

        def _regulate_lang(lang):
            if lang == 'zhs':
                return 'zh'
            return lang

        result = ''

        for query in split_chunks(paragraph.content, max_len=2000):
            salt = random.randint(32768, 65536)
            sign = md5(f'{self.api_id}{query}{salt}{self.api_key}'.encode(
                'utf-8')).hexdigest()

            headers = {'Content-Type': 'application/x-www-form-urlencoded'}
            payload = {'appid': self.api_id,
                       'q': query,
                       'from': _regulate_lang(paragraph.lang),
                       'to': _regulate_lang(self.to_lang),
                       'salt': salt,
                       'sign': sign}

            await asyncio.sleep(1)
            async with httpx.AsyncClient() as client:
                resp = client.post(api_endpoint, params=payload,
                                 headers=headers).json()
            if 'error_msg' in resp:
                raise ValueError(resp['error_msg'])
            result += ' '.join([_['dst'] for _ in resp['trans_result']])

        paragraph.content = result
        return paragraph


class GPTTranslation(PipelineStage):
    """GPT Translation
    @zhs 基于 GPT 的机器翻译
    """

    def __init__(self, endpoint='', prompt='', schema='messages') -> None:
        """
        Args:
            endpoint (str):
                API Endpoint URL
                @zhs API 网址
            prompt (str):
                Prompt text
                @zhs 提示语
            schema (str):
                Key name
                @zhs API 端点接受的名称
        """
        self.endpoint = endpoint
        self.prompt = prompt or '将下列文本翻译为中文：'
        if '{' not in self.prompt:
            self.prompt += '\n\n```{content}```'
        self.schema = schema
        super().__init__()

    async def call(self, message) -> str:
        if message == 'test':
            return 'test'
        async with httpx.AsyncClient() as client:
            resp = await client.post(self.endpoint, json={self.schema: [{
                "role": "user", "content": message
            }]}, headers={'Content-Type': 'application/json', 'Agent': 'jindai-mt/1.0'})
            resp = resp.json()
        assert resp and resp['success'], f'Failed with response: {resp}'
        return resp['choices'][0]['message']['content']

    async def resolve(self, paragraph: Paragraph) -> Paragraph:
        text = self.prompt.format(**paragraph.as_dict()).strip()
        text = await self.call(text)
        return Paragraph(paragraph, content=text)


class GoogleTranslation(PipelineStage):
    """Google Translation"""

    def __init__(self, to_lang) -> None:
        """
        Args:
            to_lang (LANG):
                Target language
                @zhs 目标语言标识
        """
        self.to_lang = to_lang
        super().__init__()

    async def translate(self, lang_from, lang_to, text) -> str:
        def TL(a):
            k = ""
            b = 406644
            b1 = 3293161072
            jd = "."
            sb = "+-a^+6"
            Zb = "+-3^+b+-f"
            e, f, g = [], 0, 0
            while g < len(a):
                m = ord(a[g])
                if m < 128:
                    e.append(m)
                    f += 1
                else:
                    if m < 2048:
                        e.append((m >> 6) | 192)
                    else:
                        if m & 64512 == 55296 and g + 1 < len(a) and a[g + 1].charCodeAt(0) & 64512 == 56320:
                            m = 65536 + ((m & 1023) << 10) + (a[g + 1].charCodeAt(0) & 1023)
                            e.append((m >> 18) | 240)
                            e.append(((m >> 12) & 63) | 128)
                            g += 1
                        else:
                            e.append((m >> 12) | 224)
                            e.append(((m >> 6) & 63) | 128)
                    e.append((m & 63) | 128)
                g += 1
            a = b
            f = 0
            while f < len(e):
                a += e[f]
                a = RL(a, sb)
                f += 1
            a = RL(a, Zb)
            a ^= b1 or 0
            if a < 0:
                a = (a & 2147483647) + 2147483648
            a %= 1000000
            return str(a) + jd + str(a ^ b)

        def RL(a, b):
            t = "a"
            Yb = "+"
            c = 0
            while c < len(b) - 2:
                d = b[c + 2]
                d = ord(d) - 87 if d >= t else int(d)
                d = a >> d if b[c + 1] == Yb else a << d
                a = (a + d) & 4294967295 if b[c] == Yb else a ^ d
                c += 3
            return a
        
        param = f"sl={lang_from}&tl={lang_to}"
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"https://translate.google.com/translate_a/single?client=gtx&{param}&hl=zh-CN&dt=at&dt=bd&dt=ex&dt=ld&dt=md&dt=qca&dt=rw&dt=rm&dt=ss&dt=t&source=bh&ssel=0&tsel=0&kc=1&tk={TL(text)}&q={urllib.parse.quote(text)}", params=param, headers={ "responseType": "json" })
        resp.raise_for_status()
        tgt = ""
        results = resp.json()
        for result in results:
            if isinstance(result, list) and len(result) == 1 and result[0]:
                tgt += result[0]
        return tgt
    
    async def resolve(self, paragraph: Paragraph) -> str:
        paragraph.content = await self.translate(paragraph.content, paragraph.lang[:2], self.to_lang)
        return paragraph


class MachineTranslation(PipelineStage):
    """Machine Translation on Local Machine
    @zhs 本地机器翻译"""

    def __init__(self, to_lang='zhs', model='opus-mt') -> None:
        """
        Args:
            to_lang (LANG):
                Target language
                @zhs 目标语言标识
            model (opus-mt|mbart50_m2m):
                Model for translation
                @zhs 机器翻译所使用的模型 (opus-mt 较快速度, mbart50_m2m 较高准确度)
        """
        super().__init__()

        self.model = safe_import('easynmt').EasyNMT(model)

        self.opencc = None
        if to_lang == 'zhs':
            to_lang = 'zh'
        elif to_lang == 'zht':
            to_lang = 'zh'
            self.opencc = safe_import(
                'opencc', 'opencc-python-reimplemented').OpenCC('s2t')

        self.to_lang = to_lang

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        translated = self.model.translate(
            paragraph.content,
            source_lang=paragraph.lang if paragraph.lang not in (
                'zhs', 'zht') else 'zh',
            target_lang=self.to_lang)
        if self.opencc:
            translated = self.opencc.convert(translated)
        paragraph.content = translated
        return paragraph


class MachineTranslationPlugin(Plugin):
    """Plugin for machin translations
    """

    def __init__(self, pmanager, **config) -> None:
        super().__init__(pmanager, **config)
        self.register_pipelines(globals())
