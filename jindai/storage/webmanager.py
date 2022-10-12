"""Web manager"""

import urllib
from io import BytesIO, IOBase, UnsupportedOperation

import requests

from .storage import StorageManager


class WebManager(StorageManager):
    """Read/write with web requests
    """

    class _ResponseStream(IOBase):
        """Response stream
        """

        def __init__(self, req, attempts=3, proxies=None, verify=False, timeout=60):
            super().__init__()
            self._pos = 0
            self._seekable = True
            self.req = req
            self.attempts = attempts
            self.verify = verify
            self.timeout = timeout
            self.proxies = proxies

            with self._urlopen() as resp:
                self.content_length = int(resp.headers.get('content-length', -1))
                if self.content_length < 0:
                    self._seekable = False
                self.st_size = self.content_length

        def seek(self, offset, whence=0):
            if not self.seekable():
                raise UnsupportedOperation
            if whence == 0:
                self._pos = 0
            elif whence == 1:
                pass
            elif whence == 2:
                self._pos = self.content_length
            self._pos += offset
            return self._pos

        def seekable(self, *args, **kwargs):
            return self._seekable

        def readable(self, *args, **kwargs):
            return not self.closed

        def writable(self, *args, **kwargs):
            return False

        def read(self, amt=-1):
            """Read bytes

            Args:
                amt (int, optional): Defaults to -1.

            Returns:
                bytes: bytes read
            """
            if not self._seekable:
                return self._urlopen().content

            if self._pos >= self.content_length:
                return b""
            if amt < 0:
                end = self.content_length - 1
            else:
                end = min(self._pos + amt - 1, self.content_length - 1)
            byte_range = (self._pos, end)
            self._pos = end + 1

            with self._urlopen(byte_range) as f:
                return f.content

        def readall(self):
            return self.read(-1)

        def tell(self):
            return self._pos

        def _urlopen(self, byte_range=None):

            if byte_range:
                self.req.headers['Range'] = '{}-{}'.format(*byte_range)
            else:
                self.req.headers.pop('Range', '')

            for _ in range(self.attempts):
                with WebManager.session() as s:
                    resp = s.send(
                        self.req, stream=True, proxies=self.proxies,
                        verify=self.verify, timeout=self.timeout)
                    if resp.status_code != 200:
                        raise OSError(f'HTTP {resp.status_code}: reading {self.req.url}')

    @staticmethod
    def _build_proxies(proxies):
        """Build proxies dict

        Args:
            proxies (str | dict): proxy specification

        Returns:
            dict: proxies dict
        """
        if proxies and not isinstance(proxies, dict):
            return {
                'http': proxies,
                'https': proxies
            }
        else:
            return proxies or {}

    @staticmethod
    def session():
        """Create a new requests session

        Returns:
            _type_: _description_
        """
        return requests.session()

    def _build_request(self, url: str, method='GET', referer: str = '',
                       headers=None, data=None, **_):
        """Build a request

        :param url: target url
        :type url: str
        :param method: request  method, defaults to 'GET'
        :type method: str, optional
        :param referer: referer url, defaults to ''
        :type referer: str, optional
        :param headers: headers, defaults to None
        :type headers: dict, optional
        :param data: request body data, defaults to None
        :type data: bytes, optional
        :return: request context
        :rtype: requests.Request
        """

        if headers is None:
            headers = {}

        headers.update(**{
            "user-agent": "Mozilla/5.1 (Windows NT 6.0) Gecko/20180101 Firefox/23.5.1"
        })

        if isinstance(url, tuple):
            url, referer = url

        if referer is not None:
            headers["referer"] = referer.encode('utf-8')

        if data and isinstance(data, bytes):
            headers['content-type'] = 'application/octet-stream'

        return requests.Request(url=url, method=method, headers=headers,
                                data=data)

    def __init__(self, *_, attempts=3, verify=False, timeout=30, seekable=False):
        """
        Args:
            attempts (int, optional): Maximal retries when fetching data from server. Defaults to 3.
            verify (bool, optional): Verify SSL Certs. Defaults to False.
            timeout (int, optional): Timeout in seconds. Defaults to 30.
        """
        self.attempts = attempts
        self.verify = verify
        self.timeout = timeout
        self.seekable = seekable

    def write(self, path: str, data: bytes, method='POST', proxy=None, **params) -> BytesIO:
        req = self._build_request(path, method, data=data, **params).prepare()
        with self.session() as s:
            resp = s.send(req, proxies=WebManager._build_proxies(
                proxy), verify=self.verify, timeout=self.timeout)
        if resp.status_code != 200:
            raise OSError(f'HTTP {resp.status_code}: writing {path}')
        return True

    def readbuf(self, path: str, proxy=None, **params) -> IOBase:
        req = self._build_request(path, **params).prepare()
        return WebManager._ResponseStream(req, self.attempts, proxies=WebManager._build_proxies(proxy), verify=self.verify, timeout=self.timeout)

    def read(self, path: str, proxy=None, **params) -> BytesIO:
        req = self._build_request(path, **params).prepare()
        with self.session() as s:
            return BytesIO(s.send(
                req, proxies=WebManager._build_proxies(proxy),
                verify=self.verify, timeout=self.timeout
            ).content)

    def exists(self, path: str) -> bool:
        return requests.get(path, timeout=30).status_code == 200

    def join(self, base_path: str, *path_segs: str) -> str:
        return urllib.parse.urljoin(base_path, '/'.join(path_segs))

    def listdir(self, path: str) -> list:
        return []

    def stat(self, path: str) -> dict:
        res = super().stat(path)
        res['type'] = 'file'
        return res


class DataSchemeManager(StorageManager):
    """Handle with data: scheme
    """

    def exists(self, path: str) -> bool:
        return True

    def read(self, path, **params):
        with urllib.request.urlopen(path) as response:
            data = response.read()
        return data
