"""
AMap LBS API
高德地图地理位置服务 API
"""
from typing import Iterable
import pandas as pd
import os
import json
import requests
import math

from jindai.models import Paragraph
from jindai.pipeline import DataSourceStage, PipelineStage
from jindai.plugin import Plugin
from jindai.models import Paragraph
from jindai.helpers import safe_import


gcj2wgs = safe_import('geojson_utils').gcj02towgs84
pluginConfig = {}


class AMapCityCodeQuery(DataSourceStage):
    
    def apply_params(self, content: str):
        """
        Args:
            content (str): Query keywords
                @zhs 查询条件
        """
        self.query = [q for q in content.split() if q]
        
    def fetch(self):
        df = pd.read_excel(os.path.join(os.path.dirname(__file__), 'AMap_adcode_citycode.xlsx'))
        for _, data in df.iterrows():
            for field in data:
                for q in self.query:
                    if q in str(field):
                        yield Paragraph(content=q[0], adcode=q[1], citycode=q[2])


class AMapPOISearch(DataSourceStage):
        
    def apply_params(self, content: str, adcode: str, category: str = ''):
        """
        Args:
            content (str): Query keywords
                @zhs 查询关键字
            adcode (str): ADCode for city/region
                @zhs 要查询的范围编码
            category (LINES): 要查询的行业范围编码
        """
        self.content = content
        self.adcode = adcode
        self.category = '|'.join(self.parse_lines(category))
        self.geoutil = safe_import('geojson_utils')
        
    def fetch(self) -> Iterable[Paragraph]:
        
        gcjconv = GCJtoWGS(field='coordinate', out_format='lat_lng').resolve

        total_pages = 1
        page = 0
        url_template = f"https://restapi.amap.com/v3/place/text?key={pluginConfig['apikey']}&keywords={self.content}&citylimit=true&city={self.adcode}&types={self.category}&page=%d"

        while page < total_pages:
            try:
                self.logger("{}/{}".format(page, total_pages))
                j = requests.get(url_template % page).content
                j = json.loads(j)
                
                for poi in j["pois"]:
                    lng, lat = GCJtoWGS.convert(poi["location"])
                    
                    feature = {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [[lng, lat]],
                        },
                        "properties": {
                            "name": poi["name"],
                            "region": self.adcode,
                            "biz": poi["typecode"] + " " + poi["type"],
                        },
                    }
                    p = Paragraph(content=poi['name'], 
                                    adcode=self.adcode,
                                    category=poi["typecode"] + " " + poi["type"],
                                    coordinate=poi['location'],
                                    geojson=feature)
                    gcjconv(p)
                    yield p
                
                total_pages = int(math.ceil(int(j["count"]) / 20))
            except requests.ConnectionError:
                pass
            finally:
                page += 1
                
                
class GCJtoWGS(PipelineStage):
    """
    GCJ to WGS Coordinate Conversion
    @zhs GCJ 到 WGS-84 坐标系转换
    """
    
    def __init__(self, field='coordinate', in_format='', out_format='', name='') -> None:
        """
        Args:
            field (str): Coordinate to read from
                @zhs 要转换的坐标字段
            in_format (lng_lat|lat_lng): Lng/Lat order
                @zhs 输入经纬度顺序
            out_format (lng_lat|lat_lng): Lng/Lat order
                @zhs 输出经纬度顺序
        """
        super().__init__(name)
        self.field = field
        self.in_format = in_format
        self.out_format = out_format
        
    @staticmethod
    def convert(coords, in_format=''):
        if not coords:
            return
        if isinstance(coords, str):
            coords = coords.split(',')
        if isinstance(coords, (list, tuple)):
            lng, lat = [float(_) for _ in coords]
        if in_format == 'lat_lng':
            lat, lng = lng, lat
        lng, lat = gcj2wgs(lng, lat)
        return [lng, lat]
    
    def resolve(self, paragraph: Paragraph) -> Paragraph:
        lng, lat = self.convert(paragraph[self.field], self.in_format)
        if self.out_format == 'lat_lng':
            paragraph[self.field] = [lat, lng]
        else:
            paragraph[self.field] = [lng, lat]


class AMapLBSPlugin(Plugin):
    
    def __init__(self, pmanager, **conf) -> None:
        super().__init__(pmanager, **conf)
        pluginConfig.update(**conf)
        self.register_pipelines(globals())
