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
    """
    Query AMap City Codes
    @zhs 查询高德地图城市代码表
    """

    def apply_params(self, content: str):
        """
        Args:
            content (str): Query keywords
                @zhs 查询条件
        """
        self.query = [q for q in content.split() if q]

    def fetch(self):
        df = pd.read_excel(os.path.join(os.path.dirname(
            __file__), 'AMap_adcode_citycode.xlsx'))
        for _, data in df.iterrows():
            for field in data:
                for q in self.query:
                    if q in str(field):
                        yield Paragraph(content=q[0], adcode=q[1], citycode=q[2])


class AMapPOISearch(DataSourceStage):
    """
    Search POIs with AMap
    @zhs 查询高德地图位置信息，可根据关键字、城市代码和类别信息限定
    """

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
        url_template = f"https://restapi.amap.com/v3/place/text?key={pluginConfig['amap_key']}&keywords={self.content}&citylimit=true&city={self.adcode}&types={self.category}&page=%d"

        while page < total_pages:
            try:
                self.log("{}/{}".format(page, total_pages))
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


class _GeoCodingStage(PipelineStage):
    """
    Geocoding parent class, do not use directly
    """

    def __init__(self, field: str = 'coordinate', out_format: str = 'lat_lng'):
        """
        Args:
            field (str): Field name for coordinate
                @zhs 坐标字段
            out_format (lng_lat|lat_lng): Lng/Lat order
                @zhs 输出经纬度顺序
        """
        self.out_format = out_format
        self.field = field

    def assign_coordinates(self, paragraph, lat, lng):
        if self.out_format == 'lat_lng':
            paragraph[self.field] = [lat, lng]
        else:
            paragraph[self.field] = [lng, lat]
        return paragraph


class AMapGeoCode(_GeoCodingStage):
    """
    Geo-coding with AMap
    @zhs 高德地图地理编码
    """

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        url = f"https://restapi.amap.com/v3/geocode/geo?key={pluginConfig['amap_key']}&address={self.content}"
        resp = requests.get(url).json()
        location = resp['geocodes'][0]['location']
        lng, lat = GCJtoWGS.convert(location)
        return self.assign_coordinates(paragraph, lat, lng)


class GCJtoWGS(_GeoCodingStage):
    """
    GCJ to WGS Coordinate Conversion
    @zhs GCJ 到 WGS-84 坐标系转换
    """

    def __init__(self, *args, in_format='', **kwargs) -> None:
        """
        Args:
            field (str): Coordinate to read from
                @zhs 要转换的坐标字段
            in_format (lng_lat|lat_lng): Lng/Lat order
                @zhs 输入经纬度顺序
            out_format (lng_lat|lat_lng): Lng/Lat order
                @zhs 输出经纬度顺序
        """
        super().__init__(*args, **kwargs)
        self.in_format = in_format

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
        return self.assign_coordinates(paragraph, lat, lng)


class GoogleMapGeoCode(_GeoCodingStage):
    """
    Geo-coding with Google Maps
    @zhs 查询谷歌地图位置信息
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        googlemaps = safe_import('googlemaps')
        self.gmaps = googlemaps.Client(key=pluginConfig['google_key'])

    def resolve(self, paragraph: Paragraph):
        coordinates = self.gmaps.geocode(paragraph.content)[
            'geometry']['location']
        return self.assign_coordinates(paragraph, lat, lng)


class BingMapGeoCode(_GeoCodingStage):
    """
    Geo-coding with Bing Maps
    @zhs 查询必应地图位置信息
    """

    def resolve(self, paragraph: Paragraph):
        url = f'http://dev.virtualearth.net/REST/v1/Locations?q={paragraph.content}&output=json&key={pluginConfig["bing_key"]}'
        outp = requests.get(url).json()
        lng, lat = outp['resourceSets'][0]['resources'][0]['point']['coordinates']
        return self.assign_coordinates(paragraph, lat, lng)
    

class OSMPOISearch(DataSourceStage):
    """
    POI Search with OSM (OpenStreetMap)
    """

    def apply_params(self, content: str = '', tags: str = '', ):
        """
        Args:
            tags (LINES): Tags
            place (str): City name to search in
        """
        self.tags = self.parse_lines(tags)
        self.ox = safe_import('omnx')
        #   tags = [
        # # 'feature descriptions', 'proposed features', 'features/translations', 
        # 'accommodation', 'addresses', 'agriculture', 'amenities', 
        # # 'barriers', 'boundaries', 
        # 'clothes', 'commerce', 'conservation', 'disabilities', 'education', 'emergencies', 'environment',
        # 'hazards', 'health', 'heritage', 'historic', 'infrastructure', 'leisure',
        # 'man made', 'meta features', 'micromapping', 'military',
        # # 'names',
        # 'offices',
        # 'places', 
        # 'police', 'properties', 'religion', 'social facilities', 'sports', 'transport',
        # # 'water', 'templates for mapping features',
        # ]
        self.city = self.ox.geocode_to_gdf(content)

    def guesses(self, tag):
        yield tag
        if tag.endswith('s'):
            tag = tag[:-1]
            yield tag
            if tag.endswith('ie'):
                tag = tag[:-2] + 'y'
            elif tag.endswith('e'):
                tag = tag[:-1]
            yield tag
    
    def fetch(self) -> Iterable[Paragraph]:
        for tag in self.tags:
            for guess in set(self.guesses(tag)):
                for p in self.ox.geometries_from_polygon(
                    self.city['geometry'].all(), tags={guess: True}):
                    yield Paragraph(**p)


class LBSPlugin(Plugin):

    def __init__(self, pmanager, **conf) -> None:
        super().__init__(pmanager, **conf)
        pluginConfig.update(**conf)
        self.register_pipelines(globals())
