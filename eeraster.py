import numpy as np
import pandas as pd
import ee
import eeddf
from typing import Tuple, List


_LONGITUDE_COLUMN_NAME = "long"
_LATITUDE_COLUMN_NAME = "lat"
_demXfab = None
_dem = None

"""
`eeRaster` represents an Earth Engine raster. 
The `eeRaster` class has a single method called `value_at`
which takes a float `lat` and `lon` as parameters and returns a float.

The `value_at` method returns the value of the raster at the specified
latitude and longitude.
"""
class eeRaster:
    """
    A class representing an Earth Engine raster.

    Args:
        imageCollection (ee.ImageCollection): The Earth Engine ImageCollection
        that represents the raster. This should be a single band normally
        specified as:
        ee.ImageCollection('projects/<project>/<collection>').select("<band>")
    """
    def __init__(self, imageCollection: ee.ImageCollection):
      self._imageCollection = imageCollection

    @staticmethod
    def demXfab():
      """
      Returns an eeRaster representing digital elevation minus forest and
      buildings. See https://gee-community-catalog.org/projects/fabdem/
      """
      global _demXfab
      eeddf.initialize_ddf()
      if (_demXfab is None):
        _demXfab = eeRaster(ee.ImageCollection(
          'projects/sat-io/open-datasets/FABDEM').select("b1"))
      return _demXfab

    @staticmethod
    def dem():
      """
      Returns an eeRaster representing digital elevation (copernicus).
      See https://gee-community-catalog.org/projects/glo30/
      """
      eeddf.initialize_ddf()
      global _dem
      if (_dem is None):
        _dem = eeRaster(ee.ImageCollection(
          'projects/sat-io/open-datasets/GLO-30').select("b1"))
      return _dem

    def value_at(self, lat: float, lon: float) -> float:
        """
        Returns the value of the raster at the specified latitude and longitude.

        Args:
            lat (float): The latitude coordinate.
            lon (float): The longitude coordinate.

        Returns:
            float: The value of the raster at the specified latitude and longitude.
        """
        return self.values_at([[lat, lon]])[0]

    def values_at(self, coordinates:List[Tuple[float, float]]) -> List[float]:
      """
      Returns the values of the raster at the specified latitudes and longitudes.

      Args:
          coordinates: A list of tuples consisting of [latitude, longitude].

      Returns:
          List[float]: The values of the raster at the specified latitudes and
          longitudes.
      """
      df = self.values_at_df(pd.DataFrame(coordinates, columns=[
        _LATITUDE_COLUMN_NAME, _LONGITUDE_COLUMN_NAME]))

      value_list = df["value"].tolist()
      return value_list

    def values_at_df(self, coordinates:pd.DataFrame, column_name: str = "value") -> pd.DataFrame:
      """
      Returns the values of the raster at the specified latitudes and longitudes.

      Args:
          coordinates: A list of tuples consisting of [latitude, longitude].

      Returns:
          List[float]: The values of the raster at the specified latitudes and
          longitudes.
      """
      coordinates[_LATITUDE_COLUMN_NAME] = coordinates[_LATITUDE_COLUMN_NAME].astype(float).round(5)
      coordinates[_LONGITUDE_COLUMN_NAME] = coordinates[_LONGITUDE_COLUMN_NAME].astype(float).round(5)

      collection = coordinates.apply(lambda row: ee.Feature(ee.Geometry.Point(
        [row[_LONGITUDE_COLUMN_NAME], row[_LATITUDE_COLUMN_NAME]])), axis=1).to_list()
      points = ee.FeatureCollection(collection)
      image = self._imageCollection.filterBounds(points).mosaic().setDefaultProjection('EPSG:3857')
      info = image.reduceRegions(points, ee.Reducer.median()).getInfo()

      features = info['features']
      dictarr = []

      for f in features:
        attr = f['properties']
        attr[_LATITUDE_COLUMN_NAME] = f['geometry']['coordinates'][1]
        attr[_LONGITUDE_COLUMN_NAME] = f['geometry']['coordinates'][0]
        dictarr.append(attr)

      df = pd.DataFrame(dictarr).rename(columns={'median': column_name})
      df[_LATITUDE_COLUMN_NAME] = df[_LATITUDE_COLUMN_NAME].round(5)
      df[_LONGITUDE_COLUMN_NAME] = df[_LONGITUDE_COLUMN_NAME].round(5)

      df_final = coordinates.merge(df, how='left')
      if column_name not in df_final.columns:
        df_final[column_name] = np.nan

      return df_final



