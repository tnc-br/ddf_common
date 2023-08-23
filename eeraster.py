import numpy as np
import pandas as pd
import ee
import eeddf
import math
import raster
from typing import Tuple, List
from multiprocessing import Pool


_PARALLEL_OPS = 30
_CHUNK_SIZE = 80
_LONGITUDE_COLUMN_NAME = "long"
_LATITUDE_COLUMN_NAME = "lat"
_demXfab = None
_dem = None

#global pool for all rasters in cases of simultaneous parallelization
_pool = None
def getPool():
  global _pool
  if (_pool is None):
    _pool = Pool(processes=_PARALLEL_OPS)
  return _pool

def _query_mp(image, coordinates: pd.DataFrame, column_name: str) -> pd.DataFrame:
  collection = coordinates.apply(lambda row: ee.Feature(ee.Geometry.Point(
    [row[_LONGITUDE_COLUMN_NAME], row[_LATITUDE_COLUMN_NAME]])), axis=1).to_list()

  points = ee.FeatureCollection(collection)
  info = image.reduceRegions(points, ee.Reducer.first(), scale=30, crs='EPSG:3857').getInfo()

  features = info['features']
  dictarr = []

  for f in features:
    attr = f['properties']
    attr[_LATITUDE_COLUMN_NAME] = f['geometry']['coordinates'][1]
    attr[_LONGITUDE_COLUMN_NAME] = f['geometry']['coordinates'][0]
    dictarr.append(attr)

  df = pd.DataFrame(dictarr).rename(columns={'first': column_name})
  df[_LATITUDE_COLUMN_NAME] = df[_LATITUDE_COLUMN_NAME].round(5)
  df[_LONGITUDE_COLUMN_NAME] = df[_LONGITUDE_COLUMN_NAME].round(5)

  df_final = coordinates.merge(df, how='left')
  if column_name not in df_final.columns:
    df_final[column_name] = np.nan

  return df_final

"""
`eeRaster` represents an Earth Engine raster. 
The `eeRaster` class has a single method called `value_at`
which takes a float `lat` and `lon` as parameters and returns a float.

The `value_at` method returns the value of the raster at the specified
latitude and longitude.
"""
class eeRaster(raster.AmazonGeoTiffBase):
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

    def values_at_df(self, coordinates:pd.DataFrame, column_name: str = "value") -> pd.DataFrame:
      """
      Returns the values of the raster at the specified latitudes and longitudes.

      Args:
          coordinates: A list of tuples consisting of [latitude, longitude].

      Returns:
          List[float]: The values of the raster at the specified latitudes and
          longitudes.
      """
      if len(coordinates) == 0:
        return coordinates
      coordinates[_LATITUDE_COLUMN_NAME] = coordinates[_LATITUDE_COLUMN_NAME].astype(float).round(5)
      coordinates[_LONGITUDE_COLUMN_NAME] = coordinates[_LONGITUDE_COLUMN_NAME].astype(float).round(5)

      query_list = []
      start = 0
      image = self._imageCollection.mosaic()
      while (start < len(coordinates)):
        end = start + _CHUNK_SIZE
        query_list.append([image, coordinates.iloc[start:end, :], column_name])
        start = end

      all_dfs = getPool().starmap(_query_mp, query_list)
      return pd.concat(all_dfs, ignore_index=True)

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



