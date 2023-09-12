import numpy as np
import pandas as pd
import ee
import eeddf
import math
import os
import raster
import time
from typing import Tuple, List, Dict, Any
from multiprocessing import Pool
from google.cloud import storage
from osgeo import gdal



_PARALLEL_OPS = 30
_CHUNK_SIZE = 80
_LONGITUDE_COLUMN_NAME = "long"
_LATITUDE_COLUMN_NAME = "lat"
_CRS = 'EPSG:3857'
_demXfab = None
_dem = None
_BUCKET_NAME = "unprocessed-isoscapes"

# Location of the pre-prod Oxygen isoscape asset. 
# Isoscapes saved here need to be manually copied to prod environments.
STAGING_OXYGEN_EE_ASSET = "projects/river-sky-386919/assets/isoscapes/d18O_isocape"

#global pool for all rasters in cases of simultaneous parallelization
_pool = None
def _getPool():
  global _pool
  if (_pool is None):
    _pool = Pool(processes=_PARALLEL_OPS)
  return _pool

def _query_mp(image, coordinates: pd.DataFrame, column_name: str, 
  crs_value: str, lon_name: str, lat_name:str) -> pd.DataFrame:
  collection = coordinates.apply(lambda row: ee.Feature(ee.Geometry.Point(
    [row[lon_name], row[lat_name]])), axis=1).to_list()

  points = ee.FeatureCollection(collection)
  info = image.reduceRegions(points, ee.Reducer.first(), scale=30, crs= crs_value).getInfo()

  features = info['features']
  dictarr = []

  for f in features:
    attr = f['properties']
    attr[lat_name] = f['geometry']['coordinates'][1]
    attr[lon_name] = f['geometry']['coordinates'][0]
    dictarr.append(attr)

  df = pd.DataFrame(dictarr).rename(columns={'first': column_name})
  df[lat_name] = df[lat_name].round(5)
  df[lon_name] = df[lon_name].round(5)

  df_final = coordinates.merge(df, how='left')
  if column_name not in df_final.columns:
    df_final[column_name] = np.nan

  return df_final

def _block_until_ee_operation_completion(operation_name: str):
  """
  Query EarthEngine operation manager until the operation finished. 
  Throws assertion error if the operation does not succeed.

  Args:
    operation_name: id of the operation.
  """
  while (
    ee.data.getOperation(operation_name)['metadata']['state'] \
     in ['PENDING', 'RUNNING']):
     time.sleep(5)

  assert ee.data.getOperation(operation_name)['metadata']['state'] == 'SUCCEEDED'

def set_ee_options(parallel_ops: int = _PARALLEL_OPS,
 chunk_size: int = _CHUNK_SIZE, crs: str = _CRS,
 lon_name: str = _LONGITUDE_COLUMN_NAME, lat_name:str = _LATITUDE_COLUMN_NAME):
  """
  Sets the execution options, such as parallelization level as well as the
  Coordinate Reference System and columns to use for longitude/latitude within
  the DataFrame methods.
  Note that this call sets options globally.

  Args:
      parallel_ops: The number of simultaneous calls to EE (number of processes).
      chunk_size: The number of lat/lon points queried in a single batch.
      crs: The Coordinate Reference System to use.
      lon_name: The name of the dataframe column holding longitude.
      lat_name: The name of the dataframe column holding latitude.
  """
  _PARALLEL_OPS = parallel_ops
  _CHUNK_SIZE = chunk_size
  _CRS = crs
  _LONGITUDE_COLUMN_NAME = lon_name
  _LATITUDE_COLUMN_NAME = lat_name

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
      
      #We round the input to EE and the output from EE to 5 decimals because
      #earth engine very often sends back lat/lon that are off from what was
      #sent by what looks like could be epsilon, meaning a bit representation
      #issue affected the roundtrip.

      #five decimals puts the precision at 11 meters, which is actually smaller
      #than the scale used for the query at 30 meters.
      coordinates[_LATITUDE_COLUMN_NAME] = coordinates[_LATITUDE_COLUMN_NAME].astype(float).round(5)
      coordinates[_LONGITUDE_COLUMN_NAME] = coordinates[_LONGITUDE_COLUMN_NAME].astype(float).round(5)

      query_list = []
      start = 0
      image = self._imageCollection.mosaic()
      while (start < len(coordinates)):
        end = start + _CHUNK_SIZE
        query_list.append([image, coordinates.iloc[start:end, :], column_name,
         _CRS, _LONGITUDE_COLUMN_NAME, _LATITUDE_COLUMN_NAME])
        start = end

      all_dfs = _getPool().starmap(_query_mp, query_list)
      return pd.concat(all_dfs, ignore_index=True)

def set_props(ee_asset_path: str, properties: Dict[str, Any]):
  """
  set_props function
  ------------------------
  This function sets properties on an Earth Engine isoscape.
  This function can also be used to edit an existing property.
  ------------------------
  Parameters:
  ee_asset_path : str
      GeoTIFF filename in Earth Engine (with the full path) of the isoscape to be stamped
      e.g: projects/river-sky-386919/assets/isoscapes/d18O_isoscape
  properties: Dict[str, Any]
      A dictionary of metadata to apply to the Earth Engine asset
      e.g. {"p_value" : 0.95}
  """
  eeddf.initialize_ddf()
  ee.data.updateAsset(ee_asset_path, {'properties': properties}, ['properties'])

def get_props(ee_asset_path: str):
  """
  get_props function
  ------------------------
  This function returns all the properties on an Earth Engine isoscape
  ------------------------
  Parameter:
  ee_asset_path : str
      GeoTIFF filename in Earth Engine (with the full path) of the isoscape to be stamped
      e.g: projects/river-sky-386919/assets/isoscapes/d18O_isoscape
  """
  eeddf.initialize_ddf()
  isoscape_metadata = ee.data.getAsset(ee_asset_path)
  return isoscape_metadata['properties']

def del_prop(ee_asset_path: str, property_name: str):
  """
  del_prop function
  ------------------------
  This function deletes a property from an Earth Engine isoscape
  ------------------------
  Parameters:
  filename : str
      GeoTIFF filename (with the full path) of the stamped isoscape
      e.g: projects/river-sky-386919/assets/isoscapes/d18O_isoscape
  property_name : str
      Name of the metadata to be deleted
      e.g: "p_value"
  """  

def ingest_isoscape(
    isoscape_path: str,
    ee_dst_path: str=STAGING_OXYGEN_EE_ASSET,
    allow_overwrite: bool=False):
  """
  ingest_isoscape function
  ------------------------
  Ingests the isoscape in earth engine, and stamps it with `properties`. Requires access
  to the "unprocessed-isoscape" cloud storage bucket and to the EE API for project river-sky-386919.
  ------------------------
  Parameters:
  filename : isoscape_path
      GeoTIFF filename (with the full path) of the stamped isoscape
      e.g: /content/gdrive/Shared drives/TNC Fellowship 🌳/4. Isotope Research & Signals/code/amazon_rainforest_files/amazon_rasters/variational/ensemble_with_carbon_brisoisorix/fixed_isorix_carbon_ensemble.tiff
  properties: Dict[str, Any]
      A dictionary of metadata to apply to the Earth Engine asset
      e.g. {"p_value" : 0.95}
  ee_dst_path: Path of the new asset to create in Earth Engine.
      e.g.  projects/river-sky-386919/assets/staging/test_isoscape
  allow_overwrite: bool
      If true, allows user to overwrite an existing isoscape with the same filename and ee_dst_path. 
  """  
  eeddf.initialize_ddf()

  # 1. Upload to Google Cloud Storage. This is unfortuntaly required to upload to EE using their API.
  storage_client = storage.Client()
  
  global _BUCKET_NAME
  bucket = storage_client.bucket(_BUCKET_NAME)

  _, isoscape_filename = os.path.split(isoscape_path)
  blob = bucket.blob(isoscape_filename)
  
  blob.upload_from_filename(isoscape_path, if_generation_match=(None if allow_overwrite else 0))

  # 2. Copy the file from Google Cloud Storage to Earth Engine. 
  ee_request_id = ee.data.newTaskId()[0]
  params = {
    "name": ee_dst_path,
    "tilesets": [{"sources": [{"uris": [f"gs://{_BUCKET_NAME}/{isoscape_filename}"]}]}]
  }

  print("Starting ingestion to ", ee_dst_path)
  op = ee.data.startIngestion(
    request_id=ee_request_id,
    params=params,
    allow_overwrite=allow_overwrite)
  _block_until_ee_operation_completion(op['name'])

  # 3. Copy the metadata to the EE asset as properties.
  dataset = gdal.Open(isoscape_path)
  metadata = dataset.GetMetadata()
  set_props(ee_dst_path, metadata)

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



