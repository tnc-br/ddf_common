from abc import ABC, abstractmethod
from dataclasses import dataclass
from osgeo import gdal, gdal_array
import numpy as np
import pandas as pd
from typing import Tuple, List
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import math
import glob
import os
import matplotlib.animation as animation
from sklearn.compose import ColumnTransformer

import model


GDRIVE_BASE = "/content/gdrive"
RASTER_BASE = "/MyDrive/amazon_rainforest_files/amazon_rasters/" #@param
MODEL_BASE = "/MyDrive/amazon_rainforest_files/amazon_isoscape_models/" #@param
SAMPLE_DATA_BASE = "/MyDrive/amazon_rainforest_files/amazon_sample_data/" #@param
ANIMATIONS_BASE = "/MyDrive/amazon_rainforest_files/amazon_animations/" #@param
TEST_DATA_BASE = "/MyDrive/amazon_rainforest_files/amazon_test_data/" #@param
_LONGITUDE_COLUMN_NAME = "long"
_LATITUDE_COLUMN_NAME = "lat"

# Module for helper functions for manipulating data and datasets.
class AmazonGeoTiffBase(ABC): # Inherit from ABC(Abstract base class)
    def value_at(self, lon: float, lat: float) -> float:
        """
        Returns the value of the raster at the specified latitude and longitude.

        Args:
            lat (float): The latitude coordinate.
            lon (float): The longitude coordinate.

        Returns:
            float: The value of the raster at the specified latitude and longitude.
        """
        return self.values_at([[lon, lat]])[0]

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
        _LONGITUDE_COLUMN_NAME, _LATITUDE_COLUMN_NAME]))

      value_list = df["value"].tolist()
      return value_list

    @abstractmethod
    def values_at_df(self, df: pd.DataFrame, column_name: str = "value") -> pd.DataFrame:
      """
      Returns the values of the raster at the specified latitudes and longitudes.

      Args:
          coordinates: A list of tuples consisting of [latitude, longitude].

      Returns:
          List[float]: The values of the raster at the specified latitudes and
          longitudes.
      """
      pass

@dataclass
class AmazonGeoTiff(AmazonGeoTiffBase):
  """Represents a geotiff from our dataset."""
  gdal_dataset: gdal.Dataset
  image_value_array: np.ndarray # ndarray of floats
  image_mask_array: np.ndarray # ndarray of uint8
  masked_image: np.ma.masked_array
  yearly_masked_image: np.ma.masked_array
  name: str = "" # The name of the raster and column for dataframe.

  def value_at(self, x: float, y: float) -> float:
    return get_data_at_coords(self, x, y, -1)

  def values_at_df(self, df: pd.DataFrame, column_name: str = "value") -> pd.DataFrame:
    for i, row in df.iterrows():
      lat = df.loc[i, "lat"]
      lon = df.loc[i, "long"]
      df.loc[i, column_name] =  _try_get_data_at_coords(self, lon, lat, -1)  
    return df


@dataclass
class Bounds:
  """Represents geographic bounds and size information."""
  minx: float
  maxx: float
  miny: float
  maxy: float
  pixel_size_x: float
  pixel_size_y: float
  raster_size_x: float
  raster_size_y: float

  def to_matplotlib(self) -> List[float]:
    return [self.minx, self.maxx, self.miny, self.maxy]

def get_raster_path(filename: str) -> str:
  global GDRIVE_BASE
  global RASTER_BASE
  root = GDRIVE_BASE if GDRIVE_BASE else ""
  return f"{root}{RASTER_BASE}{filename}"

def get_model_path(filename: str) -> str:
  global GDRIVE_BASE
  global MODEL_BASE

  root = GDRIVE_BASE if GDRIVE_BASE else ""
  return f"{root}{MODEL_BASE}{filename}"

def get_sample_db_path(filename: str) -> str:
  global GDRIVE_BASE
  global SAMPLE_DATA_BASE

  root = GDRIVE_BASE if GDRIVE_BASE else ""
  return f"{root}{SAMPLE_DATA_BASE}{filename}"

def get_animations_path(filename: str) -> str:
  global GDRIVE_BASE
  global ANIMATIONS_BASE

  root = GDRIVE_BASE if GDRIVE_BASE else ""
  return f"{root}{ANIMATIONS_BASE}{filename}"

def mount_gdrive():
  if not os.path.exists(GDRIVE_BASE):
    # Access data stored on Google Drive
    if GDRIVE_BASE:
      try:
        from google.colab import drive
        drive.mount(GDRIVE_BASE)
      except ModuleNotFoundError:
        print('WARNING, GDRIVE NOT MOUNTED! USING LOCAL FS!!!')

      if not os.path.exists(GDRIVE_BASE):
        print('CREATING A LOCAL FOLDER FOR SOURCE!!!!')
        os.makedirs(GDRIVE_BASE, exist_ok=True)


def print_raster_info(raster):
  dataset = raster
  print("Driver: {}/{}".format(dataset.GetDriver().ShortName,
                              dataset.GetDriver().LongName))
  print("Size is {} x {} x {}".format(dataset.RasterXSize,
                                      dataset.RasterYSize,
                                      dataset.RasterCount))
  print("Projection is {}".format(dataset.GetProjection()))
  geotransform = dataset.GetGeoTransform()
  if geotransform:
      print("Origin = ({}, {})".format(geotransform[0], geotransform[3]))
      print("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))

  for band in range(dataset.RasterCount):
    band = dataset.GetRasterBand(band+1)
    #print("Band Type={}".format(gdal.GetDataTypeName(band.DataType)))

    min = band.GetMinimum()
    max = band.GetMaximum()
    if not min or not max:
        (min,max) = band.ComputeRasterMinMax(False)
    #print("Min={:.3f}, Max={:.3f}".format(min,max))

    if band.GetOverviewCount() > 0:
        print("Band has {} overviews".format(band.GetOverviewCount()))

    if band.GetRasterColorTable():
        print("Band has a color table with {} entries".format(band.GetRasterColorTable().GetCount()))

def load_named_raster(path: str, name: str, use_only_band_index: int = -1) -> AmazonGeoTiff:
  raster = load_raster(path, use_only_band_index)
  raster.name = name
  return raster

def load_raster(path: str, use_only_band_index: int = -1) -> AmazonGeoTiff:
  """
  TODO: Refactor (is_single_band, etc., should be a better design)
  --> Find a way to simplify this logic. Maybe it needs to be more abstract.
  """
  mount_gdrive()
  dataset = gdal.Open(path, gdal.GA_ReadOnly)
  try:
    print_raster_info(dataset)
  except AttributeError as e:
    raise OSError("Failed to print raster. This likely means it did not load properly from "+ path)
  image_datatype = dataset.GetRasterBand(1).DataType
  mask_datatype = dataset.GetRasterBand(1).GetMaskBand().DataType
  image = np.zeros((dataset.RasterYSize, dataset.RasterXSize, 12),
                  dtype=gdal_array.GDALTypeCodeToNumericTypeCode(image_datatype))
  mask = np.zeros((dataset.RasterYSize, dataset.RasterXSize, 12),
                  dtype=gdal_array.GDALTypeCodeToNumericTypeCode(image_datatype))

  if use_only_band_index == -1:
    if dataset.RasterCount != 12 and dataset.RasterCount != 1:
      raise ValueError(f"Expected 12 raster bands (one for each month) or one annual average, but found {dataset.RasterCount}")
    if dataset.RasterCount == 1:
      use_only_band_index = 0

  is_single_band = use_only_band_index != -1

  if is_single_band and use_only_band_index >= dataset.RasterCount:
    raise IndexError(f"Specified raster band index {use_only_band_index}"
    f" but there are only {dataset.RasterCount} rasters")

  for band_index in range(12):
    band = dataset.GetRasterBand(use_only_band_index+1 if is_single_band else band_index+1)
    image[:, :, band_index] = band.ReadAsArray()
    mask[:, :, band_index] = band.GetMaskBand().ReadAsArray()
  masked_image = np.ma.masked_where(mask == 0, image)
  yearly_masked_image = masked_image.mean(axis=2)

  return AmazonGeoTiff(dataset, image, mask, masked_image, yearly_masked_image)

def get_extent(dataset):
  geoTransform = dataset.GetGeoTransform()
  minx = geoTransform[0]
  maxy = geoTransform[3]
  maxx = minx + geoTransform[1] * dataset.RasterXSize
  miny = maxy + geoTransform[5] * dataset.RasterYSize
  return Bounds(minx, maxx, miny, maxy, geoTransform[1], geoTransform[5], dataset.RasterXSize, dataset.RasterYSize)

def plot_band(geotiff: AmazonGeoTiff, month_index, figsize=None):
  if figsize:
    plt.figure(figsize=figsize)
  im = plt.imshow(geotiff.masked_image[:,:,month_index], extent=get_extent(geotiff.gdal_dataset).to_matplotlib(), interpolation='none')
  plt.colorbar(im)

def animate(geotiff: AmazonGeoTiff, nSeconds, fps):
  fig = plt.figure( figsize=(8,8) )

  months = []
  labels = []
  for m in range(12):
    months.append(geotiff.masked_image[:,:,m])
    labels.append(f"Month: {m+1}")
  a = months[0]
  extent = get_extent(geotiff.gdal_dataset).to_matplotlib()
  ax = fig.add_subplot()
  im = fig.axes[0].imshow(a, interpolation='none', aspect='auto', extent = extent)
  txt = fig.text(0.3,0,"", fontsize=24)
  fig.colorbar(im)

  def animate_func(i):
    if i % fps == 0:
      print( '.', end ='' )

    im.set_array(months[i])
    txt.set_text(labels[i])
    return [im, txt]

  anim = animation.FuncAnimation(
                                fig,
                                animate_func,
                                frames = nSeconds * fps,
                                interval = 1000 / fps, # in ms
                                )
  plt.close()

  return anim

def save_numpy_to_geotiff(bounds: Bounds, prediction: np.ma.MaskedArray, path: str):
  """Copy metadata from a base geotiff and write raster data + mask from `data`"""
  driver = gdal.GetDriverByName("GTiff")
  metadata = driver.GetMetadata()
  if metadata.get(gdal.DCAP_CREATE) != "YES":
      raise RuntimeError("GTiff driver does not support required method Create().")
  if metadata.get(gdal.DCAP_CREATECOPY) != "YES":
      raise RuntimeError("GTiff driver does not support required method CreateCopy().")

  dataset = driver.Create(path, bounds.raster_size_x, bounds.raster_size_y, prediction.shape[2], eType=gdal.GDT_Float64)
  dataset.SetGeoTransform([bounds.minx, bounds.pixel_size_x, 0, bounds.maxy, 0, bounds.pixel_size_y])
  dataset.SetProjection('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4326"]]')

  #dataset = driver.CreateCopy(path, base.gdal_dataset, strict=0)
  if len(prediction.shape) != 3 or prediction.shape[0] != bounds.raster_size_x or prediction.shape[1] != bounds.raster_size_y:
    raise ValueError("Shape of prediction does not match base geotiff")
  #if prediction.shape[2] > base.gdal_dataset.RasterCount:
  #  raise ValueError(f"Expected fewer than {dataset.RasterCount} bands in prediction but found {prediction.shape[2]}")

  prediction_transformed = np.flip(np.transpose(prediction, axes=[1,0,2]), axis=0)
  for band_index in range(dataset.RasterCount):
    band = dataset.GetRasterBand(band_index+1)
    if band.CreateMaskBand(0) == gdal.CE_Failure:
      raise RuntimeError("Failed to create mask band")
    mask_band = band.GetMaskBand()
    band.WriteArray(np.choose(prediction_transformed[:, :, band_index].mask, (prediction_transformed[:, :, band_index].data,np.array(band.GetNoDataValue()),)))
    mask_band.WriteArray(np.logical_not(prediction_transformed[:, :, band_index].mask))

def coords_to_indices(bounds: Bounds, x: float, y: float):
  if x < bounds.minx or x > bounds.maxx or y < bounds.miny or y > bounds.maxy:
    #raise ValueError("Coordinates out of bounds")
    return None, None
    
  # X => lat, Y => lon
  x_idx = bounds.raster_size_y - int(math.ceil((y - bounds.miny) / abs(bounds.pixel_size_y)))
  y_idx = int((x - bounds.minx) / abs(bounds.pixel_size_x))

  return x_idx, y_idx

def _try_get_data_at_coords(dataset: AmazonGeoTiff, x: float, y: float, month: int) -> float:
  # x = longitude
  # y = latitude
  bounds = get_extent(dataset.gdal_dataset)
  x_idx, y_idx = coords_to_indices(bounds, x, y)
  if not x_idx and not y_idx:
    return None
  if month == -1:
    value = dataset.yearly_masked_image[x_idx, y_idx]
  else:
    value = dataset.masked_image[x_idx, y_idx, month]
  if np.ma.is_masked(value):
    return None
  else:
    return value

def get_data_at_coords(dataset: AmazonGeoTiff, x: float, y: float, month: int) -> float:
  # x = longitude
  # y = latitude
  value = _try_get_data_at_coords(dataset, x, y, month)
  if value is None:
    raise ValueError(f"Coordinates ({y},{x}) are masked in {dataset.name}")
  else:
    return value

def is_valid_point(lat: float, lon: float, reference_isocape: AmazonGeoTiff) -> bool:  
  return _try_get_data_at_coords(reference_isocape, lon, lat, -1) is not None

brazil_map_geotiff_ = None
def brazil_map_geotiff() -> AmazonGeoTiff:
  global brazil_map_geotiff_
  if not brazil_map_geotiff_:
    brazil_map_geotiff_ = load_raster(get_raster_path("brasil_clim_raster.tiff"))
  return brazil_map_geotiff_

relative_humidity_geotiff_ = None
def relative_humidity_geotiff() -> AmazonGeoTiff:
  global relative_humidity_geotiff_
  if not relative_humidity_geotiff_:
    relative_humidity_geotiff_ = load_named_raster(get_raster_path("R.rh_Stack.tif"), "rh")
  return relative_humidity_geotiff_

temperature_geotiff_ = None
def temperature_geotiff() -> AmazonGeoTiff:
  global temperature_geotiff_
  if not temperature_geotiff_:
    temperature_geotiff_ = load_named_raster(get_raster_path("Temperatura_Stack.tif"), "temp")
  return temperature_geotiff_

vapor_pressure_deficit_geotiff_ = None
def vapor_pressure_deficit_geotiff() -> AmazonGeoTiff:
  global vapor_pressure_deficit_geotiff_
  if not vapor_pressure_deficit_geotiff_:
    vapor_pressure_deficit_geotiff_ = load_named_raster(get_raster_path("R.vpd_Stack.tif"), "vpd")
  return vapor_pressure_deficit_geotiff_

atmosphere_isoscape_geotiff_ = None
def atmosphere_isoscape_geotiff() -> AmazonGeoTiff:
  global atmosphere_isoscape_geotiff_
  if not atmosphere_isoscape_geotiff_:
    atmosphere_isoscape_geotiff_ = load_named_raster(get_raster_path("Iso_Oxi_Stack.tif"), "atmosphere_oxygen_ratio")
  return atmosphere_isoscape_geotiff_

cellulose_isoscape_geotiff_ = None
def cellulose_isoscape_geotiff() -> AmazonGeoTiff:
  global cellulose_isoscape_geotiff_
  if not cellulose_isoscape_geotiff_:
    cellulose_isoscape_geotiff_ = load_named_raster(get_raster_path("iso_O_cellulose.tif"), "cellulose_oxygen_ratio")
  return cellulose_isoscape_geotiff_

pet_geotiff_ = None
def pet_geotiff() -> AmazonGeoTiff:
  global pet_geotiff_
  if not pet_geotiff_:
    pet_geotiff_ = load_named_raster(get_raster_path("pet_Stack_mean.tiff"), "pet")
  return pet_geotiff_

dem_geotiff_ = None
def dem_geotiff() -> AmazonGeoTiff:
  global dem_geotiff_
  if not dem_geotiff_:
    dem_geotiff_ = load_named_raster(get_raster_path("dem_pa_brasil_raster.tiff"), "dem",  use_only_band_index=0)
  return dem_geotiff_

pa_geotiff_ = None
def pa_geotiff() -> AmazonGeoTiff:
  global pa_geotiff_
  if not pa_geotiff_:
    pa_geotiff_ = load_named_raster(get_raster_path("dem_pa_brasil_raster.tiff"), "pa",  use_only_band_index=1)
  return pa_geotiff_

krig_means_isoscape_geotiff_ = None
def krig_means_isoscape_geotiff() -> AmazonGeoTiff:
  global krig_means_isoscape_geotiff_
  if not krig_means_isoscape_geotiff_:
    krig_means_isoscape_geotiff_ = load_named_raster(get_raster_path("uc_davis_d18O_cel_ordinary_random_grouped_means.tiff"), "ordinary_krig_means")
  return krig_means_isoscape_geotiff_

krig_variances_isoscape_geotiff_ = None
def krig_variances_isoscape_geotiff() -> AmazonGeoTiff:
  global krig_variances_isoscape_geotiff_
  if not krig_variances_isoscape_geotiff_:
    krig_variances_isoscape_geotiff_ = load_named_raster(get_raster_path("uc_davis_d18O_cel_ordinary_random_grouped_vars.tiff"), "ordinary_krig_vars")
  return krig_variances_isoscape_geotiff_  

precipitation_regression_isoscape_geotiff_ = None
def precipitation_regression_isoscape_geotiff() -> AmazonGeoTiff:
  global precipitation_regression_isoscape_geotiff_
  if not precipitation_regression_isoscape_geotiff_:
    precipitation_regression_isoscape_geotiff_ = load_named_raster(get_raster_path("isoscape_fullmodel_d18O_prec_REGRESSION.tiff"), "precipitation_regression_isoscape_geotiff")
  return precipitation_regression_isoscape_geotiff_

craig_gordon_isoscape_geotiff_ = None
def craig_gordon_isoscape_geotiff() -> AmazonGeoTiff:
  global craig_gordon_isoscape_geotiff_
  if not craig_gordon_isoscape_geotiff_:
    craig_gordon_isoscape_geotiff_ = load_named_raster(get_raster_path("Iso_Oxi_Stack_mean_TERZER.tif"), "craig_gordon_isoscape_geotiff")
  return craig_gordon_isoscape_geotiff_

brisoscape_geotiff_ = None
def brisoscape_geotiff() -> AmazonGeoTiff:
  global brisoscape_geotiff_
  if not brisoscape_geotiff_:
    brisoscape_geotiff_ = load_named_raster(get_raster_path("brisoscape_mean_ISORIX.tif"), "brisoscape_geotiff")
  return brisoscape_geotiff_

d13C_mean_geotiff_ = None
def d13C_mean_geotiff() -> AmazonGeoTiff:
  global d13C_mean_geotiff_
  if not d13C_mean_geotiff_:
    d13C_mean_geotiff_ = load_named_raster(get_raster_path("d13C_cel_map_BRAZIL_stack.tiff"), "d13C_mean", use_only_band_index=0)
  return d13C_mean_geotiff_

d13C_var_geotiff_ = None
def d13C_var_geotiff() -> AmazonGeoTiff:
  global d13C_var_geotiff_
  if not d13C_var_geotiff_:
    d13C_var_geotiff_ = load_named_raster(get_raster_path("d13C_cel_map_BRAZIL_stack.tiff"), "d13C_var", use_only_band_index=1)
  return d13C_var_geotiff_

d13C_mean_amazon_only_geotiff_ = None
def d13C_mean_amazon_only_geotiff() -> AmazonGeoTiff:
  global d13C_mean_amazon_only_geotiff_
  if not d13C_mean_amazon_only_geotiff_:
    d13C_mean_amazon_only_geotiff_ = load_named_raster(get_raster_path("d13C_cel_amazon_stack_terra_null.tiff"), "d13C_mean_amazon", use_only_band_index=0)
  return d13C_mean_amazon_only_geotiff_

d13C_var_amazon_only_geotiff_ = None
def d13C_var_amazon_only_geotiff() -> AmazonGeoTiff:
  global d13C_var_amazon_only_geotiff_
  if not d13C_var_amazon_only_geotiff_:
    d13C_var_amazon_only_geotiff_ = load_named_raster(get_raster_path("d13C_cel_amazon_stack_terra_null.tiff"), "d13C_var_amazon", use_only_band_index=1)
  return d13C_var_amazon_only_geotiff_

# Reference rasters used to get the shape of brazil.
brazil_template_ = None
def brazil_template() -> AmazonGeoTiff:
  global brazil_template_
  if not brazil_template_:
    brazil_template = load_named_raster(get_raster_path("amazon_template.tiff"), "brazil_template", use_only_band_index=0)
  return brazil_template_

amazon_template_ = None
def amazon_template() -> AmazonGeoTiff:
  global amazon_template_
  if not amazon_template_:
    amazon_template = load_named_raster(get_raster_path("amazon_template.tiff"), "brazil_template", use_only_band_index=0)
  return amazon_template_


# A collection of column names to functions that load the corresponding geotiffs.
column_name_to_geotiff_fn = {
  "VPD" : vapor_pressure_deficit_geotiff,
  "RH": relative_humidity_geotiff,
  "PET": pet_geotiff,
  "DEM": dem_geotiff,
  "PA": pa_geotiff,
  "Mean Annual Temperature": temperature_geotiff,
  "Mean Annual Precipitation": brazil_map_geotiff,
  "Iso_Oxi_Stack_mean_TERZER": craig_gordon_isoscape_geotiff,
  "isoscape_fullmodel_d18O_prec_REGRESSION": precipitation_regression_isoscape_geotiff,
  "brisoscape_mean_ISORIX": brisoscape_geotiff,
  "d13C_cel_mean": d13C_mean_geotiff,
  "d13C_cel_var": d13C_var_geotiff,
  "ordinary_kriging_linear_d18O_predicted_mean" : krig_means_isoscape_geotiff,
  "ordinary_kriging_linear_d18O_predicted_variance" : krig_variances_isoscape_geotiff
}

# Converts a resolution into a Bounds class, storing info about pixel size and raster size.
def create_bounds_from_res(res_x: int, res_y: int, base_bounds: Bounds):
  # Use base_bounds to get the min/max lat and lon. Scale everything else
  # to fit the new resolution.
  new_bounds = base_bounds
  new_bounds.pixel_size_x *= (base_bounds.raster_size_x/res_x)
  new_bounds.pixel_size_y *= (base_bounds.raster_size_y/res_y)
  new_bounds.raster_size_x = res_x
  new_bounds.raster_size_y = res_y
  return new_bounds

def stamp_isoscape(filename: str, metadata_name: str, metadata_value: str):
  """
  stamp_isoscape function
  ------------------------
  This function stamps an isoscape with key/value metadata.
  This function can also be used to edit an existing stamp:
    - inform the existing metadata_name
    - inform a new desired metadata_value 
  ------------------------
  Parameters:
  filename : str
      GeoTIFF filename (with the full path) of the isoscape to be stamped
      e.g: /content/gdrive/Shared drives/TNC Fellowship 🌳/4. Isotope Research & Signals/code/amazon_rainforest_files/amazon_rasters/variational/ensemble_with_carbon_brisoisorix/fixed_isorix_carbon_ensemble.tiff
  metadata_name : str
      Name of the metadata to be stamped 
      e.g: "p_value"
  metadata_value : str
      Value of the metadada related to the metadata_name informed
      e,g: "0.05"
  """
  dataset = gdal.Open(filename)
  metadata = dataset.GetMetadata()
  metadata[metadata_name] = metadata_value
  dataset.SetMetadata(metadata)
  dataset.FlushCache()

def show_stamps(filename: str):
  """
  show_stamps function
  ------------------------
  This function returns all the metadata from an isoscape
  ------------------------
  Parameter:
  filename : str
      GeoTIFF filename (with the full path) of the stamped isoscape
      e.g: /content/gdrive/Shared drives/TNC Fellowship 🌳/4. Isotope Research & Signals/code/amazon_rainforest_files/amazon_rasters/variational/ensemble_with_carbon_brisoisorix/fixed_isorix_carbon_ensemble.tiff
  """
  dataset = gdal.Open(filename)
  metadata = dataset.GetMetadata()
  return(metadata)

def del_stamp(filename: str, metadata_name: str):
  """
  del_stamp function
  ------------------------
  This function deletes a key/value metadata
  ------------------------
  Parameters:
  filename : str
      GeoTIFF filename (with the full path) of the stamped isoscape
      e.g: /content/gdrive/Shared drives/TNC Fellowship 🌳/4. Isotope Research & Signals/code/amazon_rainforest_files/amazon_rasters/variational/ensemble_with_carbon_brisoisorix/fixed_isorix_carbon_ensemble.tiff
  metadata_name : str
      Name of the metadata to be deleted
      e.g: "p_value"
  """  
  dataset = gdal.Open(filename)
  metadata = dataset.GetMetadata()
  del metadata[metadata_name]
  dataset.SetMetadata(metadata)
  dataset.FlushCache()
