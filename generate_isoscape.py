import raster
import model
import eeraster

import typing
import numpy as np
import pandas as pd
from osgeo import gdal, gdal_array
from tqdm import tqdm

def save_numpy_to_geotiff(bounds: raster.Bounds, prediction: np.ma.MaskedArray, path: str):
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

  if len(prediction.shape) != 3 or prediction.shape[0] != bounds.raster_size_x or prediction.shape[1] != bounds.raster_size_y:
    raise ValueError("Shape of prediction does not match base geotiff")
  
  prediction_transformed = np.flip(np.transpose(prediction, axes=[1,0,2]), axis=0)
  for band_index in range(dataset.RasterCount):
    band = dataset.GetRasterBand(band_index+1)
    if band.CreateMaskBand(0) == gdal.CE_Failure:
      raise RuntimeError("Failed to create mask band")
    mask_band = band.GetMaskBand()
    band.WriteArray(np.choose(prediction_transformed[:, :, band_index].mask, (prediction_transformed[:, :, band_index].data,np.array(band.GetNoDataValue()),)))
    mask_band.WriteArray(np.logical_not(prediction_transformed[:, :, band_index].mask))

def get_predictions_at_each_pixel(
    model: model.Model,
    geotiffs: dict[str, raster.AmazonGeoTiffBase],
    bounds: raster.Bounds,
    geometry_mask: raster.AmazonGeoTiff=None):
  """Uses `model` to make mean/variance predictions for every pixel in `bounds`.
  Queries are constructed by querying every geotiff in `geotiffs` for information 
  at that pixel and passing the parameters to the model. 
  
  Parameters are standardized using feature_transformer, a set of standardizers used
  to fit training data.
  
  `model`: Tensorflow model used to make predictions
  `feature_transformer`: ScikitLearn ColumnTransformer storing transformations of columns
                         Input must be transformed prior to predictions
  `geotiffs`: The set of geotiffs required to make the prediction
  `bounds`: Every pixel within these bounds will have a prediction made on it
  `geometry_mask`: If specified, only make predictions within this mask and within `bounds`."""

  # Initialize a blank plane representing means and variance.
  predicted_np = np.ma.array(
      np.zeros([bounds.raster_size_x, bounds.raster_size_y, 2], dtype=float),
      mask=np.ones([bounds.raster_size_x, bounds.raster_size_y, 2], dtype=bool))

  for x_idx in tqdm(range(0, bounds.raster_size_x)):
    rows = []
    for y_idx in range(0, bounds.raster_size_y):
      row = {}
      row["lat"] = bounds.miny + (y_idx * -bounds.pixel_size_y)
      row["long"] = bounds.minx + (x_idx * bounds.pixel_size_x)
      rows.append(row)
    
    X = pd.DataFrame.from_dict(rows)
    for geotiff_label, geotiff in geotiffs.items():
      X = geotiff.values_at_df(X, geotiff_label)
      X = X[X[geotiff_label].notna()]
      if not len(X.index):
        break

    if (len(X.index)):
      predictions = model.predict_on_batch(X)

      means_np = predictions[:, 0]
      for prediction, y_idx in zip(means_np, X.index):
        predicted_np.mask[x_idx, y_idx, 0] = False
        predicted_np.data[x_idx, y_idx, 0] = prediction
      vars_np = predictions[:, 1]
      for prediction, y_idx in zip (vars_np, X.index):
        predicted_np.mask[x_idx, y_idx, 1] = False
        predicted_np.data[x_idx, y_idx, 1] = prediction

  return predicted_np

def dispatch_rasters(
    required_rasters: typing.List[str],
    use_earth_engine_assets : bool=False,
    local_fallback: bool=False):
  """
  dispatch_rasters function
  --------------------------------------------------
  Loads rasters either from Google Earth Engine or from a local/gdrive
  depending on parameters
  --------------------------------------------------
  Parameters:
  required_rasters:
    The list of rasters to load, e.g. ["DEM", "PA", "VPD"]
  use_earth_engine_assets: 
    Whether to load the rasters from earth engine. If False, loads from local/gdrive.
  local_fallback:
    If use_earth_engine_assets=True, whether to fallback to gdrive if rasters could not
    be found in ee. 
  """

  rasters_to_dispatch = {}

  if use_earth_engine_assets:
    for raster_name in required_rasters:
      if raster_name in eeraster.column_name_to_ee_asset_fn:
        rasters_to_dispatch[raster_name] = eeraster.column_name_to_ee_asset_fn[raster_name]()

  if not use_earth_engine_assets or local_fallback:
    for raster_name in required_rasters:
      
      # Skip loading locally if it was found in EE.
      if raster_name in rasters_to_dispatch:
        continue
      
      # Load it from local (or gdrive).
      if raster_name in raster.column_name_to_geotiff_fn:
        rasters_to_dispatch[raster_name] = raster.column_name_to_geotiff_fn[raster_name]()

  # Identify missing rasters.
  missing = set(rasters_to_dispatch.keys()) -  set(required_rasters)
  if missing:
    raise ValueError("The following training columns do not have an associated raster: ", missing)

  return rasters_to_dispatch

  

def generate_isoscapes_from_variational_model(
    model: model.Model,
    res_x: int, 
    res_y: int,
    output_geotiff: str,
    amazon_only: bool=False):
  """
  generate_isoscapes_from_variational_model function
  --------------------------------------------------
  This function generates an isoscape using a model, according
  to the resolution specs. It queries the model for every 
  pixel in a (res_x x res_y) tiff of the landscape.
  --------------------------------------------------
  Parameters:
  model: model.Model 
    Pretrained model used to make predictions on every pixel.
  res_x: int
    The output x resolution
  res_y: int
    The output y resolution
  output_geotiff: str
    Name of the file to output. 
  amazon_only: bool
    Whether to only generate a raster of the Amazon region as opposed to
    all of Brazil.
  """
  required_geotiffs = model.training_column_names()
  required_geotiffs.remove('lat')
  required_geotiffs.remove('long')
  
  input_geotiffs = dispatch_rasters(
    required_geotiffs,
    use_earth_engine_assets=True,
    local_fallback=True)

  arbitrary_geotiff = raster.dem_geotiff()
  if amazon_only:
    arbitrary_geotiff = raster.d13C_mean_amazon_only_geotiff()
  base_bounds = raster.get_extent(arbitrary_geotiff.gdal_dataset)
  output_resolution = raster.create_bounds_from_res(res_x, res_y, base_bounds) 

  np = get_predictions_at_each_pixel(
    model, input_geotiffs, output_resolution, 
    geometry_mask=arbitrary_geotiff)
  save_numpy_to_geotiff(
      output_resolution, np, raster.get_raster_path(output_geotiff+".tiff"))