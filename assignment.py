import scipy.stats
import raster

import numpy as np

# Computes a PDF given a 2D geospatial mean and variance raster, and a sample
def compute_pdf(sample, means_raster, vars_raster):
  pdf = scipy.stats.norm.pdf(x=sample, loc=means_raster.yearly_masked_image.data, scale=np.sqrt(vars_raster.yearly_masked_image.data))
  pdf /= np.nansum(pdf)
  return pdf

def compute_pdf_isoscapes(
  reported_latitude: float,
  reported_longitude: float,
  oxygen_means_isoscape_filename: str = None,
  oxygen_vars_isoscape_filename:str = None,
):
  if oxygen_means_isoscape_filename and oxygen_vars_isoscape_filename:
    if oxygen_means_isoscape_filename == oxygen_vars_isoscape_filename:
      oxygen_means_isoscape = raster.load_raster(
          raster.get_raster_path(oxygen_means_isoscape_filename), use_only_band_index=0)
      oxygen_vars_isoscape = raster.load_raster(
          raster.get_raster_path(oxygen_means_isoscape_filename), use_only_band_index=1)
    else:
      oxygen_means_isoscape = raster.load_raster(
        raster.get_raster_path(oxygen_means_isoscape_filename), use_only_band_index=0)
      oxygen_vars_isoscape = raster.load_raster(
        raster.get_raster_path(oxygen_vars_isoscape_filename), use_only_band_index=0)
  else:
    raise ValueError("No isoscape(s) provided")
  sample = oxygen_means_isoscape.value_at(reported_longitude, reported_latitude)

  pdf = compute_pdf(sample, oxygen_means_isoscape, oxygen_vars_isoscape)

  bounds = raster.get_extent(oxygen_means_isoscape.gdal_dataset)
  output_resolution = raster.create_bounds_from_res(pdf.shape[0], pdf.shape[1], bounds) 

  return output_resolution, pdf


def compute_fraud_probability(
    reported_latitude: float,
    reported_longitude: float,
    allowed_mask: np.array,
    means_raster: raster.AmazonGeoTiff,
    var_raster: raster.AmazonGeoTiff):
  sample = means_raster.value_at(lat=reported_latitude, lon=reported_longitude)
  pdf = compute_pdf(sample, means_raster, var_raster)
  bounds = raster.get_extent(means_raster.gdal_dataset)
  return 1 - float(np.nansum(np.multiply(pdf, allowed_mask)))