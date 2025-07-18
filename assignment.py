import scipy.stats
import raster

# Computes a PDF given a 2D geospatial mean and variance raster, and a sample
def compute_pdf(sample, means_raster, var_raster):
  pdf = scipy.stats.norm.pdf(x=sample, loc=means_raster.yearly_masked_image.data, scale=np.sqrt(vars_raster.yearly_masked_image.data))
  pdf /= np.nansum(pdf)
  return pdf

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