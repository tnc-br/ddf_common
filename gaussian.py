###############################################################################
# MIT License
# 
# Copyright (c) 2023 The Nature Conservancy - Brazil
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###############################################################################

# Module for helper functions for gaussian blur.

from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import numpy as np
import raster
from tqdm import tqdm

def get_2d_gaussian(center_lon: float, center_lat: float, stdev: float):
  """Quick-and-dirty function to get a PDF for sampling from an image
  to turn it into a distribution. Intended for use with isoscapes.

  Major room for improvement! This framing assumes no distortion from the
  projection, i.e. that 1 deg latitude == 1 deg longitude == 111 km everywhere.
  This should probably be fine for Brazil, for now, since it's near the Equator.
  """
  rv = multivariate_normal([center_lat, center_lon], [[stdev, 0], [0, stdev]])

  return rv

# x = longitude
# y = latitude
def plot_gaussian(rv, shape: raster.Bounds):
  """Informative, for debugging and visualizing get_2d_gaussian()."""
  x = np.linspace(shape.minx, shape.maxx, shape.raster_size_x)
  y = np.linspace(shape.maxy, shape.miny, shape.raster_size_y) # inverted y axis
  X, Y = np.meshgrid(x,y)
  target = np.empty((shape.raster_size_y, shape.raster_size_x,2,), dtype=float)
  target[:, :, 0] = Y
  target[:, :, 1] = X
  pd = rv.pdf(target)
  plt.imshow(pd)
  plt.colorbar()

# TODO: For each pixel in the predicted isoscape
# Yeah, this is basically Gaussian blur, huh...
# BUT, it does give us a distribution.
def gaussian_kernel(input: raster.AmazonGeoTiff, stdev_in_degrees: float=1):
  bounds = raster.get_extent(input.gdal_dataset)
  means = np.ma.zeros((bounds.raster_size_x, bounds.raster_size_y,), dtype=float)
  means.mask = np.ones((bounds.raster_size_x, bounds.raster_size_y), dtype=bool)
  variances = np.ma.zeros((bounds.raster_size_x, bounds.raster_size_y,), dtype=float)
  variances.mask = np.ones((bounds.raster_size_x, bounds.raster_size_y), dtype=bool)
  for map_y in tqdm(range(0, bounds.raster_size_y, 1)):
    y_coord = map_y * abs(bounds.pixel_size_y) + bounds.miny
    for map_x in range(0, bounds.raster_size_x, 1):
      x_coord = map_x * abs(bounds.pixel_size_x) + bounds.minx
      rv = get_2d_gaussian(y_coord, x_coord, stdev_in_degrees)
      rsamp = rv.rvs(1000)
      values = []
      for coordinate_pair in rsamp:
        try:
          #print(coordinate_pair)
          values.append(raster.get_data_at_coords(input, coordinate_pair[0], coordinate_pair[1], 0))
        except ValueError:
          pass
        if len(values) == 30:
          break
      if len(values) == 30:
        # Set the mean and stdev pixels
        #print(x_coord, y_coord)
        means[map_x, map_y] = np.mean(values)
        variances[map_x, map_y] = np.var(values)
        # Apply sample corrective factor to variance
        variances[map_x, map_y] *= len(values) / (len(values)-1)
        means.mask[map_x, map_y] = False
        variances.mask[map_x, map_y] = False
  return means, variances

