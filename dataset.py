# Module for helper functions for manipulating data and datasets.

from dataclasses import dataclass
from enum import Enum
from geopy import distance
from numpy.random import MT19937, RandomState, SeedSequence
from tqdm import tqdm
import datetime
import ee
import math
import numpy as np
import pandas as pd
import pytest
import random

from partitioned_dataset import PartitionedDataset
import raster

def gen_tabular_dataset(monthly: bool, samples_per_site: int) -> pd.DataFrame:
  return gen_tabular_dataset_with_coords(monthly, samples_per_site,
                            [(-70,-5,),(-67.5,0,),(-66,-4.5,),(-63,-9.5,),
                             (-63,-9,),(-62,-6,),(-60,-2.5,),(-60,1,),
                              (-60,-12.5,),(-59,-2.5,),(-57.5,-4,),
                               (-55,-3.5,),(-54,-1,),(-52.5,-13,),(-51.5,-2.5,)],
                                         0.5)

def add_features_from_rasters(df: pd.DataFrame, rasters: list) -> pd.DataFrame:
  '''
  Given a pd.DataFrame df:
  1. enumerates each row looking for a 'lat' and 'lon' column
  2.   for each raster, looks up the lat and lon value
  3.     adds that value to the feature_df
  4. returns the concat of new feature_df with original df
  '''
  feature_dict = {}
  for raster in rasters:
    feature_dict[raster.name] = []
  
  dupe_columns = []
  for row in df.itertuples():
    lat = getattr(row, _LATITUDE_COLUMN_NAME)
    lon = getattr(row, _LONGITUDE_COLUMN_NAME)
    for raster in rasters:
      feature_dict[raster.name].append(raster.value_at(lon, lat))

  # prefer the rasters as the source
  df = df.drop([r.name for r in rasters], axis=1, errors='ignore')
  return pd.concat([pd.DataFrame(feature_dict), df], axis=1)


def gen_tabular_dataset_with_coords(monthly: bool, samples_per_site: int, 
  sample_site_coordinates: list, sample_radius: float) -> pd.DataFrame:
  features = [raster.relative_humidity_geotiff(), 
    raster.temperature_geotiff(), 
    raster.vapor_pressure_deficit_geotiff(), 
    raster.atmosphere_isoscape_geotiff(), 
    raster.cellulose_isoscape_geotiff()]
  image_feature_names = ["rh", "temp", "vpd", "atmosphere_oxygen_ratio", "cellulose_oxygen_ratio"]
  feature_names = [_LATITUDE_COLUMN_NAME,_LONGITUDE_COLUMN_NAME, "month_of_year"] + image_feature_names
  rs = RandomState(MT19937(SeedSequence(42)))

  feature_values = {}
  for name in feature_names:
    feature_values[name] = []

  for coord in tqdm(sample_site_coordinates):
    month_start = 0 if monthly else -1
    month_end = 12 if monthly else 0
    for month in range(month_start, month_end):
      samples_collected = 0
      while samples_collected < samples_per_site:
        row = {}
        sample_x, sample_y = 2*(rs.rand(2) - 0.5) * sample_radius
        sample_x += coord[0]
        sample_y += coord[1]

        try:
          for feature, feature_name in zip(features, image_feature_names):
            row[feature_name] = raster.get_data_at_coords(
              feature, sample_x, sample_y, month)
          row["month_of_year"] = month
          row[_LONGITUDE_COLUMN_NAME] = sample_x
          row[_LATITUDE_COLUMN_NAME] = sample_y
          samples_collected += 1

        except ValueError as e:
          # masked and out-of-bounds coordinates
          # note that if sample_radius is zero, we know that looping again
          # will still result in a ValueError. In this case, we trigger the
          # loop to finish by incrementing the end condition (samples_collected)
          # In that case, we end up ignoring the point.
          # This can also happen with a small positive radius, but one whose
          # coords return all masked/out of bounds, but this is less likely.
          if sample_radius == 0:
            samples_collected += 1
          continue
        for key, value in row.items():
          feature_values[key].append(value)

  samples = pd.DataFrame(feature_values)

  if not monthly:
    samples.drop("month_of_year", axis=1, inplace=True)

  return samples

def load_sample_data(reference_csv_filename: str) -> pd.DataFrame:
  """This method loads reference sample data from a CSV, determines the unique
  locations in that CSV and then calls the existing method
  gen_tabular_dataset_with_coords to enrich those locations with features for
  "rh", "temp", "vpd", and  "atmosphere_oxygen_ratio".
  Finally, it joins the CSV provided d18O means values from the CSV with the
  per-loc features"""

  df = pd.read_csv(raster.get_sample_db_path(reference_csv_filename),
   encoding="ISO-8859-1", sep=',')
  df = df[['Code', _LATITUDE_COLUMN_NAME, _LONGITUDE_COLUMN_NAME, 'd18O_cel']]
  df = df[df['d18O_cel'].notna()]

  grouped = df.groupby([_LATITUDE_COLUMN_NAME, _LONGITUDE_COLUMN_NAME])

  # means is the reference sample calculated mean of d18O at each lat/lon
  means = grouped.mean().reset_index()

  # locations is now the list of unique lat and longs
  locations = list(zip(means[_LONGITUDE_COLUMN_NAME], means[_LATITUDE_COLUMN_NAME]))

  sample_data = gen_tabular_dataset_with_coords(monthly=False,
   samples_per_site=1, sample_site_coordinates=locations, sample_radius = 0)
  # drop the simulated cellulose_oxygen_ratio.
  # TODO(https://github.com/tnc-br/ddf_common/issues/5), refactor the code to only do features.
  sample_data = sample_data.drop('cellulose_oxygen_ratio', axis = 1)

  # Here we merge the features "rh", "temp", "vpd", and  "atmosphere_oxygen_ratio"
  # with the means based on lat/long
  sample_data = pd.merge(sample_data, means, how="inner", 
    left_on=[_LATITUDE_COLUMN_NAME, _LONGITUDE_COLUMN_NAME],
    right_on=[_LATITUDE_COLUMN_NAME, _LONGITUDE_COLUMN_NAME])
  sample_data = sample_data.rename(
    columns={'d18O_cel': 'cellulose_oxygen_ratio' }).reset_index()
  sample_data.drop('index', inplace=True, axis=1)


  return sample_data

# If a reference_csv_filename is requested, we use that over any simulated data.
def aggregate_reference_data(reference_csv_filename: str) -> pd.DataFrame:
  if reference_csv_filename:
    return load_sample_data(reference_csv_filename)
  # This is the historical simulated input that has 17 samples x 15 sites
  return  gen_tabular_dataset(monthly=False, samples_per_site=17)

def partitioned_reference_data(reference_csv_filename: str) -> PartitionedDataset:
  partition_data = partitioned_dataset.partition(
                             aggregate_reference_data(reference_csv_filename),
                             PartitionStrategy.FIXED)
  partitioned_dataset.print_split(partition_data)
  return partition_data

def load_reference_samples(filters: list[ee.Filter] = []) -> pd.DataFrame:
  '''
  Given an optional list of filters, returns a DataFrame containing all current
  reference sample from Earth Engine. You must have the proper authorization
  to access this data, which is obtained by belonging to an organization added
  to TimberID.org
  '''
  import google
  from google.colab import auth
  auth.authenticate_user()

  credentials, project_id = google.auth.default()
  ee.Initialize(credentials, project='river-sky-386919')
  fc = ee.FeatureCollection('projects/river-sky-386919/assets/timberID/trusted_samples')
  for filter_fc in filters:
    fc = fc.filter(filter_fc)
  info = fc.getInfo()
  features = info['features']
  dictarr = []

  for f in features:
      attr = f['properties']
      attr[_LATITUDE_COLUMN_NAME] = f['geometry']['coordinates'][1]
      attr[_LONGITUDE_COLUMN_NAME] = f['geometry']['coordinates'][0]
      dictarr.append(attr)

  return pd.DataFrame(dictarr)

def preprocess_sample_data(df: pd.DataFrame,
                           feature_columns: list[str],
                           label_columns: list[str],
                           aggregate_columns: list[str],
                           keep_grouping: bool) -> pd.DataFrame:
  '''
  Given a pd.DataFrame df:
  1. Filters in relevant columns using feature_columns, label_columns
  2. Calculates the mean and variance of each column in label_columns grouping
     by a key made of aggregate_columns
  3. If keep_grouping = True, we export groupings by key aggregate_columns
     otherwise we return the original sample with their matching means/variances.
  '''
  df.dropna(subset=feature_columns + label_columns, inplace=True)
  df = df[feature_columns + label_columns]

  if aggregate_columns:
    grouped = df.groupby(aggregate_columns)

    for col in label_columns:
      means = grouped.mean().reset_index()
      means.rename(columns={col: f"{col}_mean"}, inplace=True)
      means = means[aggregate_columns + [f"{col}_mean"]]

      variances = grouped.var().reset_index()
      variances.rename(columns={col: f"{col}_variance"}, inplace=True)
      variances = variances[aggregate_columns + [f"{col}_variance"]]

      df = pd.merge(df, means, how="inner",
                    left_on=aggregate_columns, right_on=aggregate_columns)
      df = pd.merge(df, variances, how="inner",
                    left_on=aggregate_columns, right_on=aggregate_columns)
      df.drop(columns=[col], inplace=True)

    if keep_grouping:
      # The first entry is the same as all entries in the grouping for the
      # aggregate_columns. Any other column will have different values but
      # we only take the first one.
      df = df.groupby(aggregate_columns).first().reset_index()

  return df

#Utility function for randomly sampling a point around a sample site
def _is_valid_point(lat: float, lon: float, reference_isocape: raster.AmazonGeoTiff):
  return raster.is_valid_point(lat, lon, reference_isocape)

# Pick a random point around (lat, lon) within max_distance_km. If edge_only is
# true, only pick points exactly max_distance_km away from (lat, lon).
def _random_nearby_point(lat: float, lon: float, max_distance_km: float, edge_only=False):
  # Pick a random angle pointing outward from the origin.
  # 0 == North, 90 == East, 180 == South, 270 == West
  angle = 360 * random.random()

  # sqrt() is required for an equal radial distribution, otherwise samples
  # cluster around origin.
  dist = max_distance_km if edge_only else max_distance_km * math.sqrt(random.random())

  # WGS-84 is the most accurate ellipsoidal model of Earth, but we should double
  # check to make sure this matches the model used by our sample collectors.
  point = distance.geodesic(
      ellipsoid='WGS-84', kilometers=dist).destination((lat, lon), bearing=angle)
  return point.latitude, point.longitude

# Given a list of real_points, returns true if (lat, lon) is within threshold
# of any of those points.
def _is_nearby_real_point(lat: float, lon: float, real_points, threshold_km: float):
  for point, _ in real_points:
    if distance.geodesic((lat, lon), point).km < threshold_km:
      return True
  return False

#This function creates a dataset based on real samples adding a Fraud column
def create_fraudulent_samples(real_samples_data: pd.DataFrame, mean_isoscapes: list[raster.AmazonGeoTiff],elements: list[str],max_trusted_radius: float,max_fraud_radius:float,min_fraud_radius:float) -> pd.DataFrame:
  '''
  This function creates a dataset based on real samples adding a Fraud column, where True represents a real lat/lon and False represents a fraudulent lat/lon
  Input:
  - real_samples_data: dataset containing real samples
  - elements: element that will be used in the ttest: Oxygen (e.g: d18O_cel), Carbon or Nitrogen.
  - mean_isoscapes: Isoscapes of mean values of isotope values from elements
  - max_trusted_radius, In km, the maximum distance from a real point where its value is still considered legitimate.
  - max_fraud_radius: In km, the maximum distance from a real point to randomly sample a fraudalent coordinate.
  - min_fraud_radius: In km, the minimum distance from a real point to randomly sample a fraudalent coordinate.
  Output: 
  - fake_data: pd.DataFrame with lat, long, isotope_value and fraudulent columns
  '''
  real_samples = real_samples_data.groupby(['lat','long'])[elements]
  real_samples_code = real_samples_data.groupby(['lat','long','Code'])[elements]

  count = 0
  lab_samp = real_samples

  if max_fraud_radius <= min_fraud_radius:
    raise ValueError("max_fraud_radius {} <= min_fraud_radius {}".format(
        max_fraud_radius, min_fraud_radius))
    
  fake_sample = pd.DataFrame(columns=['Code',
          'lat',
          'long',
          'fraud'] + elements)

  # Max number of times to attempt to generate random coordinates.
  max_random_sample_attempts = 1000
  count = 0

  for coord, lab_samp in real_samples_code:
    if lab_samp.size <= 1:
      continue
    lat, lon, attempts = 0, 0, 0
    while((not all([_is_valid_point(lat, lon, mean_iso) for mean_iso in mean_isoscapes]) or
          _is_nearby_real_point(lat, lon, real_samples, min_fraud_radius)) and
          attempts < max_random_sample_attempts):
      lat, lon = _random_nearby_point(coord[0], coord[1], max_fraud_radius)
      attempts += 1
    if attempts == max_random_sample_attempts:
      continue
    for i in range(lab_samp.shape[0]):
        new_row = {'Code': f"fake_mad{count}", 'lat': lat, 'long': lon,'fraud': True }
        for element in elements:
          new_row[element] = lab_samp[element].iloc[i]
        fake_sample.loc[len(fake_sample)] = new_row
    count += 1

  return fake_sample