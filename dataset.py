# Module for helper functions for manipulating data and datasets.

from dataclasses import dataclass
from enum import Enum
from geopy import distance
from itertools import permutations
from numpy.random import MT19937, RandomState, SeedSequence
from shapely import Polygon
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import datetime
import ee
import eeddf
from typing import List, Dict
import math
import numpy as np
import pandas as pd
import pytest
import random
import datetime
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass

SAMPLE_COUNT_COLUMN_NAME_SUFFIX = 'count'

from partitioned_dataset import PartitionedDataset
import partitioned_dataset
import raster

@dataclass
class PartitionedDataset:
  '''
  Container of dataframes representing the train, test and validation sets of a sample.
  '''
  train: pd.DataFrame
  test: pd.DataFrame
  validation: pd.DataFrame

@dataclass
class DatasetGeographicPartitions:
  '''
  Describes the bounds of a geographic area within a dataset.
  '''
  min_longitude: float
  max_longitude: float
  min_latitude: float
  max_latitude: float


class PartitionStrategy(Enum):
  '''
  The strategies you can partition datasets to.
  '''
  FIXED = 1
  RANDOM = 2


@dataclass
class FixedPartitionStrategy:
  '''
  Defines the parameters for the FIXED partition strategy
  '''
  train_fixed_bounds: DatasetGeographicPartitions
  validation_fixed_bounds: DatasetGeographicPartitions
  test_fixed_bounds: DatasetGeographicPartitions


_FIXED_PARTITION_STRATEGY = FixedPartitionStrategy(
  # Train
  DatasetGeographicPartitions(
    min_longitude=-62.5,
    max_longitude=float('inf'),
    min_latitude=-5,
    max_latitude=float('inf'),
  ),
  # Validation
  DatasetGeographicPartitions(
    min_longitude=float('-inf'),
    max_longitude=-62.5,
    min_latitude=-5,
    max_latitude=float('inf')
  ),
  # Test
  DatasetGeographicPartitions(
    min_longitude=float('-inf'),
    max_longitude=float('inf'),
    min_latitude=float('-inf'),
    max_latitude=-5
  )
)

@dataclass
class RandomPartitionStrategy:
  '''
  Defines the parameters for the RANDOM partition strategy
  '''
  train_fraction: float
  validation_fraction: float
  test_fraction: float
  random_seed: int


_RANDOM_PARTITION_STRATEGY = RandomPartitionStrategy(
  0.8, 0.1, 0.1, None
)

# Standard column names in reference samples.
_LONGITUDE_COLUMN_NAME = "long"
_LATITUDE_COLUMN_NAME = "lat"

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
                             partitioned_dataset.PartitionStrategy.FIXED)
  partitioned_dataset.print_split(partition_data)
  return partition_data

def load_reference_samples(org_name: str = 'google' , filters: list[ee.Filter] = [], test_environment:bool = False) -> pd.DataFrame:
  '''
  Given an optional list of filters, returns a DataFrame containing all current
  reference sample from Earth Engine. You must have the proper authorization
  to access this data, which is obtained by belonging to an organization added
  to TimberID.org
  org_name: specifies the organization the user belongs to.
  test_environment: if true, returns data from the test environment instead of production
  '''
  eeddf.initialize_ddf(test_environment)
  fc = ee.FeatureCollection('projects/' + eeddf.ee_project_name() + '/assets/ee_org/' + org_name + '/trusted_samples')
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
  df = df[feature_columns + label_columns]

  if aggregate_columns:
    grouped = df.groupby(aggregate_columns)

    for col in label_columns:
      counts = grouped.count().reset_index()
      counts.rename(
        columns={col: f"{col}_{SAMPLE_COUNT_COLUMN_NAME_SUFFIX}"},
        inplace=True)
      counts = counts[aggregate_columns + [f"{col}_{SAMPLE_COUNT_COLUMN_NAME_SUFFIX}"]]
  
      means = grouped.mean().reset_index()
      means.rename(columns={col: f"{col}_mean"}, inplace=True)
      means = means[aggregate_columns + [f"{col}_mean"]]

      variances = grouped.var().reset_index()
      variances.rename(columns={col: f"{col}_variance"}, inplace=True)
      variances = variances[aggregate_columns + [f"{col}_variance"]]

      df = pd.merge(df, counts, how="inner",
                  left_on=aggregate_columns, right_on=aggregate_columns)
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
def create_fraudulent_samples(real_samples_data: pd.DataFrame,
  mean_isoscapes: list[raster.AmazonGeoTiff],
  elements: list[str],
  max_fraud_radius:float,
  trusted_buffer_radius:float,
  sample_drop_rate:float=0.0) -> pd.DataFrame:
  '''
  This function creates a dataset based on real samples adding a Fraud column, where True represents a real lat/lon and False represents a fraudulent lat/lon
  Input:
  - real_samples_data: dataset containing real samples
  - elements: element that will be used in the ttest: Oxygen (e.g: d18O_cel), Carbon or Nitrogen.
  - mean_isoscapes: Isoscapes of mean values of isotope values from elements
  - max_fraud_radius: In km, the maximum distance from a real point to randomly sample a fraudalent coordinate.
  - trusted_buffer_radius: In km, the minimum distance from a real point to randomly sample a fraudalent coordinate.
  - sample_drop_rate: How often, randomly, should we drop some real samples in fraud sample generation.
  Output: 
  - fake_data: pd.DataFrame with lat, long, isotope_value and fraudulent columns
  '''
  real_samples = real_samples_data.groupby(['lat','long'])[elements]
  real_samples_code = real_samples_data.groupby(['lat','long','Code'])[elements]

  count = 0
  lab_samp = real_samples

  if max_fraud_radius <= trusted_buffer_radius:
    raise ValueError("max_fraud_radius {} <= trusted_buffer_radius {}".format(
        max_fraud_radius, trusted_buffer_radius))
    
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
    if sample_drop_rate > 0 and count > (real_samples.size().shape[0] * sample_drop_rate):
      break
    lat, lon, attempts = 0, 0, 0
    while((not all([_is_valid_point(lat, lon, mean_iso) for mean_iso in mean_isoscapes]) or
          _is_nearby_real_point(lat, lon, real_samples, trusted_buffer_radius)) and
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

def _valid_in_all_rasters(
  lat: float,
  lon: float,
  rasters: List[raster.AmazonGeoTiff]) -> bool:
  '''
    Args:
        lat (float): The latitude coordinate.
        lon (float): The longitude coordinate.
        rasters (List[AmazonGeoTiff]): lat, lon are checked to be valid coords
          in each of these rasters

    Returns:
        bool: True if (lat, lon) is valid in all rasters, False if invalid in
          at least one.
  '''
  for r in rasters:
    if not raster.is_valid_point(lat, lon, r):
      return False
  return True

def nudge_invalid_coords(
    df: pd.DataFrame,
      rasters: List[raster.AmazonGeoTiff],
    max_degrees_deviation: int=2):
  '''
    Given a Pandas DataFrame with latitude and longitude columns, maybe 
    perturb the latitude and longitude (i.e. nudge) values to fit within the 
    bounds of every AmazonGeoTiff in `rasters`. 

    This may be necessary as some rasters have slightly different coordinate 
    systems than the one used by data providers. Samples very close to borders 
    are particularly susceptible to being out-of-bounds and will need nudging.

    Args:
        df (pd.DataFrame): A dataframe with "lat" and "long" columns.
        rasters (List[AmazonGeoTiff]): A list of rasters to use to decide whether
          to nudge coordindates. If a row in the dataframe does not have a value
          in this raster, nudge it.
        max_degrees_deviation: The maximum angle a coordinate can be nudged. 

    Returns:
        pd.DataFrame: Returns a dataframe with nudged coordinates.
  '''
  for i, row in df.iterrows():
    # Get the lat and long for the current row.
    lat = df.loc[i, "lat"]
    lon = df.loc[i, "long"]

    if _valid_in_all_rasters(lat, lon, rasters):
      continue

    # nudge 0.01 degrees at a time.
    for nudge in [x/100.0 for x in range(1, max_degrees_deviation*100)]:
      if _valid_in_all_rasters(lat + nudge, lon + nudge, rasters):
        df.loc[i, "lat"] = lat + nudge
        df.loc[i, "long"] = lon + nudge
        break
      elif _valid_in_all_rasters(lat - nudge, lon - nudge, rasters):
        df.loc[i, "lat"] = lat - nudge
        df.loc[i, "long"] = lon - nudge
        break
      elif _valid_in_all_rasters(lat + nudge, lon - nudge, rasters):
        df.loc[i, "lat"] = lat + nudge
        df.loc[i, "long"] = lon - nudge
        break
      elif _valid_in_all_rasters(lat - nudge, lon + nudge, rasters):
        df.loc[i, "lat"] = lat - nudge
        df.loc[i, "long"] = lon + nudge
        break
    if df.loc[i, "lat"] == lat and df.loc[i, "long"] == lon:
      raise ValueError("Failed to nudge coordinates into valid space")

  return df

def load_dataset(path: str, 
    mean_label: str, 
    var_label: str, 
    columns_to_keep: List[str], 
    side_raster_input):
  df = pd.read_csv(path, encoding="ISO-8859-1", sep=',')
  df = df[df[var_label].notna()]
  df.reset_index(inplace=True)

  df = nudge_invalid_coords(
      df, list(side_raster_input.values()), max_degrees_deviation=2)

  for name, geotiff in side_raster_input.items():
    df[name] = df.apply(lambda row: geotiff.value_at(row['long'], row['lat']), axis=1)
  for name, geotiff in side_raster_input.items():
    df = df[df[name].notnull()]

  X = df.drop(df.columns.difference(columns_to_keep), axis=1)
  Y = df[[mean_label, var_label]]

  return X, Y

@dataclass
class FeaturesToLabels:
  def __init__(self, X: pd.DataFrame, Y: pd.DataFrame):
    self.X = X
    self.Y = Y

  def as_tuple(self):
    return (self.X, self.Y)


def create_feature_scaler(X: pd.DataFrame,
                          columns_to_passthrough,
                          columns_to_scale,
                          columns_to_standardize) -> ColumnTransformer:
  feature_scaler = ColumnTransformer(
      [(column+'_scaler', MinMaxScaler(), [column]) for column in columns_to_scale] +
      [(column+'_standardizer', StandardScaler(), [column]) for column in columns_to_standardize],
      remainder='passthrough')
  feature_scaler.fit(X)
  return feature_scaler

def create_label_scaler(Y: pd.DataFrame, mean_label: str, var_label: str) -> ColumnTransformer:
  label_scaler = ColumnTransformer([
      ('var_minmax_scaler', MinMaxScaler(), [var_label]),
      ('mean_std_scaler', StandardScaler(), [mean_label])],
      remainder='passthrough')
  label_scaler.fit(Y)
  return label_scaler

def scale(X: pd.DataFrame, feature_scaler):
  # transform() outputs numpy arrays :(  need to convert back to DataFrame.
  X_standardized = pd.DataFrame(feature_scaler.transform(X),
                        index=X.index, columns=X.columns)
  return X_standardized

  # Just a class organization, holds each scaled dataset and the scaler used.
# Useful for unscaling predictions.
@dataclass
class ScaledPartitions():
  def __init__(self,
               feature_scaler: ColumnTransformer,
               label_scaler: ColumnTransformer,
               train: FeaturesToLabels, val: FeaturesToLabels,
               test: FeaturesToLabels):
    self.feature_scaler = feature_scaler
    self.label_scaler = label_scaler
    self.train = train
    self.val = val
    self.test = test


def load_and_scale(config: Dict,
                   mean_label: str,
                   var_label: str,
                   columns_to_passthrough: List[str],
                   columns_to_scale: List[str],
                   columns_to_standardize: List[str],
                   extra_columns_from_geotiffs: Dict[str, raster.AmazonGeoTiff]) -> ScaledPartitions:
  columns_to_keep = columns_to_passthrough + columns_to_scale + columns_to_standardize
  X_train, Y_train = load_dataset(config['TRAIN'], mean_label, var_label, columns_to_keep, extra_columns_from_geotiffs)

  # Optionally load test datasets if given. If not given, evaluation would not be performed on the final model.
  (X_test, Y_test) = (None, None) if 'TEST' not in config else load_dataset(config['TEST'], mean_label, var_label, columns_to_keep, extra_columns_from_geotiffs)

  # Optionally load validation dataset. 
  (X_val, Y_val) = (None, None) if 'VALIDATION' not in config else load_dataset(config['VALIDATION'], mean_label, var_label, columns_to_keep, extra_columns_from_geotiffs)
  
  # Fit the scaler:
  feature_scaler = create_feature_scaler(
      X_train,
      columns_to_passthrough,
      columns_to_scale,
      columns_to_standardize)
  label_scaler = create_label_scaler(Y_train, mean_label=mean_label, var_label=var_label)

  # Apply the scaler:
  train = FeaturesToLabels(scale(X_train, feature_scaler), Y_train)
  test = None if X_test is None else FeaturesToLabels(scale(X_test, feature_scaler), Y_test)
  val = None if X_val is None else FeaturesToLabels(scale(X_val, feature_scaler), Y_val)
  return ScaledPartitions(feature_scaler, label_scaler, train, val, test)
