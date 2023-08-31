# Module for helper functions for manipulating data and datasets.

import datetime
import ee
import math
import numpy as np
import pandas as pd
import pytest
import random
import raster

from dataclasses import dataclass
from enum import Enum
from geopy import distance
from itertools import permutations
from numpy.random import MT19937, RandomState, SeedSequence
from shapely import Polygon
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

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
  FURTHEST_POINTS = 3

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

@dataclass
class FurthestPointsPartitionStrategy:
  '''
  Defines the parameters for the FURTHEST_POINTS partition strategy
  - top_n: How many furthest points to consider as centroids to sample
  '''
  top_n: int 
  train_fraction: float
  validation_fraction: float
  test_fraction: float
  random_seed: int
  max_attempts: int

_FURTHEST_POINTS_STRATEGY = FurthestPointsPartitionStrategy(
  top_n=5,
  train_fraction=0.8,
  validation_fraction=0.1,
  test_fraction=0.1,
  random_seed=123,
  max_attempts=1000
)

# Standard column names in reference samples.
_LONGITUDE_COLUMN_NAME = "long"
_LATITUDE_COLUMN_NAME = "lat"

def _partition_data_fixed(sample_data: pd.DataFrame,
                          strategy: FixedPartitionStrategy) -> PartitionedDataset:
  '''
  Return data split between the fixed rectangle train_validation_test_bounds
  of latitude and longitude for each of the rows in sample_data. Ranges of partitions are [min, max).
  '''
  train_bounds = strategy.train_fixed_bounds
  validation_bounds = strategy.validation_fixed_bounds
  test_bounds = strategy.test_fixed_bounds

  train_data = sample_data[
    (sample_data[_LATITUDE_COLUMN_NAME] >= train_bounds.min_latitude) &
    (sample_data[_LONGITUDE_COLUMN_NAME] >= train_bounds.min_longitude) &
    (sample_data[_LATITUDE_COLUMN_NAME] < train_bounds.max_latitude) &
    (sample_data[_LONGITUDE_COLUMN_NAME] < train_bounds.max_longitude)]
  validation_data = sample_data[
    (sample_data[_LATITUDE_COLUMN_NAME] >= validation_bounds.min_latitude) &
    (sample_data[_LONGITUDE_COLUMN_NAME] >= validation_bounds.min_longitude) &
    (sample_data[_LATITUDE_COLUMN_NAME] < validation_bounds.max_latitude) &
    (sample_data[_LONGITUDE_COLUMN_NAME] < validation_bounds.max_longitude)]
  test_data = sample_data[
    (sample_data[_LATITUDE_COLUMN_NAME] >= test_bounds.min_latitude) &
    (sample_data[_LONGITUDE_COLUMN_NAME] >= test_bounds.min_longitude) &
    (sample_data[_LATITUDE_COLUMN_NAME] < test_bounds.max_latitude) &
    (sample_data[_LONGITUDE_COLUMN_NAME] < test_bounds.max_longitude)]

  return PartitionedDataset(train=train_data, test=test_data, validation=validation_data)

def _partition_data_random(sample_data: pd.DataFrame,
                           strategy: RandomPartitionStrategy):
  '''
  Return sample_data split randomly into train/validation/test buckets based on
  the provided strategy.
  '''
  sample_data.sample(frac=1, random_state=strategy.random_seed)
  n_train = int(sample_data.shape[0] * strategy.train_fraction)
  n_validation = int(sample_data.shape[0] * strategy.validation_fraction)

  train_data = sample_data.iloc[:n_train]
  validation_data = sample_data.iloc[n_train:n_train+n_validation]
  test_data = sample_data.iloc[n_train+n_validation:]

  return PartitionedDataset(train=train_data, test=test_data, validation=validation_data)

def _nearest_neighbors(
  center_coordinate: list[float],
  sample_data: pd.DataFrame,
  n_neighbors: int
):
  '''
  From sample_data, select the n_neighbors closest to the specified
  center_coordinate.
  '''
  assert(len(center_coordinate) == 2)
  # Sometimes rounding of split fractions with dataframe shapes
  # may end up with bigger or smaller splits.
  n_neighbors = int(
    min(sample_data.shape[0], n_neighbors))
  if n_neighbors < 3:
    return pd.DataFrame({
      _LONGITUDE_COLUMN_NAME: [],
      _LATITUDE_COLUMN_NAME: [],
    })
  near = NearestNeighbors(n_neighbors=n_neighbors).fit(
    sample_data[[_LONGITUDE_COLUMN_NAME, _LATITUDE_COLUMN_NAME]]
  )
  indices = near.kneighbors(
    [center_coordinate], return_distance=False)[0]
  return sample_data.iloc[indices, :]

def _partition_by_nearest_neighbors(
  sample_data: pd.DataFrame,
  strategy: FurthestPointsPartitionStrategy,
  train_coord: tuple,
  validation_coord: tuple,
  test_coord: tuple
):
  '''
  Returns a PartitionedDataset from splits of a sample_data pd.DataFrame by the train,
  validation and test split fractions, choosing the points based on the nearest points
  for each of the train_coord, validation_coord and test_coord arguments.
  It first gets the nearest neighbors for train_coord, then validation_coord
  and finally test_coord.
  '''
  train_split = _nearest_neighbors(
    center_coordinate=train_coord,
    sample_data=sample_data,
    n_neighbors=sample_data.shape[0]*strategy.train_fraction
  )
  filtered_sample_data = sample_data[
          ~sample_data.index.isin(
            train_split.index.values
          )]
  
  validation_split = _nearest_neighbors(
    center_coordinate=validation_coord,
    sample_data=filtered_sample_data,
    n_neighbors=sample_data.shape[0]*strategy.validation_fraction 
  )
  filtered_sample_data = filtered_sample_data[
    ~filtered_sample_data.index.isin(
      validation_split.index.values
    )]
  
  test_split = _nearest_neighbors(
    center_coordinate=test_coord,
    sample_data=filtered_sample_data,
    n_neighbors=filtered_sample_data.shape[0] 
  )

  return PartitionedDataset(
    train=train_split,
    test=test_split,
    validation=validation_split
  )

def _polygon(
  coordinates: list[list]
):
  '''
  Given an unordered list of longitude and latitude coordinates
  represented by tuples, returns a polygon with the shape of the
  coordinates.
  '''
  if len(coordinates) < 3:
    return None
  # Compute centroid
  centroid = (sum([c[0] for c in coordinates])/len(coordinates),
              sum([c[1] for c in coordinates])/len(coordinates))
  # Sort by polar angle. This arranges the coordinates so that they
  # delimit an outward shape.
  sorted_coordinates = list(coordinates)
  sorted_coordinates.sort(
    key=lambda c: math.atan2(
      c[1] - centroid[1],
      c[0] - centroid[0])
  )
  return Polygon(sorted_coordinates)

def _valid_polygons(
  train_polygon: Polygon,
  validation_polygon: Polygon,
  test_polygon: Polygon
):
  '''
  Returns true if the train, validation, and test polygons
  have a larger area than or equal to 0.1 and they don't overlap
  over each other. Returns false otherwise. 
  '''
  small_polygon = False
  for polygon in [train_polygon, validation_polygon, test_polygon]:
    if polygon.area < 0.1:
      return False

  if (train_polygon.intersects(validation_polygon) or
     train_polygon.intersects(test_polygon) or
     validation_polygon.intersects(test_polygon)):
    return False

  return True

def _shuffled_unique_coordinates(
  sample_data: pd.DataFrame,
  furthest_coordinates: list[list[float]],
  strategy: FurthestPointsPartitionStrategy
):
  '''
  Returns shuffled unique coordinates from sample_data that match
  the expected train, validation and test fractions.
  '''
  unique_coordinates_df = sample_data.groupby(
    by=['long', 'lat']
  ).first().reset_index()
  shuffled_coordinates_df = unique_coordinates_df.sample(frac=1.0,
    random_state=strategy.random_seed)
  coordinates = list(
    shuffled_coordinates_df[
      [_LONGITUDE_COLUMN_NAME, _LATITUDE_COLUMN_NAME]
    ].values)
  assert(len(coordinates) >= strategy.top_n)
  min_size = (int(len(coordinates) * strategy.train_fraction) +
    int(len(coordinates) * strategy.validation_fraction) +
    int(len(coordinates) * strategy.test_fraction))
  assert(int(len(coordinates) * strategy.train_fraction) > 0)
  assert(int(len(coordinates) * strategy.validation_fraction) > 0)
  assert(int(len(coordinates) * strategy.test_fraction) > 0)
  assert(
    min_size <= len(coordinates)
  ), (f"You need {min_size} aka {int(len(coordinates) * strategy.train_fraction)} train + "
    f"{int(len(coordinates) * strategy.validation_fraction)} validation + "
    f"{int(len(coordinates) * strategy.test_fraction)} test samples but you have a sample of {len(coordinates)}")
  
  return coordinates

def _maybe_partition_furthest_points(
  sample_data: pd.DataFrame,
  furthest_coordinates: list[list[float]],
  strategy: FurthestPointsPartitionStrategy
):
  '''
  Returns a PartitionedDataset that has a valid train/validation/test split
  that has:
  - Non-overlapping splits
  - Splits which polygon areas have less than 0.1 area
  It attempts to generate these partitions randomly for at most strategy.max_attempts.
  If unsuccesful, returns None.
  '''
  partitioned_dataset = None
  attempts = 0
  while (attempts <= strategy.max_attempts and
         partitioned_dataset is None):
    random.seed(strategy.random_seed)
    random.shuffle(furthest_coordinates)
    # We need permutations of 3 numbers that correspond to train, validation and test.
    coord_permutations = list(permutations(furthest_coordinates, 3))
    random.shuffle(coord_permutations)
    # Pick one random permutation.
    sampled_permutation = coord_permutations[0]

    partitioned_dataset = _partition_by_nearest_neighbors(
      sample_data=sample_data,
      strategy=strategy,
      train_coord=sampled_permutation[0],
      validation_coord=sampled_permutation[1],
      test_coord=sampled_permutation[2]
    )
    train_polygon = _polygon(partitioned_dataset.train[
      [_LONGITUDE_COLUMN_NAME, _LATITUDE_COLUMN_NAME]
    ].values)
    validation_polygon = _polygon(partitioned_dataset.validation[
      [_LONGITUDE_COLUMN_NAME, _LATITUDE_COLUMN_NAME]
    ].values)
    test_polygon = _polygon(partitioned_dataset.test[
      [_LONGITUDE_COLUMN_NAME, _LATITUDE_COLUMN_NAME]
    ].values)
    are_valid_polygons = _valid_polygons(
      train_polygon=train_polygon, validation_polygon=validation_polygon,
      test_polygon=test_polygon)
    if (train_polygon is None or
        validation_polygon is None or
        test_polygon is None or
        not are_valid_polygons):
      partitioned_dataset = None
    attempts += 1
  return partitioned_dataset

def _partition_data_furthest_points(
  sample_data: pd.DataFrame,
  strategy: FurthestPointsPartitionStrategy
):
  coordinates = _shuffled_unique_coordinates(sample_data, strategy)
  
  # Compute centroid of the coordinates
  centroid = [sum([c[0] for c in coordinates])/len(coordinates),
              sum([c[1] for c in coordinates])/len(coordinates)]
  
  # Sort coordinates from furthest to closest to the centroid.
  distances_to_centroid = []
  for coord in coordinates:
    distances_to_centroid.append((
      math.dist(centroid, coord), (coord[0], coord[1])))
  distances_to_centroid.sort(reverse=True)
  furthest_coordinates = [d[1] for d in distances_to_centroid[:strategy.top_n]]

  partitioned_dataset_or_none = _maybe_partition_furthest_points(
    sample_data=sample_data,
    furthest_coordinates=furthest_coordinates,
    strategy=strategy)
  assert(partitioned_dataset_or_none is not None)

  return partitioned_dataset

def partition(sample_data: pd.DataFrame,
              partition_strategy: PartitionStrategy) -> PartitionedDataset:
  '''
  Splits pd.DataFrame sample_data based on the partition_strategy provided.
  '''
  if partition_strategy == PartitionStrategy.FIXED:
    return _partition_data_fixed(sample_data, _FIXED_PARTITION_STRATEGY)
  elif partition_strategy == PartitionStrategy.RANDOM:
    return _partition_data_random(sample_data, _RANDOM_PARTITION_STRATEGY)
  elif partition_strategy == PartitionStrategy.FURTHEST_POINTS:
    return _partition_data_k_means_furthest_points(
      sample_data, _FURTHEST_POINTS_STRATEGY
    )
  else:
    raise ValueError(f"Unknown partition strategy: {partition_strategy}")

def print_split(dataset: PartitionedDataset) -> None:
  total_len = len(dataset.train)+len(dataset.validation)+len(dataset.test)
  print(f"Train: {100*len(dataset.train)/total_len:.2f}% ({len(dataset.train)})")
  print(f"Test: {100*len(dataset.test)/total_len:.2f}% ({len(dataset.test)})")
  print(f"Validation: {100*len(dataset.validation)/total_len:.2f}% ({len(dataset.validation)})")

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
  partition_data = partition(aggregate_reference_data(reference_csv_filename),
                             PartitionStrategy.FIXED)
  print_split(partition_data)
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