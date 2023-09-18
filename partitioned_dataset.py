# Module that has helper classes and functions to partition dataset.

from dataclasses import dataclass
from enum import Enum
from itertools import permutations
from shapely import Polygon
from sklearn.neighbors import NearestNeighbors
import math
import pandas as pd
import random
import matplotlib.pyplot as plt

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
  Defines the parameters for the FURTHEST_POINTS partition strategy.
  What this strategy consists of is, attempting max_attempts times:

  1. Get top_n coordinates from the sample that are the furthest away
     from the centroid of the overall sample.
  2. Sample randomly 3 of these top_n coordinates. This will be the
     centroid of the train/validation/test splits.
  3. Get the sample_size*fraction nearest neighbors to each centroid.
  4. Check that new splits don't overlap or have an area smaller than 0.1

  Attributes:
  - top_n: How many furthest points to consider as centroids to sample
           that will be the centroid of nearest neighbors for train/validation/test
           splits respectivelty.
  - train/validation/test_fraction: how much of the unique coordinate samples to
           assign to each split
  - max_attempts: How many teams to sample furthest points as centroids randomly to
                  attempt creating a valid split. See _maybe_partition_furthest_points
                  for more details.
  '''
  top_n: int 
  train_fraction: float
  validation_fraction: float
  test_fraction: float
  random_seed: int
  max_attempts: int

_FURTHEST_POINTS_STRATEGY = FurthestPointsPartitionStrategy(
  top_n=5,
  train_fraction=0.6,
  validation_fraction=0.2,
  test_fraction=0.2,
  random_seed=1500,
  max_attempts=1000
)

# Standard column names in reference samples.
_LONGITUDE_COLUMN_NAME = "long"
_LATITUDE_COLUMN_NAME = "lat"

@dataclass(eq=True, frozen=True, order=True)
class Coordinate:
  '''
  Represents a geospatial coordinate.
  A immutable class that we're able to order and compare based
  on its fields.
  '''
  longitude: float
  latitude: float

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
  center_coordinate: Coordinate,
  sample_data: pd.DataFrame,
  n_neighbors: int) -> pd.DataFrame:
  '''
  From sample_data, select the n_neighbors closest to the specified
  center_coordinate.
  '''
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
    [[center_coordinate.longitude, center_coordinate.latitude]],
    return_distance=False)[0]
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
  coordinates: list[list]) -> Polygon:
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
  test_polygon: Polygon) -> bool:
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
    # Sometimes geometries intersect only at the vertices and we don't
    # want to exclude this kind of candidates.
    if (train_polygon.intersection(validation_polygon).area >= 0.1 or
        train_polygon.intersection(test_polygon).area >= 0.1 or
        validation_polygon.intersection(test_polygon).area >= 0.1):
      return False

  return True

def _shuffled_unique_coordinates(
  sample_data: pd.DataFrame,
  strategy: FurthestPointsPartitionStrategy) -> list[Coordinate]:
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
  if len(coordinates) < strategy.top_n:
    raise ValueError
  min_size = (int(len(coordinates) * strategy.train_fraction) +
    int(len(coordinates) * strategy.validation_fraction) +
    int(len(coordinates) * strategy.test_fraction))
  if ((int(len(coordinates) * strategy.train_fraction) <= 0) or
     (int(len(coordinates) * strategy.validation_fraction) <= 0) or
     (int(len(coordinates) * strategy.test_fraction) <= 0)):
     raise ValueError
  
  if (min_size > len(coordinates)):
    print(f"You need {min_size} aka {int(len(coordinates) * strategy.train_fraction)} train + "
      f"{int(len(coordinates) * strategy.validation_fraction)} validation + "
      f"{int(len(coordinates) * strategy.test_fraction)} test samples " +
      f"but you have a sample of {len(coordinates)}")
    raise ValueError
  
  coordinates = [Coordinate(longitude=c[0], latitude=c[1]) for c in coordinates]
  return coordinates

def _maybe_partition_furthest_points(
  sample_data: pd.DataFrame,
  furthest_coordinates: list[Coordinate],
  strategy: FurthestPointsPartitionStrategy) -> PartitionedDataset:
  '''
  Returns a PartitionedDataset that has a valid train/validation/test split
  where each split is made of coordinates from sample_data which are grouped into
  clusters where furthest_points are the centers of each train/validation/split.
  that has:
  - Non-overlapping splits
  - Splits where the polygon generated by their coordinates has an area of
    at least 0.1 degrees (lat/lon)
  It attempts to generate these partitions randomly for at most strategy.max_attempts.
  If unsuccesful, returns None.
  '''
  partitioned_dataset = None
  for _ in range(strategy.max_attempts):
    if partitioned_dataset is not None:
      break
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
  return partitioned_dataset

def _partition_data_furthest_points(
  sample_data: pd.DataFrame,
  strategy: FurthestPointsPartitionStrategy) -> PartitionedDataset:
  '''
  Returns a Partitioned dataset that consists of splits of the coordinates 
  longitude and latitude) of sample_data into train/validation/test splits.
  Each split has a centroid.
  Each centroid comes from a random sub-sample of the strategy.top_n
  furthest points (Eucledian distance) from the overall centroid of
  the coordinates in sample_data.
  If a split can't be generated where the splits are not overlapping,
  or are too small (< 0.1 long/lat degrees of area), we raise an exception.
  '''
  coordinates = _shuffled_unique_coordinates(sample_data, strategy)
  
  # Compute centroid of the coordinates
  centroid = [sum([c.longitude for c in coordinates])/len(coordinates),
              sum([c.latitude for c in coordinates])/len(coordinates)]
  
  # Sort coordinates from furthest to closest to the centroid.
  distances_to_centroid = []
  for coord in coordinates:
    distances_to_centroid.append((
      math.dist(centroid, [coord.longitude, coord.latitude]), coord))
  distances_to_centroid.sort(reverse=True)
  furthest_coordinates = [d[1] for d in distances_to_centroid[:strategy.top_n]]

  partitioned_dataset_or_none = _maybe_partition_furthest_points(
    sample_data=sample_data,
    furthest_coordinates=furthest_coordinates,
    strategy=strategy)
  if (partitioned_dataset_or_none is None):
    raise ValueError

  return partitioned_dataset_or_none

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
    return _partition_data_furthest_points(
      sample_data, _FURTHEST_POINTS_STRATEGY
    )
  else:
    raise ValueError(f"Unknown partition strategy: {partition_strategy}")

def print_split(dataset: PartitionedDataset) -> None:
  total_len = len(dataset.train)+len(dataset.validation)+len(dataset.test)
  print(f"Train: {100*len(dataset.train)/total_len:.2f}% ({len(dataset.train)})")
  print(f"Test: {100*len(dataset.test)/total_len:.2f}% ({len(dataset.test)})")
  print(f"Validation: {100*len(dataset.validation)/total_len:.2f}% ({len(dataset.validation)})")
