# Module for helper functions for manipulating data and datasets.
from dataclasses import dataclass
import pandas as pd
import raster
from numpy.random import MT19937, RandomState, SeedSequence
from tqdm import tqdm
from geopy import distance
import math
import pytest
import numpy as np
import random

@dataclass
class PartitionedDataset:
  train: pd.DataFrame
  test: pd.DataFrame
  validation: pd.DataFrame

def partition(df) -> PartitionedDataset:
  train = df[df["lon"] < -55]
  test = df[(df["lon"] >= -55) & (df["lat"] > -2.85)]
  validation = df[(df["lon"] >= -55) & (df["lat"] <= -2.85)]
  return PartitionedDataset(train, test, validation)

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

def gen_tabular_dataset_with_coords(monthly: bool, samples_per_site: int, 
  sample_site_coordinates: list, sample_radius: float) -> pd.DataFrame:
  features = [raster.relative_humidity_geotiff(), 
    raster.temperature_geotiff(), 
    raster.vapor_pressure_deficit_geotiff(), 
    raster.atmosphere_isoscape_geotiff(), 
    raster.cellulose_isoscape_geotiff()]
  image_feature_names = ["rh", "temp", "vpd", "atmosphere_oxygen_ratio", "cellulose_oxygen_ratio"]
  feature_names = ["lat", "lon", "month_of_year"] + image_feature_names
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
          row["lon"] = sample_x
          row["lat"] = sample_y
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
  df = df[['Code', 'lat', 'long', 'd18O_cel']]
  df = df[df['d18O_cel'].notna()]

  grouped = df.groupby(['lat', 'long'])

  # means is the reference sample calculated mean of d18O at each lat/lon
  means = grouped.mean().reset_index()

  # locations is now the list of unique lat and longs
  locations = list(zip(means["long"], means["lat"]))

  sample_data = gen_tabular_dataset_with_coords(monthly=False,
   samples_per_site=1, sample_site_coordinates=locations, sample_radius = 0)
  # drop the simulated cellulose_oxygen_ratio. TODO, refactor the code so that
  # it only does features.
  sample_data = sample_data.drop('cellulose_oxygen_ratio', axis = 1)

  # Here we merge the features "rh", "temp", "vpd", and  "atmosphere_oxygen_ratio"
  # with the means based on lat/long
  sample_data = pd.merge(sample_data, means, how="inner", 
    left_on=['lat', 'lon'], right_on=['lat', 'long'])
  sample_data = sample_data.drop('long', axis=1).rename(
    columns={'d18O_cel': 'cellulose_oxygen_ratio' }).reset_index()
  sample_data.drop('index', inplace=True, axis=1)


  return sample_data

# If a reference_csv_filename is requested, we use that over any simulated data.
def yearly_data_255_trees(reference_csv_filename: str) -> pd.DataFrame:
  if reference_csv_filename:
    return load_sample_data(reference_csv_filename)
  # This is the historical simulated input that has 17 samples x 15 sites
  return  gen_tabular_dataset(monthly=False, samples_per_site=17)

def yearly_255_trees_partitioned(reference_csv_filename: str) -> PartitionedDataset:
  partition_data = partition(yearly_data_255_trees(reference_csv_filename))
  print_split(partition_data)
  return partition_data

#Utility function for randomly sampling a point around a sample site
def _is_valid_point(lat: float, lon: float, reference_isocape: raster.AmazonGeoTiff):
  try:
    raster.get_data_at_coords(reference_isocape, lon, lat, 0)
    return True
  except ValueError:
    return False

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
def create_fraudulent_samples(real_samples_data: pd.DataFrame, mean_iso: raster.AmazonGeoTiff,element: str,max_trusted_radius: float,max_fraud_radius:float,min_fraud_radius:float) -> pd.DataFrame:
  '''
  This function creates a dataset based on real samples adding a Fraud column, where True represents a real lat/lon and False represents a fraudulent lat/lon
  Input:
  - real_samples_data: dataset containing real samples
  - element: element that will be used in the ttest: Oxygen (e.g: d18O_cel), Carbon or Nitrogen.
  - mean_iso: isoscape averages
  - max_trusted_radius, In km, the maximum distance from a real point where its value is still considered legitimate.
  - max_fraud_radius: In km, the maximum distance from a real point to randomly sample a fraudalent coordinate.
  - min_fraud_radius: In km, the minimum distance from a real point to randomly sample a fraudalent coordinate.
  Output: 
  - fake_data: pd.DataFrame with lat, long, isotope_value and fraudulent columns
  '''

  real_samples_data.dropna(subset=[element], how='all', inplace=True)
  real_samples = real_samples_data.groupby(['lat','long'])[element]
  real_samples_code = real_samples_data.groupby(['lat','long','Code'])[element]

  count = 0
  lab_samp = real_samples

  if max_fraud_radius <= min_fraud_radius:
    raise ValueError("max_fraud_radius {} <= min_fraud_radius {}".format(
        max_fraud_radius, min_fraud_radius))
    
  fake_sample = pd.DataFrame(columns=['Code',
          'lat',
          'long',
          element,
          'fraud'])

  # Max number of times to attempt to generate random coordinates.
  max_random_sample_attempts = 1000
  count = 0

  for coord, lab_samp in real_samples_code:
    if lab_samp.size <= 1 :
      continue
    count += 1
    lat, lon, attempts = 0, 0, 0
    while((not _is_valid_point(lat, lon, mean_iso) or
          _is_nearby_real_point(lat, lon, real_samples, min_fraud_radius)) and
          attempts < max_random_sample_attempts):
      lat, lon = _random_nearby_point(coord[0], coord[1], max_fraud_radius)
      if lab_samp.size < 5:
        pass
      else:
      #generating 5 samples per code
        for i in range(5):
          new_row = {'Code': f"fake_mad{count}", 'lat': lat, 'long': lon,element: lab_samp.iloc[i],'fraud': True }
          fake_sample.loc[len(fake_sample)] = new_row
      attempts += 1

  return fake_sample

