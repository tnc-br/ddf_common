# Module for helper functions for manipulating data and datasets.
from dataclasses import dataclass
import pandas as pd
import raster
from numpy.random import MT19937, RandomState, SeedSequence
import numpy as np
from tqdm import tqdm

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

def add_features_from_rasters(df: pd.DataFrame, rasters: list) -> pd.DataFrame:
  '''
  Given a pd.DataFrame df:
  1. enumerates each row looking for a 'lat' and 'lon' column
  2.   for each raster, looks up the lat and lon value
  3.     adds that value to the new feature_df
  4. returns the concat of both original df and new feature_df
  '''
  feature_dict = {}
  for raster in rasters:
    feature_dict[raster.name] = []
  
  for row in df.itertuples():
    lat = getattr(row, "lat")
    lon = getattr(row, "lon")
    for raster in rasters:
      feature_dict[raster.name].append(raster.value_at(lon, lat))

  return pd.concat([pd.DataFrame(feature_dict), df], axis=1)


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
  # drop the simulated cellulose_oxygen_ratio.
  # TODO(https://github.com/tnc-br/ddf_common/issues/5), refactor the code to only do features.
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
def aggregate_reference_data(reference_csv_filename: str) -> pd.DataFrame:
  if reference_csv_filename:
    return load_sample_data(reference_csv_filename)
  # This is the historical simulated input that has 17 samples x 15 sites
  return  gen_tabular_dataset(monthly=False, samples_per_site=17)

def partitioned_reference_data(reference_csv_filename: str) -> PartitionedDataset:
  partition_data = partition(aggregate_reference_data(reference_csv_filename))
  print_split(partition_data)
  return partition_data


def preprocess_sample_data(df: pd.DataFrame,
                           feature_columns: list[str],
                           label_columns: list[str],
                           aggregate_columns: list[str],
                           keep_grouping: bool) -> pd.DataFrame:
  '''
  Given a pd.DataFRame df:
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
