# Module for helper functions for manipulating data and datasets.
from dataclasses import dataclass
import pandas as pd
import raster
from numpy.random import MT19937, RandomState, SeedSequence
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
    min_lattitude: float
    max_lattitude: float


# For FIXED data partition only, the bounds of each partition for the split
_TRAIN_FIXED_BOUNDS = DatasetGeographicPartitions(
    min_longitude=-62.5,
    max_longitude=float('inf'),
    min_lattitude=-5,
    max_lattitude=float('inf'),
)
_VALIDATION_FIXED_BOUNDS = DatasetGeographicPartitions(
    min_longitude=float('-inf'),
    max_longitude=-62.5,
    min_lattitude=-5,
    max_lattitude=float('inf')
)
_TEST_FIXED_BOUNDS = DatasetGeographicPartitions(
    min_longitude=float('-inf'),
    max_longitude=float('inf'),
    min_lattitude=float('-inf'),
    max_lattitude=-5
)
_TRAIN_VALIDATION_TEST_BOUNDS = [
    _TRAIN_FIXED_BOUNDS, _VALIDATION_FIXED_BOUNDS, _TEST_FIXED_BOUNDS]

# For RANDOM data partition only, the fraction of the dataset allocated for train, validation
# and test sets.
TRAIN_VALIDATION_TEST_RATIOS = [0.8, 0.1, 0.1]


def _partition_data_fixed(sample_data: pd.DataFrame,
                          train_validation_test_bounds: list[DatasetGeographicPartitions]) -> PartitionedDataset:
    '''
    Return data split between the fixed rectangle train_validation_test_bounds
    of lattitude and longitude for each of the rows in sample_data. Ranges of partitions are [min, max).
    '''
    train_bounds = train_validation_test_bounds[0]
    validation_bounds = train_validation_test_bounds[1]
    test_bounds = train_validation_test_bounds[2]

    train_data = sample_data[
        (sample_data['lat'] >= train_bounds.min_lattitude) & (sample_data['long'] >= train_bounds.min_longitude) &
        (sample_data['lat'] < train_bounds.max_lattitude) & (sample_data['long'] < train_bounds.max_longitude)]
    validation_data = sample_data[
        (sample_data['lat'] >= validation_bounds.min_lattitude) & (sample_data['long'] >= validation_bounds.min_longitude) &
        (sample_data['lat'] < validation_bounds.max_lattitude) & (sample_data['long'] < validation_bounds.max_longitude)]
    test_data = sample_data[
        (sample_data['lat'] >= test_bounds.min_lattitude) & (sample_data['long'] >= test_bounds.min_longitude) &
        (sample_data['lat'] < test_bounds.max_lattitude) & (sample_data['long'] < test_bounds.max_longitude)]

    return PartitionedDataset(train=train_data, test=test_data, validation=validation_data)


def _partition_data_random(sample_data, train_validation_test_ratios):
    '''
    Return sample_data split randomly into train/validation/test buckets based on
    train_validation_test_ratios.
    '''
    sample_data.sample(frac=1)
    n_train = int(sample_data.shape[0] * train_validation_test_ratios[0])
    n_validation = int(sample_data.shape[0] * train_validation_test_ratios[1])

    train_data = sample_data.iloc[:n_train]
    validation_data = sample_data.iloc[n_train:n_train+n_validation]
    test_data = sample_data.iloc[n_train+n_validation:]

    return PartitionedDataset(train=train_data, test=test_data, validation=validation_data)


def partition(sample_data: pd.DataFrame,
              partition_strategy: str) -> PartitionedDataset:
    '''
    Splits pd.DataFrame sample_data based on the partition_strategy provided.
    Valid argument values: "FIXED", "RANDOM"
    '''
    if partition_strategy == "FIXED":
        return _partition_data_fixed(sample_data, _TRAIN_VALIDATION_TEST_BOUNDS)
    elif partition_strategy == "RANDOM":
        return _partition_data_random(sample_data, TRAIN_VALIDATION_TEST_RATIOS)
    else:
        raise ValueError(f"Unknown partition strategy: {partition_strategy}")


def print_split(dataset: PartitionedDataset) -> None:
    total_len = len(dataset.train)+len(dataset.validation)+len(dataset.test)
    print(
        f"Train: {100*len(dataset.train)/total_len:.2f}% ({len(dataset.train)})")
    print(f"Test: {100*len(dataset.test)/total_len:.2f}% ({len(dataset.test)})")
    print(
        f"Validation: {100*len(dataset.validation)/total_len:.2f}% ({len(dataset.validation)})")


def gen_tabular_dataset(monthly: bool, samples_per_site: int) -> pd.DataFrame:
    return gen_tabular_dataset_with_coords(monthly, samples_per_site,
                                           [(-70, -5,), (-67.5, 0,), (-66, -4.5,), (-63, -9.5,),
                                            (-63, -9,), (-62, -6,
                                                         ), (-60, -2.5,), (-60, 1,),
                                               (-60, -12.5,), (-59, -
                                                               2.5,), (-57.5, -4,),
                                               (-55, -3.5,), (-54, -1,), (-52.5, -13,), (-51.5, -2.5,)],
                                           0.5)


def gen_tabular_dataset_with_coords(monthly: bool, samples_per_site: int,
                                    sample_site_coordinates: list, sample_radius: float) -> pd.DataFrame:
    features = [raster.relative_humidity_geotiff(),
                raster.temperature_geotiff(),
                raster.vapor_pressure_deficit_geotiff(),
                raster.atmosphere_isoscape_geotiff(),
                raster.cellulose_isoscape_geotiff()]
    image_feature_names = ["rh", "temp", "vpd",
                           "atmosphere_oxygen_ratio", "cellulose_oxygen_ratio"]
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
                                                  samples_per_site=1, sample_site_coordinates=locations, sample_radius=0)
    # drop the simulated cellulose_oxygen_ratio.
    # TODO(https://github.com/tnc-br/ddf_common/issues/5), refactor the code to only do features.
    sample_data = sample_data.drop('cellulose_oxygen_ratio', axis=1)

    # Here we merge the features "rh", "temp", "vpd", and  "atmosphere_oxygen_ratio"
    # with the means based on lat/long
    sample_data = pd.merge(sample_data, means, how="inner",
                           left_on=['lat', 'lon'], right_on=['lat', 'long'])
    sample_data = sample_data.drop('long', axis=1).rename(
        columns={'d18O_cel': 'cellulose_oxygen_ratio'}).reset_index()
    sample_data.drop('index', inplace=True, axis=1)

    return sample_data

# If a reference_csv_filename is requested, we use that over any simulated data.


def aggregate_reference_data(reference_csv_filename: str) -> pd.DataFrame:
    if reference_csv_filename:
        return load_sample_data(reference_csv_filename)
    # This is the historical simulated input that has 17 samples x 15 sites
    return gen_tabular_dataset(monthly=False, samples_per_site=17)


def partitioned_reference_data(reference_csv_filename: str) -> PartitionedDataset:
    partition_data = partition(
        aggregate_reference_data(reference_csv_filename))
    print_split(partition_data)
    return partition_data
