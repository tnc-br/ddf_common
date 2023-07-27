from dataclasses import dataclass
import raster
import scipy.stats
import math
import pandas as pd

_LONGITUDE_COLUMN_NAME = 'long'
_LATITUDE_COLUMN_NAME = 'lat'
_FRAUDULENT_COLUMN_NAME = 'fraud'


@dataclass
class HypothesisTest:
    '''
    Represents a hypothesis test of a sample to an isoscape
    '''
    longitude: float
    latitude: float
    p_value: float
    p_value_threshold: float


def sample_ttest(longitude: float,
                 latitude: float,
                 isotope_values: pd.Series,
                 means_isoscape: raster.AmazonGeoTiff,
                 variances_isoscape: raster.AmazonGeoTiff,
                 sample_size_per_location: int,
                 p_value_target: float):
    '''
    longitude: Of the sample
    latitude: Of the sample
    isotope_values: Of the sample
    means_isoscape: Isoscape that maps geographic coordinates to a mean isotope value.
    variances_isoscape: Isoscape that maps geographic coordinates to the variance of
                        isotope valuesat that location.
    sample_size_per_location: Number of samples per geographic location used to calculate
                              mean and variance in isoscapes.
    p_value_target: desired p_value for the t-test (e.sample_data: 0.05)
    '''
    if isotope_values.size <= 1:
        raise ValueError  # Isotope values needs to be more than 1.

    isotope_mean = isotope_values.mean()
    isotope_variance = isotope_values.var()*(isotope_values.size /
                                             (isotope_values.size - 1))
    isotope_sample_count = isotope_values.size

    # Values from prediction.
    predicted_isotope_mean = raster.get_data_at_coords(
        means_isoscape, longitude, latitude, 0)
    predicted_isotope_variance = raster.get_data_at_coords(
        variances_isoscape, longitude, latitude, 0)
    predicted_isotope_sample_count = sample_size_per_location

    # t-student Test
    _, p_value = scipy.stats.ttest_ind_from_stats(
        predicted_isotope_mean,
        math.sqrt(predicted_isotope_variance),
        predicted_isotope_sample_count,
        isotope_mean,
        math.sqrt(isotope_variance),
        isotope_sample_count,
        equal_var=False, alternative="two-sided"
    )

    return HypothesisTest(longitude, latitude, p_value, p_value_target)

def fraud_metrics(sample_data: pd.DataFrame,
                  isotope_column_name: str,
                  means_isoscape: raster.AmazonGeoTiff,
                  variances_isoscape: raster.AmazonGeoTiff,
                  sample_size_per_location: int,
                  p_value_target: float):
    '''
    Calculates the accuracy, precision, recall based on true positives and negatives,
    and the false positive and negatives. (go/ddf-glossary)

    sample_data: pd.DataFrame with lat, long, isotope_value and fraudulent columns
    means_isoscape: Isoscape that maps geographic coordinates to a mean isotope value.
    variances_isoscape: Isoscape that maps geographic coordinates to the variance of
                        isotope valuesat that location.
    sample_size_per_location: Number of samples per geographic location used to calculate
                              mean and variance in isoscapes.
    p_value_target: desired p_value for the t-test (e.sample_data: 0.05)
    '''
    sample_data = sample_data.groupby([
        _LONGITUDE_COLUMN_NAME,
        _LATITUDE_COLUMN_NAME,
        _FRAUDULENT_COLUMN_NAME])[isotope_column_name]
      
    # Counts the number of locations in the sample have more than one row in dataset.
    rows = 0

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    for group_key, isotope_values in sample_data:
        if isotope_values.size <= 1:
            continue
        hypothesis_test = sample_ttest(group_key[0], group_key[1], isotope_values, means_isoscape,
                                       variances_isoscape, sample_size_per_location, p_value_target)

        if not group_key[2]:
            if hypothesis_test.p_value >= p_value_target:
                true_negative += 1
            else:
                false_positive += 1
        else:
            if hypothesis_test.p_value >= p_value_target:
                false_negative += 1
            else:
                true_positive += 1

        rows += 1

    if rows == 0:
        return (0, 0, 0)

    accuracy = (true_negative + true_positive)/rows

    precision = 0
    if (true_positive + false_positive) > 0:
        precision = true_positive / (true_positive + false_positive)

    recall = 0
    if (false_negative + true_positive) > 0:
        recall = true_positive / (false_negative + true_positive)

    return (accuracy, precision, recall)
