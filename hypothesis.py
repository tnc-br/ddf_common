from dataclasses import dataclass
from sklearn.metrics import precision_recall_curve
import raster
import scipy.stats
import math
import pandas as pd
import numpy as np
import dataset

_TREE_CODE_COLUMN_NAME = 'Code'
_LONGITUDE_COLUMN_NAME = 'long'
_LATITUDE_COLUMN_NAME = 'lat'
_FRAUD_LABEL_COLUMN_NAME = 'fraud'
_FRAUD_P_VALUE_COLUMN_NAME = 'fraud_p_value'

@dataclass
class HypothesisTest:
    '''
    Represents a hypothesis test of a sample to an isoscape
    '''
    longitude: float
    latitude: float
    p_value: float
    p_value_threshold: float

@dataclass
class FraudMetrics:
    '''
    Collection of metrics for fraud detection
    '''
    isotope_column_names: list[str]
    accuracy: float
    precision: float
    recall: float

def sample_ttest(longitude: float,
                 latitude: float,
                 isotope_means: list[float],
                 isotope_variances: list[float],
                 isotope_counts: list[int],
                 means_isoscapes: list[raster.AmazonGeoTiff],
                 variances_isoscapes: list[raster.AmazonGeoTiff],
                 isoscape_sample_size_per_location: int,
                 p_value_target: float) -> HypothesisTest:
    '''
    Returns Hypothesis test with longitude, latitude, combined_p_value and p_value_target.
    Calculates p values from predicted isotope values from the mean and variance isoscapes
    and combines them by multiplication as combined_p_value.
    longitude: Of the sample
    latitude: Of the sample
    isotope_values: Of the sample
    isotope_counts: Of the sample
    means_isoscape: List of isoscapes where each maps geographic coordinates to a mean isotope value.
    variances_isoscape: List of isoscapes where each maps geographic coordinates to the variance of
                        isotope values at that location.
    isoscape_sample_size_per_location: Number of samples per geographic location used to calculate
                              mean and variance in isoscapes.
    p_value_target: desired p_value for the t-test (e.sample_data: 0.05)
    '''
    p_values = []
    for i, isotope_mean in enumerate(isotope_means):
      isotope_variance = isotope_variances[i]
      data_sample_size = isotope_counts[i]
      means_isoscape = means_isoscapes[i]
      variances_isoscape = variances_isoscapes[i]

      # Values from prediction.
      predicted_isotope_mean = raster.get_data_at_coords(
          means_isoscape, longitude, latitude)
      predicted_isotope_variance = raster.get_data_at_coords(
          variances_isoscape, longitude, latitude)
      
      if (predicted_isotope_mean is None or
        predicted_isotope_variance is None):
        return HypothesisTest(
          longitude, latitude, None, p_value_target)

      # t-student Test
      _, p_value = scipy.stats.ttest_ind_from_stats(
          mean1=predicted_isotope_mean,
          std1=math.sqrt(predicted_isotope_variance),
          nobs1=isoscape_sample_size_per_location,
          mean2=isotope_mean,
          std2=math.sqrt(isotope_variance),
          nobs2=data_sample_size,
          equal_var=False, alternative="two-sided"
      )

      p_values.append(p_value)  
    combined_p_value = np.array(p_values).prod()  

    return HypothesisTest(longitude, latitude, combined_p_value, p_value_target)

def get_predictions_grouped(sample_data: pd.DataFrame,
                    isotope_means_column_names: list[str],
                    isotope_variances_column_names: list[str],
                    isotope_counts_column_names: list[str],
                    means_isoscapes: list[raster.AmazonGeoTiff],
                    variances_isoscapes: list[raster.AmazonGeoTiff],
                    sample_size_per_location: int):
  '''
  Calculates the p values of a hypothesis test for the elements specified by
  isotope_column_names using values from means_isoscapes and variances_isoscapes.
  This method assumes that sample_data is grouped by aggregate_columns.

  sample_data: pd.DataFrame with lat, long, isotope_value counts, means and variances
               and fraudulent columns
  isotope_means_column_names: Names of the columns in sample_data that have isotope mean
                              values. They must correspond in order to the element order of
                              means_isoscapes and variances_isoscapes.
  isotope_variances_column_names: Names of the columns in sample_data that have isotope variance
                              values. They must correspond in order to the element order of
                              means_isoscapes and variances_isoscapes.
  isotope_counts_column_names: Names of the columns in sample_data that have isotope count
                              values. They must correspond in order to the element order of
                              means_isoscapes and variances_isoscapes.
  means_isoscapes: Isoscapes where each maps geographic coordinates to a mean isotope value.
  variances_isoscapes: Isoscapes where each maps geographic coordinates to the variance of
                      isotope values at that location.
  sample_size_per_location: Number of samples per geographic location used to calculate
                            mean and variance in means_isoscapes and variances_isoscapes.
  '''
  predictions = sample_data
  predictions[_FRAUD_P_VALUE_COLUMN_NAME] = predictions.apply(lambda row: sample_ttest(
      longitude=row[_LONGITUDE_COLUMN_NAME],
      latitude=row[_LATITUDE_COLUMN_NAME],
      isotope_means=row[isotope_means_column_names],
      isotope_variances=row[isotope_variances_column_names],
      means_isoscapes=means_isoscapes,
      variances_isoscapes=variances_isoscapes,
      isoscape_sample_size_per_location=sample_size_per_location,
      data_sample_sizes=row[isotope_counts_column_names],
      p_value_target=None
    ).p_value, axis=1)

  return predictions

def get_predictions(sample_data: pd.DataFrame,
                    isotope_column_names: list[str],
                    means_isoscapes: list[raster.AmazonGeoTiff],
                    variances_isoscapes: list[raster.AmazonGeoTiff],
                    sample_size_per_location: int):
  '''
  Calculates the p values of a hypothesis test for the elements specified by
  isotope_column_names using values from means_isoscapes and variances_isoscapes.
  This method assumes that the data is not grouped by aggregate_columns.

  sample_data: pd.DataFrame with lat, long, isotope_value and fraudulent columns
  isotope_column_names: Name of the columns in sample_data that has raw isotope values.
                        They should follow the size and order of elements of means_isoscapes
                        and variances_isoscapes.
  means_isoscapes: Isoscapes where each maps geographic coordinates to a mean isotope value.
  variances_isoscapes: Isoscapes where each maps geographic coordinates to the variance of
                      isotope valuesat that location.
  sample_size_per_location: Number of samples per geographic location used to calculate
                            mean and variance in isoscapes.
  '''
  assert(
    len(isotope_column_names) == len(means_isoscapes) and
    len(isotope_column_names) == len(variances_isoscapes))
  aggregate_columns = [
      _TREE_CODE_COLUMN_NAME,
      _LONGITUDE_COLUMN_NAME,
      _LATITUDE_COLUMN_NAME,
      _FRAUD_LABEL_COLUMN_NAME]

  feature_columns = list(sample_data.columns.values)
  for col in isotope_column_names:
    feature_columns.remove(col)

  sample_data = dataset.preprocess_sample_data(
    df=sample_data,
    feature_columns=feature_columns,
    label_columns=isotope_column_names,
    aggregate_columns=aggregate_columns,
    keep_grouping=True
  )

  return get_predictions_grouped(
    sample_data=sample_data,
    isotope_means_column_names=[f"{col}_mean" for col in isotope_column_names],
    isotope_variances_column_names=[f"{col}_variance" for col in isotope_column_names],
    isotope_counts_column_names=[f"{col}_{dataset.SAMPLE_COUNT_COLUMN_NAME_SUFFIX}" for col in isotope_column_names],
    means_isoscapes=means_isoscapes,
    variances_isoscapes=variances_isoscapes,
    sample_size_per_location=sample_size_per_location)

def fraud_metrics(sample_data: pd.DataFrame,
                  isotope_column_names: list[str],
                  means_isoscapes: list[raster.AmazonGeoTiff],
                  variances_isoscapes: list[raster.AmazonGeoTiff],
                  sample_size_per_location: int,
                  p_value_target: float,
                  group_data: bool = True):
    '''
    Calculates the accuracy, precision, recall based on true positives and negatives,
    and the false positive and negatives. (go/ddf-glossary)

    sample_data: pd.DataFrame with lat, long, isotope_value and fraudulent columns
    isotope_column_names: Names of the columns that correspond to separate elements to consider
                          which need to follow the length and order of means_isoscapes
                          and variances_isoscapes.
    means_isoscapes: Isoscapes where each maps geographic coordinates to a mean isotope value.
    variances_isoscapes: Isoscapes where each maps geographic coordinates to the variance of
                        isotope valuesat that location.
    sample_size_per_location: Number of samples per geographic location used to calculate
                              mean and variance in isoscapes.
    p_value_target: desired p_value for the t-test (e.sample_data: 0.05)
    group_data: Whether or not the data needs to be grouped.
    '''
    if group_data:
      predictions = get_predictions(sample_data,
                    isotope_column_names,
                    means_isoscapes,
                    variances_isoscapes,
                    sample_size_per_location)
    else:
      predictions = get_predictions_grouped(sample_data,
                    isotope_column_names,
                    means_isoscapes,
                    variances_isoscapes,
                    sample_size_per_location)
    
    # A low p-value in our t-test indicates that two distributions (the ground truth and sample being tested)
    # are dissimilar, which should cause a positive (fraud) result."
    # https://screenshot.googleplex.com/8gphW7cydwLeBEB
    true_positives = (
      predictions[(predictions[_FRAUD_LABEL_COLUMN_NAME] == True) &
                  (predictions[_FRAUD_P_VALUE_COLUMN_NAME] < p_value_target)].shape[0])
    true_negatives = (
      predictions[(predictions[_FRAUD_LABEL_COLUMN_NAME] == False) &
                  (predictions[_FRAUD_P_VALUE_COLUMN_NAME] >= p_value_target)].shape[0])
    false_positives = (
      predictions[(predictions[_FRAUD_LABEL_COLUMN_NAME] == False) &
                  (predictions[_FRAUD_P_VALUE_COLUMN_NAME] < p_value_target)].shape[0])
    false_negatives = (
      predictions[(predictions[_FRAUD_LABEL_COLUMN_NAME] == True) &
                  (predictions[_FRAUD_P_VALUE_COLUMN_NAME] >= p_value_target)].shape[0])
    
    rows = predictions.shape[0]
    if rows == 0:
      return FraudMetrics(isotope_column_names=isotope_column_names,
                          accuracy=0, precision=0, recall=0)
      
    accuracy = (true_negatives + true_positives)/rows

    precision = 0
    if (true_positives + false_positives) > 0:
      precision = true_positives / (true_positives + false_positives)

    recall = 0
    if (false_negatives + true_positives) > 0:
      recall = true_positives / (false_negatives + true_positives)

    return FraudMetrics(isotope_column_names=isotope_column_names,
                        accuracy=accuracy,
                        precision=precision,
                        recall=recall)
