from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import raster
import pandas as pd
import dataset
import hypothesis
import numpy as np
from typing import Dict, Any, List

def calculate_rmse(df, means_isoscape, vars_isoscape, mean_true_name, var_true_name, mean_pred_name, var_pred_name):
  '''
  Calculates the mean, variance and overall (mean and variance) RMSE of df using
  the provided columns mean_true_name, var_true_name, mean_pred_name, var_pred_name
  can take any value except 'truth' and 'prediction'
  '''
  # Make sure names do not collide.
  assert(
      len([mean_true_name, var_true_name, mean_pred_name, var_pred_name, 'truth', 'prediction']) ==
      len(set([mean_true_name, var_true_name, mean_pred_name, var_pred_name, 'truth', 'prediction'])))

  df[mean_pred_name] = df.apply(lambda row:raster.get_data_at_coords(means_isoscape, row['long'],row['lat'],-1), axis=1)
  df[var_pred_name] = df.apply(lambda row:raster.get_data_at_coords(vars_isoscape, row['long'],row['lat'],-1), axis=1)

  predictions = list(df.apply(lambda row: [row[mean_pred_name], row[var_pred_name]], axis=1).values)
  truths = list(df.apply(lambda row: [row[mean_true_name], row[var_true_name]], axis=1).values)

  return (mean_squared_error(df[mean_true_name].values, df[mean_pred_name].values, squared=False),
         mean_squared_error(df[var_true_name].values, df[var_pred_name].values, squared=False),
         mean_squared_error(truths, predictions, squared=False))

def isoscape_precision_recall_thresholds(
    test_dataset: pd.DataFrame,
    isotope_column_names: list[str],
    means_isoscapes: list[raster.AmazonGeoTiff],
    vars_isoscapes: list[raster.AmazonGeoTiff]) -> list[list[float]]:
  predictions = hypothesis.get_predictions(
    sample_data=test_dataset,
    isotope_column_names=isotope_column_names,
    means_isoscapes=means_isoscapes,
    variances_isoscapes=vars_isoscapes,
    sample_size_per_location=5)

  predictions.dropna(subset=['fraud', 'fraud_p_value'], inplace=True)

  y_true = predictions['fraud']
  # Fraud p value is lower the more positive a prediction/label is.
  # Inverting it gives us the probability of positive label class (fraud).
  y_pred = 1 - predictions['fraud_p_value']

  return precision_recall_curve(y_true, y_pred)

def generate_fake_samples(
  start_max_fraud_radius: int, 
  end_max_fraud_radius: int,
  radius_pace: int,
  max_trusted_radius: int,
  min_fraud_radius: int,
  real_samples_data: pd.DataFrame,
  elements: List[str],
  reference_isoscapes: List[raster.AmazonGeoTiff]):
  fake_samples = {}
  for max_radius in range(start_max_fraud_radius, end_max_fraud_radius+1, radius_pace):
    fake_samples[max_radius] = dataset.create_fraudulent_samples(
      real_samples_data,
      reference_isoscapes,
      elements,
      max_trusted_radius,
      max_radius,
      min_fraud_radius)
  return fake_samples

def find_p_value(
    precision: list[float],
    recall: list[float],
    thresholds: list[float],
    precision_target: float,
    recall_target: float) -> list[float]:
  assert(precision_target or recall_target)
  if precision_target:
    target_pos = np.argwhere(precision[:-1] >= precision_target)
  else:
    target_pos = np.argwhere(recall[:-1] >= recall_target)
  # No precision/recall is greater than or equal to the target
  if len(target_pos) < 1:
    if precision_target:
      target_pos = [[np.argmax(precision[:-1])]]
    else:
      target_pos = [[np.argmax(recall[:-1])]]

  precision_target_found = precision[:-1][target_pos[0]]
  recall_target_found = recall[:-1][target_pos[0]]
  p_value_found = (1-thresholds)[target_pos[0]]

  return precision_target_found, recall_target_found, p_value_found

def evaluate_fake_true_mixture(
  dist_to_fake_samples: Dict,
  real: pd.DataFrame,
  mean_isoscapes: List[raster.AmazonGeoTiff],
  var_isoscapes: List[raster.AmazonGeoTiff],
  isotope_column_names: List[str],
  precision_target: float,
  recall_target: float
):
  auc_scores = {}
  p_values_found = {}
  precision_targets_found = {}
  recall_targets_found ={}

  for radius, fake_sample in dist_to_fake_samples.items():
    test_dataset = pd.concat([real, pd.DataFrame(fake_sample)], ignore_index=True)
    test_dataset = dataset.nudge_invalid_coords(
        df=test_dataset,
        rasters=mean_isoscapes + var_isoscapes
    )

    precision, recall, thresholds = isoscape_precision_recall_thresholds(
        test_dataset=test_dataset,
        isotope_column_names=isotope_column_names,
        means_isoscapes=mean_isoscapes,
        vars_isoscapes=var_isoscapes
    )

    auc_score = auc(recall, precision)
    auc_scores[radius] = auc_score

    precision_target_found, recall_target_found, p_value_found = find_p_value(
        precision=precision,
        recall=recall,
        thresholds=thresholds,
        precision_target=precision_target,
        recall_target=recall_target
    )

    p_values_found[radius] = p_value_found[0]
    precision_targets_found[radius] = precision_target_found[0]
    recall_targets_found[radius] = recall_target_found[0]
  return auc_scores, p_values_found, precision_targets_found, recall_targets_found

def evaluate(
  means_isoscape: raster.AmazonGeoTiff,
  vars_isoscape: raster.AmazonGeoTiff,
  original_dataset: pd.DataFrame,
  isotope_column_name: str,
  eval_dataset: pd.DataFrame,
  mean_label: str,
  var_label: str,
  sample_size_per_location: int,
  precision_target: float,
  recall_target: float,
  start_max_fraud_radius: int,
  end_max_fraud_radius: int,
  radius_pace: int,
  max_fraud_radius: int,
  min_trusted_radius: int) -> Dict[str, Any]:
  '''
  Runs a minimal one-sided evaluation pipeline. 
  '''

  # Sanitize
  eval_dataset = eval_dataset.dropna(subset=[var_label])
  mean_predicted_label = mean_label + "_predicted"
  var_predicted_label = var_label + "_predicted"

  # RMSE
  rmse = {}
  rmse['mean_rmse'], rmse['var_rmse'], rmse['overall_rmse'] = calculate_rmse(
      eval_dataset, means_isoscape, vars_isoscape, mean_label, var_label,
      mean_predicted_label, var_predicted_label)

  # Group and set up fake data
  eval_dataset['fraud'] = False
  eval_dataset['cel_count'] = sample_size_per_location
  inferences_df = hypothesis.get_predictions_grouped(
      eval_dataset, [mean_label], [var_label], ['cel_count'],
      [means_isoscape], [vars_isoscape], sample_size_per_location)

  inferences_df.dropna(subset=[var_label, var_predicted_label], inplace=True)

  real_samples_data = pd.merge(
    eval_dataset[['Code','lat','long', mean_label, var_label]],
    original_dataset, how="inner", 
    left_on=['Code', 'lat', 'long'], right_on=['Code', 'lat', 'long'])
  real = real_samples_data[['Code','lat','long'] + [isotope_column_name]]
  real = real.assign(fraud=False)

  dist_to_fake_samples = generate_fake_samples(
    start_max_fraud_radius=start_max_fraud_radius,
    end_max_fraud_radius=end_max_fraud_radius,
    radius_pace=radius_pace,
    max_trusted_radius=max_fraud_radius,
    min_fraud_radius=min_trusted_radius, 
    real_samples_data=real_samples_data,
    elements=[isotope_column_name],
    reference_isoscapes=[means_isoscape, vars_isoscape])
  
  # Test the isoscape against the mixture of real and fake samples. 
  auc_scores, p_values_found, precision_targets_found, recall_targets_found = evaluate_fake_true_mixture(
    dist_to_fake_samples=dist_to_fake_samples, 
    real=real,
    mean_isoscapes=[means_isoscape],
    var_isoscapes=[vars_isoscape],
    isotope_column_names=[isotope_column_name],
    precision_target=precision_target,
    recall_target=recall_target)

  return rmse, auc_scores, p_values_found, precision_targets_found, recall_targets_found