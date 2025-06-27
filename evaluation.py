from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
import raster
import pandas as pd
import dataset
from dataclasses import dataclass, field
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

  return (root_mean_squared_error(df[mean_true_name].values, df[mean_pred_name].values),
         root_mean_squared_error(df[var_true_name].values, df[var_pred_name].values),
         root_mean_squared_error(truths, predictions))

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

def isoscape_roc_auc_score(
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

  return roc_auc_score(y_true, y_pred)

def generate_fake_samples(
  start_max_fraud_radius: int, 
  end_max_fraud_radius: int,
  radius_pace: int,
  trusted_buffer_radius: int,
  real_samples_data: pd.DataFrame,
  elements: List[str],
  reference_isoscapes: List[raster.AmazonGeoTiff],
  fake_sample_drop_rate:float=0.0,
  fake_samples_per_sample:int=1):
  fake_samples = {}
  for max_radius in range(start_max_fraud_radius, end_max_fraud_radius+1, radius_pace):
    fake_samples[max_radius] = dataset.create_fraudulent_samples(
      real_samples_data,
      reference_isoscapes,
      elements,
      max_radius,
      trusted_buffer_radius,
      fake_sample_drop_rate,
      fake_samples_per_sample)
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
) -> EvalResults:
  auc_scores = {}
  p_values_found = {}
  precision_targets_found = {}
  recall_targets_found = {}
  pr_curves = {}
  auc_roc_scores = {}

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

    pr_curves[radius] = {
      "precision": precision.tolist(), 
      "recall": recall.tolist(),
      "thresholds": thresholds.tolist()}

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

    radius_roc_auc_score = isoscape_roc_auc_score(
        test_dataset=test_dataset,
        isotope_column_names=isotope_column_names,
        means_isoscapes=mean_isoscapes,
        vars_isoscapes=var_isoscapes
    )
    auc_roc_scores[radius] = radius_roc_auc_score


  return EvalResults(auc_scores, p_values_found, precision_targets_found, recall_targets_found, pr_curves, auc_roc_scores)

@dataclass
class EvalResults:
  '''
  Container for results from `evaluate`
  '''
  rmse: Dict[str, float]
  auc_scores: Dict[int, float] = field(default_factory=dict)
  p_values_found: Dict[int, float] = field(default_factory=dict)
  precision_targets_found: Dict[int, float] = field(default_factory=dict)
  recall_targets_found: Dict[int, float] = field(default_factory=dict)
  pr_curves: Dict[int, Dict[str, List[float]]] = field(default_factory=dict)
  auc_roc_scores: Dict[int, float] = field(default_factory=dict)

  def convert_to_bq_dict(self):
    bq_dict = self.rmse
    bq_dict['per_radius_eval'] = []
    for radius in self.auc_scores.keys():
      radius_result = {}
      radius_result['radius'] = radius
      radius_result['auc'] = self.auc_scores[radius]
      radius_result['p_value'] = self.p_values_found[radius]
      radius_result['precision_target'] = self.precision_targets_found[radius]
      radius_result['recall_target'] = self.recall_targets_found[radius]
      radius_result['pr_curve'] = self.pr_curves[radius]
      radius_result['auc_roc_score'] = self.auc_roc_scores[radius]
      bq_dict['per_radius_eval'].append(radius_result)
    return bq_dict

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
  trusted_buffer_radius: int,
  fake_sample_drop_rate:float=0.0,
  fake_samples_per_sample:int=1) -> Dict[str, Any]:
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
    trusted_buffer_radius=trusted_buffer_radius, 
    real_samples_data=real_samples_data,
    elements=[isotope_column_name],
    reference_isoscapes=[means_isoscape, vars_isoscape],
    fake_sample_drop_rate=fake_sample_drop_rate,
    fake_samples_per_sample=fake_samples_per_sample)
  
  # Test the isoscape against the mixture of real and fake samples. 
  return evaluate_fake_true_mixture(
    dist_to_fake_samples=dist_to_fake_samples, 
    real=real,
    mean_isoscapes=[means_isoscape],
    var_isoscapes=[vars_isoscape],
    isotope_column_names=[isotope_column_name],
    precision_target=precision_target,
    recall_target=recall_target)

def evaluate_multiple_elements(
  means_isoscapes: List[raster.AmazonGeoTiff],
  vars_isoscapes: List[raster.AmazonGeoTiff],
  original_dataset: pd.DataFrame,
  isotope_column_names: List[str],
  eval_dataset: pd.DataFrame,
  mean_labels: List[str],
  var_labels: List[str],
  count_labels: List[str],
  sample_size_per_location: int,
  precision_target: float,
  recall_target: float,
  start_max_fraud_radius: int,
  end_max_fraud_radius: int,
  radius_pace: int,
  trusted_buffer_radius: int,
  fake_sample_drop_rate:float=0.0,
  fake_samples_per_sample:int=1) -> Dict[str, Any]:
  '''
  Runs a one-sided evaluation pipeline with multiple elements. 
  '''
  assert len(means_isoscapes) == len(vars_isoscapes) == len(isotope_column_names)
  rmse = {}
  eval_dataset = eval_dataset.dropna(subset=var_labels)
  var_predicted_labels = []
  for i in range(len(isotope_column_names)):
    # Sanitize
    mean_predicted_label = mean_labels[i] + "_predicted"
    var_predicted_label = var_labels[i] + "_predicted"
    var_predicted_labels.append(var_predicted_label)

    # RMSE
    isotope_rmse = {}
    isotope_rmse['mean_rmse'], isotope_rmse['var_rmse'], isotope_rmse['overall_rmse'] = calculate_rmse(
        eval_dataset, means_isoscapes[i], vars_isoscapes[i], mean_labels[i], var_labels[i],
        mean_predicted_label, var_predicted_label)
    rmse[isotope_column_names[i]] = isotope_rmse

  # Group and set up fake data
  eval_dataset['fraud'] = False
  eval_dataset['cel_count'] = sample_size_per_location
  inferences_df = hypothesis.get_predictions_grouped(
      eval_dataset, mean_labels, var_labels, count_labels,
      means_isoscapes, vars_isoscapes, sample_size_per_location)

  inferences_df.dropna(subset=var_labels + var_predicted_labels, inplace=True)

  real_samples_data = pd.merge(
    eval_dataset[['Code','lat','long'] + mean_labels + var_labels + count_labels],
    original_dataset, how="inner", 
    left_on=['Code', 'lat', 'long'], right_on=['Code', 'lat', 'long'])
  real = real_samples_data[['Code','lat','long'] + isotope_column_names]
  real = real.assign(fraud=False)

  dist_to_fake_samples = generate_fake_samples(
    start_max_fraud_radius=start_max_fraud_radius,
    end_max_fraud_radius=end_max_fraud_radius,
    radius_pace=radius_pace,
    trusted_buffer_radius=trusted_buffer_radius, 
    real_samples_data=real_samples_data,
    elements=isotope_column_names,
    reference_isoscapes=means_isoscapes + vars_isoscapes,
    fake_sample_drop_rate=fake_sample_drop_rate,
    fake_samples_per_sample=fake_samples_per_sample)
  
  # Test the isoscape against the mixture of real and fake samples. 
  return evaluate_fake_true_mixture(
    dist_to_fake_samples=dist_to_fake_samples, 
    real=real,
    mean_isoscapes=means_isoscapes,
    var_isoscapes=vars_isoscapes,
    isotope_column_names=isotope_column_names,
    precision_target=precision_target,
    recall_target=recall_target)