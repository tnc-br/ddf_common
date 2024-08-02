from sklearn.metrics import mean_squared_error
import raster
import pandas as pd
import dataset
from typing import Dict, Any

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

def evaluate(
  means_isoscape: raster.AmazonGeoTiff,
  vars_isoscape: raster.AmazonGeoTiff,
  original_dataset: pd.DataFrame,
  eval_dataset: pd.DataFrame,
  mean_label: str,
  var_label: str,
  sample_size_per_location: int,
  precision_target: float,
  recall_target: float) -> Dict[str, Any]:
  '''
  Runs a minimal one-sided evaluation pipeline. 
  '''

  # Sanitize
  eval_dataset = eval_dataset.dropna(subset=[var_label])

  mean_predicted_label = mean_label + "_predicted"
  var_predicted_label = var_label + "_predicted"

  eval_results = {}

  eval_results['mean_rmse'], eval_results['var_rmse'], eval_results['overall_rmse'] = calculate_rmse(
      eval_dataset, means_isoscape, vars_isoscape, mean_label, var_label,
      mean_predicted_label, var_predicted_label)

  eval_dataset['fraud'] = False
  eval_dataset['cel_count'] = sample_size_per_location
  inferences_df = hypothesis.get_predictions_grouped(
      eval_dataset, [mean_label], [var_label], ['cel_count'],
      [means_isoscape], [vars_isoscape], sample_size_per_location)

  inferences_df.dropna(subset=[var_label, var_predicted_label], inplace=True)
  eval_results['mse'] = mean_squared_error(
      inferences_df[var_label],
      inferences_df[var_predicted_label],
      squared=False)

  # elements = ['d18O_cel', 'd15N_wood', 'd13C_wood']
  # isotope_column_names = ['d18O_cel', 'd15N_wood', 'd13C_wood']
  elements = ['d18O_cel']
  isotope_column_names = ['d18O_cel']

  real_samples_data = pd.merge(
    eval_dataset[['Code','lat','long', mean_label, var_label]],
    original_dataset, how="inner", 
    left_on=['Code', 'lat', 'long'], right_on=['Code', 'lat', 'long'])
  real = real_samples_data[['Code','lat','long'] + elements]
  real = real.assign(fraud=False)
  return eval_results