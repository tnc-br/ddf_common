from sklearn.metrics import mean_squared_error
import raster
import pandas as pd
import dataset

def get_data_rmse(
  raster: raster.AmazonGeoTiff,
  row: pd.DataFrame):
  try:
    return raster.get_data_at_coords(means_isoscape, row['long'],row['lat'],-1)
  except ValueError:
    return None

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

  df[mean_pred_name] = df.apply(lambda row: get_data_rmse(means_isoscape, row), axis=1)
  df[var_pred_name] = df.apply(lambda row: get_data_rmse(vars_isoscape, row), axis=1)

  df = df.dropna(subset=[mean_pred_name, var_pred_name, mean_true_name, var_true_name])

  predictions = list(df.apply(lambda row: [row[mean_pred_name], row[var_pred_name]], axis=1).values)
  truths = list(df.apply(lambda row: [row[mean_true_name], row[var_true_name]], axis=1).values)

  return (mean_squared_error(df[mean_true_name].values, df[mean_pred_name].values, squared=False),
         mean_squared_error(df[var_true_name].values, df[var_pred_name].values, squared=False),
         mean_squared_error(truths, predictions, squared=False))