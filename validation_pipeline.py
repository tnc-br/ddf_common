from evaluation import evaluate_multiple_elements
from dataclasses import dataclass
from datetime import datetime

import os
import raster
import pandas as pd

_OXYGEN_ISOTOPE_LABEL = "d18O_cel"
_NITROGEN_ISOTOPE_LABEL = 'd15N_wood'
_CARBON_ISOTOPE_LABEL = 'd13C_wood'

def get_oxygen_isoscapes(oxygen_means_isoscape_filename: str,
                         oxygen_vars_isoscape_filename: str):
  if oxygen_means_isoscape_filename == oxygen_vars_isoscape_filename:
      oxygen_means_isoscape = raster.load_raster(
          raster.get_raster_path(oxygen_means_isoscape_filename), use_only_band_index=0)
      oxygen_vars_isoscape = raster.load_raster(
          raster.get_raster_path(oxygen_means_isoscape_filename), use_only_band_index=1)
    else:
      oxygen_means_isoscape = raster.load_raster(
        raster.get_raster_path(oxygen_means_isoscape_filename), use_only_band_index=0)
      oxygen_vars_isoscape = raster.load_raster(
        raster.get_raster_path(oxygen_vars_isoscape_filename), use_only_band_index=0)
  return oxygen_means_isoscape, oxygen_vars_isoscape

def get_nitrogen_isoscapes(
  nitrogen_means_isoscape_filename: str,
  nitrogen_vars_isoscape_filename: str
):
  nitrogen_means_isoscape = raster.load_raster(
          raster.get_raster_path(nitrogen_means_isoscape_filename), use_only_band_index=0)
  nitrogen_vars_isoscape = raster.load_raster(
          raster.get_raster_path(nitrogen_vars_isoscape_filename), use_only_band_index=0)
  return nitrogen_means_isoscape, nitrogen_vars_isoscape

def get_carbon_isoscapes(
  carbon_means_isoscape_filename: str,
  carbon_vars_isoscape_filename: str
):
  carbon_means_isoscape = raster.load_raster(raster.get_raster_path(carbon_isoscape_filename), use_only_band_index=0)
  carbon_vars_isoscape = raster.load_raster(raster.get_raster_path(carbon_isoscape_filename), use_only_band_index=1)
  return carbon_means_isoscape, carbon_vars_isoscape

@dataclass
class FraudGenerationParams:
  start_max_fraud_radius: float
  end_max_fraud_radius: float
  radius_pace: float
  trusted_buffer_radius: float
  fake_sample_drop_rate: float
  fake_samples_per_sample: int

def validation_pipeline(
    source_dataset_filename: str,
    test_set_filename: str,
    fraud_generation_params: FraudGenerationParams,
    precision_target: float = 0.95,
    recall_target: float = None,
    oxygen_means_isoscape_filename: str = None,
    oxygen_vars_isoscape_filename:str = None,
    carbon_isoscape_filename: str = None,
    nitrogen_means_isoscape_filename: str = None,
    nitrogen_vars_isoscape_filename: str = None):
  assert(oxygen_means_isoscape_filename or
         oxygen_vars_isoscape_filename or
         carbon_isoscape_filename or
         nitrogen_means_isoscape_filename or
         nitrogen_vars_isoscape_filename)
  assert(source_dataset_filename)
  assert(precision_target or recall_target)

  # Load Carbon, Oxygen and Nitrogen Isoscapes if specified
  means_isoscapes = []
  vars_isoscapes = []
  column_names = []
  # Oxygen
  if oxygen_means_isoscape_filename and oxygen_vars_isoscape_filename:
    oxygen_means_isoscape, oxygen_vars_isoscape = get_oxygen_isoscapes(oxygen_means_isoscape_filename,
      oxygen_vars_isoscape_filename)
    means_isoscapes.append(oxygen_means_isoscape)
    vars_isoscapes.append(oxygen_vars_isoscape)
    column_names.append(_OXYGEN_ISOTOPE_LABEL)
  # Nitrogen
  if nitrogen_means_isoscape_filename and nitrogen_vars_isoscape_filename:
    nitrogen_means_isoscape, nitrogen_vars_isoscape = get_nitrogen_isoscapes(nitrogen_means_isoscape_filename,
      nitrogen_vars_isoscape_filename)
    means_isoscapes.append(nitrogen_means_isoscape)
    vars_isoscapes.append(nitrogen_vars_isoscape)
    column_names.append(_NITROGEN_ISOTOPE_LABEL)
  # Carbon
  if carbon_isoscape_filename:
    carbon_means_isoscape, carbon_vars_isoscape = get_carbon_isoscapes(carbon_means_isoscape_filename,
      carbon_vars_isoscape_filename)
    means_isoscapes.append(carbon_means_isoscape)
    vars_isoscapes.append(carbon_vars_isoscape)
    column_names.append(_CARBON_ISOTOPE_LABEL)

  # Load Original Dataset
  original_dataset = pd.read_csv(raster.get_sample_db_path(source_dataset_filename), index_col=0)
  # Load test dataset
  eval_dataset = pd.read_csv(raster.get_sample_db_path(test_set_filename), index_col=0)

  results = evaluate_multiple_elements(
    means_isoscapes=means_isoscapes,
    vars_isoscapes=vars_isoscapes,
    original_dataset=original_dataset,
    isotope_column_names=column_names,
    eval_dataset=eval_dataset,
    mean_labels=[f'{i}_mean' for i in column_names],
    var_labels=[f'{i}_variance' for i in column_names],
    count_labels=[f'{i}_count' for i in column_names],
    sample_size_per_location=5,
    precision_target=precision_target,
    recall_target=recall_target,
    start_max_fraud_radius=fraud_generation_params.start_max_fraud_radius,
    end_max_fraud_radius=fraud_generation_params.end_max_fraud_radius,
    radius_pace=fraud_generation_params.radius_pace,
    trusted_buffer_radius=fraud_generation_params.trusted_buffer_radius,
    fake_sample_drop_rate=fraud_generation_params.fake_sample_drop_rate,
    fake_samples_per_sample=fraud_generation_params.fake_samples_per_sample
  )

  return results

def stamp(filename:str, auc_scores, p_values_found, precisions_target_found, recalls_target_found, end_max_fraud_radius):
  """
    Adds precision, recall, and p-value thresholds to isoscape metadata for every radius tested in the validation pipeline.
    Stamping isoscapes:
    1. p-value threshold where precision = 95% (considered the last radius in the loop to stamp)
    2. the recall at that level
    3. the AUC
    4. the parameters used for validation (% fraud and radius)
    5. the date.time of validation
    Input:
      filename : str
      GeoTIFF filename (with the full path) of the isoscape to be stamped
      e.g: /content/gdrive/Shared drives/TNC Fellowship 🌳/4. Isotope Research & Signals/code/amazon_rainforest_files/amazon_rasters/variational/ensemble_with_carbon_brisoisorix/fixed_isorix_carbon_ensemble.tiff
  """

  for radius in auc_scores.keys():
    #p-value threshold where precision = precision_target_found
    raster.stamp_isoscape(filename, "P_VALUE_THRESHOLD_"+str(radius),  p_values_found[radius])
    raster.stamp_isoscape(filename, "PRECISION_"+str(radius), precisions_target_found[radius])
    raster.stamp_isoscape(filename, "RECALL_"+str(radius), recalls_target_found[radius])
    raster.stamp_isoscape(filename, "AUC_"+str(radius), auc_scores[radius])

    if radius == end_max_fraud_radius:
      raster.stamp_isoscape(filename, "P_VALUE_THRESHOLD",  p_values_found[radius])
      raster.stamp_isoscape(filename, "PRECISION", precisions_target_found[radius])
      raster.stamp_isoscape(filename, "RECALL", recalls_target_found[radius])
      raster.stamp_isoscape(filename, "AUC", auc_scores[radius])

  #The date/time of validation
  now = datetime.now()
  dt_string = now.strftime("%m/%d/%Y %H:%M:%S")
  metadata_name = "DATE_TIME"
  metadata_value = dt_string
  raster.stamp_isoscape(filename, metadata_name, metadata_value)

  isoscape_filename =  os.path.basename(filename).strip(".tiff")
  raster.stamp_isoscape(filename, "REFERENCE_ISOSCAPE_NAME", isoscape_filename)