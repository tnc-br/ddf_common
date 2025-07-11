from evaluation import evaluate_multiple_elements
from dataclasses import dataclass

_OXYGEN_ISOTOPE_LABEL = "d18O_cel"
_NITROGEN_ISOTOPE_LABEL = 'd15N_wood'
_CARBON_ISOTOPE_LABEL = 'd13C_wood'


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
    means_isoscapes.append(oxygen_means_isoscape)
    vars_isoscapes.append(oxygen_vars_isoscape)
    column_names.append(_OXYGEN_ISOTOPE_LABEL)
  # Nitrogen
  if nitrogen_means_isoscape_filename and nitrogen_vars_isoscape_filename:
    means_isoscapes.append(raster.load_raster(
          raster.get_raster_path(nitrogen_means_isoscape_filename), use_only_band_index=0))
    vars_isoscapes.append(raster.load_raster(
          raster.get_raster_path(nitrogen_vars_isoscape_filename), use_only_band_index=0))
    column_names.append(_NITROGEN_ISOTOPE_LABEL)
  # Carbon
  if carbon_isoscape_filename:
    means_isoscapes.append(raster.load_raster(raster.get_raster_path(carbon_isoscape_filename), use_only_band_index=0))
    vars_isoscapes.append(raster.load_raster(raster.get_raster_path(carbon_isoscape_filename), use_only_band_index=1))
    column_names.apppend(_CARBON_ISOTOPE_LABEL)

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