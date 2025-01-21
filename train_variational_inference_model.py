import model
import dataset
import raster
import generate_isoscape
import evaluation
import bqddf
from dataclasses import dataclass, field
from typing import List, Dict, Any
from joblib import dump
import pandas as pd
from google.api_core.exceptions import GoogleAPIError

# Container for parameters for training VI model

class VIModelTrainingParams:
    def __init__(
        self,
        training_id: str,
        num_epochs: int,
        num_layers: int,
        num_nodes_per_layer: int,
        training_batch_size: int,
        learning_rate: float,
        dropout_rate: float,
        mean_label: str,
        var_label: str,
        # E.g. relu, linear, or one of these:
        # https://www.tensorflow.org/api_docs/python/tf/keras/activations
        #
        # No sanitation is done on this param.
        activation_func: str,
        
        # Wait this many epochs without loss improvement before stopping training
        early_stopping_patience: int,

        # If true, loss = KL(real, predicted) + KL(predicted, real), else
        # it's just KL(real, predicted)
        double_sided_kl: bool,

        # If 0, compute loss by comparing distributions directly
        kl_num_samples_from_pred_dist: int,

        # Features to standardize.
        features_to_standardize: List[str],

        # Unscaled, unnormallized raw feature data.
        features_to_passthrough: List[str],

        # Number of cross validation folds. If 0, no cross validation is disabled. If
        # >0, epochs becomes epochs per fold. 
        n_cv_folds: int,

        resolution_x: int,
        resolution_y: int,

        # Arbitrary tags passed in by experimenter. 
        tags: List[str],
        **kwargs
    ):
        self.training_id = training_id
        self.num_epochs = num_epochs
        self.num_layers = num_layers
        self.num_nodes_per_layer = num_nodes_per_layer
        self.training_batch_size = training_batch_size
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.mean_label = mean_label
        self.var_label = var_label
        self.activation_func = activation_func
        self.early_stopping_patience = early_stopping_patience
        self.double_sided_kl = double_sided_kl
        self.kl_num_samples_from_pred_dist = kl_num_samples_from_pred_dist
        self.features_to_standardize = features_to_standardize
        self.features_to_passthrough = features_to_passthrough
        self.n_cv_folds = n_cv_folds
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.tags = tags
        self.additional_params = {key: value for key, value in kwargs.items()}

    def convert_to_bq_dict(self):
      as_dict = {
        'training_id': self.training_id,
        'num_epochs': self.num_epochs,
        'num_layers': self.num_layers,
        'num_nodes_per_layer': self.num_nodes_per_layer,
        'training_batch_size': self.training_batch_size,
        'learning_rate': self.learning_rate,
        'dropout_rate': self.dropout_rate,
        'activation_func': self.activation_func,
        'early_stopping_patience': self.early_stopping_patience,
        'double_sided_kl': self.double_sided_kl,
        'kl_num_samples_from_pred_dist': self.kl_num_samples_from_pred_dist,
        'features': self.features_to_standardize + self.features_to_passthrough,
        'resolution_x': self.resolution_x,
        'resolution_y': self.resolution_y,
        'tags': self.tags
      }
      plus_adl_params = as_dict.copy()
      plus_adl_params['as_json'] = as_dict | self.additional_params
      return plus_adl_params

@dataclass 
class VIModelEvalParams:
    # Number of isoscape ratio measurements for sample. Should be 5.
    samples_per_location: int

    # Generates a PR curve, and uses the p_value at precision_target
    # for evaluation. Can not be used at the same time as recall_target.
    precision_target: float

    # Like precision target but for recall. Can not be used at the same time
    # as precision target.
    recall_target: float

    # Run evaluation generating fake samples at various radii from a real sample.
    # Run the first eval at `start_max_fraud_radius`, and increment by `radius pace`
    # and run it again until `end_max_fraud_radius` is reached. 
    start_max_fraud_radius: int
    end_max_fraud_radius: int
    radius_pace: int

    # Forbid generating fake samples this close to a real sample.  
    trusted_buffer_radius: int

    # Which elements in the eval dataset to test for. 
    elements_to_eval: List[str]

    def convert_to_bq_dict(self):
      return {
        'samples_per_location': self.samples_per_location,
        'precision_target': self.precision_target,
        'recall_target': self.recall_target,
        'start_max_fraud_radius': self.start_max_fraud_radius,
        'end_max_fraud_radius': self.end_max_fraud_radius,
        'radius_pace': self.radius_pace,
        'trusted_buffer_radius': self.trusted_buffer_radius,
        'elements_to_eval': self.elements_to_eval,
      }

def check_training_run_exists(training_id: str):
  exists = bqddf.get_training_result_from_flattened(training_id).total_rows
  if exists:
    raise GoogleAPIError(f"training_id {training_id} already exists. " +
                          "Choose a different training_id (overwrites not supported).")

def train_variational_inference_model(
    training_params: VIModelTrainingParams, 
    eval_params: VIModelEvalParams,
    files: Dict,
    isoscape_save_location: str,
    model_save_location: str,
    eval_only: bool):

    if not eval_only:
      # Crash if a run with this training_id already happened.
      check_training_run_exists(training_params.training_id)
      
      # Columns not found in the training data, but their corresponding value have
      # strong signals.
      potentially_extra_columns = [
          "brisoscape_mean_ISORIX",
          "d13C_cel_mean",
          "d13C_cel_var",
          "ordinary_kriging_linear_d18O_predicted_mean",
          "ordinary_kriging_linear_d18O_predicted_variance",
      ]

      #Load the geotiff it the training_params request it.
      extra_columns_from_geotiffs = {}
      for feature in training_params.features_to_passthrough + training_params.features_to_standardize:
          if feature in potentially_extra_columns:
              extra_columns_from_geotiffs[feature] = raster.column_name_to_geotiff_fn[feature]()

      data = dataset.load_and_scale(
          files, 
          training_params.mean_label, 
          training_params.var_label, 
          training_params.features_to_passthrough, 
          [],
          training_params.features_to_standardize, 
          extra_columns_from_geotiffs)

      vi_model, rmse = model.train(
          data, 
          run_id=training_params.training_id, 
          epochs=training_params.num_epochs,
          hidden_layers=[training_params.num_nodes_per_layer]*training_params.num_layers,
          training_batch_size=training_params.training_batch_size,
          learning_rate=training_params.learning_rate,
          dropout_rate=training_params.dropout_rate,
          double_sided_kl=training_params.double_sided_kl,
          kl_num_samples_from_pred_dist=training_params.kl_num_samples_from_pred_dist,
          mean_label=training_params.mean_label,
          var_label=training_params.var_label,
          activation_func=training_params.activation_func,
          patience=training_params.early_stopping_patience,
          n_cv_folds=training_params.n_cv_folds,
          model_checkpoint=model_save_location)

      # Package the scaling info and model weights together.
      vi_model.save(model_save_location)
      dump(data.feature_scaler, f"{model_save_location.strip('.keras')}.pkl")
      packaged_model = model.TFModel(model_save_location, f"{model_save_location.strip('.keras')}.pkl")

      generate_isoscape.generate_isoscapes_from_variational_model(
          packaged_model, 
          training_params.resolution_x,
          training_params.resolution_y,
          isoscape_save_location, 
          amazon_only=False) 

    # Evaluation setup
    means_isoscape = raster.load_raster(isoscape_save_location, use_only_band_index=0)
    vars_isoscape = raster.load_raster(isoscape_save_location, use_only_band_index=1)

    eval_dataset = pd.read_csv(files['EVAL'], index_col=0)
    original_dataset = pd.read_csv(files['ORIGINAL'], index_col=0)
    
    eval_results = evaluation.evaluate(
        means_isoscape,
        vars_isoscape,
        original_dataset,
        eval_params.elements_to_eval,
        eval_dataset,
        training_params.mean_label,
        training_params.var_label,
        eval_params.samples_per_location,
        eval_params.precision_target,
        eval_params.recall_target,
        eval_params.start_max_fraud_radius,
        eval_params.end_max_fraud_radius,
        eval_params.radius_pace,
        eval_params.trusted_buffer_radius)

    training_run = training_params.convert_to_bq_dict()
    training_run['dataset_id'] = files['TRAIN']

    eval_metadata = eval_params.convert_to_bq_dict()
    eval_metadata |= eval_results.convert_to_bq_dict()

    bqddf.insert_harness_run(training_run, eval_metadata)
        
    return eval_results
    