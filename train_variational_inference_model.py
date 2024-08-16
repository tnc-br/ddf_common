import model
import dataset
import raster
import generate_isoscape
import evaluation
from dataclasses import dataclass
from typing import List, Dict
from joblib import dump

# Container for parameters for training VI model
@dataclass
class VIModelTrainingParams:
    training_id: str
    num_epochs: int
    num_layers: int
    num_nodes_per_layer: int
    training_batch_size: int
    learning_rate: float
    dropout_rate: float

    mean_label: str
    var_label: str

    # E.g. relu, linear, or one of these:
    # https://www.tensorflow.org/api_docs/python/tf/keras/activations
    #
    # No sanitation is done on this param.
    activation_func: str
    
    # Wait this many epochs without loss improvement before stopping training
    early_stopping_patience: int

    # If true, loss = KL(real, predicted) + KL(predicted, real), else
    # it's just KL(real, predicted)
    double_sided_kl: bool

    # If 0, compute loss by comparing distributions directly
    kl_num_samples_from_pred_dist: int

    # Features to standardize.
    features_to_standardize: List[str]

    # Unscaled, unnormallized raw feature data.
    features_to_passthrough: List[str]

    resolution_x: int
    resolution_y: int

@dataclass 
class VIModelEvalParams:
    samples_per_location: int
    precision_target: float
    recall_target: float
    start_max_fraud_radius: int
    end_max_fraud_radius: int
    radius_pace: int
    max_fraud_radius: int
    min_trusted_radius: int
    elements_to_eval: List[str]

def train_variational_inference_model(
    params: VIModelTrainingParams, 
    eval_params: VIModelEvalParams,
    files: Dict,
    isoscape_save_location: str,
    model_save_location: str):

    # Columns not found in the training data, but their corresponding value have
    # strong signals.
    potentially_extra_columns = [
        "brisoscape_mean_ISORIX",
        "d13C_cel_mean",
        "d13C_cel_var",
        "ordinary_kriging_linear_d18O_predicted_mean",
        "ordinary_kriging_linear_d18O_predicted_variance",
    ]

    # Load the geotiff it the params request it.
    extra_columns_from_geotiffs = {}
    for feature in params.features_to_passthrough + params.features_to_standardize:
        if feature in potentially_extra_columns:
            extra_columns_from_geotiffs[feature] = raster.column_name_to_geotiff_fn[feature]()

    data = dataset.load_and_scale(
        files, 
        params.mean_label, 
        params.var_label, 
        params.features_to_passthrough, 
        [],
        params.features_to_standardize, 
        extra_columns_from_geotiffs)

    # vi_model, rmse = model.train(
    #     data, 
    #     run_id=params.training_id, 
    #     epochs=params.num_epochs,
    #     hidden_layers=[params.num_nodes_per_layer]*params.num_layers,
    #     training_batch_size=params.training_batch_size,
    #     learning_rate=params.learning_rate,
    #     dropout_rate=params.dropout_rate,
    #     double_sided_kl=params.double_sided_kl,
    #     kl_num_samples_from_pred_dist=params.kl_num_samples_from_pred_dist,
    #     mean_label=params.mean_label,
    #     var_label=params.var_label,
    #     activation_func=params.activation_func,
    #     patience=params.early_stopping_patience,
    #     model_checkpoint=model_save_location)

    # # Package the scaling info and model weights together.
    # vi_model.save(model_save_location)
    # dump(data.feature_scaler, f"{model_save_location.strip('.keras')}.pkl")
    # packaged_model = model.TFModel(model_save_location, f"{model_save_location.strip('.keras')}.pkl")

    # generate_isoscape.generate_isoscapes_from_variational_model(
    #     packaged_model, 
    #     params.resolution_x,
    #     params.resolution_y,
    #     isoscape_save_location, 
    #     amazon_only=False) 


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
        params.mean_label,
        params.var_label,
        eval_params.samples_per_location,
        eval_params.precision_target,
        eval_params.recall_target,
        eval_params.start_max_fraud_radius,
        eval_params.end_max_fraud_radius,
        eval_params.radius_pace,
        eval_params.max_fraud_radius,
        eval_params.min_trusted_radius)
        
    return eval_results
    # TODO: Write to BQ