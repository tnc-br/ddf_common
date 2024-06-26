import model
import dataset
import raster
import generate_isoscape
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

    mean_label: str
    var_label: str
    

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

def train_variational_inference_model(
    params: VIModelTrainingParams, 
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

    vi_model, rmse = model.train(
        data, 
        run_id=params.training_id, 
        epochs=params.num_epochs,
        hidden_layers=[params.num_nodes_per_layer]*params.num_layers,
        training_batch_size=params.training_batch_size,
        learning_rate=params.learning_rate,
        double_sided_kl=params.double_sided_kl,
        kl_num_samples_from_pred_dist=params.kl_num_samples_from_pred_dist,
        mean_label=params.mean_label,
        var_label=params.var_label,
        patience=params.early_stopping_patience,
        model_checkpoint=model_save_location)

    # Package the scaling info and model weights together.
    vi_model.save(f"{model_save_location}.tf", save_format='tf')
    dump(data.feature_scaler, f"{model_save_location}.pkl")
    packaged_model = model.TFModel(f"{model_save_location}.tf", f"{model_save_location}.pkl")


    generate_isoscape.generate_isoscapes_from_variational_model(
        packaged_model, 
        params.resolution_x,
        params.resolution_y,
        isoscape_save_location, 
        amazon_only=False) 
    
    # TODO: Write to BQ