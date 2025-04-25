from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from abc import abstractmethod
from functools import partial
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from matplotlib import pyplot as plt
from tensorflow.python.ops import math_ops
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.initializers import glorot_normal
import tensorflow_probability as tfp
import keras
import tensorflow_probability as tfp
import joblib
import typing
import logging
import pandas as pd
import numpy as np
from dataset import ScaledPartitions
from typing import List
from typing import Callable
from copy import deepcopy
from joblib import dump
import statistics

tf.keras.utils.set_random_seed(18731)

class Model:
    '''
    Abstract class representing a model, trained in any way.
    '''

    @abstractmethod
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def predict_on_batch(self, X: pd.DataFrame):
        '''
        Makes a prediction on a collection of inputs.
        '''
        pass
    
    @abstractmethod
    def training_column_names(self) -> typing.List[str]:
        '''
        List of named columns used to train this model, in order of training, if applicable.
        '''
        pass

    @abstractmethod
    def fit(self, X_train, Y_train, validation_data, **kwargs):
        '''
        Fit the model using the kwargs as configuration. 
        '''
    @abstractmethod
    def save_model(self, filename, **kwargs):
        '''
        Save the model.
        '''

class TFModel(Model):
    '''
    A TF trained model with scaled input training data. The scalers
    are stored in `transformer` and are applied to input before predictions.
    '''
    def __init__(self, tf_model_path: str, transformer_path: str) -> None:
        Model.__init__(self)
        self.vi_model =  self._load_tf_model(tf_model_path)
        self.transformer = self._load_transformer(transformer_path)

    def _load_transformer(self, transformer_path: str) -> ColumnTransformer:
        return joblib.load(transformer_path)
    
    def _apply_transformer(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.transformer.transform(X),
                            index=X.index, columns=X.columns)
    
    def _load_tf_model(self, tf_model_path: str) -> tf.keras.Model:
        return tf.keras.models.load_model(tf_model_path)
    
    def predict_on_batch(self, X: pd.DataFrame):
        return self.vi_model.predict_on_batch(X)

    def training_column_names(self) -> typing.List[str]:
        return self.transformer.feature_names_in_.tolist()

    def fit(self, X_train, Y_train, validation_data, **kwargs):
        return self.vi_model.fit(X_train, Y_train, validation_data=validation_data, **kwargs)

    def save_model(self, filename, **kwargs):
        assert '.keras' in filename 
        self.vi_model.save(filename)
        kwargs.pop('feature_scaler')
        


def sample_normal_distribution(
    mean: tf.Tensor,
    stdev: tf.Tensor,
    n: int) -> tf.Tensor:
    '''
    Given a batch of normal distributions described by a mean and stdev in
    a tf.Tensor, sample n elements from each distribution and return the mean
    and standard deviation per sample.
    '''
    batch_size = tf.shape(mean)[0]

    # Output tensor is (n, batch_size, 1)
    sample_values = tfp.distributions.Normal(
        loc=mean,
        scale=stdev).sample(
            sample_shape=n)
    # Reshaped tensor will be (batch_size, n)
    sample_values = tf.transpose(sample_values)
    # Get the mean per sample in the batch.
    sample_mean = tf.transpose(tf.math.reduce_mean(sample_values, 2))
    sample_stdev = tf.transpose(tf.math.reduce_std(sample_values, 2))

    return sample_mean, sample_stdev


# log(σ2/σ1) + ( σ1^2+(μ1−μ2)^2 ) / 2* σ^2   − 1/2
@keras.saving.register_keras_serializable(package="Custom", name="KLCustomLoss")
class KLCustomLoss:
    def __init__(self, double_sided, num_to_sample):
        self.double_sided = double_sided
        self.num_to_sample = num_to_sample

    def kl_divergence(self, real, predicted):
        '''
        real: tf.Tensor of the real mean and standard deviation of sample to compare
        predicted: tf.Tensor of the predicted mean and standard deviation to compare
        sample: Whether or not to sample the predicted distribution to get a new
                mean and standard deviation.
        '''
        if real.shape != predicted.shape:
            raise ValueError(
                f"real.shape {real.shape} != predicted.shape {predicted.shape}")

        real_value = tf.gather(real, [0], axis=1)
        real_std = tf.math.sqrt(tf.gather(real, [1], axis=1))


        predicted_value = tf.gather(predicted, [0], axis=1)
        predicted_std = tf.math.sqrt(tf.gather(predicted, [1], axis=1))
        # If num_to_sample>0, sample from the distribution defined by the predicted mean
        # and standard deviation to use for mean and stdev used in KL divergence loss.
        if self.num_to_sample is not None:
            predicted_value, predicted_std = sample_normal_distribution(
                mean=predicted_value, stdev=predicted_std, n=self.num_to_sample)

        kl_loss = -0.5 + tf.math.log(predicted_std/real_std) + \
        (tf.square(real_std) + tf.square(real_value - predicted_value))/ \
        (2*tf.square(predicted_std))

        return tf.math.reduce_mean(kl_loss)
    
    def __call__(self, real, predicted):
        return self.kl_divergence(real, predicted) + (self.kl_divergence(predicted, real) if self.double_sided else 0.0)

    def get_config(self):
        return {
          "double_sided": self.double_sided.numpy().item(),
          "num_to_sample": self.num_to_sample.numpy().item()}

    @classmethod
    def from_config(cls, config):
        # Convert config values back to TensorFlow tensors if needed
        config["double_sided"] = tf.constant(config["double_sided"], dtype=tf.bool)
        config["num_to_sample"] = tf.constant(config["num_to_sample"], dtype=tf.int32) 
        return cls(**config)  # Create an instance using the config



def get_early_stopping_callback(patience: int, min_steps: int):
  return EarlyStopping(monitor='val_loss', patience=patience, min_delta=0.001,
                       verbose=1, restore_best_weights=True, start_from_epoch=min_steps)

# I was experimenting with models that took longer to train, and used this
# checkpointing callback to periodically save the model. It's optional.
def get_checkpoint_callback(model_file):
  return ModelCheckpoint(
      model_file,
      monitor='val_loss', verbose=0, save_best_only=True, mode='min')

# You could use:
@tf.keras.utils.register_keras_serializable()
class SoftplusLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.math.log(1 + tf.exp(inputs))

def cross_val_with_best_model(
    build_model_fn: Callable[[], Model],
    evaluate_fn: Callable[[Model, pd.DataFrame, pd.DataFrame], float],
    sp: ScaledPartitions,
    n_cv_folds: int,
    output_model_filename: str=None,
    **fit_kwargs) -> typing.Tuple[typing.Any, Model]:
    '''
        Runs cross validation on the model, returning the history and fitted best model. 
        build_model_fn: Function that builds, compiles, and returns a model.
        evaluate_fn: Function for scoring and comparing models across different folds.
        sp: Stands for "Scaled Partitions", representing your dataset. This function only looks at the "train" 
            partition and uses this to create validation sets.
        n_cv_folds: Number of cross validation folds. These folds are sources from the "train" partition.
        evaluate_fn.

        Returns the fitted model and any training artifacts generated by calling fit().

    '''

    kf = KFold(n_splits=n_cv_folds)
    predictions_per_fold = {}
    loss_per_fold = {}

    for fold, (train_index, val_index) in enumerate(kf.split(sp.train.X)):
        print(f"Training fold #{fold} ||| (train_index_start: {train_index}, val_index_start: {val_index})")
        X_train, X_val = sp.train.X.loc[train_index], sp.train.X.loc[val_index]
        Y_train, Y_val = sp.train.Y.loc[train_index], sp.train.Y.loc[val_index]

        model = build_model_fn()
        model.fit(
            X_train=X_train, Y_train=Y_train,
            validation_data=(X_val, Y_val), 
            **fit_kwargs)

        predictions = model.predict_on_batch(X_val)
        predictions_per_fold[fold] = pd.DataFrame(predictions, index=X_val.index)
        fold_mean_rmse, fold_var_rmse = np.sqrt(
            mean_squared_error(Y_val, predictions, multioutput='raw_values'))
        print(f'''mean_rmse for fold #{fold}: {fold_mean_rmse}''')
        print(f'''var_rmse for fold #{fold}: {fold_var_rmse}''')

    # Concatenate the predictions in the dictionary vertically, compare to whole
    # dataset to get rmse.
    all_predictions = pd.concat(predictions_per_fold.values(), axis=0)
    mean_rmse, var_rmse = np.sqrt(
        mean_squared_error(sp.train.Y, all_predictions, multioutput='raw_values'))

    cv_artifacts = {
      'mean_rmse': mean_rmse, 
      'var_rmse': var_rmse
    }

    # Return the model trained on all data.
    print("Training on all data")
    final_model = build_model_fn()
    training_artifacts = final_model.fit(
        X_train=X_train, Y_train=Y_train,
        validation_data=(X_train, Y_train),
         **fit_kwargs)

    return training_artifacts, final_model, cv_artifacts


def train_or_update_variational_model(
        sp: ScaledPartitions,
        hidden_layers: List[int],
        epochs: int,
        batch_size: int,
        lr: float,
        dropout_rate: float,
        patience: int,
        double_sided_kl: bool,
        kl_num_samples_from_pred_dist: int,
        activation_func: str,
        n_cv_folds: int,
        min_steps: int,
        model_file=None,
        use_checkpoint=False) -> Model:
    callbacks_list = [get_early_stopping_callback(patience, min_steps),
                      get_checkpoint_callback(model_file)]
    def build_model():
        tf.keras.utils.set_random_seed(18731)
        if not use_checkpoint:
            inputs = keras.Input(shape=(sp.train.X.shape[1],))
            x = inputs
            for layer_size in hidden_layers:
                x = keras.layers.Dense(
                    layer_size, activation=activation_func, kernel_initializer=glorot_normal)(x)
                if dropout_rate > 0:
                    x = keras.layers.Dropout(rate=dropout_rate)(x)

            mean_output = keras.layers.Dense(
                1, name='mean_output', kernel_initializer=glorot_normal)(x)

            # We can not have negative variance. Apply very little variance.
            var_output = keras.layers.Dense(
                1, name='var_output', kernel_initializer=glorot_normal)(x)

            # Invert the normalization on our outputs
            mean_scaler = sp.label_scaler.named_transformers_['mean_std_scaler']
            extracted_mean = np.float32(mean_scaler.mean_)
            extracted_var_for_scaling_mean = np.float32(mean_scaler.var_)
            final_mean_output = keras.layers.Lambda(
                lambda x: x * mean_scaler.var_ + mean_scaler.mean_,
                name='final_mean_output' 
            )(mean_output)

            var_scaler = sp.label_scaler.named_transformers_['var_minmax_scaler']
            unscaled_var = var_output * var_scaler.scale_ + var_scaler.min_
            final_variance_output = SoftplusLayer(name='final_variance_output')(unscaled_var)

            # Output mean, tuples.
            model = keras.Model(inputs=inputs, outputs=[final_mean_output, final_variance_output])

            optimizer = keras.optimizers.Adam(learning_rate=lr)
            model.compile( 
                optimizer=optimizer, 
                loss={
                    'final_mean_output': 'mean_squared_error',  
                    'final_variance_output': 'mean_squared_error' 
                },
                metrics={
                    'final_mean_output': tf.keras.metrics.RootMeanSquaredError(name='rmse_mean'),
                    'final_variance_output': tf.keras.metrics.RootMeanSquaredError(name='rmse_var')
                }
            )
            model.summary()
        else:
            model = keras.models.load_model(
                model_file,
                custom_objects={"KLCustomLoss": KLCustomLoss})
        model.save(model_file)
        dump(sp.feature_scaler, f"{model_file.strip('.keras')}.pkl")
        packaged_model = TFModel(model_file, f"{model_file.strip('.keras')}.pkl")
        return packaged_model
    
    if sp.val: 
        # No cross validation
        if n_cv_folds:
            raise ValueError("Cross validation (n_cv_folds>0) not possible if manually specifying a validation dataset.")
        
        # Pass in validation set and build model in a single run.
        model = build_model()
        history = model.fit(sp.train.X, sp.train.Y, verbose=1, validation_data=sp.val.as_tuple(), shuffle=True,
                            epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)
        return history, model.vi_model, None
    else:
        # Yes cross validation
        if not n_cv_folds:
          raise ValueError("If no validation dataset is specified, number of cross validation folds (n_cv_folds) must be set.")
        
        # Cross val function takes abstract model, so pass in fit params
        # as kwargs and custom evaluation function.
        fit_kwargs = {'epochs': epochs,
                      'batch_size': batch_size,
                      'callbacks': callbacks_list,
                      'shuffle': True}
        evaluate_fn = lambda tf_model, X, Y: tf_model.vi_model.evaluate(X, Y) 
        history, packaged_model, cv_results = cross_val_with_best_model(
            build_model, evaluate_fn, sp, n_cv_folds, **fit_kwargs)
        return history, packaged_model.vi_model, cv_results

def render_plot_loss(history, name):
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title(name + ' model loss')
  plt.ylabel('loss')
  plt.yscale("log")
  plt.ylim((0, 10))
  plt.xlabel('epoch')
  plt.legend(['loss', 'val_loss'], loc='upper left')
  plt.show()

def train(
    sp: ScaledPartitions,
    run_id: str, 
    epochs: int,
    hidden_layers: List[int], 
    training_batch_size: int,
    learning_rate: float,
    dropout_rate: float,
    double_sided_kl: bool,
    kl_num_samples_from_pred_dist: int,
    activation_func: str,
    mean_label: str,
    var_label: str,
    patience: int,
    min_steps: int,
    n_cv_folds: int, 
    model_checkpoint: str):
  print("==================")
  print(run_id)
  rmse = None
  history, model, maybe_cv_results = train_or_update_variational_model(
    sp, hidden_layers=hidden_layers, epochs=epochs, batch_size=training_batch_size,
    lr=learning_rate, dropout_rate=dropout_rate, patience=patience, double_sided_kl=double_sided_kl,
    kl_num_samples_from_pred_dist=kl_num_samples_from_pred_dist, activation_func=activation_func,
    min_steps=min_steps, n_cv_folds=n_cv_folds, model_file=model_checkpoint, use_checkpoint=False)
  
  if maybe_cv_results:
    print('Avg mean RMSE across folds: ', maybe_cv_results['mean_rmse'])
    print('Avg var RMSE across folds:', maybe_cv_results['var_rmse'])
    rmse = maybe_cv_results

  render_plot_loss(history, run_id+" kl_loss")
  best_epoch_index = history.history['val_loss'].index(min(history.history['val_loss']))
  print('Val loss:', history.history['val_loss'][best_epoch_index])
  print('Train loss:', history.history['loss'][best_epoch_index])

  if sp.test:
    print('Test loss:', model.evaluate(x=sp.test.X, y=sp.test.Y, verbose=0))  
    predictions = model.predict_on_batch(sp.test.X)
    predictions = pd.DataFrame(predictions, columns=[mean_label, var_label])
    rmse = {
      'mean_rmse': np.sqrt(mean_squared_error(sp.test.Y[mean_label], predictions[mean_label])),
      'var_rmse': np.sqrt(mean_squared_error(sp.test.Y[var_label], predictions[var_label])),
    }
    print("dO18 RMSE: "+ str(rmse))
  return model, rmse