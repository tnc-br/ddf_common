from sklearn.compose import ColumnTransformer
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
import joblib
import typing
import pandas as pd
from dataset import ScaledPartitions
from typing import List

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
        return tf.keras.saving.load_model(tf_model_path)
    
    def predict_on_batch(self, X: pd.DataFrame):
        X = self._apply_transformer(X)
        return self.vi_model.predict_on_batch(X)

    def training_column_names(self) -> typing.List[str]:
        return self.transformer.feature_names_in_.tolist()

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

import tensorflow_probability as tfp

# log(σ2/σ1) + ( σ1^2+(μ1−μ2)^2 ) / 2* σ^2   − 1/2
def kl_divergence_helper(real, predicted, num_to_sample=0):
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
    if num_to_sample:
      predicted_value, predicted_std = sample_normal_distribution(
          mean=predicted_value, stdev=predicted_std, n=num_to_sample)

    kl_loss = -0.5 + tf.math.log(predicted_std/real_std) + \
     (tf.square(real_std) + tf.square(real_value - predicted_value))/ \
     (2*tf.square(predicted_std))

    return tf.math.reduce_mean(kl_loss)

def kl_divergence(real, predicted, double_sided, num_to_sample):
  if double_sided:
    return kl_divergence_helper(real, predicted, num_to_sample) + \
     kl_divergence_helper(predicted, real, num_to_sample)
  return kl_divergence_helper(real, predicted, num_to_sample)

def get_early_stopping_callback(patience: int):
  return EarlyStopping(monitor='val_loss', patience=patience, min_delta=0.001,
                       verbose=1, restore_best_weights=True, start_from_epoch=0)

tf.keras.utils.set_random_seed(18731)

# I was experimenting with models that took longer to train, and used this
# checkpointing callback to periodically save the model. It's optional.
def get_checkpoint_callback(model_file):
  return ModelCheckpoint(
      model_file,
      monitor='val_loss', verbose=0, save_best_only=True, mode='min')

def train_or_update_variational_model(
        sp: ScaledPartitions,
        hidden_layers: List[int],
        epochs: int,
        batch_size: int,
        lr: float,
        patience: int,
        double_sided_kl: bool,
        kl_num_samples_from_pred_dist: int,
        model_file=None,
        use_checkpoint=False):
  callbacks_list = [get_early_stopping_callback(patience),
                    get_checkpoint_callback(model_file)]
  if not use_checkpoint:
    inputs = keras.Input(shape=(sp.train.X.shape[1],))
    x = inputs
    for layer_size in hidden_layers:
      x = keras.layers.Dense(
          layer_size, activation='relu', kernel_initializer=glorot_normal)(x)
    mean_output = keras.layers.Dense(
        1, name='mean_output', kernel_initializer=glorot_normal)(x)

    # We can not have negative variance. Apply very little variance.
    var_output = keras.layers.Dense(
        1, name='var_output', kernel_initializer=glorot_normal)(x)

    # Invert the normalization on our outputs
    mean_scaler = sp.label_scaler.named_transformers_['mean_std_scaler']
    untransformed_mean = mean_output * mean_scaler.var_ + mean_scaler.mean_

    var_scaler = sp.label_scaler.named_transformers_['var_minmax_scaler']
    unscaled_var = var_output * var_scaler.scale_ + var_scaler.min_
    untransformed_var = keras.layers.Lambda(lambda t: tf.math.log(1 + tf.exp(t)))(unscaled_var)

    # Output mean,  tuples.
    outputs = keras.layers.concatenate([untransformed_mean, untransformed_var])
    model = keras.Model(inputs=inputs, outputs=outputs)

    optimizer = keras.optimizers.Adam(learning_rate=lr)
    kl_divergence_with_params = partial(
        kl_divergence, double_sided=double_sided_kl, num_to_sample=kl_num_samples_from_pred_dist)
    model.compile(optimizer=optimizer, loss=kl_divergence_with_params)
    model.summary()
  else:
    model = keras.models.load_model(
        get_model_save_location(model_file),
        custom_objects={"kl_divergence": kl_divergence})
  history = model.fit(sp.train.X, sp.train.Y, verbose=1, validation_data=sp.val.as_tuple(), shuffle=True,
                      epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)
  return history, model

  from sklearn.metrics import mean_squared_error

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
    double_sided_kl: bool,
    kl_num_samples_from_pred_dist: int,
    mean_label: str,
    var_label: str,
    patience: int, 
    model_checkpoint: str):
  print("==================")
  print(run_id)
  history, model = train_or_update_variational_model(
    sp, hidden_layers=hidden_layers, epochs=epochs, batch_size=training_batch_size,
    lr=learning_rate, patience=patience, double_sided_kl=double_sided_kl,
    kl_num_samples_from_pred_dist=kl_num_samples_from_pred_dist,
    model_file=model_checkpoint, use_checkpoint=False)
  render_plot_loss(history, run_id+" kl_loss")

  best_epoch_index = history.history['val_loss'].index(min(history.history['val_loss']))
  print('Val loss:', history.history['val_loss'][best_epoch_index])
  print('Train loss:', history.history['loss'][best_epoch_index])
  print('Test loss:', model.evaluate(x=sp.test.X, y=sp.test.Y, verbose=0))

  predictions = model.predict_on_batch(sp.test.X)
  predictions = pd.DataFrame(predictions, columns=[mean_label, var_label])
  rmse = np.sqrt(mean_squared_error(sp.test.Y[mean_label], predictions[mean_label]))
  print("dO18 RMSE: "+ str(rmse))
  return model, rmse

