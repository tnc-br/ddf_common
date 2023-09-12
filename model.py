from sklearn.compose import ColumnTransformer
from abc import abstractmethod
import tensorflow as tf
import joblib
import pandas as pd

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