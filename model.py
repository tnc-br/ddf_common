from sklearn.compose import ColumnTransformer
import tensorflow as tf
import joblib
import pandas as pd

class Model:
    def __init__(self) -> None:
        pass
    
    def predict_on_batch(X: pd.DataFrame):
        pass

class TFModel(Model):
    def __init__(self, transformer_path: str, tf_model_path: str) -> None:
        Model.__init__()
        self.vi_model =  load_tf_model(tf_model_path)
        self.transformer = apply_transformer(transformer_path)

    def load_transformer(transformer_path: str) -> ColumnTransformer:
        return joblib.load(transformer_path)
    
    def apply_transformer(X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.transformer.transform(X),
                            index=X.index, columns=X.columns)
    
    def load_tf_model(tf_model_path: str) -> tf.keras.Model:
        return tf.keras.saving.load_model(tf_model_path)
    
    def predict_on_batch(X: DataFrame):
        return self.vi_model.predict_on_batch(X)