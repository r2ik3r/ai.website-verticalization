import tensorflow as tf

def save_model(model: tf.keras.Model, path: str):
    model.save(path, include_optimizer=True)

def load_model(path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(path)
