import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_model(emb_dim: int, num_labels: int, hidden: int = 512, dropout: float = 0.3):
    inp = keras.Input(shape=(emb_dim,), name="emb")
    x = layers.Dense(hidden, activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(hidden//2, activation="relu")(x)
    x = layers.Dropout(dropout)(x)

    y_labels = layers.Dense(num_labels, activation="sigmoid", name="labels")(x)
    y_scores = layers.Dense(num_labels, activation="sigmoid", name="scores")(x)  # scores in 0–1

    model = keras.Model(inp, [y_labels, y_scores])
    losses = {
        "labels": keras.losses.BinaryCrossentropy(),
        "scores": keras.losses.MeanSquaredError()
    }
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=losses,
                  metrics={"labels": [keras.metrics.AUC(name="auc"),
                                      keras.metrics.Precision(name="precision"),
                                      keras.metrics.Recall(name="recall")]})
    return model


def to_bin_vector(scores_int, bins: int = 10):
    import numpy as np
    y = np.zeros((len(scores_int), bins), dtype="float32")
    for i, s in enumerate(scores_int):
        s = max(1, min(bins, int(s)))
        y[i, s - 1] = 1.0
    return y
