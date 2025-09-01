# src/verticalizer/models/kerasmultilabel.py (replace file)

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

def _focal_bce(gamma: float = 2.0):
    bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    def loss(y_true, y_pred):
        b = bce(y_true, y_pred)
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        w = tf.pow(1.0 - p_t, gamma)
        return tf.reduce_mean(w * b)
    return loss

def build_model(embdim: int, numlabels: int, hidden: int = 512, dropout: float = 0.3, labels_loss: str = "bce", gamma: float = 2.0):
    inp = keras.Input(shape=(embdim,), name="emb")
    x = layers.Dense(hidden, activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(hidden * 2, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    ylabels = layers.Dense(numlabels, activation="sigmoid", name="labels")(x)
    yscores = layers.Dense(numlabels, activation="sigmoid", name="scores")(x)
    model = keras.Model(inp, [ylabels, yscores])
    if labels_loss == "focal":
        loss_labels = _focal_bce(gamma=gamma)
    else:
        loss_labels = keras.losses.BinaryCrossentropy()
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss={"labels": loss_labels, "scores": keras.losses.MeanSquaredError()},
        loss_weights={"labels": 1.0, "scores": 0.3},
        metrics={"labels": [keras.metrics.AUC(name="auc"), keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")]},
    )
    return model

def to_bin_vector(scores_int: int, bins: int = 10):
    import numpy as np
    y = np.zeros((len(scores_int), bins), dtype="float32")
    for i, s in enumerate(scores_int):
        s = max(1, min(bins, int(s)))
        y[i, s - 1] = 1.0
    return y