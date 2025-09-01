# src/verticalizer/models/keras_multilabel.py
from tensorflow import keras
from tensorflow.keras import layers

def build_model(emb_dim: int, num_labels: int, hidden: int = 512, dropout: float = 0.3):
    """
    Build a multi-label classification + per-vertical score regression model.
    - labels head: sigmoid(num_labels), trained with BCE
    - scores head: sigmoid(num_labels), trained with MSE, mapped to 1..10
    """
    inp = keras.Input(shape=(emb_dim,), name="emb")
    x = layers.Dense(hidden, activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(hidden // 2, activation="relu")(x)
    x = layers.Dropout(dropout)(x)

    y_labels = layers.Dense(num_labels, activation="sigmoid", name="labels")(x)
    y_scores = layers.Dense(num_labels, activation="sigmoid", name="scores")(x)

    model = keras.Model(inp, [y_labels, y_scores])
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss={
            "labels": keras.losses.BinaryCrossentropy(),
            "scores": keras.losses.MeanSquaredError()
        },
        # Prioritise classification so model learns to separate categories before fine-tuning scores
        loss_weights={"labels": 1.0, "scores": 0.3},
        metrics={
            "labels": [
                keras.metrics.AUC(name="auc"),
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall")
            ]
        }
    )
    return model


def to_bin_vector(scores_int, bins: int = 10):
    import numpy as np
    y = np.zeros((len(scores_int), bins), dtype="float32")
    for i, s in enumerate(scores_int):
        s = max(1, min(bins, int(s)))
        y[i, s - 1] = 1.0
    return y
