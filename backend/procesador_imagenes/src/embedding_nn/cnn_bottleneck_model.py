from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D,
    Flatten, Dense, Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LeakyReLU


def construir_cnn_bottleneck(
    image_size,
    embedding_dim,
    lr
):
    inputs = Input(shape=(*image_size, 1))

    x = Conv2D(32, (3,3), activation="relu", padding="same")(inputs)
    x = MaxPooling2D()(x)

    x = Conv2D(64, (3,3), activation="relu", padding="same")(x)
    x = MaxPooling2D()(x)

    x = Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x = MaxPooling2D()(x)

    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)

    #BOTTLENECK (EMBEDDING)
    x = Dense(embedding_dim, name="embedding")(x)
    embedding = LeakyReLU(alpha=0.1)(x)

    output = Dense(1, activation="sigmoid")(embedding)

    model = Model(inputs, output)

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # modelo SOLO para embeddings
    embedding_model = Model(inputs, embedding)

    return model, embedding_model
