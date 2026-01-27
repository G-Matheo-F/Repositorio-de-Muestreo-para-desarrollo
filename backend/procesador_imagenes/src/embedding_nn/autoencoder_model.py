from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam


def construir_autoencoder(input_dim, embedding_dim, hidden_dim, lr):
    entrada = Input(shape=(input_dim,))
    x = Dense(hidden_dim, activation="relu")(entrada)
    embedding = Dense(embedding_dim, activation="relu", name="embedding")(x)
    x = Dense(hidden_dim, activation="relu")(embedding)
    salida = Dense(input_dim, activation="sigmoid")(x)

    autoencoder = Model(entrada, salida)
    encoder = Model(entrada, embedding)

    autoencoder.compile(
        optimizer=Adam(learning_rate=lr),
        loss="mse"
    )

    return autoencoder, encoder
