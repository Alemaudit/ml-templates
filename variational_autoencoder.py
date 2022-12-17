# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 15:05:23 2019

@author: aless
"""
import tensorflow as tf
from tensorflow import keras



def get_vae(n_variables: int, neck: int, n_latent: int) -> keras.models.Model:

    # Encoder

    inputs = keras.layers.Input(shape=(n_variables,), name="kpi_input")

    projection = keras.layers.Dense(
        units=neck,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(0.0001),
        name="projection",
    )(inputs)

    z_mean = keras.layers.Dense(
        units=n_latent,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(0.0001),
        name="means",
    )(projection)

    z_log_sigma = keras.layers.Dense(
        units=n_latent,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(0.0001),
        name="log_stds",
    )(projection)

    def vae_loss(x, y):
        xent_loss = keras.losses.binary_crossentropy(x, y)
        kl_loss = -0.5 * tf.math.reduce_mean(
            1 + z_log_sigma - tf.math.square(z_mean) - tf.math.exp(z_log_sigma), axis=-1
        )
        return xent_loss + kl_loss

    # Sampler

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = tf.random.normal(shape=(n_latent,))
        return z_mean + tf.math.exp(z_log_sigma) * epsilon


    gauss = keras.layers.Lambda(sampling, output_shape=(n_latent,), name="sampler")(
        [z_mean, z_log_sigma]
    )

    # Decoder

    inv_projection = keras.layers.Dense(
        units=neck,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(0.0001),
        name="inv_projection",
    )(gauss)

    output = keras.layers.Dense(units=n_variables, activation="linear", name="output")(
        inv_projection
    )
    vae = keras.models.Model(inputs=inputs, outputs=output, name="ovl_vae")
    encoder = keras.models.Model(inputs, z_mean)
    return vae, encoder, vae_loss


if __name__ == "__main__":
    import numpy as np

    n_vars = 10

    X = np.random.random((n_vars, 10000))

    vae, encoder, vae_loss = get_vae(n_variables=n_vars, neck=3, n_latent=5)

    vae.compile(
        optimizer=keras.optimizers.RMSprop(lr=0.001, epsilon=1e-8),
        loss=vae_loss,
        metrics=["mean_squared_error"],
    )

    TB = keras.callbacks.TensorBoard(
        log_dir="./.logs",
        histogram_freq=0,
        batch_size=X.shape[0],
        write_graph=True,
        write_grads=False,
        write_images=False,
        embeddings_freq=0,
        embeddings_layer_names=None,
        embeddings_metadata=None,
        embeddings_data=None,
        update_freq="epoch",
    )

    ES = keras.callbacks.EarlyStopping(
        monitor="mean_squared_error",
        min_delta=0,
        patience=0,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
    )

    vae_history = vae.fit(
        X, X, epochs=200, batch_size=200, verbose=0, callbacks=[TB, ES]
    )
